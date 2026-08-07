#!/usr/bin/env python3
"""Dense tile training for the pinned author MambaHSI implementation (M4)."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.dataset import FullImageDataset, reflect_indices
from src.io_mat import load_label
from src.metrics import metrics_from_confusion
from src.models import MambaHSI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def class_weights(label_path: Path, num_classes: int) -> torch.Tensor:
    labels, _ = load_label(label_path, "train_label")
    counts = np.bincount(labels[labels > 0], minlength=num_classes + 1)[1:].astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def tile_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= overlap:
        raise ValueError("tile_size must be larger than tile_overlap")
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, tile_size - overlap))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(tile_size: int) -> np.ndarray:
    if tile_size == 1:
        return np.ones((1, 1), dtype=np.float32)
    axis = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
    return axis[:, None] * axis[None, :]


@torch.inference_mode()
def evaluate_tiled(
    model: MambaHSI,
    cube: np.ndarray,
    labels: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, object], float]:
    """Overlap-blended whole-scene validation on official val pixels only."""
    model.eval()
    height, width = labels.shape
    logits_sum = np.zeros((height, width, model.num_classes), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    window = blend_window(tile_size)
    started = time.perf_counter()
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype))
        if amp
        else nullcontext
    )
    for row0 in tile_starts(height, tile_size, overlap):
        real_h = min(tile_size, height - row0)
        rows = reflect_indices(np.arange(row0, row0 + tile_size), height)
        for col0 in tile_starts(width, tile_size, overlap):
            real_w = min(tile_size, width - col0)
            cols = reflect_indices(np.arange(col0, col0 + tile_size), width)
            block = np.asarray(cube[np.ix_(rows, cols)], dtype=np.float32)
            tensor = torch.from_numpy(block.transpose(2, 0, 1)[None]).to(
                device, non_blocking=True
            )
            with amp_context():
                logits = model(tensor)
            tile_logits = logits[0, :, :real_h, :real_w].permute(1, 2, 0).float().cpu().numpy()
            tile_weight = window[:real_h, :real_w]
            logits_sum[row0 : row0 + real_h, col0 : col0 + real_w] += (
                tile_logits * tile_weight[..., None]
            )
            weight_sum[row0 : row0 + real_h, col0 : col0 + real_w] += tile_weight
    if np.any(weight_sum <= 0):
        raise RuntimeError("Dense validation left uncovered pixels")
    prediction = np.argmax(logits_sum / weight_sum[..., None], axis=2).astype(np.int64)
    truth = labels.astype(np.int64) - 1
    mask = labels > 0
    encoded = truth[mask] * model.num_classes + prediction[mask]
    confusion = np.bincount(encoded, minlength=model.num_classes**2).reshape(
        model.num_classes, model.num_classes
    )
    return metrics_from_confusion(confusion), time.perf_counter() - started


def focal_dense(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None,
    gamma: float = 2.0,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not torch.any(valid):
        raise RuntimeError("Training tile contains no supervised pixels")
    # CUDA nll_loss2d has no deterministic implementation. Selecting the
    # official supervised pixels first gives the mathematically identical
    # [N,C] loss while retaining strict deterministic algorithms.
    selected_logits = logits.permute(0, 2, 3, 1)[valid]
    selected_targets = targets[valid]
    losses = nn.functional.cross_entropy(
        selected_logits, selected_targets, weight=weight, reduction="none"
    )
    pt = torch.exp(-losses)
    return (((1.0 - pt) ** gamma) * losses).mean()


def dense_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not torch.any(valid):
        raise RuntimeError("Training tile contains no supervised pixels")
    selected_logits = logits.permute(0, 2, 3, 1)[valid]
    return nn.functional.cross_entropy(selected_logits, targets[valid], weight=weight)


def append_summary(path: Path, row: dict[str, object]) -> None:
    fields = [
        "model", "seed", "status", "best_epoch", "val_oa", "val_aa",
        "val_kappa", "parameters", "training_seconds",
        "validation_inference_seconds", "config_changes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def train_m4(config: dict[str, object]) -> None:
    rank, world_size, local_rank, device = setup_distributed()
    seed = int(config["seed"])
    seed_everything(seed, bool(config.get("deterministic", True)))
    if int(config.get("batch_size", 1)) != 1:
        raise ValueError("The pinned author MambaHSI requires batch_size=1")

    output_dir = resolve_path(config["output_root"]) / f"seed{seed}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        import yaml
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    cube_path = resolve_path(config["data"]["cache_cube"])
    train_label_path = resolve_path(config["data"]["train_label"])
    val_label_path = resolve_path(config["data"]["val_label"])
    if not cube_path.exists():
        raise FileNotFoundError(f"Missing {cube_path}; generate standardized 98-band cache first")
    dataset = FullImageDataset(
        cube_path, train_label_path, "train_label", int(config["tile_size"]),
        augment=bool(config.get("train_augment", True)), seed=seed,
        ignore_index=IGNORE_INDEX,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=int(config.get("num_workers", 0)), pin_memory=True,
    )
    params = config.get("mambahsi", {})
    model = MambaHSI(
        in_channels=int(config["bands"]), num_classes=int(config["num_classes"]),
        hidden_dim=int(params.get("hidden_dim", 128)),
        mamba_type=str(params.get("mamba_type", "both")),
        token_num=int(params.get("token_num", 4)),
        group_num=int(params.get("group_num", 4)),
        use_residual=bool(params.get("use_residual", True)),
        use_att=bool(params.get("use_att", True)),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    ddp_model: nn.Module = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank])

    weights = None
    if config.get("class_weight") == "sqrt_inverse_frequency":
        weights = class_weights(train_label_path, model.num_classes).to(device)
    optimizer = torch.optim.Adam(
        ddp_model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    use_amp = bool(config.get("amp", True))
    amp_dtype_name = str(config.get("amp_dtype", "float16")).lower()
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError(f"Unsupported amp_dtype {amp_dtype_name}")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    val_labels, _ = load_label(val_label_path, "val_label")
    val_cube = np.load(cube_path, mmap_mode="r") if rank == 0 else None

    best_oa = -1.0
    best_epoch = -1
    best_metrics: dict[str, object] | None = None
    best_infer_seconds = 0.0
    stale_epochs = 0
    training_start = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        ddp_model.train()
        loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
        for tiles, targets in loader:
            tiles = tiles.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = ddp_model(tiles)
                if config.get("focal_loss", False):
                    loss = focal_dense(logits, targets, weights)
                else:
                    loss = dense_cross_entropy(logits, targets, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            supervised = (targets != IGNORE_INDEX).sum()
            loss_sum[0] += loss.detach().double() * supervised
            loss_sum[1] += supervised
        if world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.barrier()
        train_loss = float((loss_sum[0] / loss_sum[1]).item())

        stop = torch.zeros(1, dtype=torch.int32, device=device)
        if rank == 0:
            assert val_cube is not None
            metrics, infer_seconds = evaluate_tiled(
                model, val_cube, val_labels, int(config["tile_size"]),
                int(config["tile_overlap"]), device, use_amp,
                amp_dtype,
            )
            oa = float(metrics["oa"])
            print(
                f"epoch={epoch + 1} loss={train_loss:.6f} val_OA={oa:.6f} "
                f"val_AA={float(metrics['aa']):.6f} "
                f"val_Kappa={float(metrics['kappa']):.6f} infer_s={infer_seconds:.2f}",
                flush=True,
            )
            assert writer is not None
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("val/OA", oa, epoch + 1)
            writer.add_scalar("val/AA", float(metrics["aa"]), epoch + 1)
            writer.add_scalar("val/Kappa", float(metrics["kappa"]), epoch + 1)
            for class_index, accuracy in enumerate(metrics["per_class_accuracy"], 1):
                writer.add_scalar(f"val/class_{class_index}_accuracy", accuracy, epoch + 1)
            if oa > best_oa:
                best_oa, best_epoch = oa, epoch + 1
                best_metrics, best_infer_seconds = metrics, infer_seconds
                stale_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": best_epoch,
                     "metrics": metrics, "config": config},
                    output_dir / "best.pt",
                )
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )
            else:
                stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                stop[0] = 1
        if world_size > 1:
            dist.broadcast(stop, src=0)
        if int(stop.item()):
            break

    training_seconds = time.perf_counter() - training_start
    if rank == 0:
        assert best_metrics is not None and writer is not None
        writer.close()
        append_summary(
            PROJECT_ROOT / "outputs" / "summary.csv",
            {
                "model": f"{config['model']}_{config['model_name']}", "seed": seed,
                "status": "complete", "best_epoch": best_epoch,
                "val_oa": best_metrics["oa"], "val_aa": best_metrics["aa"],
                "val_kappa": best_metrics["kappa"], "parameters": parameter_count,
                "training_seconds": training_seconds,
                "validation_inference_seconds": best_infer_seconds,
                "config_changes": (
                    f"tile_size={config['tile_size']};tile_overlap={config['tile_overlap']};"
                    f"amp_dtype={amp_dtype_name}"
                ),
            },
        )
        print(f"m4 complete: best val OA={best_oa:.6f}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
