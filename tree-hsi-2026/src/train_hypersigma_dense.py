#!/usr/bin/env python3
"""Dense dual-view HyperSIGMA fine-tuning with spatial refit protocols."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import reflect_indices
from src.io_mat import load_label
from src.metrics import metrics_from_confusion
from src.models import HyperSIGMADense


IGNORE_INDEX = -100


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def spatial_holdout(labels: np.ndarray, block_size: int, fraction: float, seed: int) -> np.ndarray:
    rows, cols = np.nonzero(labels > 0)
    block_ids = np.unique((rows // block_size) * 100_000 + cols // block_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(block_ids)
    chosen = set(block_ids[: max(1, round(len(block_ids) * fraction))].tolist())
    # A pure global block draw can omit rare species. Add one whole spatial
    # block for every missing class; never fall back to adjacent random pixels.
    for cls in range(1, 18):
        class_rows, class_cols = np.nonzero(labels == cls)
        class_blocks = np.unique(
            (class_rows // block_size) * 100_000 + class_cols // block_size
        )
        if not any(int(block) in chosen for block in class_blocks):
            chosen.add(int(class_blocks[int(rng.integers(0, len(class_blocks)))]))
    grid_r, grid_c = np.indices(labels.shape)
    encoded = (grid_r // block_size) * 100_000 + grid_c // block_size
    mask = np.isin(encoded, np.fromiter(chosen, dtype=np.int64)) & (labels > 0)
    present = set(np.unique(labels[mask]).tolist())
    missing = set(range(1, 18)).difference(present)
    if missing:
        raise RuntimeError(f"Spatial holdout construction failed for {sorted(missing)}")
    return mask


def build_supervision(config: dict[str, object]) -> tuple[np.ndarray, np.ndarray | None, str]:
    data = config["data"]
    train, _ = load_label(resolve_path(data["train_label"]), "train_label")
    val, _ = load_label(resolve_path(data["val_label"]), "val_label")
    mode = str(config.get("training_mode", "strict"))
    if mode == "strict":
        return train, val, "official train -> official val"
    combined = np.where(train > 0, train, val).astype(np.uint8)
    if mode == "refit90":
        holdout = spatial_holdout(
            val,
            int(config.get("holdout_block_size", 128)),
            float(config.get("holdout_fraction", 0.1)),
            int(config["seed"]),
        )
        supervision = combined.copy()
        supervision[holdout] = 0
        validation = np.where(holdout, val, 0).astype(np.uint8)
        return supervision, validation, "train + spatial 90% val -> spatial 10% val"
    if mode == "refit_all":
        return combined, None, "train + all val; fixed-epoch final refit"
    raise ValueError(f"Unsupported training_mode={mode}")


class DualViewTileDataset(Dataset):
    def __init__(
        self,
        spatial_path: Path,
        spectral_path: Path,
        labels: np.ndarray,
        tile_size: int,
        samples_per_epoch: int,
        seed: int,
        augment: bool,
    ) -> None:
        self.spatial = np.load(spatial_path, mmap_mode="r")
        self.spectral = np.load(spectral_path, mmap_mode="r")
        self.labels = labels
        self.tile_size = tile_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.augment = augment
        self.epoch = 0
        self.by_class = [np.argwhere(labels == cls).astype(np.int32) for cls in range(1, 18)]
        if any(len(coords) == 0 for coords in self.by_class):
            raise ValueError("Every class must have at least one supervised pixel")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(
            self.seed * 1_000_003 + self.epoch * 10_007 + index
        )
        cls_index = index % 17
        coords = self.by_class[cls_index]
        anchor_row, anchor_col = coords[int(rng.integers(0, len(coords)))]
        offset_r = int(rng.integers(0, self.tile_size)) if self.augment else self.tile_size // 2
        offset_c = int(rng.integers(0, self.tile_size)) if self.augment else self.tile_size // 2
        row0, col0 = int(anchor_row) - offset_r, int(anchor_col) - offset_c
        src_rows = np.arange(row0, row0 + self.tile_size)
        src_cols = np.arange(col0, col0 + self.tile_size)
        rows = reflect_indices(src_rows, self.labels.shape[0])
        cols = reflect_indices(src_cols, self.labels.shape[1])
        spatial = np.asarray(self.spatial[np.ix_(rows, cols)], dtype=np.float32)
        spectral = np.asarray(self.spectral[np.ix_(rows, cols)], dtype=np.float32)
        target = np.full((self.tile_size, self.tile_size), IGNORE_INDEX, np.int64)
        valid_r = (src_rows >= 0) & (src_rows < self.labels.shape[0])
        valid_c = (src_cols >= 0) & (src_cols < self.labels.shape[1])
        block = self.labels[np.ix_(src_rows[valid_r], src_cols[valid_c])]
        encoded = block.astype(np.int64) - 1
        target[np.ix_(valid_r, valid_c)] = np.where(
            block > 0, encoded, IGNORE_INDEX
        )

        if self.augment:
            gain = float(np.exp(rng.normal(0.0, 0.12)))
            shadow = float(rng.uniform(0.55, 0.9)) if rng.random() < 0.25 else 1.0
            knots = rng.normal(0.0, 0.05, size=8)
            curve = np.interp(np.linspace(0, 7, 98), np.arange(8), knots)
            spectral = spectral * (gain * shadow) * np.exp(curve)[None, None, :]
            spectral += rng.normal(0.0, 0.01, size=spectral.shape).astype(np.float32)
            spatial = spatial * (gain * shadow)
            if rng.random() < 0.15:
                start = int(rng.integers(0, 94))
                spectral[:, :, start : start + int(rng.integers(1, 5))] = 0.0
            turns = int(rng.integers(0, 4))
            spatial = np.rot90(spatial, turns, axes=(0, 1))
            spectral = np.rot90(spectral, turns, axes=(0, 1))
            target = np.rot90(target, turns, axes=(0, 1))
            if rng.random() < 0.5:
                spatial, spectral, target = spatial[::-1], spectral[::-1], target[::-1]
            if rng.random() < 0.5:
                spatial, spectral, target = spatial[:, ::-1], spectral[:, ::-1], target[:, ::-1]
        return (
            torch.from_numpy(np.ascontiguousarray(spatial.astype(np.float32).transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(spectral.astype(np.float32).transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(target)),
        )


def class_weights(labels: np.ndarray, power: float = 0.5) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"class_weight_power must be in [0, 1], got {power}")
    counts = np.bincount(labels[labels > 0], minlength=18)[1:].astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def parameter_groups(model: HyperSIGMADense, config: dict[str, object]) -> list[dict[str, object]]:
    hs = config["hypersigma"]
    groups: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_encoder = ".spat_encoder." in name or ".spec_encoder." in name
        depth = model.encoder_depth(name)
        if not is_encoder:
            kind, base_lr = "head", float(hs["head_lr"])
        elif any(token in name for token in ("patch_embed", "pos_embed", "spat_map", "spec_embed")):
            kind, base_lr = "embedding", float(hs["embedding_lr"])
        else:
            kind = "block" if depth is not None else "encoder_other"
            decay_power = 11 - (depth if depth is not None else 11)
            base_lr = float(hs["backbone_lr"]) * float(hs["layer_decay"]) ** decay_power
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or "pos_embed" in name
        groups.append({
            "params": [parameter], "lr": base_lr,
            "base_lr": base_lr, "kind": kind, "depth": depth,
            "weight_decay": 0.0 if no_decay else float(config["weight_decay"]),
        })
    return groups


def set_epoch_lrs(
    optimizer: torch.optim.Optimizer, epoch: int, epochs: int, config: dict[str, object]
) -> None:
    hs = config["hypersigma"]
    warm = int(hs.get("head_warmup_epochs", 5))
    partial = int(hs.get("top4_until_epoch", 15))
    warmup_steps = max(1, round(epochs * float(hs.get("warmup_ratio", 0.05))))
    if epoch < warmup_steps:
        schedule = (epoch + 1) / warmup_steps
    else:
        progress = (epoch - warmup_steps) / max(1, epochs - warmup_steps)
        schedule = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        active = True
        if epoch < warm and group["kind"] in {"block", "encoder_other"}:
            active = False
        elif epoch < partial and group["kind"] == "block" and (group["depth"] or 0) < 8:
            active = False
        group["lr"] = float(group["base_lr"]) * schedule if active else 0.0


def tile_origins(shape: tuple[int, int], tile: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(0, shape[0], tile)
        for col in range(0, shape[1], tile)
    ]


@torch.inference_mode()
def evaluate(
    model: HyperSIGMADense,
    spatial: np.ndarray,
    spectral: np.ndarray,
    labels: np.ndarray,
    tile_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_tiles: int | None,
    seed: int,
) -> tuple[dict[str, object], float]:
    model.eval()
    origins = tile_origins(labels.shape, tile_size)
    origins = [
        origin for origin in origins
        if np.any(labels[origin[0] : origin[0] + tile_size, origin[1] : origin[1] + tile_size] > 0)
    ]
    if max_tiles is not None and len(origins) > max_tiles:
        rng = np.random.default_rng(seed)
        origins = [origins[i] for i in np.sort(rng.choice(len(origins), max_tiles, replace=False))]
    confusion = np.zeros((17, 17), dtype=np.int64)
    started = time.perf_counter()
    for row0, col0 in origins:
        real_h = min(tile_size, labels.shape[0] - row0)
        real_w = min(tile_size, labels.shape[1] - col0)
        rows = reflect_indices(np.arange(row0, row0 + tile_size), labels.shape[0])
        cols = reflect_indices(np.arange(col0, col0 + tile_size), labels.shape[1])
        x_spat = torch.from_numpy(np.asarray(spatial[np.ix_(rows, cols)], np.float32).transpose(2, 0, 1)[None]).to(device)
        x_spec = torch.from_numpy(np.asarray(spectral[np.ix_(rows, cols)], np.float32).transpose(2, 0, 1)[None]).to(device)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(x_spat, x_spec)
        prediction = logits.argmax(1)[0, :real_h, :real_w].cpu().numpy()
        target = labels[row0 : row0 + real_h, col0 : col0 + real_w]
        valid = target > 0
        encoded = (target[valid].astype(np.int64) - 1) * 17 + prediction[valid]
        confusion += np.bincount(encoded, minlength=289).reshape(17, 17)
    return metrics_from_confusion(confusion), time.perf_counter() - started


def append_summary(row: dict[str, object]) -> None:
    path = PROJECT_ROOT / "outputs/summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if empty:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank, world, local_rank, device = setup_distributed()
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    supervision, validation, protocol = build_supervision(config)
    output_dir = resolve_path(config["output_root"]) / f"seed{seed}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Protocol: {protocol}; train pixels={int((supervision > 0).sum())}", flush=True)
    if world > 1:
        dist.barrier()

    data = config["data"]
    spatial_path = resolve_path(data["spatial_cache"])
    spectral_path = resolve_path(data["spectral_cache"])
    dataset = DualViewTileDataset(
        spatial_path, spectral_path, supervision,
        int(config["tile_size"]), int(config["samples_per_epoch"]), seed,
        bool(config.get("train_augment", True)),
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), sampler=sampler,
        num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
    )
    hs = config["hypersigma"]
    model = HyperSIGMADense(
        spatial_channels=30, spectral_channels=98,
        image_size=int(config["tile_size"]), num_classes=17,
        spatial_patch_size=int(hs["spatial_patch_size"]),
        spatial_checkpoint=resolve_path(hs["spatial_checkpoint"]),
        spectral_checkpoint=resolve_path(hs["spectral_checkpoint"]),
    ).to(device)
    if rank == 0:
        print(f"HyperSIGMA weights: {model.pretrained_report}", flush=True)
    optimizer = torch.optim.AdamW(parameter_groups(model, config), betas=(0.9, 0.999))
    ddp: nn.Module = model
    if world > 1:
        ddp = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    weights = (
        class_weights(
            supervision,
            power=float(config.get("class_weight_power", 0.5)),
        ).to(device)
        if config.get("class_weight")
        else None
    )
    if rank == 0:
        if weights is None:
            print("Class weighting: disabled", flush=True)
        else:
            print(
                f"Class weighting: inverse-frequency power="
                f"{float(config.get('class_weight_power', 0.5)):.3f}; "
                f"weights={weights.cpu().tolist()}",
                flush=True,
            )
    amp_dtype = torch.bfloat16 if config.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    val_spatial = np.load(spatial_path, mmap_mode="r") if rank == 0 and validation is not None else None
    val_spectral = np.load(spectral_path, mmap_mode="r") if rank == 0 and validation is not None else None

    epochs = int(config["epochs"])
    if str(config.get("training_mode")) == "refit_all" and config.get("epochs_from_checkpoint"):
        selection = torch.load(
            resolve_path(config["epochs_from_checkpoint"]),
            map_location="cpu", weights_only=False,
        )
        epochs = int(selection["epoch"])
        if rank == 0:
            print(f"Final refit epochs inherited from selection checkpoint: {epochs}", flush=True)
    best_oa, best_epoch, stale = -1.0, 0, 0
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        set_epoch_lrs(optimizer, epoch, epochs, config)
        ddp.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for x_spat, x_spec, target in loader:
            x_spat, x_spec, target = x_spat.to(device), x_spec.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = ddp(x_spat, x_spec)
                valid = target != IGNORE_INDEX
                loss = nn.functional.cross_entropy(
                    logits.permute(0, 2, 3, 1)[valid], target[valid],
                    weight=weights, label_smoothing=float(config.get("label_smoothing", 0.0)),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            totals[0] += loss.detach().double() * valid.sum()
            totals[1] += valid.sum()
        if world > 1:
            dist.all_reduce(totals)
        train_loss = float(totals[0] / totals[1])
        should_validate = validation is not None and (
            (epoch + 1) % int(config.get("validate_every", 2)) == 0 or epoch + 1 == epochs
        )
        stop = torch.zeros(1, dtype=torch.uint8, device=device)
        if rank == 0:
            assert writer is not None
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            if should_validate:
                metrics, infer_s = evaluate(
                    model, val_spatial, val_spectral, validation,
                    int(config["tile_size"]), device, amp_dtype,
                    int(config.get("monitor_tiles", 128)), seed,
                )
                oa = float(metrics["oa"])
                writer.add_scalar("val/OA", oa, epoch + 1)
                writer.add_scalar("val/AA", float(metrics["aa"]), epoch + 1)
                print(
                    f"epoch={epoch + 1} loss={train_loss:.6f} monitor_OA={oa:.6f} "
                    f"monitor_AA={float(metrics['aa']):.6f} infer_s={infer_s:.1f}", flush=True
                )
                if oa > best_oa:
                    best_oa, best_epoch, best_metrics, stale = oa, epoch + 1, metrics, 0
                    torch.save({"model": model.state_dict(), "epoch": best_epoch, "config": config}, output_dir / "best.pt")
                else:
                    stale += 1
                if stale >= int(config["patience"]):
                    stop[0] = 1
            else:
                print(f"epoch={epoch + 1} loss={train_loss:.6f}", flush=True)
        if world > 1:
            dist.broadcast(stop, src=0)
        if int(stop.item()):
            break

    if rank == 0:
        assert writer is not None
        if validation is None:
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "config": config}, output_dir / "best.pt")
        else:
            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            best_metrics, full_s = evaluate(
                model, val_spatial, val_spectral, validation,
                int(config["tile_size"]), device, amp_dtype, None, seed,
            )
            (output_dir / "full_val_metrics.json").write_text(json.dumps(best_metrics, indent=2))
            print(
                f"full_val best_epoch={best_epoch} OA={best_metrics['oa']:.6f} "
                f"AA={best_metrics['aa']:.6f} Kappa={best_metrics['kappa']:.6f} infer_s={full_s:.1f}", flush=True
            )
        writer.close()
        row = {
            "model": "m5_hypersigma_dense_max", "seed": seed, "status": "complete",
            "best_epoch": best_epoch,
            "val_oa": "" if best_metrics is None else best_metrics["oa"],
            "val_aa": "" if best_metrics is None else best_metrics["aa"],
            "val_kappa": "" if best_metrics is None else best_metrics["kappa"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": time.perf_counter() - started,
            "validation_inference_seconds": "",
            "config_changes": protocol,
        }
        append_summary(row)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
