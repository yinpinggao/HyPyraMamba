#!/usr/bin/env python3
"""Unified author-source training entry point for models M1-M5."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PatchDataset, reflect_indices
from src.io_mat import load_label
from src.metrics import metrics_from_confusion
from src.models import GAHT, HybridSN, HyperSIGMA, SSFTT


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        # Full M5 validation covers 690k expensive ViT patches. When GPUs are
        # shared, ranks can reach the final metric reduction more than the
        # NCCL default 10 minutes apart; that is load imbalance, not a hang.
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(hours=2),
        )
    return rank, world_size, local_rank, device


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def class_weights(label_path: Path, label_key: str, num_classes: int) -> torch.Tensor:
    labels, _ = load_label(label_path, label_key)
    counts = np.bincount(labels[labels > 0], minlength=num_classes + 1)[1:].astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None,
    gamma: float = 2.0,
) -> torch.Tensor:
    losses = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-losses)
    return (((1.0 - pt) ** gamma) * losses).mean()


def warmup_cosine_multiplier(
    step: int, *, warmup_steps: int, total_steps: int
) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))


@torch.inference_mode()
def evaluate_dense(
    model: HybridSN,
    cube: np.ndarray,
    labels: np.ndarray,
    tile_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, object], float]:
    model.eval()
    confusion = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    halo = model.halo
    height, width = labels.shape
    start_time = time.perf_counter()
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp
        else nullcontext()
    )
    for row0 in range(0, height, tile_size):
        row1 = min(row0 + tile_size, height)
        rows = reflect_indices(np.arange(row0 - halo, row1 + halo), height)
        for col0 in range(0, width, tile_size):
            col1 = min(col0 + tile_size, width)
            cols = reflect_indices(np.arange(col0 - halo, col1 + halo), width)
            block = np.asarray(cube[np.ix_(rows, cols)], dtype=np.float32)
            tensor = torch.from_numpy(block.transpose(2, 0, 1)[None]).to(
                device, non_blocking=True
            )
            with amp_context():
                logits = model.forward_dense(tensor)
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64) + 1
            truth = labels[row0:row1, col0:col1]
            mask = truth > 0
            if np.any(mask):
                encoded = (truth[mask].astype(np.int64) - 1) * model.num_classes + (
                    prediction[mask] - 1
                )
                confusion += np.bincount(
                    encoded, minlength=model.num_classes**2
                ).reshape(model.num_classes, model.num_classes)
    elapsed = time.perf_counter() - start_time
    return metrics_from_confusion(confusion), elapsed


@torch.inference_mode()
def evaluate_patches(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    amp: bool,
    world_size: int,
) -> tuple[dict[str, object], float]:
    model.eval()
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    start_time = time.perf_counter()
    for patches, targets in loader:
        patches = patches.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            logits = model(patches)
        predictions = logits.argmax(dim=1)
        encoded = targets * num_classes + predictions
        confusion += torch.bincount(
            encoded, minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    elapsed = torch.tensor(time.perf_counter() - start_time, device=device)
    if world_size > 1:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return metrics_from_confusion(confusion.cpu().numpy()), float(elapsed.item())


def build_model(config: dict[str, object]) -> nn.Module:
    common = {
        "in_channels": int(config["bands"]),
        "patch_size": int(config["patch_size"]),
        "num_classes": int(config["num_classes"]),
    }
    model_id = str(config["model"])
    if model_id == "m1":
        return HybridSN(**common)
    if model_id == "m2":
        return SSFTT(**common)
    if model_id == "m3":
        gaht = config.get("gaht", {})
        return GAHT(
            **common,
            n_groups=gaht.get("n_groups", (2, 2, 2)),
            depths=gaht.get("depths", (1, 2, 1)),
            embed_dims=gaht.get("embed_dims", (256, 128, 64)),
            num_heads=gaht.get("num_heads", (8, 4, 2)),
            mlp_ratios=gaht.get("mlp_ratios", (1, 1, 1)),
        )
    if model_id == "m5":
        hypersigma = config.get("hypersigma", {})
        if not bool(hypersigma.get("interpolate_pos_encoding", True)):
            raise ValueError("M5 requires interpolate_pos_encoding=true")
        return HyperSIGMA(
            **common,
            spatial_patch_size=int(hypersigma.get("spatial_patch_size", 2)),
            spatial_checkpoint=resolve_path(hypersigma["spatial_checkpoint"]),
            spectral_checkpoint=resolve_path(hypersigma["spectral_checkpoint"]),
            load_pretrained=True,
        )
    raise NotImplementedError(f"Model {model_id} is not enabled yet")


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "seed",
        "status",
        "best_epoch",
        "val_oa",
        "val_aa",
        "val_kappa",
        "parameters",
        "training_seconds",
        "validation_inference_seconds",
        "config_changes",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.model is not None:
        config["model"] = args.model
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["model"] == "m4":
        from src.train_dense import train_m4

        train_m4(config)
        return
    if config["model"] not in {"m1", "m2", "m3", "m5"}:
        raise NotImplementedError(f"Unsupported model {config['model']}")

    rank, world_size, local_rank, device = setup_distributed()
    seed = int(config["seed"])
    seed_everything(seed, bool(config.get("deterministic", True)))
    output_dir = resolve_path(config["output_root"]) / f"seed{seed}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if world_size > 1:
        dist.barrier()

    cube_path = resolve_path(config["data"]["cache_cube"])
    train_label_path = resolve_path(config["data"]["train_label"])
    val_label_path = resolve_path(config["data"]["val_label"])
    if not cube_path.exists():
        raise FileNotFoundError(
            f"Missing {cube_path}; run src/preprocess.py before training"
        )
    dataset = PatchDataset(
        cube_path,
        train_label_path,
        "train_label",
        patch_size=int(config["patch_size"]),
        augment=bool(config.get("train_augment", True)),
        seed=seed,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(config).to(device)
    hypersigma_config = config.get("hypersigma", {})
    m5_finetune_mode = str(hypersigma_config.get("finetune_mode", "full")).lower()
    if isinstance(model, HyperSIGMA):
        if m5_finetune_mode == "peft":
            # requires_grad must be finalized before DDP constructs its reducer.
            model.set_peft_trainable()
        elif m5_finetune_mode != "full":
            raise ValueError(
                f"Unsupported HyperSIGMA finetune_mode={m5_finetune_mode!r}"
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if rank == 0:
        print(
            f"model parameters: total={parameter_count:,} "
            f"trainable={trainable_parameter_count:,} "
            f"({100.0 * trainable_parameter_count / parameter_count:.3f}%)",
            flush=True,
        )
        if isinstance(model, HyperSIGMA):
            print(f"HyperSIGMA pretrained: {model.pretrained_report}", flush=True)
            print(f"HyperSIGMA finetune_mode={m5_finetune_mode}", flush=True)
    ddp_model: nn.Module = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=str(config["model"]) == "m5",
        )

    weights = None
    if config.get("class_weight") == "sqrt_inverse_frequency":
        weights = class_weights(train_label_path, "train_label", model.num_classes).to(device)
    label_smoothing = float(config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            ddp_model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    elif optimizer_name == "adamw":
        if not isinstance(model, HyperSIGMA):
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
        elif m5_finetune_mode == "peft":
            optimizer = torch.optim.AdamW(
                model.peft_parameter_groups(
                    head_lr=float(hypersigma_config.get("head_lr", config["learning_rate"])),
                    patch_lr=float(
                        hypersigma_config.get(
                            "patch_embedding_lr", config["learning_rate"]
                        )
                    ),
                    norm_lr=float(
                        hypersigma_config.get("norm_bias_lr", config["learning_rate"])
                    ),
                ),
                weight_decay=float(config["weight_decay"]),
            )
        else:
            backbone_multiplier = float(
                hypersigma_config.get("backbone_lr_multiplier", 0.1)
            )
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": list(model.backbone_parameters()),
                        "lr": float(config["learning_rate"]) * backbone_multiplier,
                    },
                    {
                        "params": list(model.head_parameters()),
                        "lr": float(config["learning_rate"]),
                    },
                ],
                weight_decay=float(config["weight_decay"]),
            )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            ddp_model.parameters(),
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config["weight_decay"]),
        )
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name}")
    total_optimizer_steps = int(config["epochs"]) * len(loader)
    scheduler_name = str(config.get("scheduler", "none")).lower()
    scheduler = None
    warmup_steps = 0
    if scheduler_name == "cosine":
        warmup_ratio = float(hypersigma_config.get("warmup_ratio", 0.0))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
        warmup_steps = int(round(total_optimizer_steps * warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                warmup_cosine_multiplier,
                warmup_steps=warmup_steps,
                total_steps=total_optimizer_steps,
            ),
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler {scheduler_name}")
    if rank == 0:
        print(
            f"optimizer={optimizer_name} scheduler={scheduler_name} "
            f"steps={total_optimizer_steps} warmup_steps={warmup_steps} "
            f"group_lrs={[group['lr'] for group in optimizer.param_groups]}",
            flush=True,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", True)))
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    val_labels, _ = load_label(val_label_path, "val_label")
    use_dense_validation = str(config["model"]) == "m1"
    val_cube = np.load(cube_path, mmap_mode="r") if rank == 0 and use_dense_validation else None
    val_loader = None
    full_val_dataset = None
    validation_fraction = 1.0
    if not use_dense_validation:
        val_dataset = PatchDataset(
            cube_path,
            val_label_path,
            "val_label",
            patch_size=int(config["patch_size"]),
            augment=False,
            seed=seed,
        )
        full_val_dataset = val_dataset
        if isinstance(model, HyperSIGMA):
            validation_fraction = float(
                config.get("hypersigma", {}).get("validation_fraction", 1.0)
            )
        if not 0.0 < validation_fraction <= 1.0:
            raise ValueError(
                f"validation_fraction must be in (0, 1], got {validation_fraction}"
            )
        if validation_fraction < 1.0:
            rng = np.random.default_rng(seed)
            selected_parts: list[np.ndarray] = []
            for class_index in range(int(config["num_classes"])):
                class_indices = np.flatnonzero(val_dataset.targets == class_index)
                sample_count = max(
                    1, int(round(len(class_indices) * validation_fraction))
                )
                selected_parts.append(
                    np.sort(rng.choice(class_indices, size=sample_count, replace=False))
                )
            validation_indices = np.sort(np.concatenate(selected_parts)).astype(np.int64)
        else:
            validation_indices = np.arange(len(val_dataset), dtype=np.int64)
        if rank == 0:
            selected_counts = np.bincount(
                val_dataset.targets[validation_indices],
                minlength=int(config["num_classes"]),
            )
            print(
                f"validation monitor: fraction={validation_fraction:.3f} "
                f"pixels={len(validation_indices)}/{len(val_dataset)} "
                f"per_class={selected_counts.tolist()}",
                flush=True,
            )
        # Unlike DistributedSampler, this exact partition never pads/duplicates
        # validation pixels when the dataset length is not divisible by world size.
        rank_validation_indices = validation_indices[rank::world_size].tolist()
        val_subset = Subset(val_dataset, rank_validation_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=int(config.get("val_batch_size", 512)),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
            drop_last=False,
        )

    best_oa = -1.0
    best_epoch = -1
    best_metrics: dict[str, object] | None = None
    best_infer_seconds = 0.0
    epochs_without_improvement = 0
    training_start = time.perf_counter()
    m5_warmup_epochs = int(hypersigma_config.get("warmup_head_epochs", 0))
    gradient_clip = float(config.get("gradient_clip", 0.0))
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    for epoch in range(int(config["epochs"])):
        if isinstance(model, HyperSIGMA) and m5_finetune_mode == "full":
            model.set_backbone_trainable(epoch >= m5_warmup_epochs)
            if rank == 0 and epoch in {0, m5_warmup_epochs}:
                stage = "head-only warmup" if epoch == 0 and m5_warmup_epochs else "full fine-tune"
                print(f"m5 stage: {stage} at epoch {epoch + 1}", flush=True)
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        ddp_model.train()
        loss_sum = torch.zeros(2, device=device, dtype=torch.float64)
        for patches, targets in loader:
            patches = patches.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=bool(config.get("amp", True))
            ):
                logits = ddp_model(patches)
                if config.get("focal_loss", False):
                    loss = focal_cross_entropy(logits, targets, weights)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if gradient_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            loss_sum[0] += loss.detach().double() * targets.numel()
            loss_sum[1] += targets.numel()
        if world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        train_loss = float((loss_sum[0] / loss_sum[1]).item())

        if world_size > 1:
            dist.barrier()
        stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
        metrics = None
        infer_seconds = 0.0
        if use_dense_validation and rank == 0:
            metrics, infer_seconds = evaluate_dense(
                model,
                val_cube,
                val_labels,
                int(config["val_tile_size"]),
                device,
                bool(config.get("amp", True)),
            )
        elif not use_dense_validation:
            assert val_loader is not None
            metrics, infer_seconds = evaluate_patches(
                model,
                val_loader,
                int(config["num_classes"]),
                device,
                bool(config.get("amp", True)),
                world_size,
            )
        if rank == 0:
            assert metrics is not None
            oa = float(metrics["oa"])
            metric_prefix = "val" if validation_fraction == 1.0 else "val_subset"
            print(
                f"epoch={epoch + 1} loss={train_loss:.6f} "
                f"{metric_prefix}_OA={oa:.6f} "
                f"{metric_prefix}_AA={float(metrics['aa']):.6f} "
                f"{metric_prefix}_Kappa={float(metrics['kappa']):.6f} "
                f"infer_s={infer_seconds:.2f}",
                flush=True,
            )
            assert writer is not None
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("val/OA", oa, epoch + 1)
            writer.add_scalar("val/AA", float(metrics["aa"]), epoch + 1)
            writer.add_scalar("val/Kappa", float(metrics["kappa"]), epoch + 1)
            for class_index, accuracy in enumerate(metrics["per_class_accuracy"], start=1):
                writer.add_scalar(f"val/class_{class_index}_accuracy", accuracy, epoch + 1)
            if oa > best_oa:
                best_oa = oa
                best_epoch = epoch + 1
                best_metrics = metrics
                best_infer_seconds = infer_seconds
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "metrics": best_metrics,
                        "config": config,
                    },
                    output_dir / "best.pt",
                )
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(best_metrics, indent=2), encoding="utf-8"
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["patience"]):
                stop_tensor[0] = 1
        if world_size > 1:
            dist.broadcast(stop_tensor, src=0)
        if int(stop_tensor.item()) == 1:
            break

    run_full_validation = (
        isinstance(model, HyperSIGMA)
        and validation_fraction < 1.0
        and bool(config.get("hypersigma", {}).get("full_validation_at_end", True))
    )
    if run_full_validation:
        if world_size > 1:
            dist.barrier()
        checkpoint_path = output_dir / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        assert full_val_dataset is not None
        full_val_subset = Subset(
            full_val_dataset, range(rank, len(full_val_dataset), world_size)
        )
        full_val_loader = DataLoader(
            full_val_subset,
            batch_size=int(config.get("val_batch_size", 512)),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
            drop_last=False,
        )
        full_metrics, full_infer_seconds = evaluate_patches(
            model,
            full_val_loader,
            int(config["num_classes"]),
            device,
            bool(config.get("amp", True)),
            world_size,
        )
        if rank == 0:
            selection_metrics = checkpoint["metrics"]
            checkpoint["selection_metrics"] = selection_metrics
            checkpoint["metrics"] = full_metrics
            torch.save(checkpoint, checkpoint_path)
            (output_dir / "full_val_metrics.json").write_text(
                json.dumps(full_metrics, indent=2), encoding="utf-8"
            )
            print(
                f"full_val best_epoch={best_epoch} OA={float(full_metrics['oa']):.6f} "
                f"AA={float(full_metrics['aa']):.6f} "
                f"Kappa={float(full_metrics['kappa']):.6f} "
                f"infer_s={full_infer_seconds:.2f}",
                flush=True,
            )
            best_metrics = full_metrics
            best_infer_seconds = full_infer_seconds

    training_seconds = time.perf_counter() - training_start
    if rank == 0:
        assert best_metrics is not None
        assert writer is not None
        writer.close()
        append_summary(
            PROJECT_ROOT / "outputs" / "summary.csv",
            {
                "model": f"{config['model']}_{config['model_name']}",
                "seed": seed,
                "status": "complete",
                "best_epoch": best_epoch,
                "val_oa": best_metrics["oa"],
                "val_aa": best_metrics["aa"],
                "val_kappa": best_metrics["kappa"],
                "parameters": parameter_count,
                "training_seconds": training_seconds,
                "validation_inference_seconds": best_infer_seconds,
                "config_changes": (
                    f"validation_fraction={validation_fraction};"
                    f"full_validation_at_end={run_full_validation};"
                    f"finetune_mode={m5_finetune_mode};"
                    f"trainable_parameters={trainable_parameter_count};"
                    f"label_smoothing={label_smoothing};"
                    f"scheduler={scheduler_name};"
                    f"gradient_clip={gradient_clip}"
                ),
            },
        )
        print(
            f"{config['model']} complete: best monitor OA={best_oa:.6f}; "
            f"reported full val OA={float(best_metrics['oa']):.6f}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
