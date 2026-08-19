#!/usr/bin/env python3
"""Continue the official HyperSIGMA MAE objective on competition imagery."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = PROJECT_ROOT / "third_party" / "HyperSIGMA" / "Pretrain"


def setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


class UnlabeledTileDataset(Dataset):
    def __init__(
        self, paths: list[Path], tile_size: int, samples_per_epoch: int, seed: int
    ) -> None:
        self.cubes = [np.load(path, mmap_mode="r") for path in paths]
        self.tile_size = tile_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(
            self.seed * 1_000_003 + self.epoch * 10_007 + index
        )
        cube = self.cubes[int(rng.integers(0, len(self.cubes)))]
        row = int(rng.integers(0, cube.shape[0] - self.tile_size + 1))
        col = int(rng.integers(0, cube.shape[1] - self.tile_size + 1))
        tile = np.asarray(
            cube[row : row + self.tile_size, col : col + self.tile_size],
            dtype=np.float32,
        )
        tile = np.rot90(tile, int(rng.integers(0, 4)), axes=(0, 1))
        if rng.random() < 0.5:
            tile = tile[::-1]
        if rng.random() < 0.5:
            tile = tile[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))


def build_model(branch: str, image_size: int) -> nn.Module:
    old_mmengine = sys.modules.get("mmengine")
    old_mmengine_dist = sys.modules.get("mmengine.dist")
    if old_mmengine_dist is None:
        mmengine = ModuleType("mmengine")
        mmengine_dist = ModuleType("mmengine.dist")
        mmengine_dist.get_dist_info = lambda: (0, 1)
        mmengine.dist = mmengine_dist
        sys.modules["mmengine"] = mmengine
        sys.modules["mmengine.dist"] = mmengine_dist
    sys.path.insert(0, str(PRETRAIN_ROOT))
    try:
        args = SimpleNamespace(image_size=image_size, use_ckpt="False", num_tokens=100)
        if branch == "spatial":
            module = importlib.import_module("models_mae_Spat")
            return module.spat_mae_b(args, inchannels=30)
        module = importlib.import_module("models_mae_Spec")
        return module.spec_mae_b(args, inchannels=100)
    finally:
        sys.path.pop(0)
        if old_mmengine_dist is None:
            sys.modules.pop("mmengine.dist", None)
            if old_mmengine is None:
                sys.modules.pop("mmengine", None)
            else:
                sys.modules["mmengine"] = old_mmengine


def load_compatible(model: nn.Module, path: Path) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    target = model.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.removeprefix("module.") in target
        and target[key.removeprefix("module.")].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": len(result.missing_keys),
        "unexpected": len(result.unexpected_keys),
    }


def resample_98_to_100(x: torch.Tensor) -> torch.Tensor:
    batch, bands, height, width = x.shape
    if bands != 98:
        raise ValueError(f"Expected 98 native bands, got {bands}")
    lines = x.permute(0, 2, 3, 1).reshape(-1, 1, bands)
    lines = F.interpolate(lines, size=100, mode="linear", align_corners=True)
    return lines.reshape(batch, height, width, 100).permute(0, 3, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("spatial", "spectral"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/cache")
    parser.add_argument(
        "--cache-path",
        action="append",
        default=None,
        help="Explicit unlabeled cache path; repeat for source/scene caches.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rank, world, local_rank, device = setup_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    suffix = "pca30" if args.branch == "spatial" else "native98"
    paths = (
        [Path(item).resolve() for item in args.cache_path]
        if args.cache_path
        else [
            args.cache_dir / f"{scene}_hsmax_{suffix}_float16.npy"
            for scene in ("train", "scene1", "scene2")
        ]
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing robust caches: {missing}")
    dataset = UnlabeledTileDataset(
        paths, args.image_size, args.samples_per_epoch, args.seed
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    model = build_model(args.branch, args.image_size)
    report = load_compatible(model, args.checkpoint.resolve())
    model.to(device)
    ddp: nn.Module = model
    if world > 1:
        ddp = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    optimizer = torch.optim.AdamW(
        ddp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        ddp.train()
        loss_total = torch.zeros(2, dtype=torch.float64, device=device)
        for tiles in loader:
            tiles = tiles.to(device, non_blocking=True)
            if args.branch == "spectral":
                tiles = resample_98_to_100(tiles)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = ddp(tiles, mask_ratio=args.mask_ratio)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total[0] += loss.detach().double()
            loss_total[1] += 1.0
        if world > 1:
            dist.all_reduce(loss_total)
        scheduler.step()
        if rank == 0:
            print(
                f"branch={args.branch} epoch={epoch + 1}/{args.epochs} "
                f"loss={float(loss_total[0] / loss_total[1]):.6f}", flush=True
            )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "branch": args.branch,
                "source_checkpoint": str(args.checkpoint.resolve()),
                "load_report": report,
                "epochs": args.epochs,
                "elapsed_seconds": time.perf_counter() - started,
            },
            args.output,
        )
        args.output.with_suffix(".json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"Saved domain-adapted {args.branch} MAE to {args.output}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
