#!/usr/bin/env python3
"""Whole-scene inference for frozen author-model checkpoints M1-M3."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import reflect_indices
from src.models import GAHT, HybridSN, HyperSIGMA, MambaHSI, SSFTT


SCENES = {
    "scene1": (3104, 4507),
    "scene2": (3409, 2181),
}

TTA_VIEWS = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def spatial_transform(tensor: torch.Tensor, view: str) -> torch.Tensor:
    """Apply one dihedral image view to the last two tensor dimensions."""
    if view == "identity":
        return tensor
    if view == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if view == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if view == "rot90":
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if view == "rot180":
        return torch.rot90(tensor, 2, dims=(-2, -1))
    if view == "rot270":
        return torch.rot90(tensor, 3, dims=(-2, -1))
    raise ValueError(f"Unknown TTA view: {view}")


def inverse_spatial_transform(tensor: torch.Tensor, view: str) -> torch.Tensor:
    """Map spatial logits from a transformed view back to source coordinates."""
    if view in {"identity", "hflip", "vflip", "rot180"}:
        return spatial_transform(tensor, view)
    if view == "rot90":
        return spatial_transform(tensor, "rot270")
    if view == "rot270":
        return spatial_transform(tensor, "rot90")
    raise ValueError(f"Unknown TTA view: {view}")


def build_model(config: dict[str, object]) -> HybridSN | SSFTT | GAHT | MambaHSI | HyperSIGMA:
    """Instantiate only the project's adapters around pinned author sources."""
    model_id = str(config["model"])
    if model_id == "m4":
        params = config.get("mambahsi", {})
        return MambaHSI(
            in_channels=int(config["bands"]),
            num_classes=int(config["num_classes"]),
            hidden_dim=int(params.get("hidden_dim", 128)),
            mamba_type=str(params.get("mamba_type", "both")),
            token_num=int(params.get("token_num", 4)),
            group_num=int(params.get("group_num", 4)),
            use_residual=bool(params.get("use_residual", True)),
            use_att=bool(params.get("use_att", True)),
        )
    common = {
        "in_channels": int(config["bands"]),
        "patch_size": int(config["patch_size"]),
        "num_classes": int(config["num_classes"]),
    }
    if model_id == "m1":
        return HybridSN(**common)
    if model_id == "m2":
        return SSFTT(**common)
    if model_id == "m3":
        gaht = config.get("gaht", {})
        if not isinstance(gaht, dict):
            raise TypeError(f"GAHT config must be a mapping, got {type(gaht).__name__}")
        return GAHT(
            **common,
            n_groups=gaht.get("n_groups", (2, 2, 2)),
            depths=gaht.get("depths", (1, 2, 1)),
            embed_dims=gaht.get("embed_dims", (256, 128, 64)),
            num_heads=gaht.get("num_heads", (8, 4, 2)),
            mlp_ratios=gaht.get("mlp_ratios", (1, 1, 1)),
        )
    if model_id == "m5":
        params = config.get("hypersigma", {})
        return HyperSIGMA(
            **common,
            spatial_patch_size=int(params.get("spatial_patch_size", 2)),
            load_pretrained=False,
        )
    raise ValueError(f"Unsupported checkpoint model: {model_id}")


def patch_logits(
    model: SSFTT | GAHT | HyperSIGMA,
    patches: torch.Tensor,
    amp_context,
    tta: bool,
) -> torch.Tensor:
    """Average author-model classification logits across optional TTA views."""
    views = TTA_VIEWS if tta else ("identity",)
    logits_sum: torch.Tensor | None = None
    for view in views:
        with amp_context():
            logits = model(spatial_transform(patches, view)).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return logits_sum / len(views)


def mamba_tile_logits(
    model: MambaHSI,
    tensor: torch.Tensor,
    amp_context,
    tta: bool,
) -> torch.Tensor:
    views = TTA_VIEWS if tta else ("identity",)
    logits_sum: torch.Tensor | None = None
    for view in views:
        with amp_context():
            logits = model(spatial_transform(tensor, view)).float()
        logits = inverse_spatial_transform(logits, view)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return logits_sum / len(views)


def overlap_starts(size: int, tile_size: int, overlap: int) -> list[int]:
    if not 0 <= overlap < tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size")
    if size <= tile_size:
        return [0]
    starts = list(range(0, size - tile_size + 1, tile_size - overlap))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)
    return starts


@torch.inference_mode()
def infer_mamba_probabilities(
    model: MambaHSI,
    cube: np.ndarray,
    output_path: Path,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp: bool,
    output_dtype: np.dtype,
    tta: bool,
    rank: int,
    world_size: int,
) -> None:
    """Distributed overlap-blended tile inference for author MambaHSI."""
    if cube.ndim != 3 or cube.shape[2] != model.in_channels:
        raise ValueError(f"Expected HxWx{model.in_channels}, got {cube.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output_path.parent / f".{output_path.stem}_tile_parts"
    if rank == 0:
        part_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    height, width = cube.shape[:2]
    row_starts = overlap_starts(height, tile_size, overlap)
    col_starts = overlap_starts(width, tile_size, overlap)
    tiles = [(row0, col0) for row0 in row_starts for col0 in col_starts]
    model.eval()
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if amp and device.type == "cuda"
        else nullcontext
    )
    started = time.perf_counter()
    for tile_index, (row0, col0) in enumerate(tiles):
        if tile_index % world_size != rank:
            continue
        rows = reflect_indices(np.arange(row0, row0 + tile_size), height)
        cols = reflect_indices(np.arange(col0, col0 + tile_size), width)
        block = np.asarray(cube[np.ix_(rows, cols)], dtype=np.float32)
        tensor = torch.from_numpy(block.transpose(2, 0, 1)[None]).to(
            device, non_blocking=True
        )
        logits = mamba_tile_logits(model, tensor, amp_context, tta)
        tile_logits = logits[0].permute(1, 2, 0).cpu().numpy().astype(np.float16)
        np.save(part_dir / f"tile_{tile_index:05d}.npy", tile_logits)
        print(
            f"  rank={rank} {output_path.name}: tile {tile_index + 1}/{len(tiles)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        logits_sum = np.zeros((height, width, model.num_classes), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        axis = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
        window = axis[:, None] * axis[None, :]
        for tile_index, (row0, col0) in enumerate(tiles):
            part_path = part_dir / f"tile_{tile_index:05d}.npy"
            tile_logits = np.load(part_path).astype(np.float32)
            real_h = min(tile_size, height - row0)
            real_w = min(tile_size, width - col0)
            tile_weight = window[:real_h, :real_w]
            logits_sum[row0 : row0 + real_h, col0 : col0 + real_w] += (
                tile_logits[:real_h, :real_w] * tile_weight[..., None]
            )
            weight_sum[row0 : row0 + real_h, col0 : col0 + real_w] += tile_weight
            part_path.unlink()
        if np.any(weight_sum <= 0):
            raise RuntimeError("M4 overlap blending left uncovered pixels")
        probabilities = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=output_dtype,
            shape=(height, width, model.num_classes),
        )
        for row0 in range(0, height, 64):
            row1 = min(row0 + 64, height)
            averaged = logits_sum[row0:row1] / weight_sum[row0:row1, :, None]
            averaged -= averaged.max(axis=2, keepdims=True)
            np.exp(averaged, out=averaged)
            averaged /= averaged.sum(axis=2, keepdims=True)
            probabilities[row0:row1] = averaged.astype(output_dtype)
        probabilities.flush()
        del probabilities, logits_sum, weight_sum
        part_dir.rmdir()
    if world_size > 1:
        dist.barrier()


def dense_logits(
    model: HybridSN,
    tensor: torch.Tensor,
    amp_context,
    tta: bool,
) -> torch.Tensor:
    """Average dense logits after undoing each view's spatial transform."""
    views = TTA_VIEWS if tta else ("identity",)
    logits_sum: torch.Tensor | None = None
    for view in views:
        with amp_context():
            logits = model.forward_dense(spatial_transform(tensor, view)).float()
        logits = inverse_spatial_transform(logits, view)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return logits_sum / len(views)


@torch.inference_mode()
def infer_dense_probabilities(
    model: HybridSN,
    cube: np.ndarray,
    output_path: Path,
    tile_size: int,
    device: torch.device,
    amp: bool,
    output_dtype: np.dtype,
    tta: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Write HxWxC probabilities incrementally, preserving scene coordinates."""
    if cube.ndim != 3 or cube.shape[2] != model.in_channels:
        raise ValueError(
            f"Expected HxWx{model.in_channels} cache, got {tuple(cube.shape)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        probabilities = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=output_dtype,
            shape=(cube.shape[0], cube.shape[1], model.num_classes),
        )
        del probabilities
    if world_size > 1:
        dist.barrier()
    probabilities = np.lib.format.open_memmap(output_path, mode="r+")
    model.eval()
    halo = model.halo
    height, width = cube.shape[:2]
    use_amp = amp and device.type == "cuda"
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp
        else nullcontext
    )
    started = time.perf_counter()
    tile_index = 0
    for row0 in range(0, height, tile_size):
        row1 = min(row0 + tile_size, height)
        rows = reflect_indices(np.arange(row0 - halo, row1 + halo), height)
        for col0 in range(0, width, tile_size):
            assigned = tile_index % world_size == rank
            tile_index += 1
            if not assigned:
                continue
            col1 = min(col0 + tile_size, width)
            cols = reflect_indices(np.arange(col0 - halo, col1 + halo), width)
            block = np.asarray(cube[np.ix_(rows, cols)], dtype=np.float32)
            tensor = torch.from_numpy(block.transpose(2, 0, 1)[None]).to(
                device, non_blocking=True
            )
            logits = dense_logits(model, tensor, amp_context, tta)
            tile_probabilities = torch.softmax(logits, dim=1)
            tile = tile_probabilities.squeeze(0).permute(1, 2, 0).cpu().numpy()
            expected_shape = (row1 - row0, col1 - col0, model.num_classes)
            if tile.shape != expected_shape:
                raise RuntimeError(f"Dense output {tile.shape} != expected {expected_shape}")
            probabilities[row0:row1, col0:col1] = tile.astype(
                output_dtype, copy=False
            )
        probabilities.flush()
        print(
            f"  rank={rank} {output_path.name}: rows {row0}:{row1}/{height} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    del probabilities
    if world_size > 1:
        dist.barrier()


@torch.inference_mode()
def infer_patch_probabilities(
    model: SSFTT | GAHT | HyperSIGMA,
    cube: np.ndarray,
    output_path: Path,
    batch_size: int,
    device: torch.device,
    amp: bool,
    output_dtype: np.dtype,
    tta: bool = False,
    rank: int = 0,
    world_size: int = 1,
    resume: bool = False,
) -> None:
    """Infer every pixel using vectorized reflect-padded patch batches."""
    if cube.ndim != 3 or cube.shape[2] != model.in_channels:
        raise ValueError(
            f"Expected HxWx{model.in_channels} cache, got {tuple(cube.shape)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = cube.shape[:2]
    output_existed = output_path.exists()
    if rank == 0 and not (resume and output_existed):
        probabilities = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=output_dtype,
            shape=(height, width, model.num_classes),
        )
        del probabilities
    if world_size > 1:
        dist.barrier()
    probabilities = np.lib.format.open_memmap(output_path, mode="r+")
    model.eval()
    radius = model.halo
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    pixel_count = height * width
    range_start = pixel_count * rank // world_size
    range_stop = pixel_count * (rank + 1) // world_size
    if resume and output_existed:
        expected_shape = (height, width, model.num_classes)
        if probabilities.shape != expected_shape or probabilities.dtype != output_dtype:
            raise ValueError(
                f"Cannot resume {output_path}: got shape={probabilities.shape} "
                f"dtype={probabilities.dtype}, expected shape={expected_shape} "
                f"dtype={output_dtype}"
            )
        flat_probabilities = probabilities.reshape(pixel_count, model.num_classes)
        resume_start = range_stop
        scan_size = 262_144
        for scan_start in range(range_start, range_stop, scan_size):
            scan_stop = min(scan_start + scan_size, range_stop)
            written = np.any(
                flat_probabilities[scan_start:scan_stop] != 0, axis=1
            )
            if not bool(written.all()):
                resume_start = scan_start + int(np.flatnonzero(~written)[0])
                break
        # Restart at a batch boundary. Rewriting the preceding partial batch is
        # harmless and protects against a process dying during a memmap write.
        resume_start = max(
            range_start,
            range_start + ((resume_start - range_start) // batch_size) * batch_size,
        )
        print(
            f"  rank={rank} {output_path.name}: resume at "
            f"{resume_start - range_start}/{range_stop - range_start} pixels",
            flush=True,
        )
    else:
        resume_start = range_start
    use_amp = amp and device.type == "cuda"
    amp_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp
        else nullcontext
    )
    started = time.perf_counter()
    report_every = max(batch_size, (1_000_000 // batch_size) * batch_size)
    for start in range(resume_start, range_stop, batch_size):
        stop = min(start + batch_size, range_stop)
        flat = np.arange(start, stop, dtype=np.int64)
        centers_row, centers_col = np.divmod(flat, width)
        rows = reflect_indices(centers_row[:, None] + offsets[None, :], height)
        cols = reflect_indices(centers_col[:, None] + offsets[None, :], width)
        # Advanced indexing performs one vectorized gather directly from the
        # float16 scene memmap. Conversion happens once for the complete batch.
        patches = np.asarray(
            cube[rows[:, :, None], cols[:, None, :], :], dtype=np.float32
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(patches.transpose(0, 3, 1, 2))
        ).to(device, non_blocking=True)
        logits = patch_logits(model, tensor, amp_context, tta)
        batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        output_rows, output_cols = np.divmod(flat, width)
        probabilities[output_rows, output_cols] = batch_probabilities.astype(
            output_dtype, copy=False
        )
        if stop == range_stop or (stop - range_start) % report_every == 0:
            probabilities.flush()
            print(
                f"  rank={rank} {output_path.name}: pixels {stop - range_start}/"
                f"{range_stop - range_start} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    del probabilities
    if world_size > 1:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="m1", choices=("m1", "m2", "m3", "m4", "m5"))
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="default: outputs/<model>/seed0/best.pt",
    )
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/cache")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), default=list(SCENES))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="average identity, H/V flips, and 90/180/270-degree rotations",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dtype", choices=("float16", "float32"), default="float16")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="continue patch inference from the first all-zero probability row",
    )
    args = parser.parse_args()

    if args.tile_size <= 0:
        parser.error("--tile-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    checkpoint_path = resolve_path(
        args.checkpoint
        if args.checkpoint is not None
        else PROJECT_ROOT / "outputs" / args.model / "seed0" / "best.pt"
    )
    cache_dir = resolve_path(args.cache_dir)
    output_dir = (
        resolve_path(args.output_dir) if args.output_dir is not None else checkpoint_path.parent
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # In heterogeneous or shared GPU pools, one inference rank can finish
        # much earlier than another. Keep the final synchronization alive long
        # enough for the slowest rank rather than using NCCL's 10-minute default.
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(hours=4),
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("model") != args.model:
        raise ValueError(
            f"--model {args.model} does not match checkpoint model {config.get('model')}"
        )
    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    output_dtype = np.dtype(args.output_dtype)

    for scene in args.scenes:
        expected_hw = SCENES[scene]
        cache_path = cache_dir / f"{scene}_pca{model.in_channels}_float16.npy"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing {cache_path}; generate it with src/preprocess.py "
                "--reuse-stats ... --test-only"
            )
        cube = np.load(cache_path, mmap_mode="r")
        if cube.shape != (*expected_hw, model.in_channels):
            raise ValueError(f"{cache_path}: shape {cube.shape} != {(*expected_hw, model.in_channels)}")
        output_path = output_dir / f"probs_{scene}.npy"
        if rank == 0 and output_path.exists() and not (args.overwrite or args.resume):
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite or --resume"
            )
        if world_size > 1:
            dist.barrier()
        print(f"rank={rank} Inferring {scene}: {cache_path} -> {output_path}", flush=True)
        if isinstance(model, HybridSN):
            infer_dense_probabilities(
                model=model,
                cube=cube,
                output_path=output_path,
                tile_size=args.tile_size,
                device=device,
                amp=bool(args.amp),
                output_dtype=output_dtype,
                tta=bool(args.tta),
                rank=rank,
                world_size=world_size,
            )
        elif isinstance(model, MambaHSI):
            infer_mamba_probabilities(
                model=model,
                cube=cube,
                output_path=output_path,
                tile_size=int(config.get("tile_size", args.tile_size)),
                overlap=int(
                    args.tile_overlap
                    if args.tile_overlap is not None
                    else config.get("tile_overlap", 32)
                ),
                device=device,
                amp=bool(args.amp),
                output_dtype=output_dtype,
                tta=bool(args.tta),
                rank=rank,
                world_size=world_size,
            )
        else:
            infer_patch_probabilities(
                model=model,
                cube=cube,
                output_path=output_path,
                batch_size=args.batch_size,
                device=device,
                amp=bool(args.amp),
                output_dtype=output_dtype,
                tta=bool(args.tta),
                rank=rank,
                world_size=world_size,
                resume=bool(args.resume),
            )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
