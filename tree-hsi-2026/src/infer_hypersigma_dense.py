#!/usr/bin/env python3
"""Multi-GPU strip-parallel inference for dense dual-view HyperSIGMA."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import reflect_indices
from src.models import HyperSIGMADense


SCENES = {"scene1": (3104, 4507), "scene2": (3409, 2181)}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def starts(size: int, tile: int, overlap: int) -> list[int]:
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    if size <= tile:
        return [0]
    values = list(range(0, size - tile + 1, tile - overlap))
    if values[-1] != size - tile:
        values.append(size - tile)
    return values


def transform(x: torch.Tensor, view: str) -> torch.Tensor:
    if view == "identity": return x
    if view == "hflip": return torch.flip(x, (-1,))
    if view == "vflip": return torch.flip(x, (-2,))
    if view == "rot90": return torch.rot90(x, 1, (-2, -1))
    if view == "rot180": return torch.rot90(x, 2, (-2, -1))
    if view == "rot270": return torch.rot90(x, 3, (-2, -1))
    raise ValueError(view)


def inverse(x: torch.Tensor, view: str) -> torch.Tensor:
    if view in {"identity", "hflip", "vflip", "rot180"}:
        return transform(x, view)
    return transform(x, "rot270" if view == "rot90" else "rot90")


@torch.inference_mode()
def infer_scene(
    model: HyperSIGMADense,
    spatial_path: Path,
    spectral_path: Path,
    output_path: Path,
    shape: tuple[int, int],
    tile: int,
    overlap: int,
    views: tuple[str, ...],
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    height, width = shape
    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=(height, width, 17)
        )
        target[:] = 0
        target.flush()
        del target
    if world > 1:
        dist.barrier()
    spatial = np.load(spatial_path, mmap_mode="r")
    spectral = np.load(spectral_path, mmap_mode="r")
    row0 = height * rank // world
    row1 = height * (rank + 1) // world
    logits_sum = np.zeros((row1 - row0, width, 17), dtype=np.float32)
    weight_sum = np.zeros((row1 - row0, width), dtype=np.float32)
    axis = np.maximum(np.hanning(tile).astype(np.float32), 0.05)
    window = axis[:, None] * axis[None, :]
    amp_dtype = torch.bfloat16
    model.eval()
    began = time.perf_counter()
    relevant_rows = [r for r in starts(height, tile, overlap) if r < row1 and r + tile > row0]
    for tile_row in relevant_rows:
        rows = reflect_indices(np.arange(tile_row, tile_row + tile), height)
        write_top = max(tile_row, row0)
        write_bottom = min(tile_row + tile, row1)
        local_top, local_bottom = write_top - row0, write_bottom - row0
        tile_top, tile_bottom = write_top - tile_row, write_bottom - tile_row
        for tile_col in starts(width, tile, overlap):
            cols = reflect_indices(np.arange(tile_col, tile_col + tile), width)
            x_spat = torch.from_numpy(np.asarray(spatial[np.ix_(rows, cols)], dtype=np.float32).transpose(2, 0, 1)[None]).to(device)
            x_spec = torch.from_numpy(np.asarray(spectral[np.ix_(rows, cols)], dtype=np.float32).transpose(2, 0, 1)[None]).to(device)
            average = None
            for view in views:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(transform(x_spat, view), transform(x_spec, view))
                logits = inverse(logits.float(), view)
                average = logits if average is None else average + logits
            tile_logits = (average / len(views))[0].permute(1, 2, 0).cpu().numpy()
            col1 = min(tile_col + tile, width)
            real_width = col1 - tile_col
            weights = window[tile_top:tile_bottom, :real_width]
            logits_sum[local_top:local_bottom, tile_col:col1] += tile_logits[tile_top:tile_bottom, :real_width] * weights[..., None]
            weight_sum[local_top:local_bottom, tile_col:col1] += weights
        print(
            f"rank={rank} rows={row0}:{row1} tile_row={tile_row} elapsed={time.perf_counter()-began:.1f}s",
            flush=True,
        )
    if np.any(weight_sum <= 0):
        raise RuntimeError(f"rank={rank} left uncovered output pixels")
    logits_sum /= weight_sum[..., None]
    logits_sum -= logits_sum.max(axis=2, keepdims=True)
    probabilities = np.exp(logits_sum)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    target = np.load(output_path, mmap_mode="r+")
    target[row0:row1] = probabilities.astype(np.float16)
    target.flush()
    del target
    if world > 1:
        dist.barrier()
    if rank == 0:
        check = np.load(output_path, mmap_mode="r")
        if check.shape != (height, width, 17) or not np.isfinite(check).all():
            raise RuntimeError(f"Invalid probability output {output_path}")
        print(f"Completed {output_path} in {time.perf_counter()-began:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tta", default="identity,hflip,vflip,rot90,rot180,rot270")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    hs = config["hypersigma"]
    model = HyperSIGMADense(
        image_size=int(config["tile_size"]), spatial_patch_size=int(hs["spatial_patch_size"]),
        spatial_checkpoint=resolve_path(hs["spatial_checkpoint"]),
        spectral_checkpoint=resolve_path(hs["spectral_checkpoint"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    views = tuple(item.strip() for item in args.tta.split(",") if item.strip())
    for scene, shape in SCENES.items():
        infer_scene(
            model,
            resolve_path(f"data/cache/{scene}_hsmax_pca30_float16.npy"),
            resolve_path(f"data/cache/{scene}_hsmax_native98_float16.npy"),
            args.output_dir.resolve() / f"probs_{scene}.npy",
            shape, int(config["tile_size"]), int(config["tile_overlap"]), views,
            rank, world, device,
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
