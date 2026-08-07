#!/usr/bin/env python3
"""Transductive, robust preprocessing for the HyperSIGMA maximum-potential run.

This intentionally differs from the strict train-mask baseline: statistics and
PCA are fitted without labels from the train cube and both test scenes.  The
result supplies PCA30 to the spatial branch and ordered native 98 bands to the
spectral branch.  PCA whitening is never used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CubeSpec:
    name: str
    path: Path
    variable: str
    height: int
    width: int


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def source_layout(source: h5py.Dataset, spec: CubeSpec, bands: int = 98) -> str:
    if source.shape == (bands, spec.height, spec.width):
        return "CHW"
    if source.shape == (bands, spec.width, spec.height):
        return "CWH"
    raise ValueError(f"Unexpected shape {source.shape} for {spec.path}")


def read_row(source: h5py.Dataset, layout: str, row: int) -> np.ndarray:
    if layout == "CHW":
        return np.asarray(source[:, row, :], dtype=np.float32).T
    return np.asarray(source[:, :, row], dtype=np.float32).T


def sample_scene(spec: CubeSpec, count: int, seed: int) -> np.ndarray:
    """Uniformly sample pixels with sequential HDF5 row-block reads.

    Reading one row at a time repeatedly decompresses the same HDF5 chunks for
    these MATLAB files and can amplify I/O by two orders of magnitude.
    """
    rng = np.random.default_rng(seed)
    count = min(int(count), spec.height * spec.width)
    flat = np.sort(
        rng.choice(spec.height * spec.width, size=count, replace=False)
    )
    rows, cols = np.divmod(flat, spec.width)
    samples = np.empty((count, 98), dtype=np.float32)
    with h5py.File(spec.path, "r") as handle:
        source = handle[spec.variable]
        layout = source_layout(source, spec)
        row_block = 64
        for row0 in range(0, spec.height, row_block):
            row1 = min(row0 + row_block, spec.height)
            selected = np.flatnonzero((rows >= row0) & (rows < row1))
            if selected.size == 0:
                continue
            if layout == "CHW":
                block = np.asarray(
                    source[:, row0:row1, :], dtype=np.float32
                ).transpose(1, 2, 0)
            else:
                block = np.asarray(
                    source[:, :, row0:row1], dtype=np.float32
                ).transpose(2, 1, 0)
            samples[selected] = block[rows[selected] - row0, cols[selected]]
    return samples


def robust_transform(
    values: np.ndarray, low: np.ndarray, high: np.ndarray, clip: float
) -> np.ndarray:
    scaled = 2.0 * (values - low) / (high - low) - 1.0
    return np.clip(scaled, -clip, clip)


def transform_scene(
    spec: CubeSpec,
    output_dir: Path,
    low: np.ndarray,
    high: np.ndarray,
    pca: PCA,
    rows_per_chunk: int,
    clip: float,
) -> None:
    native_path = output_dir / f"{spec.name}_hsmax_native98_float16.npy"
    pca_path = output_dir / f"{spec.name}_hsmax_pca30_float16.npy"
    native = np.lib.format.open_memmap(
        native_path, mode="w+", dtype=np.float16,
        shape=(spec.height, spec.width, 98),
    )
    spatial = np.lib.format.open_memmap(
        pca_path, mode="w+", dtype=np.float16,
        shape=(spec.height, spec.width, pca.n_components_),
    )
    with h5py.File(spec.path, "r") as handle:
        source = handle[spec.variable]
        layout = source_layout(source, spec)
        for row0 in range(0, spec.height, rows_per_chunk):
            row1 = min(row0 + rows_per_chunk, spec.height)
            if layout == "CHW":
                block = np.asarray(source[:, row0:row1, :], dtype=np.float32).transpose(1, 2, 0)
            else:
                block = np.asarray(source[:, :, row0:row1], dtype=np.float32).transpose(2, 1, 0)
            robust = robust_transform(block, low, high, clip).astype(np.float32)
            flat = robust.reshape(-1, 98, order="C")
            native[row0:row1] = robust.astype(np.float16)
            spatial[row0:row1] = pca.transform(flat).reshape(
                row1 - row0, spec.width, pca.n_components_, order="C"
            ).astype(np.float16)
            native.flush()
            spatial.flush()
            print(f"{spec.name}: rows {row0}:{row1}/{spec.height}", flush=True)
    del native, spatial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/cache")
    parser.add_argument("--samples-per-scene", type=int, default=150_000)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        CubeSpec("train", data_dir / "data_hsi.mat", "data", 4040, 2444),
        CubeSpec("scene1", data_dir / "test_scene1.mat", "image", 3104, 4507),
        CubeSpec("scene2", data_dir / "test_scene2.mat", "image", 3409, 2181),
    )
    sampled = []
    for index, spec in enumerate(specs):
        print(f"Sampling unlabeled pixels from {spec.name}", flush=True)
        sampled.append(
            sample_scene(spec, args.samples_per_scene, args.seed + index * 1009)
        )
    samples = np.concatenate(sampled, axis=0)
    low = np.percentile(samples, args.lower_percentile, axis=0).astype(np.float32)
    high = np.percentile(samples, args.upper_percentile, axis=0).astype(np.float32)
    invalid = high - low < 1e-6
    high[invalid] = low[invalid] + 1.0
    robust_samples = robust_transform(samples, low, high, args.clip)
    pca = PCA(
        n_components=args.pca_components,
        whiten=False,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(robust_samples)
    stats_path = output_dir / "preprocess_hypersigma_max.npz"
    np.savez(
        stats_path,
        robust_low=low,
        robust_high=high,
        clip=np.asarray(args.clip),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        pca_whiten=np.asarray(0),
        sample_count=np.asarray(len(samples)),
        includes_unlabeled_test=np.asarray(1),
    )
    del samples, robust_samples, sampled
    for spec in specs:
        transform_scene(
            spec, output_dir, low, high, pca,
            args.rows_per_chunk, args.clip,
        )
    manifest = {
        "statistics_scope": "unlabeled train cube + unlabeled test scene1 + scene2",
        "samples_per_scene": args.samples_per_scene,
        "pca_components": args.pca_components,
        "pca_whiten": False,
        "native_scaling": "per-band joint 2-98 percentile to [-1,1], clipped",
        "clip": args.clip,
        "stats": str(stats_path),
    }
    (output_dir / "preprocess_hypersigma_max.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
