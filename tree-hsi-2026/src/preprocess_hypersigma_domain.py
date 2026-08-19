#!/usr/bin/env python3
"""Source-fitted HyperSIGMA caches with scene-specific radiometric alignment.

This challenger deliberately keeps the PCA basis and robust scaling anchored
to the labelled source scene.  Each unlabeled target scene may optionally be
matched to the source per-band marginal distribution before it is projected
into that fixed PCA basis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess_hypersigma_max import (
    CubeSpec,
    robust_transform,
    sample_scene,
    source_layout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANTILES = (2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0)


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for index in range(1, result.shape[0]):
        result[index] = np.maximum(result[index], result[index - 1] + 1e-4)
    return result


def quantile_align(
    block: np.ndarray, target_quantiles: np.ndarray, source_quantiles: np.ndarray
) -> np.ndarray:
    """Map target per-band marginals into the source radiometric frame."""
    # HDF5 CWH -> HWC transposes are non-contiguous.  Force C order so the
    # flattened output below is a writable view rather than a detached copy.
    block = np.ascontiguousarray(block, dtype=np.float32)
    if target_quantiles.shape != source_quantiles.shape:
        raise ValueError("target/source quantile shape mismatch")
    if block.shape[-1] != target_quantiles.shape[1]:
        raise ValueError("block band count does not match quantile table")
    target_quantiles = _strictly_increasing(target_quantiles)
    aligned = np.empty(block.shape, dtype=np.float32, order="C")
    flat = block.reshape(-1, block.shape[-1])
    output = aligned.reshape(-1, aligned.shape[-1])
    for band in range(block.shape[-1]):
        x = flat[:, band]
        tq = target_quantiles[:, band]
        sq = source_quantiles[:, band]
        mapped = np.interp(x, tq, sq).astype(np.float32, copy=False)
        below = x < tq[0]
        above = x > tq[-1]
        if np.any(below):
            slope = (sq[1] - sq[0]) / (tq[1] - tq[0])
            mapped[below] = sq[0] + (x[below] - tq[0]) * slope
        if np.any(above):
            slope = (sq[-1] - sq[-2]) / (tq[-1] - tq[-2])
            mapped[above] = sq[-1] + (x[above] - tq[-1]) * slope
        output[:, band] = mapped
    return aligned


def affine_align(
    block: np.ndarray, target_quantiles: np.ndarray, source_quantiles: np.ndarray
) -> np.ndarray:
    """Median/IQR alignment used as a conservative radiometric alternative."""
    if target_quantiles.shape[0] < 5 or source_quantiles.shape[0] < 5:
        raise ValueError("affine alignment requires 25/50/75 percentiles")
    target_q25, target_med, target_q75 = target_quantiles[2:5]
    source_q25, source_med, source_q75 = source_quantiles[2:5]
    target_iqr = np.maximum(target_q75 - target_q25, 1.0)
    source_iqr = np.maximum(source_q75 - source_q25, 1.0)
    return (
        (np.asarray(block, dtype=np.float32) - target_med) / target_iqr
        * source_iqr
        + source_med
    )


def transform_scene(
    spec: CubeSpec,
    output_dir: Path,
    robust_low: np.ndarray,
    robust_high: np.ndarray,
    pca: PCA,
    rows_per_chunk: int,
    clip: float,
    alignment: str,
    source_quantiles: np.ndarray,
    target_quantiles: np.ndarray | None,
) -> None:
    native_path = output_dir / f"{spec.name}_domain_native98_float16.npy"
    pca_path = output_dir / f"{spec.name}_domain_pca30_float16.npy"
    native = np.lib.format.open_memmap(
        native_path,
        mode="w+",
        dtype=np.float16,
        shape=(spec.height, spec.width, 98),
    )
    spatial = np.lib.format.open_memmap(
        pca_path,
        mode="w+",
        dtype=np.float16,
        shape=(spec.height, spec.width, pca.n_components_),
    )
    with h5py.File(spec.path, "r") as handle:
        source = handle[spec.variable]
        layout = source_layout(source, spec)
        for row0 in range(0, spec.height, rows_per_chunk):
            row1 = min(row0 + rows_per_chunk, spec.height)
            if layout == "CHW":
                block = np.asarray(
                    source[:, row0:row1, :], dtype=np.float32
                ).transpose(1, 2, 0)
            else:
                block = np.asarray(
                    source[:, :, row0:row1], dtype=np.float32
                ).transpose(2, 1, 0)
            if target_quantiles is not None:
                if alignment == "quantile":
                    block = quantile_align(block, target_quantiles, source_quantiles)
                elif alignment == "affine":
                    block = affine_align(block, target_quantiles, source_quantiles)
                elif alignment != "none":
                    raise ValueError(f"Unsupported target alignment: {alignment}")
            robust = robust_transform(block, robust_low, robust_high, clip).astype(
                np.float32
            )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "hypersigma_source_scenealign",
    )
    parser.add_argument("--source-samples", type=int, default=300_000)
    parser.add_argument("--target-samples", type=int, default=150_000)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--target-align", choices=("none", "affine", "quantile"), default="quantile")
    parser.add_argument("--clip", type=float, default=3.0)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Write fitted statistics without materializing full scene caches.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "train": CubeSpec("train", data_dir / "data_hsi.mat", "data", 4040, 2444),
        "scene1": CubeSpec("scene1", data_dir / "test_scene1.mat", "image", 3104, 4507),
        "scene2": CubeSpec("scene2", data_dir / "test_scene2.mat", "image", 3409, 2181),
    }

    source_samples = sample_scene(specs["train"], args.source_samples, args.seed)
    source_quantiles = np.percentile(
        source_samples, DEFAULT_QUANTILES, axis=0
    ).astype(np.float32)
    robust_low = np.percentile(
        source_samples, args.lower_percentile, axis=0
    ).astype(np.float32)
    robust_high = np.percentile(
        source_samples, args.upper_percentile, axis=0
    ).astype(np.float32)
    invalid = robust_high - robust_low < 1e-6
    robust_high[invalid] = robust_low[invalid] + 1.0
    robust_source = robust_transform(
        source_samples, robust_low, robust_high, args.clip
    ).astype(np.float32)
    pca = PCA(
        n_components=args.pca_components,
        whiten=False,
        svd_solver="randomized",
        random_state=args.seed,
    )
    pca.fit(robust_source)

    target_quantiles: dict[str, np.ndarray] = {}
    for index, scene in enumerate(("scene1", "scene2"), start=1):
        samples = sample_scene(
            specs[scene], args.target_samples, args.seed + index * 1009
        )
        target_quantiles[scene] = np.percentile(
            samples, DEFAULT_QUANTILES, axis=0
        ).astype(np.float32)

    stats_path = output_dir / "preprocess_domain.npz"
    np.savez(
        stats_path,
        robust_low=robust_low,
        robust_high=robust_high,
        clip=np.asarray(args.clip),
        quantile_levels=np.asarray(DEFAULT_QUANTILES, dtype=np.float32),
        source_quantiles=source_quantiles,
        scene1_quantiles=target_quantiles["scene1"],
        scene2_quantiles=target_quantiles["scene2"],
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        source_sample_count=np.asarray(args.source_samples),
        target_sample_count=np.asarray(args.target_samples),
    )
    manifest = {
        "statistics_scope": "source train cube only",
        "target_alignment": args.target_align,
        "target_alignment_scope": "scene-specific unlabeled marginal statistics",
        "quantile_levels": list(DEFAULT_QUANTILES),
        "source_samples": args.source_samples,
        "target_samples_per_scene": args.target_samples,
        "pca_components": args.pca_components,
        "pca_fit_scope": "source train cube only",
        "native_scaling": "source-fitted robust 2-98 percentile to [-1,1], clipped",
        "clip": args.clip,
        "stats": str(stats_path),
        "fit_only": bool(args.fit_only),
    }
    (output_dir / "preprocess_domain.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    if args.fit_only:
        return

    transform_scene(
        specs["train"], output_dir, robust_low, robust_high, pca,
        args.rows_per_chunk, args.clip, "none", source_quantiles, None,
    )
    for scene in ("scene1", "scene2"):
        transform_scene(
            specs[scene], output_dir, robust_low, robust_high, pca,
            args.rows_per_chunk, args.clip, args.target_align,
            source_quantiles, target_quantiles[scene],
        )


if __name__ == "__main__":
    main()
