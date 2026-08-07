#!/usr/bin/env python3
"""Leakage-safe band standardization and optional PCA cache generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_mat import load_label


@dataclass(frozen=True)
class SavedPCA:
    """Minimal PCA transform reconstructed from saved training-only statistics."""

    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    whiten: bool

    @property
    def n_components_(self) -> int:
        return int(self.components.shape[0])

    def transform(self, samples: np.ndarray) -> np.ndarray:
        transformed = (samples - self.mean) @ self.components.T
        if self.whiten:
            transformed /= np.sqrt(self.explained_variance)
        return transformed


def load_saved_preprocess(path: Path) -> tuple[np.ndarray, np.ndarray, SavedPCA | None]:
    """Load a fitted transform without accessing labels or fitting any estimator."""
    with np.load(path) as payload:
        mean = np.asarray(payload["band_mean"], dtype=np.float64)
        std = np.asarray(payload["band_std"], dtype=np.float64)
        if "pca_components" not in payload.files:
            pca = None
        else:
            required = {"pca_components", "pca_mean", "pca_explained_variance"}
            missing = required.difference(payload.files)
            if missing:
                raise KeyError(f"Incomplete saved PCA in {path}: missing {sorted(missing)}")
            pca = SavedPCA(
                components=np.asarray(payload["pca_components"], dtype=np.float64),
                mean=np.asarray(payload["pca_mean"], dtype=np.float64),
                explained_variance=np.asarray(
                    payload["pca_explained_variance"], dtype=np.float64
                ),
                whiten=bool(np.asarray(payload.get("pca_whiten", 0)).item()),
            )
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError(f"Invalid band statistics in {path}: {mean.shape=} {std.shape=}")
    if np.any(std <= 0):
        raise ValueError(f"Saved band_std contains non-positive values in {path}")
    if pca is not None:
        if pca.components.shape[1] != mean.size or pca.mean.shape != mean.shape:
            raise ValueError(f"Saved PCA dimensions do not match band statistics in {path}")
        if pca.explained_variance.shape != (pca.n_components_,):
            raise ValueError(f"Invalid PCA explained variance shape in {path}")
        if pca.whiten and np.any(pca.explained_variance <= 0):
            raise ValueError(f"Cannot whiten with non-positive PCA variance in {path}")
    return mean, std, pca


def raw_train_samples(dataset: h5py.Dataset, coords: np.ndarray) -> np.ndarray:
    samples = np.empty((len(coords), dataset.shape[0]), dtype=np.float64)
    for index, (row, col) in enumerate(coords):
        samples[index] = dataset[:, int(row), int(col)]
    return samples


def transform_hdf5_cube(
    input_path: Path,
    variable: str,
    output_path: Path,
    expected_hw: tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    pca: PCA | SavedPCA | None,
    rows_per_chunk: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = expected_hw
    output_bands = pca.n_components_ if pca is not None else len(mean)
    target = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float16, shape=(height, width, output_bands)
    )
    with h5py.File(input_path, "r") as handle:
        source = handle[variable]
        if source.shape == (len(mean), height, width):
            layout = "CHW"
        elif source.shape == (len(mean), width, height):
            layout = "CWH"
        else:
            raise ValueError(f"Unexpected source shape {source.shape} for {input_path}")
        for row0 in range(0, height, rows_per_chunk):
            row1 = min(row0 + rows_per_chunk, height)
            if layout == "CHW":
                block = np.asarray(source[:, row0:row1, :], dtype=np.float32).transpose(1, 2, 0)
            else:
                block = np.asarray(source[:, :, row0:row1], dtype=np.float32).transpose(2, 1, 0)
            flat = block.reshape(-1, block.shape[-1], order="C")
            flat = (flat - mean) / std
            if pca is not None:
                flat = pca.transform(flat)
            target[row0:row1] = flat.reshape(row1 - row0, width, output_bands, order="C")
            target.flush()
            print(f"  {input_path.name}: rows {row0}:{row1}/{height}", flush=True)
    del target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--rows-per-chunk", type=int, default=64)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--reuse-stats",
        type=Path,
        default=None,
        help="Load an existing preprocess_m1.npz; never refit standardization or PCA.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Generate only the two test caches. Requires --reuse-stats.",
    )
    parser.add_argument(
        "--disable-saved-pca",
        action="store_true",
        help=(
            "With --reuse-stats, reuse only the saved train-mask band mean/std "
            "and omit its PCA transform. This creates standardized 98-band "
            "caches without refitting or touching validation/test statistics."
        ),
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.test_only and args.reuse_stats is None:
        parser.error("--test-only requires --reuse-stats so PCA cannot be refitted")
    if args.disable_saved_pca and args.reuse_stats is None:
        parser.error("--disable-saved-pca requires --reuse-stats")

    if args.reuse_stats is not None:
        stats_path = args.reuse_stats.resolve()
        mean, std, pca = load_saved_preprocess(stats_path)
        if args.disable_saved_pca:
            pca = None
        train_sample_count = None
        with np.load(stats_path) as payload:
            if "train_sample_count" in payload.files:
                train_sample_count = int(payload["train_sample_count"])
        print(f"Reusing fitted preprocessing from {stats_path}; no fit performed")
    else:
        train_label, _ = load_label(data_dir / "train_label.mat", "train_label")
        coords = np.argwhere(train_label > 0)
        with h5py.File(data_dir / "data_hsi.mat", "r") as handle:
            source = handle["data"]
            samples = raw_train_samples(source, coords)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0, ddof=0)
        std[std < 1e-6] = 1.0
        standardized = (samples - mean) / std
        pca = None
        if args.pca_components > 0:
            pca = PCA(
                n_components=args.pca_components,
                whiten=args.pca_whiten,
                svd_solver="full",
            )
            pca.fit(standardized)

        stats_path = cache_dir / "preprocess_m1.npz"
        train_sample_count = len(samples)
        payload = {
            "band_mean": mean,
            "band_std": std,
            "train_sample_count": train_sample_count,
        }
        if pca is not None:
            payload.update(
                {
                    "pca_components": pca.components_,
                    "pca_mean": pca.mean_,
                    "pca_explained_variance": pca.explained_variance_,
                    "pca_explained_variance_ratio": pca.explained_variance_ratio_,
                    "pca_whiten": np.asarray(int(args.pca_whiten)),
                }
            )
        np.savez(stats_path, **payload)
        print("Saved", stats_path)

    output_bands = pca.n_components_ if pca is not None else len(mean)
    if not args.test_only:
        train_label, _ = load_label(data_dir / "train_label.mat", "train_label")
        transform_hdf5_cube(
            data_dir / "data_hsi.mat",
            "data",
            cache_dir / f"train_pca{output_bands}_float16.npy",
            train_label.shape,
            mean,
            std,
            pca,
            args.rows_per_chunk,
        )
    if args.include_test or args.test_only:
        scenes = [
            ("test_scene1.mat", "image", (3104, 4507), f"scene1_pca{output_bands}_float16.npy"),
            ("test_scene2.mat", "image", (3409, 2181), f"scene2_pca{output_bands}_float16.npy"),
        ]
        for filename, variable, hw, output_name in scenes:
            transform_hdf5_cube(
                data_dir / filename,
                variable,
                cache_dir / output_name,
                hw,
                mean,
                std,
                pca,
                args.rows_per_chunk,
            )
    manifest = {
        "mode": "reuse" if args.reuse_stats is not None else "fit",
        "statistics_path": str(stats_path),
        "train_statistics_scope": "train_label > 0 only",
        "train_samples": train_sample_count,
        "pca_components": int(output_bands),
        "pca_whiten": bool(pca.whiten if isinstance(pca, SavedPCA) else args.pca_whiten),
        "cache_dtype": "float16",
    }
    manifest_name = (
        f"test_cache_pca{output_bands}.json"
        if args.test_only
        else f"preprocess_pca{output_bands}.json"
    )
    (cache_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
