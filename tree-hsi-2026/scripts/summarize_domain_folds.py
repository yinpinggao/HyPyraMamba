#!/usr/bin/env python3
"""Aggregate component-fold evidence and emit a fixed-epoch refit config."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import metrics_from_confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-prefix", default="m5_domain_component_fold")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m5_max_refit_all_noweight.yaml",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "hypersigma_source_scenealign",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m5_domain_refit_all.yaml",
    )
    parser.add_argument("--refit-prefix", default="m5_domain_refit_all")
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "domain_validation" / "aggregate_summary.json",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    rows = []
    aggregate_confusion = np.zeros((17, 17), dtype=np.int64)
    for fold in range(args.folds):
        fold_dir = args.fold_output_root / f"{args.run_prefix}{fold}" / "seed0"
        path = fold_dir / "selection_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed fold summary: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        row["fold"] = fold
        rows.append(row)
        metrics_path = fold_dir / "full_val_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregate_confusion += np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    epochs = [int(row["best_epoch"]) for row in rows]
    summary = {
        "folds": rows,
        "mean_oa": statistics.fmean(float(row["oa"]) for row in rows),
        "worst_oa": min(float(row["oa"]) for row in rows),
        "mean_aa": statistics.fmean(float(row["aa"]) for row in rows),
        "median_best_epoch": int(statistics.median(epochs)),
        "mean_best_epoch": statistics.fmean(epochs),
        "aggregate_out_of_fold": metrics_from_confusion(aggregate_confusion),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.summary_only:
        return

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["model_name"] = "hypersigma_dense_domain_refit_all"
    config["training_mode"] = "refit_all"
    config["epochs"] = int(statistics.median(epochs))
    config.pop("epochs_from_checkpoint", None)
    config["output_root"] = str(
        (PROJECT_ROOT / "outputs" / args.refit_prefix).resolve()
    )
    data = dict(config["data"])
    data["spatial_cache"] = str(
        (args.cache_dir / "train_domain_pca30_float16.npy").resolve()
    )
    data["spectral_cache"] = str(
        (args.cache_dir / "train_domain_native98_float16.npy").resolve()
    )
    config["data"] = data
    config["inference_data"] = {
        scene: {
            "spatial_cache": str(
                (args.cache_dir / f"{scene}_domain_pca30_float16.npy").resolve()
            ),
            "spectral_cache": str(
                (args.cache_dir / f"{scene}_domain_native98_float16.npy").resolve()
            ),
        }
        for scene in ("scene1", "scene2")
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output_config}")


if __name__ == "__main__":
    main()
