#!/usr/bin/env python3
"""Generate isolated challenger configs without touching champion configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "m5_max_refit90_noweight.yaml",
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "domain_validation" / "component512_buffer64",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "hypersigma_source_scenealign",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "domain_challenger",
    )
    parser.add_argument(
        "--control-output-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "domain_control",
    )
    parser.add_argument(
        "--domain-weight-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "m5_domain_source_scenealign",
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.control_output_dir.mkdir(parents=True, exist_ok=True)

    def write_branch(
        fold: int,
        branch: str,
        output_dir: Path,
        run_prefix: str,
        train_spatial: Path,
        train_spectral: Path,
        scene_spatial: str,
        scene_spectral: str,
        spatial_checkpoint: Path,
        spectral_checkpoint: Path,
    ) -> None:
        config = dict(base)
        config["model_name"] = f"hypersigma_dense_{branch}_component_fold{fold}"
        config["training_mode"] = "component_fold"
        config["domain_fold_path"] = str(
            (args.fold_dir / f"fold{fold}.npz").resolve()
        )
        config["output_root"] = str(
            (PROJECT_ROOT / "outputs" / f"{run_prefix}{fold}").resolve()
        )
        data = dict(config["data"])
        data["spatial_cache"] = str(train_spatial.resolve())
        data["spectral_cache"] = str(train_spectral.resolve())
        config["data"] = data
        hypersigma = dict(config["hypersigma"])
        hypersigma["spatial_checkpoint"] = str(spatial_checkpoint.resolve())
        hypersigma["spectral_checkpoint"] = str(spectral_checkpoint.resolve())
        config["hypersigma"] = hypersigma
        config["inference_data"] = {
            scene: {
                "spatial_cache": str(
                    (PROJECT_ROOT / scene_spatial.format(scene=scene)).resolve()
                ),
                "spectral_cache": str(
                    (PROJECT_ROOT / scene_spectral.format(scene=scene)).resolve()
                ),
            }
            for scene in ("scene1", "scene2")
        }
        path = output_dir / f"fold{fold}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)

    for fold in range(args.folds):
        write_branch(
            fold,
            "domain",
            args.output_dir,
            "m5_domain_component_fold",
            args.cache_dir / "train_domain_pca30_float16.npy",
            args.cache_dir / "train_domain_native98_float16.npy",
            str(args.cache_dir / "{scene}_domain_pca30_float16.npy"),
            str(args.cache_dir / "{scene}_domain_native98_float16.npy"),
            args.domain_weight_dir / "spatial_mae.pt",
            args.domain_weight_dir / "spectral_mae.pt",
        )
        write_branch(
            fold,
            "control",
            args.control_output_dir,
            "m5_component_control_fold",
            PROJECT_ROOT / "data/cache/train_hsmax_pca30_float16.npy",
            PROJECT_ROOT / "data/cache/train_hsmax_native98_float16.npy",
            "data/cache/{scene}_hsmax_pca30_float16.npy",
            "data/cache/{scene}_hsmax_native98_float16.npy",
            PROJECT_ROOT / "outputs/m5_domain_adapt/spatial_mae.pt",
            PROJECT_ROOT / "outputs/m5_domain_adapt/spectral_mae.pt",
        )


if __name__ == "__main__":
    main()
