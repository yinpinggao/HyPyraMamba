#!/usr/bin/env python3
"""Prepare leak-resistant component/buffer spatial validation folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_validation import write_component_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "HypraMamba" / "data" / "TreeSpeciesHSI",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "domain_validation" / "component512_buffer64",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--macro-block", type=int, default=512)
    parser.add_argument("--buffer", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    summary = write_component_folds(
        args.data_dir / "val_label.mat",
        args.data_dir / "train_label.mat",
        args.output_dir,
        n_folds=args.folds,
        macro_block=args.macro_block,
        buffer=args.buffer,
        seed=args.seed,
    )
    for item in summary:
        print(
            f"fold={item['fold']} validation={item['validation_pixels']} "
            f"supervision={item['supervision_pixels']} "
            f"protected={item['protected_pixels']} "
            f"classes={item['validation_classes']}"
        )


if __name__ == "__main__":
    main()
