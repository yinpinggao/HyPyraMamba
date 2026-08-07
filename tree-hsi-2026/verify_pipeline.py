#!/usr/bin/env python3
"""P0 data-alignment audit and validation-label truth-loop test."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_mat import describe_cube_layout, inspect_mat, load_label
from src.submit import (
    SceneInfo,
    flat_to_map,
    id_to_row_col,
    load_scene_info,
    map_to_flat,
    model_output_to_labels,
)


def default_data_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT.parent / "HypraMamba" / "data" / "TreeSpeciesHSI",
    ]
    for candidate in candidates:
        if (candidate / "sample_submission.csv").exists():
            return candidate
    return candidates[0]


def audit_sample(sample_path: Path, scenes: list[SceneInfo]) -> tuple[list[str], str, str]:
    scene_lookup = {scene.scene: scene for scene in scenes}
    counts: Counter[str] = Counter()
    transitions: list[str] = []
    previous_scene: str | None = None
    expected_local: dict[str, int] = Counter()
    with sample_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != 2:
            raise ValueError(f"Invalid sample schema: {reader.fieldnames}")
        id_column, pred_column = reader.fieldnames
        for global_index, row in enumerate(reader):
            pixel_id = row[id_column]
            scene_name, local_text = pixel_id.rsplit("_", 1)
            local_index = int(local_text)
            if scene_name not in scene_lookup:
                raise ValueError(f"Unknown scene in sample row {global_index}: {pixel_id}")
            if local_index != expected_local[scene_name]:
                raise ValueError(
                    f"Non-contiguous ID at row {global_index}: {pixel_id}; "
                    f"expected {expected_local[scene_name]}"
                )
            if previous_scene != scene_name:
                transitions.append(f"row {global_index}: {scene_name}_00000000")
                previous_scene = scene_name
            expected_local[scene_name] += 1
            counts[scene_name] += 1
    for scene in scenes:
        if counts[scene.scene] != scene.pixel_count:
            raise ValueError(
                f"{scene.scene}: sample has {counts[scene.scene]} rows, expected {scene.pixel_count}"
            )
    return transitions, id_column, pred_column


def write_sparse_truth_loop(
    val_label: np.ndarray,
    output_path: Path,
) -> tuple[int, int, float]:
    """Round-trip every labeled validation pixel through IDs and a real CSV."""
    height, width = val_label.shape
    val_scene = SceneInfo("val", height, width, height * width, False, "C")
    # Define the expected ID vector independently of src.submit.map_to_flat.
    # Otherwise the same wrong bijection in writer and reader can pass 100%.
    flat_truth = val_label.reshape(-1, order="C")
    selected = np.flatnonzero(flat_truth > 0)

    # Emulate model outputs (0..16), then apply the mandatory +1 conversion.
    zero_based_predictions = flat_truth[selected].astype(np.int64) - 1
    submission_labels = model_output_to_labels(zero_based_predictions, zero_based=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for flat_index, label in zip(selected, submission_labels, strict=True):
            writer.writerow([f"val_{int(flat_index):08d}", int(label)])

    recovered_flat = np.zeros(val_scene.pixel_count, dtype=np.uint8)
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scene_name, index_text = row["id"].rsplit("_", 1)
            if scene_name != "val":
                raise ValueError(row["id"])
            recovered_flat[int(index_text)] = int(row["label"])
    recovered = recovered_flat.reshape((height, width), order="C")
    eval_mask = val_label > 0
    correct = int(np.count_nonzero(recovered[eval_mask] == val_label[eval_mask]))
    total = int(np.count_nonzero(eval_mask))
    return correct, total, correct / total


def assert_independent_row_major_contract() -> None:
    """Check fixed ID values without using mapper/inverse self-consistency."""
    sentinel = np.arange(12, dtype=np.int64).reshape(3, 4)
    scene = SceneInfo("sentinel", 3, 4, 12, True, "C")
    expected = np.arange(12, dtype=np.int64)
    np.testing.assert_array_equal(map_to_flat(sentinel, scene), expected)
    expected_coords = {
        0: (0, 0),
        1: (0, 1),
        3: (0, 3),
        4: (1, 0),
        11: (2, 3),
    }
    for local_id, coordinate in expected_coords.items():
        actual = id_to_row_col(local_id, scene)
        if actual != coordinate:
            raise AssertionError(
                f"ID {local_id}: got coordinate {actual}, expected {coordinate}"
            )
    np.testing.assert_array_equal(flat_to_map(expected, scene), sentinel)


def counts_1_to_17(label: np.ndarray) -> list[int]:
    return np.bincount(label.reshape(-1), minlength=18)[1:18].astype(int).tolist()


def build_report(
    data_dir: Path,
    report_path: Path,
    train_label: np.ndarray,
    val_label: np.ndarray,
    scenes: list[SceneInfo],
    transitions: list[str],
    sample_columns: tuple[str, str],
    truth_correct: int,
    truth_total: int,
) -> None:
    files = [
        "data_hsi.mat",
        "train_label.mat",
        "val_label.mat",
        "test_scene1.mat",
        "test_scene2.mat",
    ]
    metadata_rows: list[str] = []
    for filename in files:
        for info in inspect_mat(data_dir / filename):
            metadata_rows.append(
                f"| `{filename}` | `{info.variable}` | `{info.shape}` | `{info.dtype}` | "
                f"{'v7.3 / HDF5' if info.is_hdf5 else 'classic MAT'} |"
            )

    train_counts = counts_1_to_17(train_label)
    val_counts = counts_1_to_17(val_label)
    distribution_rows = [
        f"| {cls} | {train_counts[cls - 1]} | {val_counts[cls - 1]} |"
        for cls in range(1, 18)
    ]
    scene_rows = []
    formula_rows = []
    for scene in scenes:
        scene_rows.append(
            f"| {scene.scene} | {scene.height} | {scene.width} | {scene.pixel_count} | "
            f"{scene.transpose} | `{scene.flatten_order}` |"
        )
        probes = [0, 1, scene.width - 1, scene.width, scene.pixel_count - 1]
        probe_text = ", ".join(
            f"{idx}->({id_to_row_col(idx, scene)[0]}, {id_to_row_col(idx, scene)[1]})"
            for idx in probes
        )
        formula_rows.append(f"- `{scene.scene}` probes: {probe_text}")

    overlap = int(np.count_nonzero((train_label > 0) & (val_label > 0)))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Tree Species HSI 2026: Data Alignment Report",
                "",
                "## P0 result",
                "",
                f"- Validation row-major truth-loop: **{truth_correct}/{truth_total} "
                f"= {100.0 * truth_correct / truth_total:.6f}%**.",
                "- Status: **PASS**. No training is started by this audit.",
                f"- Official train/val labeled-pixel overlap: **{overlap}**.",
                f"- Actual sample columns: `{sample_columns[0]}`, `{sample_columns[1]}`. "
                "The pipeline reads these names from the sample instead of hard-coding a conflicting schema.",
                "- The former writer/inverse-writer round trip was not independent: the same wrong "
                "transpose could cancel itself and report 100%. The current audit uses fixed sentinel "
                "IDs and a directly defined row-major validation vector.",
                "- Controlled submissions made from identical cached predictions also support this "
                "interpretation: canonical HxW + C-order scored `0.11278`, while applying the metadata "
                "transpose again scored `0.04828`. These scores are diagnostic evidence only, not a "
                "hyperparameter-tuning signal.",
                "",
                "## Real file layouts",
                "",
                "| file | variable | observed Python shape | dtype | loader |",
                "|---|---|---:|---|---|",
                *metadata_rows,
                "",
                "`h5py.is_hdf5(path)` is checked before loading. The labeled cube is stored as "
                "`(98, 4040, 2444)` and maps to label `(H, W)=(4040, 2444)` via "
                "`transpose(1, 2, 0)`. Test cubes are stored as `(98, W, H)` and map to HWC via "
                "`transpose(2, 1, 0)`.",
                "",
                "## Submission scene segmentation",
                "",
                "| scene | H | W | pixels | raw MAT axis correction | CSV order |",
                "|---|---:|---:|---:|---|---|",
                *scene_rows,
                "",
                "The sample was streamed end-to-end. Its only scene transitions are:",
                "",
                *[f"- {item}" for item in transitions],
                "",
                "## ID to `(scene, row, col)` mapping",
                "",
                "`scene_info.transpose=True` describes the raw HDF5 spatial layout: the observed "
                "test cube `(98, W, H)` is transposed once to canonical `(H, W, 98)`. It is **not** "
                "a second transpose instruction for CSV generation. Once predictions have shape "
                "`(H, W)`, sample IDs enumerate them directly in row-major C order:",
                "",
                "```text",
                "local_index = integer suffix of sample ID",
                "row = local_index // W",
                "col = local_index % W",
                "ID = f\"{scene}_{local_index:08d}\"",
                "```",
                "",
                *formula_rows,
                "",
                "Equivalently, `flat = prediction_hw.reshape(-1, order=\"C\")`. All "
                "reshape/flatten calls in the implementation pass an explicit `order`.",
                "",
                "## Label offset",
                "",
                "Network predictions are treated as `0..16`; `model_output_to_labels(..., "
                "zero_based=True)` adds exactly one and rejects anything outside `1..17`. The "
                "truth-loop exercised this conversion for every labeled validation pixel.",
                "",
                "## Class distributions",
                "",
                "Background (`0`) is excluded below. Official train contains exactly 10 pixels per class.",
                "",
                "| class | train pixels | val pixels |",
                "|---:|---:|---:|",
                *distribution_rows,
                "",
                f"- Train labeled total: {int(np.count_nonzero(train_label))}",
                f"- Validation labeled total: {int(np.count_nonzero(val_label))}",
                f"- Train background: {int(np.count_nonzero(train_label == 0))}",
                f"- Validation background: {int(np.count_nonzero(val_label == 0))}",
                "",
                "## Submission contract",
                "",
                "`src/submit.py` is the sole CSV writer. After writing it reads the CSV back and asserts:",
                "",
                "- output IDs exactly equal sample IDs and preserve their order;",
                "- every prediction is between 1 and 17;",
                "- output row count exactly equals sample row count.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "docs" / "data_alignment_report.md"
    )
    parser.add_argument(
        "--roundtrip-csv",
        type=Path,
        default=None,
        help="Keep the validation truth-loop CSV at this path (default: temporary file).",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    print("[P0] data directory:", data_dir)
    train_label, train_info = load_label(data_dir / "train_label.mat", "train_label")
    val_label, val_info = load_label(data_dir / "val_label.mat", "val_label")
    if train_label.shape != val_label.shape:
        raise ValueError("Official train and validation label shapes differ")
    if np.any((train_label > 0) & (val_label > 0)):
        raise ValueError("Official train and validation masks overlap")

    print("[P0/order] canonical HxW -> sample IDs uses reshape(order='C')")
    sentinel = np.arange(12).reshape(3, 4)
    print("[P0/order] sentinel C:", sentinel.reshape(-1, order="C").tolist())
    print("[P0/order] sentinel F:", sentinel.reshape(-1, order="F").tolist())
    assert_independent_row_major_contract()
    print("[P0/order] independent sentinel contract: PASS")

    print("[P0/MAT] train label:", train_info)
    print("[P0/MAT] val label:", val_info)
    cube_specs = [
        ("data_hsi.mat", ("data", "image", "hsi"), train_label.shape),
        ("test_scene1.mat", ("image", "data", "hsi"), (3104, 4507)),
        ("test_scene2.mat", ("image", "data", "hsi"), (3409, 2181)),
    ]
    for filename, keys, expected_hw in cube_specs:
        infos = inspect_mat(data_dir / filename)
        info = next((item for item in infos if item.variable in keys), None)
        if info is None:
            raise KeyError(f"None of {keys} found in {filename}: {infos}")
        conversion = describe_cube_layout(info.shape, expected_hw)
        print(
            f"[P0/MAT] {filename}: hdf5={info.is_hdf5} observed={info.shape} "
            f"expected_hw={expected_hw} conversion={conversion}"
        )

    scenes = load_scene_info(data_dir / "scene_info.csv")
    transitions, id_column, pred_column = audit_sample(
        data_dir / "sample_submission.csv", scenes
    )
    print(f"[P0/scenes] sample columns: {id_column!r}, {pred_column!r}")
    for transition in transitions:
        print("[P0/scenes]", transition)
    for scene in scenes:
        print(
            f"[P0/scenes] {scene.scene}: H={scene.height} W={scene.width} "
            f"count={scene.pixel_count} raw_transpose={scene.transpose} csv_order=C"
        )
        print(
            f"[P0/scenes] {scene.scene} formula evidence: ID 0 -> "
            f"{id_to_row_col(0, scene)}, ID W -> {id_to_row_col(scene.width, scene)}"
        )

    if args.roundtrip_csv is None:
        with tempfile.TemporaryDirectory(prefix="tree_hsi_p0_") as temp_dir:
            truth_path = Path(temp_dir) / "val_truth_roundtrip.csv"
            correct, total, rate = write_sparse_truth_loop(val_label, truth_path)
    else:
        correct, total, rate = write_sparse_truth_loop(val_label, args.roundtrip_csv)
    print(
        f"[P0/offset] zero-based 0..16 -> submission 1..17 exercised; "
        f"truth-loop={correct}/{total} ({rate:.12%})"
    )
    if correct != total:
        raise RuntimeError("P0 truth-loop is not 100%; training is forbidden")

    build_report(
        data_dir,
        args.report,
        train_label,
        val_label,
        scenes,
        transitions,
        (id_column, pred_column),
        correct,
        total,
    )
    print("[P0] PASS: independent row-major contract and 100% truth-loop")
    print("[P0] report:", args.report)


if __name__ == "__main__":
    main()
