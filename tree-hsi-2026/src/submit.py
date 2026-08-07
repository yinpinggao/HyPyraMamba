"""The sole CSV submission writer and pixel-ID mapping implementation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SceneInfo:
    scene: str
    height: int
    width: int
    pixel_count: int
    transpose: bool
    flatten_order: str

    def validate(self) -> None:
        if self.pixel_count != self.height * self.width:
            raise ValueError(
                f"{self.scene}: pixel_count={self.pixel_count} but H*W="
                f"{self.height * self.width}"
            )
        if self.flatten_order not in {"C", "F"}:
            raise ValueError(f"{self.scene}: invalid flatten order {self.flatten_order}")


def load_scene_info(path: str | Path) -> list[SceneInfo]:
    rows: list[SceneInfo] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = SceneInfo(
                scene=row["scene"],
                height=int(row["height"]),
                width=int(row["width"]),
                pixel_count=int(row["pixel_count"]),
                transpose=row["transpose"].strip().lower() in {"true", "1", "yes"},
                flatten_order=row["flatten_order"].strip().upper(),
            )
            item.validate()
            rows.append(item)
    if not rows:
        raise ValueError(f"No scenes found in {path}")
    return rows


def map_to_flat(label_hw: np.ndarray, scene: SceneInfo) -> np.ndarray:
    """Map canonical HxW labels to Kaggle's scene-local ID order.

    ``SceneInfo.transpose`` describes the raw MAT spatial-axis correction. It
    must not be applied again after predictions are already canonical HxW.
    """
    label_hw = np.asarray(label_hw)
    if label_hw.shape != (scene.height, scene.width):
        raise ValueError(
            f"{scene.scene}: map shape {label_hw.shape} != "
            f"({scene.height}, {scene.width})"
        )
    return label_hw.reshape(-1, order="C")


def flat_to_map(flat: np.ndarray, scene: SceneInfo) -> np.ndarray:
    """Inverse of map_to_flat."""
    flat = np.asarray(flat)
    if flat.size != scene.pixel_count:
        raise ValueError(f"{scene.scene}: flat length {flat.size} != {scene.pixel_count}")
    return flat.reshape((scene.height, scene.width), order="C")


def id_to_row_col(index: int, scene: SceneInfo) -> tuple[int, int]:
    """Return the HxW coordinate represented by a scene-local ID index."""
    if not 0 <= index < scene.pixel_count:
        raise IndexError(index)
    row, col = divmod(index, scene.width)
    return int(row), int(col)


def model_output_to_labels(prediction: np.ndarray, zero_based: bool = True) -> np.ndarray:
    """Convert network classes 0..16 to competition labels 1..17."""
    prediction = np.asarray(prediction)
    labels = prediction.astype(np.int64, copy=False) + (1 if zero_based else 0)
    if labels.size and (labels.min() < 1 or labels.max() > 17):
        raise ValueError(
            f"Submission labels must be in [1, 17], got [{labels.min()}, {labels.max()}]"
        )
    return labels.astype(np.uint8)


def inspect_sample_schema(sample_path: str | Path) -> tuple[str, str]:
    with Path(sample_path).open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if fieldnames is None or len(fieldnames) != 2:
        raise ValueError(f"Expected two sample columns, got {fieldnames}")
    return fieldnames[0], fieldnames[1]


def write_submission(
    sample_path: str | Path,
    output_path: str | Path,
    labels_by_scene: dict[str, np.ndarray],
    scenes: list[SceneInfo],
) -> None:
    """Write predictions in sample order, then read back and assert the contract."""
    sample_path = Path(sample_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_column, pred_column = inspect_sample_schema(sample_path)

    expected: dict[str, np.ndarray] = {}
    for scene in scenes:
        if scene.scene not in labels_by_scene:
            raise KeyError(f"Missing predictions for {scene.scene}")
        labels = np.asarray(labels_by_scene[scene.scene])
        if labels.shape != (scene.height, scene.width):
            raise ValueError(f"{scene.scene}: unexpected prediction shape {labels.shape}")
        if labels.min() < 1 or labels.max() > 17:
            raise ValueError(f"{scene.scene}: predictions are outside [1, 17]")
        expected[scene.scene] = map_to_flat(labels, scene)

    with sample_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=[id_column, pred_column])
        writer.writeheader()
        for row in reader:
            pixel_id = row[id_column]
            scene_name, local_text = pixel_id.rsplit("_", 1)
            local_index = int(local_text)
            if scene_name not in expected or not 0 <= local_index < expected[scene_name].size:
                raise ValueError(f"Unexpected sample ID {pixel_id}")
            label = int(expected[scene_name][local_index])
            writer.writerow({id_column: pixel_id, pred_column: label})

    # Streaming equivalents of the requested pandas assertions, without holding
    # 21.4 million string IDs in RAM.
    row_count = 0
    with sample_path.open(newline="", encoding="utf-8") as sample_handle, output_path.open(
        newline="", encoding="utf-8"
    ) as output_handle:
        sample_reader = csv.DictReader(sample_handle)
        output_reader = csv.DictReader(output_handle)
        for sample_row, output_row in zip(sample_reader, output_reader, strict=True):
            assert output_row[id_column] == sample_row[id_column]
            label = int(output_row[pred_column])
            assert 1 <= label <= 17
            row_count += 1
    assert row_count == sum(scene.pixel_count for scene in scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sole Tree HSI CSV submission entry")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--probs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = load_scene_info(args.scene_info)
    labels_by_scene: dict[str, np.ndarray] = {}
    for scene in scenes:
        probability_path = args.probs_dir / f"probs_{scene.scene}.npy"
        probabilities = np.load(probability_path, mmap_mode="r")
        expected_shape = (scene.height, scene.width, 17)
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"{probability_path}: shape {probabilities.shape} != {expected_shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{probability_path} contains non-finite probabilities")
        labels_by_scene[scene.scene] = (
            np.argmax(probabilities, axis=2).astype(np.uint8) + 1
        )
    write_submission(args.sample, args.output, labels_by_scene, scenes)
    print(f"Wrote and verified {args.output}")


if __name__ == "__main__":
    main()
