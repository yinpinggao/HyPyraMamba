"""Leak-resistant spatial validation utilities for Tree Species HSI.

The original 128-pixel random block split can cut one labelled crown into
training and validation portions.  This module creates folds by whole
same-class connected components and removes a configurable spatial buffer
around the held-out components from supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


NUM_CLASSES = 17


@dataclass(frozen=True)
class Component:
    cls: int
    component_id: int
    size: int
    row: float
    col: float
    macro_row: int
    macro_col: int


def _components(
    labels: np.ndarray, macro_block: int
) -> tuple[list[Component], dict[int, np.ndarray]]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be HxW, got {labels.shape}")
    structure = ndimage.generate_binary_structure(2, 2)
    result: list[Component] = []
    component_maps: dict[int, np.ndarray] = {}
    for cls in range(1, NUM_CLASSES + 1):
        cc, count = ndimage.label(labels == cls, structure=structure)
        component_maps[cls] = cc
        if count == 0:
            continue
        sizes = np.bincount(cc.ravel(), minlength=count + 1)
        ids = np.arange(1, count + 1)
        centers = ndimage.center_of_mass(
            np.ones_like(cc, dtype=np.uint8), cc, ids
        )
        # Component centroids are sufficient for fold assignment; the exact
        # pixel mask is reconstructed from the component labels below.
        for component_id in range(1, count + 1):
            size = int(sizes[component_id])
            if size == 0:
                continue
            row, col = centers[component_id - 1]
            result.append(
                Component(
                    cls=cls,
                    component_id=component_id,
                    size=size,
                    row=float(row),
                    col=float(col),
                    macro_row=int(row) // macro_block,
                    macro_col=int(col) // macro_block,
                )
            )
    return result, component_maps


def _assign_components(
    components: list[Component], n_folds: int, seed: int
) -> dict[tuple[int, int], int]:
    """Greedily assign whole components while balancing class mass.

    Components are never split.  The macro-region penalty discourages putting
    adjacent components of one class in the same fold, while the class-mass
    term keeps large classes from dominating one fold.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    order = np.arange(len(components))
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda i: components[i].size, reverse=True)
    class_mass = np.zeros((n_folds, NUM_CLASSES), dtype=np.int64)
    macro_counts = [dict() for _ in range(n_folds)]
    assignment: dict[tuple[int, int], int] = {}
    for index in order:
        item = components[index]
        cls_index = item.cls - 1
        macro = (item.macro_row, item.macro_col)
        scores = []
        for fold in range(n_folds):
            same_class = int(class_mass[fold, cls_index])
            macro_penalty = int(macro_counts[fold].get(macro, 0))
            # Relative class mass is the main term; adjacency is a small tie
            # breaker so this remains stable for highly imbalanced classes.
            scores.append((same_class + item.size * 0.05 * macro_penalty, fold))
        _, chosen = min(scores)
        class_mass[chosen, cls_index] += item.size
        macro_counts[chosen][macro] = macro_counts[chosen].get(macro, 0) + 1
        assignment[(item.cls, item.component_id)] = chosen
    return assignment


def make_component_folds(
    labels: np.ndarray,
    train_labels: np.ndarray | None = None,
    n_folds: int = 5,
    macro_block: int = 512,
    buffer: int = 64,
    seed: int = 0,
) -> list[dict[str, np.ndarray | dict[str, object]]]:
    """Create leak-resistant supervision/validation masks.

    ``labels`` is the official validation map.  ``train_labels`` is retained
    in supervision for every fold.  The returned validation mask contains
    whole connected components; the supervision mask removes those components
    and every labelled pixel within ``buffer`` pixels of them.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    if train_labels is None:
        train_labels = np.zeros_like(labels)
    train_labels = np.asarray(train_labels, dtype=np.uint8)
    if labels.shape != train_labels.shape:
        raise ValueError(f"label shape mismatch: {labels.shape} vs {train_labels.shape}")
    if np.any((labels > 0) & (train_labels > 0)):
        raise ValueError("train and validation labels overlap")
    if buffer < 0:
        raise ValueError("buffer must be non-negative")

    components, component_maps = _components(labels, macro_block)
    assignment = _assign_components(components, n_folds, seed)
    combined = np.where(train_labels > 0, train_labels, labels).astype(np.uint8)
    folds: list[dict[str, np.ndarray | dict[str, object]]] = []
    for fold in range(n_folds):
        holdout = np.zeros_like(labels, dtype=bool)
        for cls in range(1, NUM_CLASSES + 1):
            selected_ids = [
                item.component_id
                for item in components
                if item.cls == cls
                and assignment[(item.cls, item.component_id)] == fold
            ]
            if selected_ids:
                holdout |= np.isin(component_maps[cls], selected_ids)
        protected = (
            ndimage.distance_transform_edt(~holdout) <= buffer
            if buffer > 0
            else holdout.copy()
        )
        supervision = combined.copy()
        supervision[protected] = 0
        validation = np.where(holdout, labels, 0).astype(np.uint8)
        metadata = {
            "fold": fold,
            "n_folds": n_folds,
            "macro_block": macro_block,
            "buffer": buffer,
            "seed": seed,
            "validation_pixels": int((validation > 0).sum()),
            "supervision_pixels": int((supervision > 0).sum()),
            "protected_pixels": int(protected.sum()),
            "validation_classes": [
                int(cls) for cls in np.unique(validation[validation > 0])
            ],
        }
        folds.append(
            {
                "supervision": supervision,
                "validation": validation,
                "protected": protected.astype(np.uint8),
                "metadata": metadata,
            }
        )
    return folds


def write_component_folds(
    labels_path: str | Path,
    train_labels_path: str | Path,
    output_dir: str | Path,
    n_folds: int = 5,
    macro_block: int = 512,
    buffer: int = 64,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Load MAT labels through the project loader and write fold artifacts."""
    from src.io_mat import load_label

    labels, _ = load_label(Path(labels_path), "val_label")
    train_labels, _ = load_label(Path(train_labels_path), "train_label")
    folds = make_component_folds(
        labels, train_labels, n_folds, macro_block, buffer, seed
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for item in folds:
        metadata = item["metadata"]
        assert isinstance(metadata, dict)
        path = output / f"fold{int(metadata['fold'])}.npz"
        np.savez_compressed(
            path,
            supervision=item["supervision"],
            validation=item["validation"],
            protected=item["protected"],
        )
        metadata = dict(metadata)
        metadata["path"] = str(path.resolve())
        summary.append(metadata)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
