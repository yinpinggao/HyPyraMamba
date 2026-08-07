"""Confusion-matrix metrics for 17-class pixel classification."""

from __future__ import annotations

import numpy as np


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    confusion = np.asarray(confusion, dtype=np.int64)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class_den = confusion.sum(axis=1)
    per_class = np.divide(
        np.diag(confusion),
        per_class_den,
        out=np.zeros(confusion.shape[0], dtype=np.float64),
        where=per_class_den > 0,
    )
    oa = correct / total if total else 0.0
    valid = per_class_den > 0
    aa = float(per_class[valid].mean()) if np.any(valid) else 0.0
    expected = float((confusion.sum(axis=0) * confusion.sum(axis=1)).sum()) / max(total**2, 1)
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "oa": float(oa),
        "aa": aa,
        "kappa": float(kappa),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": confusion.tolist(),
        "evaluated_pixels": total,
    }

