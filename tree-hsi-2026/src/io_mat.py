"""MATLAB I/O with explicit HDF5/v7.3 and spatial-axis handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.io import loadmat, whosmat


@dataclass(frozen=True)
class MatArrayInfo:
    path: Path
    variable: str
    shape: tuple[int, ...]
    dtype: str
    is_hdf5: bool


def _visible_keys(keys: Iterable[str]) -> list[str]:
    return [key for key in keys if not key.startswith("__") and key != "#refs#"]


def inspect_mat(path: str | Path) -> list[MatArrayInfo]:
    """Return variable metadata without loading a large array into memory."""
    path = Path(path)
    is_hdf5 = h5py.is_hdf5(path)
    if is_hdf5:
        infos: list[MatArrayInfo] = []
        with h5py.File(path, "r") as handle:
            for key in _visible_keys(handle.keys()):
                value = handle[key]
                if isinstance(value, h5py.Dataset):
                    infos.append(
                        MatArrayInfo(path, key, tuple(value.shape), str(value.dtype), True)
                    )
        return infos
    return [
        MatArrayInfo(path, name, tuple(shape), str(dtype), False)
        for name, shape, dtype in whosmat(path)
    ]


def load_mat_array(
    path: str | Path,
    preferred_keys: Iterable[str] = (),
) -> tuple[np.ndarray, MatArrayInfo]:
    """Load one numeric array, using h5py iff the file is MATLAB v7.3/HDF5."""
    path = Path(path)
    candidates = list(preferred_keys)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            keys = _visible_keys(handle.keys())
            key = next((name for name in candidates if name in handle), None)
            if key is None:
                if len(keys) != 1:
                    raise KeyError(f"Cannot choose a variable in {path}; found {keys}")
                key = keys[0]
            array = np.asarray(handle[key])
        return array, MatArrayInfo(path, key, tuple(array.shape), str(array.dtype), True)

    payload = loadmat(path)
    keys = _visible_keys(payload.keys())
    key = next((name for name in candidates if name in payload), None)
    if key is None:
        if len(keys) != 1:
            raise KeyError(f"Cannot choose a variable in {path}; found {keys}")
        key = keys[0]
    array = np.asarray(payload[key])
    return array, MatArrayInfo(path, key, tuple(array.shape), str(array.dtype), False)


def cube_storage_to_hwc(
    array: np.ndarray,
    expected_hw: tuple[int, int],
    expected_bands: int = 98,
) -> tuple[np.ndarray, str]:
    """Convert an observed cube to HWC without relying on implicit reshape order."""
    height, width = expected_hw
    shape = tuple(array.shape)
    layouts = {
        (expected_bands, height, width): ((1, 2, 0), "CHW -> HWC"),
        (expected_bands, width, height): ((2, 1, 0), "CWH -> HWC"),
        (height, width, expected_bands): ((0, 1, 2), "HWC (unchanged)"),
        (width, height, expected_bands): ((1, 0, 2), "WHC -> HWC"),
    }
    if shape not in layouts:
        raise ValueError(
            f"Cube shape {shape} cannot be reconciled with HxW={expected_hw} "
            f"and bands={expected_bands}"
        )
    axes, description = layouts[shape]
    return np.transpose(array, axes), description


def describe_cube_layout(
    shape: tuple[int, ...],
    expected_hw: tuple[int, int],
    expected_bands: int = 98,
) -> str:
    """Describe the required axis conversion from metadata alone."""
    height, width = expected_hw
    layouts = {
        (expected_bands, height, width): "CHW -> HWC",
        (expected_bands, width, height): "CWH -> HWC",
        (height, width, expected_bands): "HWC (unchanged)",
        (width, height, expected_bands): "WHC -> HWC",
    }
    if tuple(shape) not in layouts:
        raise ValueError(
            f"Cube shape {shape} cannot be reconciled with HxW={expected_hw} "
            f"and bands={expected_bands}"
        )
    return layouts[tuple(shape)]


def load_label(path: str | Path, key: str) -> tuple[np.ndarray, MatArrayInfo]:
    label, info = load_mat_array(path, preferred_keys=(key,))
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label map in {path}, got {label.shape}")
    if label.min() < 0 or label.max() > 17:
        raise ValueError(f"Labels in {path} are outside [0, 17]")
    return label.astype(np.uint8, copy=False), info
