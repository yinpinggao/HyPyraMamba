"""Patch and dense-tile data utilities shared by the model pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros_like(indices)
    period = 2 * size - 2
    reflected = np.mod(indices, period)
    return np.where(reflected < size, reflected, period - reflected)


def extract_patch(cube: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    rows = reflect_indices(np.arange(row - radius, row + radius + 1), cube.shape[0])
    cols = reflect_indices(np.arange(col - radius, col + radius + 1), cube.shape[1])
    return np.asarray(cube[np.ix_(rows, cols)], dtype=np.float32)


class PatchDataset(Dataset):
    def __init__(
        self,
        cube_path: str | Path,
        label_path: str | Path,
        label_key: str,
        patch_size: int = 25,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        from src.io_mat import load_label

        self.cube_path = Path(cube_path)
        self.cube = np.load(self.cube_path, mmap_mode="r")
        labels, _ = load_label(label_path, label_key)
        self.coords = np.argwhere(labels > 0).astype(np.int32)
        self.targets = labels[labels > 0].astype(np.int64) - 1
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = self.coords[index]
        patch = extract_patch(self.cube, int(row), int(col), self.patch_size)
        if self.augment:
            # Deterministic for a given sampler order, seed, and item index.
            rng = np.random.default_rng(
                self.seed * 1_000_003 + self.epoch * 10_007 + index
            )
            patch = np.rot90(patch, int(rng.integers(0, 4)), axes=(0, 1))
            if rng.random() < 0.5:
                patch = patch[::-1, :, :]
            if rng.random() < 0.5:
                patch = patch[:, ::-1, :]
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1))
        return torch.from_numpy(patch), torch.tensor(self.targets[index], dtype=torch.long)

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Vectorized batch extraction used by PyTorch's map-style DataLoader."""
        index_array = np.asarray(indices, dtype=np.int64)
        coords = self.coords[index_array]
        radius = self.patch_size // 2
        offsets = np.arange(-radius, radius + 1, dtype=np.int64)
        rows = reflect_indices(coords[:, 0, None] + offsets[None, :], self.cube.shape[0])
        cols = reflect_indices(coords[:, 1, None] + offsets[None, :], self.cube.shape[1])
        patches = np.asarray(
            self.cube[rows[:, :, None], cols[:, None, :], :], dtype=np.float32
        )
        items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for offset, index in enumerate(index_array):
            patch = patches[offset]
            if self.augment:
                rng = np.random.default_rng(
                    self.seed * 1_000_003 + self.epoch * 10_007 + int(index)
                )
                patch = np.rot90(patch, int(rng.integers(0, 4)), axes=(0, 1))
                if rng.random() < 0.5:
                    patch = patch[::-1, :, :]
                if rng.random() < 0.5:
                    patch = patch[:, ::-1, :]
            patch = np.ascontiguousarray(patch.transpose(2, 0, 1))
            items.append(
                (
                    torch.from_numpy(patch),
                    torch.tensor(self.targets[index], dtype=torch.long),
                )
            )
        return items


class FullImageDataset(Dataset):
    """Sparse-supervision tiles for dense image-level HSI models.

    Every item is anchored to one official training pixel. The spectral cube is
    reflection padded, while labels outside the real scene stay ignored so an
    edge crop can never duplicate supervision. Labels are returned as 0..C-1
    with ``ignore_index`` for the competition background value zero.
    """

    def __init__(
        self,
        cube_path: str | Path,
        label_path: str | Path,
        label_key: str,
        tile_size: int,
        augment: bool = False,
        seed: int = 0,
        ignore_index: int = -100,
    ) -> None:
        from src.io_mat import load_label

        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.cube_path = Path(cube_path)
        self.cube = np.load(self.cube_path, mmap_mode="r")
        self.labels, _ = load_label(label_path, label_key)
        if self.cube.shape[:2] != self.labels.shape:
            raise ValueError(
                f"Cube spatial shape {self.cube.shape[:2]} != labels {self.labels.shape}"
            )
        self.coords = np.argwhere(self.labels > 0).astype(np.int32)
        if len(self.coords) == 0:
            raise ValueError(f"No supervised pixels in {label_path}:{label_key}")
        self.tile_size = int(tile_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.ignore_index = int(ignore_index)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_row, anchor_col = (int(value) for value in self.coords[index])
        rng = np.random.default_rng(
            self.seed * 1_000_003 + self.epoch * 10_007 + int(index)
        )
        # Jitter the anchor anywhere inside the crop. This preserves at least
        # one supervised pixel while exposing different large-image context.
        if self.augment:
            anchor_offset_row = int(rng.integers(0, self.tile_size))
            anchor_offset_col = int(rng.integers(0, self.tile_size))
        else:
            anchor_offset_row = self.tile_size // 2
            anchor_offset_col = self.tile_size // 2
        row0 = anchor_row - anchor_offset_row
        col0 = anchor_col - anchor_offset_col
        source_rows = np.arange(row0, row0 + self.tile_size, dtype=np.int64)
        source_cols = np.arange(col0, col0 + self.tile_size, dtype=np.int64)
        cube_rows = reflect_indices(source_rows, self.cube.shape[0])
        cube_cols = reflect_indices(source_cols, self.cube.shape[1])
        tile = np.asarray(self.cube[np.ix_(cube_rows, cube_cols)], dtype=np.float32)

        target = np.full(
            (self.tile_size, self.tile_size), self.ignore_index, dtype=np.int64
        )
        valid_rows = (source_rows >= 0) & (source_rows < self.labels.shape[0])
        valid_cols = (source_cols >= 0) & (source_cols < self.labels.shape[1])
        if np.any(valid_rows) and np.any(valid_cols):
            label_block = self.labels[np.ix_(source_rows[valid_rows], source_cols[valid_cols])]
            encoded = np.where(label_block > 0, label_block.astype(np.int64) - 1, self.ignore_index)
            target[np.ix_(valid_rows, valid_cols)] = encoded

        if self.augment:
            rotations = int(rng.integers(0, 4))
            tile = np.rot90(tile, rotations, axes=(0, 1))
            target = np.rot90(target, rotations, axes=(0, 1))
            if rng.random() < 0.5:
                tile = tile[::-1]
                target = target[::-1]
            if rng.random() < 0.5:
                tile = tile[:, ::-1]
                target = target[:, ::-1]
        tile = np.ascontiguousarray(tile.transpose(2, 0, 1))
        target = np.ascontiguousarray(target)
        return torch.from_numpy(tile), torch.from_numpy(target)
