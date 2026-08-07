"""Thin adapter around the author-released GAHT implementation."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Sequence

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SOURCE = PROJECT_ROOT / "third_party" / "GAHT" / "models" / "proposed.py"
UPSTREAM_COMMIT = "d9340ded8fca9cca650a0bbae3853ae499aa638d"


@lru_cache(maxsize=1)
def _upstream_module() -> ModuleType:
    """Load the pinned author's module without copying its network definition."""
    spec = importlib.util.spec_from_file_location(
        "tree_hsi_upstream_gaht_proposed", UPSTREAM_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream GAHT source: {UPSTREAM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GAHT(nn.Module):
    """Input/interface adapter whose learned network is upstream ``MyTransformer``.

    The default group/depth schedule is the author's PU/WHU-LK schedule.  It is
    valid for the shared PCA-30 cube because the upstream implementation pads
    the spectral dimension to a multiple of the first group count.
    """

    def __init__(
        self,
        in_channels: int = 30,
        patch_size: int = 7,
        num_classes: int = 17,
        n_groups: Sequence[int] = (2, 2, 2),
        depths: Sequence[int] = (1, 2, 1),
        embed_dims: Sequence[int] = (256, 128, 64),
        num_heads: Sequence[int] = (8, 4, 2),
        mlp_ratios: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        schedules = {
            "n_groups": tuple(n_groups),
            "depths": tuple(depths),
            "embed_dims": tuple(embed_dims),
            "num_heads": tuple(num_heads),
            "mlp_ratios": tuple(mlp_ratios),
        }
        if any(len(values) != 3 for values in schedules.values()):
            raise ValueError(f"GAHT requires three-stage schedules, got {schedules}")

        upstream_class = _upstream_module().MyTransformer
        self.upstream = upstream_class(
            img_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            num_stages=3,
            n_groups=list(n_groups),
            embed_dims=list(embed_dims),
            num_heads=list(num_heads),
            mlp_ratios=list(mlp_ratios),
            depths=list(depths),
        )
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.n_groups = tuple(int(value) for value in n_groups)
        self.depths = tuple(int(value) for value in depths)

    @property
    def halo(self) -> int:
        return self.patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project datasets yield BCHW; the author implementation consumes B1CHW.
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected BCHW or B1CHW input, got {tuple(x.shape)}")
        return self.upstream(x)

