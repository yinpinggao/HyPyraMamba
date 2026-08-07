"""Thin input adapter around the author-released SSFTT implementation.

Upstream source: ``third_party/HSI_SSFTT/cls_SSFTT_IP/SSFTTnet.py``
Pinned commit: ``994f5d72e4cf22e92d9b5ab9743e33dfdf642e1d``

The author model consumes ``[B, 1, bands, height, width]``.  The shared patch
dataset emits ``[B, bands, height, width]``, so this adapter only inserts the
singleton input-channel dimension.  All learned layers remain in the imported
upstream ``SSFTTnet`` instance.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SOURCE = (
    PROJECT_ROOT / "third_party" / "HSI_SSFTT" / "cls_SSFTT_IP" / "SSFTTnet.py"
)
UPSTREAM_COMMIT = "994f5d72e4cf22e92d9b5ab9743e33dfdf642e1d"


@lru_cache(maxsize=1)
def _upstream_ssftt_class() -> type[nn.Module]:
    """Import the author's class directly from the pinned local checkout."""
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(f"Missing upstream SSFTT source: {UPSTREAM_SOURCE}")
    module_name = "_tree_hsi_upstream_ssftt"
    spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {UPSTREAM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.SSFTTnet


class SSFTT(nn.Module):
    """Compatibility wrapper containing the unmodified author model."""

    def __init__(
        self,
        in_channels: int = 30,
        patch_size: int = 13,
        num_classes: int = 17,
    ) -> None:
        super().__init__()
        # The released IP architecture hard-codes Conv2d input channels as
        # 8 * 28, which follows from a 30-band input and a depth-3 Conv3d.
        if in_channels != 30:
            raise ValueError(
                "The pinned author SSFTTnet supports exactly 30 spectral bands; "
                f"received {in_channels}."
            )
        if patch_size < 5 or patch_size % 2 == 0:
            raise ValueError("patch_size must be odd and at least 5")

        upstream_class = _upstream_ssftt_class()
        self.upstream = upstream_class(in_channels=1, num_classes=num_classes)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes

    @property
    def halo(self) -> int:
        return self.patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SSFTT adapter expects [batch, bands, height, width], "
                f"received shape {tuple(x.shape)}"
            )
        expected = (self.in_channels, self.patch_size, self.patch_size)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"SSFTT adapter expects each patch to have shape {expected}, "
                f"received {tuple(x.shape[1:])}"
            )
        return self.upstream(x.unsqueeze(1))

