"""Thin tile adapter around the author-released MambaHSI network.

No learned layer is defined here.  The network is imported directly from the
pinned checkout at ``third_party/MambaHSI/model/MambaHSI.py``.  This module only
validates the competition tensor contract and optionally restores the author's
stride-4 logits to the input tile size using the same interpolation convention
as the released training script.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SOURCE = (
    PROJECT_ROOT / "third_party" / "MambaHSI" / "model" / "MambaHSI.py"
)
UPSTREAM_COMMIT = "a705284aef14e802bb1a09b2a5e73abaefc0b2d0"


@lru_cache(maxsize=1)
def _upstream_mambahsi_class() -> type[nn.Module]:
    """Import the author's class without copying or modifying its definition."""
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(f"Missing upstream MambaHSI source: {UPSTREAM_SOURCE}")

    # Importing the author module imports mamba_ssm at module scope.  Keep the
    # error actionable because mamba_ssm/causal-conv1d wheels are ABI-specific.
    try:
        if importlib.util.find_spec("mamba_ssm") is None:
            raise ModuleNotFoundError("No module named 'mamba_ssm'")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "MambaHSI requires a precompiled mamba-ssm wheel matching Python, "
            "CUDA major, PyTorch major.minor, and torch._C._GLIBCXX_USE_CXX11_ABI; "
            "do not compile it implicitly during an experiment launch."
        ) from exc

    module_name = "_tree_hsi_upstream_mambahsi"
    spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {UPSTREAM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise ImportError(
            "Could not load the author MambaHSI module. Verify that mamba-ssm "
            "and causal-conv1d are precompiled for this exact Python/CUDA/torch/ABI "
            "environment."
        ) from exc
    return module.MambaHSI


class MambaHSI(nn.Module):
    """Competition tile contract wrapped around the unmodified author model.

    The released spatial branch flattens ``B*H*W`` into one sequence, so batch
    size one is required to keep independent tiles from interacting.  Native
    logits are at spatial stride four.  ``resize_logits=True`` applies the same
    bilinear, ``align_corners=True`` resize used by the author's loss/evaluation
    code and returns one logit vector per input pixel.
    """

    output_stride = 4

    def __init__(
        self,
        in_channels: int = 98,
        num_classes: int = 17,
        hidden_dim: int = 128,
        mamba_type: str = "both",
        token_num: int = 4,
        group_num: int = 4,
        use_residual: bool = True,
        use_att: bool = True,
        resize_logits: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 98:
            raise ValueError(
                "M4 is configured for the competition's 98 raw standardized "
                f"bands; received {in_channels}."
            )
        if num_classes != 17:
            raise ValueError(
                f"M4 is configured for the competition's 17 classes; received {num_classes}."
            )
        if hidden_dim % group_num != 0 or 128 % group_num != 0:
            raise ValueError("group_num must divide both hidden_dim and the 128-channel head")
        if mamba_type not in {"spa", "spe", "both"}:
            raise ValueError(f"Unsupported author mamba_type: {mamba_type}")

        upstream_class = _upstream_mambahsi_class()
        self.upstream = upstream_class(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            use_residual=use_residual,
            mamba_type=mamba_type,
            token_num=token_num,
            group_num=group_num,
            use_att=use_att,
        )
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.mamba_type = mamba_type
        self.resize_logits = bool(resize_logits)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Return the author's native stride-4 logits without postprocessing."""
        self._validate_input(x)
        return self.upstream(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        input_size = tuple(x.shape[-2:])
        logits = self.upstream(x)
        if self.resize_logits and tuple(logits.shape[-2:]) != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=True
            )
        return logits

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "MambaHSI tile input must be [batch, 98, height, width], "
                f"received {tuple(x.shape)}"
            )
        if x.shape[0] != 1:
            raise ValueError(
                "The author spatial Mamba merges B*H*W into one sequence; use "
                "batch_size=1 so separate tiles cannot interact."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} spectral bands, received {x.shape[1]}"
            )
        if min(x.shape[-2:]) < self.output_stride:
            raise ValueError("Tile height and width must both be at least 4 pixels")
