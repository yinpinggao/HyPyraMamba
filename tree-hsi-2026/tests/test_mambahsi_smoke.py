"""Static and small CUDA smoke checks for the author MambaHSI adapter."""

from __future__ import annotations

import pytest
import torch

from src.models.mambahsi import MambaHSI, UPSTREAM_COMMIT, UPSTREAM_SOURCE


def test_mambahsi_source_is_pinned_author_checkout() -> None:
    assert UPSTREAM_SOURCE.is_file()
    assert UPSTREAM_COMMIT == "a705284aef14e802bb1a09b2a5e73abaefc0b2d0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba-ssm CUDA smoke")
def test_mambahsi_author_forward_and_tile_resize() -> None:
    model = MambaHSI().cuda().eval()
    tile = torch.randn(1, 98, 16, 16, device="cuda")
    with torch.inference_mode():
        native = model.forward_native(tile)
        resized = model(tile)
    assert native.shape == (1, 17, 4, 4)
    assert resized.shape == (1, 17, 16, 16)
