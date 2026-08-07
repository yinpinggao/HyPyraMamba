"""Minimal forward smoke test for the pinned author SSFTT adapter."""

from pathlib import Path

import torch

from src.models.ssftt import SSFTT, UPSTREAM_COMMIT, UPSTREAM_SOURCE


def test_ssftt_author_forward() -> None:
    assert UPSTREAM_COMMIT == "994f5d72e4cf22e92d9b5ab9743e33dfdf642e1d"
    assert UPSTREAM_SOURCE == (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "HSI_SSFTT"
        / "cls_SSFTT_IP"
        / "SSFTTnet.py"
    )

    model = SSFTT(in_channels=30, patch_size=13, num_classes=17).eval()
    assert model.upstream.__class__.__name__ == "SSFTTnet"
    assert model.upstream.__class__.__module__ == "_tree_hsi_upstream_ssftt"

    with torch.inference_mode():
        logits = model(torch.randn(2, 30, 13, 13))
    assert logits.shape == (2, 17)


if __name__ == "__main__":
    test_ssftt_author_forward()
    print("SSFTT author-source forward smoke test passed")
