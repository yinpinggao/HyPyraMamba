import torch

from src.models.gaht import GAHT


def test_gaht_forward_smoke() -> None:
    model = GAHT()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 30, 7, 7))
    assert logits.shape == (2, 17)

