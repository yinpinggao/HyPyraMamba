import sys
import types

import pytest
import torch
from torch import nn


class DummyMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[-1] == self.d_model
        return x


mamba_ssm = types.ModuleType('mamba_ssm')
mamba_ssm.Mamba = DummyMamba
sys.modules['mamba_ssm'] = mamba_ssm

from model.MambaHSI import ImprovedMambaHSI, ImprovedSpeMamba


@pytest.mark.parametrize('channels', [128, 130])
def test_improved_spe_mamba_preserves_input_channels(channels):
    model = ImprovedSpeMamba(
        channels=channels,
        token_num=4,
        use_residual=True,
        group_num=4,
    )
    x = torch.randn(2, channels, 15, 17)

    y = model(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    ('output_stride', 'expected_size'),
    [
        (1, (15, 17)),
        (2, (7, 8)),
    ],
)
def test_improved_mamba_hsi_output_stride_controls_spatial_size(output_stride, expected_size):
    model = ImprovedMambaHSI(
        in_channels=30,
        hidden_dim=8,
        num_classes=5,
        output_stride=output_stride,
    )
    x = torch.randn(2, 30, 15, 17)

    logits, recon = model(x, return_aux=True)

    assert logits.shape == (2, 5, *expected_size)
    assert recon.shape == (2, 30, *expected_size)


def test_improved_mamba_hsi_rejects_invalid_output_stride():
    with pytest.raises(ValueError):
        ImprovedMambaHSI(output_stride=4)
