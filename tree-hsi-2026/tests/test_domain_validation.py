from __future__ import annotations

import numpy as np

from src.domain_validation import make_component_folds
from src.preprocess_hypersigma_domain import affine_align, quantile_align


def test_component_folds_keep_components_together_and_buffered() -> None:
    labels = np.zeros((32, 40), dtype=np.uint8)
    # Two separated components of each class, enough to populate folds.
    labels[2:6, 2:7] = 1
    labels[20:24, 2:7] = 1
    labels[2:6, 25:30] = 2
    labels[20:24, 25:30] = 2
    train = np.zeros_like(labels)
    folds = make_component_folds(
        labels, train, n_folds=2, macro_block=16, buffer=2, seed=7
    )
    held = np.zeros_like(labels, dtype=np.uint8)
    for item in folds:
        validation = item["validation"]
        supervision = item["supervision"]
        assert not np.any((validation > 0) & (supervision > 0))
        held += (validation > 0).astype(np.uint8)
    assert np.all(held[labels > 0] == 1)
    assert np.all(held[labels == 0] == 0)


def test_quantile_and_affine_alignment_preserve_shapes() -> None:
    source_q = np.array(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0], [6.0, 12.0]],
        dtype=np.float32,
    )
    target_q = source_q * 2.0 + 10.0
    values = np.array([[[10.0, 10.0], [16.0, 22.0]]], dtype=np.float32)
    backing = np.zeros((1, 2, 4), dtype=np.float32)
    backing[..., ::2] = values
    block = backing[..., ::2]
    assert not block.flags.c_contiguous
    aligned = quantile_align(block, target_q, source_q)
    affine = affine_align(block, target_q, source_q)
    assert aligned.shape == block.shape
    assert affine.shape == block.shape
    assert np.isfinite(aligned).all()
    assert np.isfinite(affine).all()
    np.testing.assert_allclose(aligned[0, 0], [0.0, 0.0], atol=1e-5)
