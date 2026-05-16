"""Tests for `reliability_diagram`.

Equal-width top-1 probability buckets over `[0, 1]`. Returns `[K, 3]` rows
with columns `(conf_b, acc_b, n_b)`. Empty buckets are `(0, 0, 0)`. The
sum of `n_b` equals the number of valid (masked) residues.
"""

from __future__ import annotations

import torch

from proteinfoundation.confidence.metrics import reliability_diagram


def test_shape_is_num_bins_by_three() -> None:
    b, n, k = 1, 32, 5
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = torch.ones(b, n)
    out = reliability_diagram(logits, labels, mask, num_bins_ece=10)
    assert out.shape == (10, 3)
    assert out.dtype == torch.float32


def test_counts_sum_to_mask_sum() -> None:
    torch.manual_seed(0)
    b, n, k = 2, 64, 7
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = (torch.rand(b, n) > 0.3).to(torch.float32)
    out = reliability_diagram(logits, labels, mask, num_bins_ece=10)
    total = mask.sum()
    assert torch.allclose(out[:, 2].sum(), total, atol=1e-4)


def test_perfect_calibration_high_confidence_bucket() -> None:
    b, n, k = 1, 64, 5
    labels = torch.randint(0, k, (b, n))
    logits = torch.full((b, n, k), -1e4)
    for j in range(n):
        logits[0, j, labels[0, j]] = 1e4
    mask = torch.ones(b, n)
    out = reliability_diagram(logits, labels, mask, num_bins_ece=10)
    last = out[-1]
    assert last[2].item() == float(n)
    assert last[1].item() == 1.0
    assert last[0].item() > 0.9


def test_empty_buckets_are_zero_rows() -> None:
    b, n, k = 1, 4, 4
    logits = torch.full((b, n, k), -1e4)
    logits[..., 0] = 1e4
    labels = torch.zeros(b, n, dtype=torch.int64)
    mask = torch.ones(b, n)
    out = reliability_diagram(logits, labels, mask, num_bins_ece=10)
    nonempty = out[:, 2] > 0
    empty_rows = out[~nonempty]
    assert torch.all(empty_rows == 0.0)
