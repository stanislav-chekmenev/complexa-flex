"""Tests for `expected_calibration_error_adaptive`.

Adaptive ECE uses equal-mass top-1-probability buckets via quantiles. Ties
on identical confidence values must not crash; all-masked input must
return zero (no NaN); output is a 0-dim fp32 in `[0, 1]`.
"""

from __future__ import annotations

import torch

from proteinfoundation.confidence.metrics import expected_calibration_error_adaptive


def test_perfect_calibration_yields_low_ece() -> None:
    b, n, k = 1, 128, 5
    labels = torch.randint(0, k, (b, n))
    logits = torch.full((b, n, k), -1e4)
    for j in range(n):
        logits[0, j, labels[0, j]] = 1e4
    mask = torch.ones(b, n)
    ece = expected_calibration_error_adaptive(logits, labels, mask, num_bins_ece=15)
    assert ece.item() < 0.05


def test_worst_calibration_yields_high_ece() -> None:
    b, n, k = 1, 128, 5
    labels = torch.zeros(b, n, dtype=torch.int64)
    logits = torch.full((b, n, k), -1e4)
    logits[..., 1] = 1e4
    mask = torch.ones(b, n)
    ece = expected_calibration_error_adaptive(logits, labels, mask, num_bins_ece=15)
    assert ece.item() > 0.5


def test_output_is_scalar_fp32_in_unit_interval() -> None:
    torch.manual_seed(0)
    b, n, k = 2, 32, 50
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = (torch.rand(b, n) > 0.3).to(torch.float32)
    ece = expected_calibration_error_adaptive(logits, labels, mask, num_bins_ece=15)
    assert torch.is_tensor(ece)
    assert ece.ndim == 0
    assert ece.dtype == torch.float32
    assert 0.0 <= ece.item() <= 1.0


def test_tied_confidences_do_not_crash() -> None:
    b, n, k = 1, 1000, 4
    logits = torch.full((b, n, k), 0.0)
    logits[..., 0] = 5.0
    labels = torch.randint(0, k, (b, n))
    mask = torch.ones(b, n)
    ece = expected_calibration_error_adaptive(logits, labels, mask, num_bins_ece=15)
    assert torch.is_tensor(ece)
    assert torch.isfinite(ece)
    assert 0.0 <= ece.item() <= 1.0


def test_all_masked_returns_zero() -> None:
    b, n, k = 1, 8, 4
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = torch.zeros(b, n)
    ece = expected_calibration_error_adaptive(logits, labels, mask, num_bins_ece=15)
    assert torch.isfinite(ece)
    assert ece.item() == 0.0
