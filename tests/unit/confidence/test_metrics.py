"""Tests for the AF2-scale-aware confidence validation metrics.

Pinned: `plddt_mae` returns MAE in *pLDDT* units (driven by `bin_centers`),
not in `[0, 1]`. Stratified buckets always have all four keys with finite,
zero-not-NaN values when a bucket is empty.
"""

from __future__ import annotations

import torch

from proteinfoundation.confidence.metrics import (
    expected_calibration_error,
    pearson_r,
    plddt_accuracy,
    plddt_mae,
    plddt_mae_stratified,
    spearman_r,
)


def _bin_centers(num_bins: int = 50, bin_min: float = 0.0, bin_max: float = 100.0) -> torch.Tensor:
    bin_width = (bin_max - bin_min) / num_bins
    return torch.tensor(
        [bin_min + bin_width * (i + 0.5) for i in range(num_bins)],
        dtype=torch.float32,
    )


def test_plddt_accuracy_hand_computed() -> None:
    k = 4
    logits = torch.zeros(1, 4, k)
    logits[0, 0, 0] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 0] = 10.0
    logits[0, 3, 3] = 10.0
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    acc = plddt_accuracy(logits, labels, mask)
    assert torch.allclose(acc, torch.tensor(2.0 / 3.0), atol=1e-6)


def test_plddt_mae_in_plddt_units() -> None:
    k = 50
    centers = _bin_centers(k)
    b, n = 1, 4
    labels = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
    mask = torch.ones(b, n)

    logits_perfect = torch.full((b, n, k), -1e4)
    for j, lab in enumerate(labels[0].tolist()):
        logits_perfect[0, j, lab] = 1e4
    mae_perfect = plddt_mae(logits_perfect, labels, mask, centers)
    assert torch.allclose(mae_perfect, torch.tensor(0.0), atol=1e-3)

    logits_shift = torch.full((b, n, k), -1e4)
    for j, lab in enumerate(labels[0].tolist()):
        logits_shift[0, j, lab + 1] = 1e4
    mae_shift = plddt_mae(logits_shift, labels, mask, centers)
    assert torch.allclose(mae_shift, torch.tensor(2.0), atol=1e-3)


def test_pearson_spearman_perfect() -> None:
    b, n = 2, 7
    target = torch.linspace(0.0, 100.0, steps=b * n).reshape(b, n)
    pred = target * 0.5 + 10.0
    mask = torch.ones(b, n)
    pr = pearson_r(pred, target, mask)
    sr = spearman_r(pred, target, mask)
    assert torch.allclose(pr, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(sr, torch.tensor(1.0), atol=1e-5)


def test_stratified_mae_bucket_keys() -> None:
    k = 50
    centers = _bin_centers(k)
    b, n = 1, 6
    labels = torch.tensor([[5, 5, 30, 40, 45, 48]], dtype=torch.int64)
    logits = torch.full((b, n, k), -1e4)
    for j, lab in enumerate(labels[0].tolist()):
        logits[0, j, lab] = 1e4
    mask = torch.ones(b, n)

    out = plddt_mae_stratified(logits, labels, mask, centers)
    expected_keys = {"mae_lt50", "mae_50_70", "mae_70_90", "mae_ge90"}
    assert set(out.keys()) == expected_keys
    for k_ in expected_keys:
        assert torch.isfinite(out[k_])

    labels_only_low = torch.zeros(1, 1, dtype=torch.int64)
    logits_only_low = torch.full((1, 1, k), -1e4)
    logits_only_low[0, 0, 0] = 1e4
    mask_only_low = torch.ones(1, 1)
    out_low = plddt_mae_stratified(logits_only_low, labels_only_low, mask_only_low, centers)
    assert torch.isfinite(out_low["mae_ge90"])
    assert out_low["mae_ge90"].item() == 0.0


def test_ece_in_unit_interval() -> None:
    torch.manual_seed(0)
    b, n, k = 2, 16, 50
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = (torch.rand(b, n) > 0.3).to(torch.float32)
    ece = expected_calibration_error(logits, labels, mask, num_bins_ece=10)
    assert torch.is_tensor(ece)
    assert ece.ndim == 0
    assert torch.isfinite(ece)
    assert 0.0 <= ece.item() <= 1.0


def test_ece_perfect_calibration() -> None:
    b, n, k = 1, 32, 5
    labels = torch.randint(0, k, (b, n))
    logits = torch.full((b, n, k), -1e4)
    for j in range(n):
        logits[0, j, labels[0, j]] = 1e4
    mask = torch.ones(b, n)
    ece = expected_calibration_error(logits, labels, mask, num_bins_ece=10)
    assert ece.item() < 0.05


def test_ece_worst_calibration() -> None:
    b, n, k = 1, 64, 5
    labels = torch.zeros(b, n, dtype=torch.int64)
    logits = torch.full((b, n, k), -1e4)
    logits[..., 1] = 1e4
    mask = torch.ones(b, n)
    ece = expected_calibration_error(logits, labels, mask, num_bins_ece=10)
    assert ece.item() > 0.5


def test_ece_mask_respected() -> None:
    b, n, k = 1, 8, 4
    logits = torch.full((b, n, k), -1e4)
    logits[..., 1] = 1e4
    labels = torch.zeros(b, n, dtype=torch.int64)
    labels[0, :4] = 1
    mask = torch.zeros(b, n)
    mask[0, :4] = 1.0
    ece = expected_calibration_error(logits, labels, mask, num_bins_ece=10)
    assert ece.item() < 0.05
