"""Contract tests for the PR-B PAE loss + metrics surface (Slice 2).

`masked_pae_cross_entropy`, `masked_smooth_l1_on_pae_expected_value`, and
`combined_pae_loss` mirror the pLDDT family 1:1 but operate over pair
positions `(B, L, L)`. `pae_ece_adaptive` is the pair-flattened equal-mass
ECE; `pae_mae_stratified_by_distance` is the new distance-stratified MAE
that consumes Cα coordinates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from proteinfoundation.confidence.losses import (
    combined_pae_loss,
    masked_pae_cross_entropy,
    masked_smooth_l1_on_pae_expected_value,
)
from proteinfoundation.confidence.metrics import (
    pae_ece_adaptive,
    pae_mae_stratified_by_distance,
)


PAE_BIN_WIDTH = 0.5


def _pae_bin_centers(num_bins: int = 64) -> torch.Tensor:
    return torch.tensor(
        [PAE_BIN_WIDTH * (i + 0.5) for i in range(num_bins)],
        dtype=torch.float32,
    )


def test_masked_pae_cross_entropy_matches_manual_reduction() -> None:
    torch.manual_seed(0)
    b, n, k = 1, 3, 4
    logits = torch.randn(b, n, n, k)
    labels = torch.randint(0, k, (b, n, n))
    mask = torch.ones(b, n, n, dtype=torch.float32)
    mask[0, 1, 2] = 0.0

    out = masked_pae_cross_entropy(logits, labels, mask)

    per_cell = F.cross_entropy(
        logits.reshape(b * n * n, k),
        labels.reshape(b * n * n),
        reduction="none",
    ).reshape(b, n, n)
    expected = (per_cell * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(out, expected, atol=1e-6)
    n_valid = int(mask.sum().item())
    assert n_valid == n * n - 1


def test_masked_smooth_l1_on_pae_expected_value_matches_manual() -> None:
    b, n, k = 1, 3, 4
    centers = torch.tensor([0.25, 0.75, 1.25, 1.75], dtype=torch.float32)
    target_bins = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 0]]], dtype=torch.int64)
    target_cont = centers[target_bins]

    pred_bins = torch.tensor([[[0, 0, 2], [1, 2, 3], [2, 3, 0]]], dtype=torch.int64)
    logits = torch.full((b, n, n, k), -1e4)
    for i in range(n):
        for j in range(n):
            logits[0, i, j, pred_bins[0, i, j]] = 1e4

    mask = torch.ones(b, n, n, dtype=torch.float32)
    mask[0, 1, 2] = 0.0

    out = masked_smooth_l1_on_pae_expected_value(logits, target_cont, mask, centers)

    pred_cont = centers[pred_bins].float()
    per_cell = F.smooth_l1_loss(pred_cont, target_cont.float(), reduction="none")
    expected = (per_cell * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_combined_pae_loss_defaults_sum() -> None:
    torch.manual_seed(3)
    b, n, k = 1, 4, 64
    centers = _pae_bin_centers(k)
    logits = torch.randn(b, n, n, k)
    labels = torch.randint(0, k, (b, n, n))
    target_cont = centers[labels]
    mask = torch.ones(b, n, n, dtype=torch.float32)
    mask[0, 0, :] = 0.0

    ce = masked_pae_cross_entropy(logits, labels, mask, label_smoothing=0.0)
    sl1 = masked_smooth_l1_on_pae_expected_value(logits, target_cont, mask, centers)

    total, parts = combined_pae_loss(
        student_logits=logits,
        pae_bin_labels=labels,
        pae_continuous=target_cont,
        mask=mask,
        bin_centers=centers,
    )
    expected = 0.9 * ce + 0.1 * sl1
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(parts["loss_ce"], ce, atol=1e-6)
    assert torch.allclose(parts["loss_smooth_l1"], sl1, atol=1e-6)


def test_combined_pae_loss_ev_weight_zero_short_circuit() -> None:
    """`smooth_l1_weight=0.0` must bit-equal `ce_weight * loss_ce`.

    The EV softmax + bin-center reduction is the load-bearing allocation
    in PAE training (`[b, n, n, k]` fp32 cast); short-circuiting saves
    it under the `ev_weight=0` recipe used by the multi-head distill.
    """
    torch.manual_seed(11)
    b, n, k = 1, 4, 64
    centers = _pae_bin_centers(k)
    logits = torch.randn(b, n, n, k)
    labels = torch.randint(0, k, (b, n, n))
    target_cont = centers[labels]
    mask = torch.ones(b, n, n, dtype=torch.float32)

    ce = masked_pae_cross_entropy(logits, labels, mask, label_smoothing=0.0)

    total, parts = combined_pae_loss(
        student_logits=logits,
        pae_bin_labels=labels,
        pae_continuous=target_cont,
        mask=mask,
        bin_centers=centers,
        ce_weight=1.0,
        smooth_l1_weight=0.0,
    )
    assert torch.equal(total, ce), "ce_weight=1.0 short-circuit must be bit-equal to CE"
    assert torch.equal(
        parts["loss_smooth_l1"], torch.zeros_like(parts["loss_smooth_l1"])
    )
    assert parts["loss_smooth_l1"].dtype == torch.float32
    assert torch.allclose(parts["loss_ce"], ce, atol=1e-6)


def test_pae_ece_adaptive_finite_on_partial_mask() -> None:
    torch.manual_seed(7)
    b, n, k = 1, 5, 4
    logits = torch.randn(b, n, n, k)
    labels = torch.randint(0, k, (b, n, n))
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    mask = ((ii + jj) % 2 == 0).to(torch.float32).unsqueeze(0)

    ece = pae_ece_adaptive(logits, labels, mask)
    assert torch.is_tensor(ece)
    assert ece.ndim == 0
    assert torch.isfinite(ece)
    assert 0.0 <= ece.item() <= 1.0


def test_pae_mae_stratified_by_distance_keys_finite() -> None:
    torch.manual_seed(11)
    b, n, k = 1, 5, 64
    centers = _pae_bin_centers(k)
    logits = torch.randn(b, n, n, k)
    labels = torch.randint(0, k, (b, n, n))
    mask = torch.ones(b, n, n, dtype=torch.float32)
    ca_coords = torch.randn(b, n, 3) * 12.0

    out = pae_mae_stratified_by_distance(
        logits=logits,
        labels=labels,
        mask=mask,
        bin_centers=centers,
        ca_coords=ca_coords,
    )
    expected_keys = {"d_lt8", "d_8_16", "d_ge16"}
    assert set(out.keys()) == expected_keys
    for key in expected_keys:
        assert torch.isfinite(out[key]), f"bucket {key} produced non-finite value"
