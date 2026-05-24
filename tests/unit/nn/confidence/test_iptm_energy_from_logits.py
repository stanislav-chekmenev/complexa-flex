"""Tests for `iptm_energy_from_logits` (log-sum-exp energy variant)."""

from __future__ import annotations

import math

import torch

from proteinfoundation.nn.confidence._metrics import (
    interface_pair_mask,
    iptm_energy_from_logits,
)


NUM_BINS = 64
BIN_WIDTH = 0.5
D0_CLIP_MIN = 19


def _default_bin_centers() -> torch.Tensor:
    return torch.tensor(
        [0.0 + BIN_WIDTH * (i + 0.5) for i in range(NUM_BINS)],
        dtype=torch.float32,
    )


def _d0(n: int, clip_min: int = D0_CLIP_MIN) -> float:
    n_eff = max(n, clip_min)
    return 1.24 * (n_eff - 15) ** (1.0 / 3.0) - 1.8


def test_uniform_zero_logits_matches_reference() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.zeros(1, 4, 4, NUM_BINS)
    out = iptm_energy_from_logits(logits, mask_eff, inter, centers)

    d0 = _d0(4)
    centers_64 = centers.to(torch.float64)
    w = 1.0 / (1.0 + (centers_64 / d0).pow(2))
    log_w = w.log()
    ref = -torch.logsumexp(log_w, dim=-1).item()
    assert abs(out.item() - ref) < 1e-4


def test_empty_interface_returns_zero() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.randn(1, 4, 4, NUM_BINS)
    out = iptm_energy_from_logits(logits, mask_eff, inter, centers)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_tm_lambda_zero_matches_log_num_bins() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.zeros(1, 4, 4, NUM_BINS)
    out = iptm_energy_from_logits(logits, mask_eff, inter, centers, tm_lambda=0.0)
    ref = -math.log(NUM_BINS)
    assert abs(out.item() - ref) < 1e-4


def test_heterogeneous_n_yields_distinct_per_sample_energies() -> None:
    """Two samples with n_valid in {20, 400}, zero logits.

    d0(20)  = 1.24 * 5^(1/3) - 1.8  ~= 0.32  (clamped to clip_min=19 -> 4^(1/3)=1.587 -> d0=0.169? actually 19-15=4 so d0=1.24*4^(1/3)-1.8 = 1.24*1.587-1.8 = 0.168)
    d0(400) = 1.24 * 385^(1/3) - 1.8 ~= 7.27

    Smaller d0 -> the (centers/d0)^2 ratio is much larger -> w_b is much
    smaller -> log w is much more negative -> weighted_logits shifts down
    -> logsumexp drops -> -logsumexp (= positional energy) rises. So
    smaller N gives LARGER per-sample energy. The metric averages over
    samples-with-mass, so the batch mean lies strictly between the two.
    """
    centers = _default_bin_centers()
    B, L_pad = 2, 400
    chain_idx = torch.zeros(B, L_pad, dtype=torch.long)
    chain_idx[0, :10] = 0
    chain_idx[0, 10:20] = 1
    chain_idx[1, :200] = 0
    chain_idx[1, 200:400] = 1

    mask_eff = torch.zeros(B, L_pad, L_pad, dtype=torch.float32)
    mask_eff[0, :20, :20] = 1.0
    mask_eff[1, :400, :400] = 1.0

    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.zeros(B, L_pad, L_pad, NUM_BINS)
    out = iptm_energy_from_logits(logits, mask_eff, inter, centers)

    centers_64 = centers.to(torch.float64)

    def _ref_per_sample(n: int) -> float:
        d0 = _d0(n)
        w = 1.0 / (1.0 + (centers_64 / d0).pow(2))
        log_w = w.log()
        return float(-torch.logsumexp(log_w, dim=-1).item())

    e_20 = _ref_per_sample(20)
    e_400 = _ref_per_sample(400)
    assert e_20 > e_400, f"smaller N should give larger energy: e_20={e_20}, e_400={e_400}"
    expected = 0.5 * (e_20 + e_400)
    assert abs(out.item() - expected) < 1e-3, (
        f"batch mean mismatch: got {out.item()}, expected {expected}"
    )
