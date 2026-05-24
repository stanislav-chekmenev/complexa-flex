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
