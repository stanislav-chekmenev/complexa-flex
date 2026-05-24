"""Tests for `iptm_from_logits` (TM-score-style interface metric).

Formula recap:
    N_eff = max(n_valid, 19)                   # per-sample
    d0    = 1.24 * (N_eff - 15)^(1/3) - 1.8    # per-sample
    w_k   = 1 / (1 + (bin_centers / d0)^2)     # per-sample, per-bin
    p     = softmax(logits)                    # over bins
    s_ij  = sum_k p_ij_k * w_k                 # [B, L, L]
    row_i = mean over interface cols j of s_ij # [B, L]
    ipTM_b = max over interface rows i of row_i
    metric = mean over samples that have >=1 interface row
"""

from __future__ import annotations

import math

import torch

from proteinfoundation.nn.confidence._metrics import (
    interface_pair_mask,
    iptm_from_logits,
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


def _row_score_uniform(centers: torch.Tensor, d0: float) -> float:
    """For a row with uniform softmax, score = mean over bins of w_k."""
    w = 1.0 / (1.0 + (centers / d0).pow(2))
    return w.mean().item()


def test_one_hot_bin_zero_dimer_matches_w0() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((1, 4, 4, NUM_BINS), -1e9)
    logits[..., 0] = 0.0
    out = iptm_from_logits(logits, mask_eff, inter, centers)

    d0 = _d0(4)
    w0 = 1.0 / (1.0 + (0.25 / d0) ** 2)
    assert abs(out.item() - w0) < 1e-3


def test_one_hot_bin_max_dimer_near_zero() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((1, 4, 4, NUM_BINS), -1e9)
    logits[..., NUM_BINS - 1] = 0.0
    out = iptm_from_logits(logits, mask_eff, inter, centers)
    assert out.item() < 1e-3


def test_per_sample_d0_distinct_for_different_n_valid() -> None:
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
    out = iptm_from_logits(logits, mask_eff, inter, centers)

    d0_20 = _d0(20)
    d0_400 = _d0(400)
    s_20 = _row_score_uniform(centers, d0_20)
    s_400 = _row_score_uniform(centers, d0_400)
    expected = (s_20 + s_400) / 2.0
    assert abs(out.item() - expected) < 1e-3
    assert abs(d0_20 - d0_400) > 1.0


def test_empty_interface_returns_zero() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.randn(1, 4, 4, NUM_BINS)
    out = iptm_from_logits(logits, mask_eff, inter, centers)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_bf16_logits_match_fp32_reference() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    torch.manual_seed(0)
    logits_fp32 = torch.randn(1, 4, 4, NUM_BINS)
    out_fp32 = iptm_from_logits(logits_fp32, mask_eff, inter, centers)
    out_bf16 = iptm_from_logits(logits_fp32.to(torch.bfloat16), mask_eff, inter, centers)
    assert torch.isfinite(out_bf16)
    assert out_bf16.dtype == torch.float32
    assert abs(out_fp32.item() - out_bf16.item()) < 1e-3
