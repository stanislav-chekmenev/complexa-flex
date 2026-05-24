"""Tests for `i_pae` (interface-restricted PAE expected value)."""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence._metrics import (
    _logits_to_continuous,
    i_pae,
    interface_pair_mask,
)


NUM_BINS = 64
BIN_WIDTH = 0.5


def _default_bin_centers() -> torch.Tensor:
    return torch.tensor(
        [0.0 + BIN_WIDTH * (i + 0.5) for i in range(NUM_BINS)],
        dtype=torch.float32,
    )


def test_one_hot_bin_zero_equals_first_center() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((1, 4, 4, NUM_BINS), -1e9)
    logits[..., 0] = 0.0
    pae_ev = _logits_to_continuous(logits, centers)
    out = i_pae(pae_ev, inter)
    assert torch.isfinite(out)
    assert abs(out.item() - 0.25) < 1e-5


def test_batch_averages_per_sample() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    mask_eff = torch.ones(2, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((2, 4, 4, NUM_BINS), -1e9)
    logits[0, ..., 0] = 0.0
    logits[1, ..., 2] = 0.0
    pae_ev = _logits_to_continuous(logits, centers)
    out = i_pae(pae_ev, inter)
    expected = (0.25 + 1.25) / 2.0
    assert abs(out.item() - expected) < 1e-5


def test_empty_mask_returns_zero_without_nan() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.randn(1, 4, 4, NUM_BINS)
    pae_ev = _logits_to_continuous(logits, centers)
    out = i_pae(pae_ev, inter)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_bf16_logits_produce_finite_fp32_output() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits_fp32 = torch.randn(1, 4, 4, NUM_BINS)
    pae_ev_fp32 = _logits_to_continuous(logits_fp32, centers)
    pae_ev_bf16 = _logits_to_continuous(logits_fp32.to(torch.bfloat16), centers)
    out_fp32 = i_pae(pae_ev_fp32, inter)
    out_bf16 = i_pae(pae_ev_bf16, inter)
    assert torch.isfinite(out_bf16)
    assert out_bf16.dtype == torch.float32
    assert abs(out_bf16.item() - out_fp32.item()) < 1e-2


def test_mixed_batch_excludes_empty_samples() -> None:
    """B=2: sample 0 is dimer with i_pae=0.25 (one-hot bin 0), sample 1
    is monomer (empty interface). Expected: 0.25 (the dimer's value),
    NOT 0.125 (which would be the dilution if we divided by the full
    batch size).
    """
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    mask_eff = torch.ones(2, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((2, 4, 4, NUM_BINS), -1e9)
    logits[0, ..., 0] = 0.0
    logits[1, ..., 0] = 0.0
    pae_ev = _logits_to_continuous(logits, centers)
    out = i_pae(pae_ev, inter)
    assert torch.isfinite(out)
    assert abs(out.item() - 0.25) < 1e-5, (
        f"expected dimer's 0.25 (samples-with-mass denom), got {out.item()}"
    )
