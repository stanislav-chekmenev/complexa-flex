"""Tests for `min_ipae` (per-row interface EV mean, min over rows)."""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence._metrics import (
    _logits_to_continuous,
    interface_pair_mask,
    min_ipae,
)


NUM_BINS = 64
BIN_WIDTH = 0.5


def _default_bin_centers() -> torch.Tensor:
    return torch.tensor(
        [0.0 + BIN_WIDTH * (i + 0.5) for i in range(NUM_BINS)],
        dtype=torch.float32,
    )


def _logits_with_per_row_target(row_targets_ev: list[float], chain_idx: torch.Tensor) -> torch.Tensor:
    """Build one-hot logits so that interface rows have known EV.

    For each row `i`, the interface columns receive a one-hot at the bin
    whose center equals `row_targets_ev[i]`. Non-interface columns are
    filled with a one-hot at bin 0 (won't be averaged into the per-row
    EV because the row mask excludes them).
    """
    L = chain_idx.shape[-1]
    logits = torch.full((1, L, L, NUM_BINS), -1e9)
    for i, ev in enumerate(row_targets_ev):
        bin_idx = int(round(ev / BIN_WIDTH - 0.5))
        logits[0, i, :, bin_idx] = 0.0
    return logits


def test_min_over_interface_rows_picks_smallest_row_ev() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = _logits_with_per_row_target([4.75, 4.75, 0.75, 4.75], chain_idx)
    pae_ev = _logits_to_continuous(logits, centers)
    out = min_ipae(pae_ev, inter)
    assert abs(out.item() - 0.75) < 1e-5


def test_empty_mask_returns_zero_without_nan() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.randn(1, 4, 4, NUM_BINS)
    pae_ev = _logits_to_continuous(logits, centers)
    out = min_ipae(pae_ev, inter)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_batch_averages_per_sample_mins() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    mask_eff = torch.ones(2, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)

    logits = torch.full((2, 4, 4, NUM_BINS), -1e9)
    logits[0] = _logits_with_per_row_target([4.75, 4.75, 0.75, 4.75], torch.tensor([[0, 0, 1, 1]]))[0]
    logits[1] = _logits_with_per_row_target([2.25, 2.25, 2.25, 2.25], torch.tensor([[0, 0, 1, 1]]))[0]
    pae_ev = _logits_to_continuous(logits, centers)
    out = min_ipae(pae_ev, inter)
    expected = (0.75 + 2.25) / 2.0
    assert abs(out.item() - expected) < 1e-5
