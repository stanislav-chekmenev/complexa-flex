"""Tests for `ipsae_family` (interface-restricted, with Ca-contact variants)."""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence._metrics import (
    interface_pair_mask,
    ipsae_family,
)


NUM_BINS = 64
BIN_WIDTH = 0.5
D0_CLIP_MIN = 19


def _default_bin_centers() -> torch.Tensor:
    return torch.tensor(
        [0.0 + BIN_WIDTH * (i + 0.5) for i in range(NUM_BINS)],
        dtype=torch.float32,
    )


def _make_ca_coords_two_chains(L_per_chain: int = 2, sep: float = 50.0) -> torch.Tensor:
    coords = torch.zeros(1, 2 * L_per_chain, 3)
    coords[0, L_per_chain:, 0] = sep
    return coords


def test_returns_expected_keys_all_finite() -> None:
    centers = _default_bin_centers()
    torch.manual_seed(0)
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.randn(1, 4, 4, NUM_BINS)
    ca_coords = torch.randn(1, 4, 3)

    out = ipsae_family(logits, mask_eff, inter, centers, ca_coords)
    expected = {
        "avg_ipsae",
        "min_ipsae",
        "max_ipsae",
        "avg_ipsae_10",
        "min_ipsae_10",
        "max_ipsae_10",
    }
    assert set(out.keys()) == expected
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} not finite"
        assert v.ndim == 0


def test_contact_variants_zero_when_chains_far_apart() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.zeros(1, 4, 4, NUM_BINS)
    ca_coords = _make_ca_coords_two_chains()

    out = ipsae_family(logits, mask_eff, inter, centers, ca_coords)
    assert out["avg_ipsae_10"].item() == 0.0
    assert out["min_ipsae_10"].item() == 0.0
    assert out["max_ipsae_10"].item() == 0.0


def test_ca_coords_none_yields_zero_contact_variants() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.randn(1, 4, 4, NUM_BINS)

    out = ipsae_family(logits, mask_eff, inter, centers, None)
    assert out["avg_ipsae_10"].item() == 0.0
    assert out["min_ipsae_10"].item() == 0.0
    assert out["max_ipsae_10"].item() == 0.0


def test_uniform_logits_avg_ipsae_matches_per_row_score_mean() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.zeros(1, 4, 4, NUM_BINS)
    out = ipsae_family(logits, mask_eff, inter, centers, None)

    n_eff = max(4, D0_CLIP_MIN)
    d0 = 1.24 * (n_eff - 15) ** (1.0 / 3.0) - 1.8
    w = 1.0 / (1.0 + (centers / d0).pow(2))
    per_pair_score = w.mean().item()
    expected_per_row = per_pair_score
    assert abs(out["avg_ipsae"].item() - expected_per_row) < 1e-6
    assert abs(out["min_ipsae"].item() - expected_per_row) < 1e-6
    assert abs(out["max_ipsae"].item() - expected_per_row) < 1e-6


def test_empty_interface_yields_zero_for_all_keys() -> None:
    centers = _default_bin_centers()
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    inter = interface_pair_mask(chain_idx, mask_eff)
    logits = torch.randn(1, 4, 4, NUM_BINS)
    ca_coords = torch.randn(1, 4, 3)

    out = ipsae_family(logits, mask_eff, inter, centers, ca_coords)
    for k, v in out.items():
        assert torch.isfinite(v)
        assert v.item() == 0.0, f"{k} should be 0 when interface is empty"
