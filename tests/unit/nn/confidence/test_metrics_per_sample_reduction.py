"""Per-sample reduction parity for the 10 pAE-derived metrics.

For every metric, `M(...).mean()` over samples with mass must equal
`M(..., reduce="batch_mean")` within 1e-6. This pins the new `reduce`
kwarg as a strict generalisation of the existing batch-mean default.
"""
from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn.confidence._metrics import (
    i_pae,
    interface_pair_mask,
    ipsae_family,
    iptm_energy_from_logits,
    iptm_from_logits,
    min_ipae,
)


@pytest.fixture
def synthetic_batch() -> dict:
    torch.manual_seed(0)
    B, L, K = 3, 24, 64
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, 12:] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask[1, 20:] = False
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_ev = 8.0 + 4.0 * torch.randn((B, L, L))
    pae_ev = (pae_ev + pae_ev.transpose(-2, -1)) / 2
    logits = torch.randn((B, L, L, K))
    centers = torch.linspace(0.25, 31.75, K)
    inter = interface_pair_mask(chain_idx, mask_eff)
    return {
        "pae_ev": pae_ev,
        "logits": logits,
        "centers": centers,
        "mask_eff": mask_eff,
        "interface_mask": inter,
        "chain_idx": chain_idx,
    }


def _assert_batch_mean_matches_per_sample_mean(scalar, per_sample, name):
    assert per_sample.shape == (3,), f"{name}: expected per-sample shape (3,), got {per_sample.shape}"
    finite = per_sample[~torch.isnan(per_sample)]
    assert finite.numel() > 0, f"{name}: all per-sample values were NaN -- fixture has no mass"
    recomputed = finite.mean()
    assert torch.allclose(scalar, recomputed, atol=1e-6), (
        f"{name}: batch_mean={scalar.item():.6f} != per-sample mean over mass={recomputed.item():.6f}"
    )


def test_i_pae_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = i_pae(sb["pae_ev"], sb["interface_mask"])
    per = i_pae(sb["pae_ev"], sb["interface_mask"], reduce="per_sample")
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "i_pae")


def test_min_ipae_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = min_ipae(sb["pae_ev"], sb["interface_mask"])
    per = min_ipae(sb["pae_ev"], sb["interface_mask"], reduce="per_sample")
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "min_ipae")


def test_iptm_from_logits_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = iptm_from_logits(sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"])
    per = iptm_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"], reduce="per_sample"
    )
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "iptm_from_logits")


def test_iptm_energy_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = iptm_energy_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"]
    )
    per = iptm_energy_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"], reduce="per_sample"
    )
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "iptm_energy_from_logits")


def test_ipsae_family_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar_d = ipsae_family(sb["pae_ev"], sb["chain_idx"], sb["mask_eff"])
    per_d = ipsae_family(
        sb["pae_ev"], sb["chain_idx"], sb["mask_eff"], reduce="per_sample"
    )
    assert set(scalar_d) == set(per_d)
    for name, scalar in scalar_d.items():
        _assert_batch_mean_matches_per_sample_mean(scalar, per_d[name], name)


def test_per_sample_returns_nan_for_no_mass_sample():
    """When a sample has no interface pair, its per-sample value must be NaN.

    `nan_strategy="ignore"` (the torchmetrics default for PearsonCorrCoef) skips
    NaN entries during accumulation; the alternative (a zero value) would silently
    pull the correlation toward the origin.
    """
    B, L = 2, 16
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[0, 8:] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_ev = torch.full((B, L, L), 10.0)
    inter = interface_pair_mask(chain_idx, mask_eff)
    per = i_pae(pae_ev, inter, reduce="per_sample")
    assert not torch.isnan(per[0]), "dimer sample must have a finite per-sample value"
    assert torch.isnan(per[1]), f"monomer sample must yield NaN under reduce='per_sample', got {per[1].item()}"
