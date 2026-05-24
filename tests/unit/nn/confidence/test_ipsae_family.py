"""Tests for `ipsae_family` — colabdesign-compatible per-PR-B-round-2.

Reference: community_models/colabdesign/af/loss.py:288-339 (`calc_d0` +
`get_ipsae_loss`). For each direction (B->A and A->B) we compute, per
sample:

    pae_mask = src[:, None] * tgt[None, :] * (pae < cutoff)    # [L, L]
    d0       = clamp_min(1.24 * (clamp_min(L_row, 27) - 15)^(1/3) - 1.8, 1.0)
    tm_term  = 1 / (1 + pae^2 / d0^2)
    mean_tm  = (pae_mask * tm_term).sum(-1) / (pae_mask.sum(-1) + 1e-8)
    ipsae_d  = mean_tm.max()

Then `{min, max, avg}(ipsae_ab, ipsae_ba)` per sample, averaged over
samples-with-mass (samples where either direction has a cutoff-passing
inter-chain pair).
"""

from __future__ import annotations

import numpy as np
import torch

from proteinfoundation.nn.confidence._metrics import (
    _calc_d0_colabdesign,
    ipsae_family,
)


def _colabdesign_ipsae_reference(
    pae_ev: np.ndarray,
    chain_idx: np.ndarray,
    mask_eff: np.ndarray,
    pae_cutoff: float,
) -> dict[str, float]:
    """fp64 numpy mirror of colabdesign.get_ipsae_loss for one cutoff."""
    B = pae_ev.shape[0]
    res_valid = mask_eff.astype(bool).any(axis=-1).astype(np.float64)
    binder_id = (chain_idx == 0).astype(np.float64) * res_valid
    target_id = (chain_idx == 1).astype(np.float64) * res_valid

    def _dir(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        per_sample = np.zeros(B, dtype=np.float64)
        has_mass = np.zeros(B, dtype=bool)
        for b in range(B):
            mask = (src[b][:, None] * tgt[b][None, :]) * (pae_ev[b] < pae_cutoff).astype(np.float64)
            row_count = mask.sum(axis=-1)
            L_eff = np.clip(row_count, a_min=27.0, a_max=None)
            d0 = 1.24 * (L_eff - 15.0) ** (1.0 / 3.0) - 1.8
            d0 = np.clip(d0, a_min=1.0, a_max=None)
            tm = 1.0 / (1.0 + pae_ev[b] ** 2 / d0[:, None] ** 2)
            mean_tm = (mask * tm).sum(axis=-1) / (row_count + 1e-8)
            per_sample[b] = mean_tm.max()
            has_mass[b] = mask.sum() > 0
        return per_sample, has_mass

    ipsae_ba, mass_ba = _dir(target_id, binder_id)
    ipsae_ab, mass_ab = _dir(binder_id, target_id)
    sample_has_mass = mass_ba | mass_ab
    min_per = np.minimum(ipsae_ab, ipsae_ba)
    max_per = np.maximum(ipsae_ab, ipsae_ba)
    avg_per = 0.5 * (ipsae_ab + ipsae_ba)

    min_per = np.where(sample_has_mass, min_per, 0.0)
    max_per = np.where(sample_has_mass, max_per, 0.0)
    avg_per = np.where(sample_has_mass, avg_per, 0.0)
    n = float(max(sample_has_mass.sum(), 1))
    return {
        "avg": float(avg_per.sum() / n),
        "min": float(min_per.sum() / n),
        "max": float(max_per.sum() / n),
    }


def test_calc_d0_matches_colabdesign_clip() -> None:
    L = torch.tensor([0.0, 10.0, 26.0, 27.0, 100.0])
    d0 = _calc_d0_colabdesign(L)
    # All L < 27 clamp to L_eff = 27 -> d0 = 1.24 * 12^(1/3) - 1.8
    cl_d0_27 = 1.24 * (27.0 - 15.0) ** (1.0 / 3.0) - 1.8
    cl_d0_27 = max(cl_d0_27, 1.0)
    assert abs(d0[0].item() - cl_d0_27) < 1e-6
    assert abs(d0[1].item() - cl_d0_27) < 1e-6
    assert abs(d0[2].item() - cl_d0_27) < 1e-6
    assert abs(d0[3].item() - cl_d0_27) < 1e-6
    cl_d0_100 = 1.24 * (100.0 - 15.0) ** (1.0 / 3.0) - 1.8
    cl_d0_100 = max(cl_d0_100, 1.0)
    assert abs(d0[4].item() - cl_d0_100) < 1e-6


def test_hand_computed_tiny_dimer_matches_colabdesign() -> None:
    """4-residue dimer, all inter-chain PAE = 3.0 A, well below cutoff 15 A.

    Per direction: pae_mask sums to 2 per row -> L_eff=27 -> d0~=1.04
    (clamped to >= 1.0; value ~= 1.04 because 1.24 * 12^(1/3) - 1.8 ~= 1.04).
    tm_term = 1 / (1 + 9 / d0^2). Per-row mean over the 2 filtered cols
    = tm_term (constant). mean_tm.max() = tm_term. Per-sample
    min/max/avg over (tm_term, tm_term) = tm_term.
    """
    B, L = 1, 4
    pae_ev = torch.full((B, L, L), 3.0)
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    ref = _colabdesign_ipsae_reference(
        pae_ev.numpy().astype(np.float64),
        chain_idx.numpy(),
        mask_eff.numpy().astype(np.float64),
        15.0,
    )
    for k_ours, k_ref in (("avg_ipsae", "avg"), ("min_ipsae", "min"), ("max_ipsae", "max")):
        assert abs(out[k_ours].item() - ref[k_ref]) < 1e-5, (
            f"{k_ours} mismatch: ours={out[k_ours].item()} ref={ref[k_ref]}"
        )


def test_pae_cutoff_filtering_drops_above_cutoff_pairs() -> None:
    """Half pairs < cutoff, half above. Only the < ones contribute."""
    B, L = 1, 4
    pae_ev = torch.tensor(
        [
            [
                [0.0, 0.0, 3.0, 25.0],
                [0.0, 0.0, 25.0, 3.0],
                [3.0, 25.0, 0.0, 0.0],
                [25.0, 3.0, 0.0, 0.0],
            ]
        ]
    )
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    ref = _colabdesign_ipsae_reference(
        pae_ev.numpy().astype(np.float64),
        chain_idx.numpy(),
        mask_eff.numpy().astype(np.float64),
        15.0,
    )
    for k_ours, k_ref in (("avg_ipsae", "avg"), ("min_ipsae", "min"), ("max_ipsae", "max")):
        assert abs(out[k_ours].item() - ref[k_ref]) < 1e-5


def test_ten_variant_uses_stricter_cutoff() -> None:
    """All inter-chain pairs at 12.0 A: > 10 (excluded by _10) but < 15 (kept by base).

    Base: 4 inter-chain pairs contribute per direction, mean_tm > 0.
    _10:  0 inter-chain pairs contribute -> mean_tm = 0 -> directional
          per_sample = 0; both directions have zero mass -> sample
          excluded from denominator -> metric = 0.
    """
    B, L = 1, 4
    pae_ev = torch.full((B, L, L), 12.0)
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    assert out["avg_ipsae"].item() > 0.0
    assert out["avg_ipsae_10"].item() == 0.0
    assert out["min_ipsae_10"].item() == 0.0
    assert out["max_ipsae_10"].item() == 0.0


def test_per_row_d0_from_filtered_count_is_per_sample() -> None:
    """Two samples with different effective filtered counts -> different d0.

    Sample 0: all 4 inter-chain pairs pass cutoff -> row_count=2 per
              filtered row -> L_eff=27 (clamp) -> d0~=1.04.
    Sample 1: only 1 inter-chain pair passes per row -> row_count=1 ->
              same L_eff=27 clamp (1 < 27). Both samples actually share
              the same d0 from clamp. To exercise per-sample variation,
              we instead vary the *pae value* and verify the reference
              fp64 trajectory matches torch.
    """
    B, L = 2, 4
    pae_ev = torch.zeros(B, L, L)
    pae_ev[0] = 3.0
    pae_ev[1, 0, 2] = 1.0
    pae_ev[1, 0, 3] = 14.0
    pae_ev[1, 1, 2] = 14.0
    pae_ev[1, 1, 3] = 1.0
    pae_ev[1, 2, 0] = 1.0
    pae_ev[1, 2, 1] = 14.0
    pae_ev[1, 3, 0] = 14.0
    pae_ev[1, 3, 1] = 1.0
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    ref = _colabdesign_ipsae_reference(
        pae_ev.numpy().astype(np.float64),
        chain_idx.numpy(),
        mask_eff.numpy().astype(np.float64),
        15.0,
    )
    for k_ours, k_ref in (("avg_ipsae", "avg"), ("min_ipsae", "min"), ("max_ipsae", "max")):
        assert abs(out[k_ours].item() - ref[k_ref]) < 1e-5, (
            f"{k_ours} mismatch: ours={out[k_ours].item()} ref={ref[k_ref]}"
        )


def test_symmetric_in_chain_assignment() -> None:
    """Swap chain_idx 0<->1: {avg, min, max} unchanged."""
    B, L = 1, 4
    torch.manual_seed(0)
    pae_ev = torch.rand(B, L, L) * 14.0
    pae_ev = 0.5 * (pae_ev + pae_ev.transpose(-1, -2))
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)
    chain_idx_a = torch.tensor([[0, 0, 1, 1]])
    chain_idx_b = torch.tensor([[1, 1, 0, 0]])

    out_a = ipsae_family(pae_ev, chain_idx_a, mask_eff)
    out_b = ipsae_family(pae_ev, chain_idx_b, mask_eff)
    for k in ("avg_ipsae", "min_ipsae", "max_ipsae", "avg_ipsae_10", "min_ipsae_10", "max_ipsae_10"):
        assert abs(out_a[k].item() - out_b[k].item()) < 1e-5, k


def test_monomer_sample_returns_zero_finite() -> None:
    """chain_idx all 0: no inter-chain pairs in either direction -> 0."""
    B, L = 1, 4
    pae_ev = torch.full((B, L, L), 3.0)
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} not finite"
        assert v.item() == 0.0, f"{k} should be 0 on monomer sample"


def test_mixed_batch_dimer_plus_monomer_not_diluted() -> None:
    """One dimer + one monomer: ipsae == dimer's value, not / 2."""
    B, L = 2, 4
    pae_ev = torch.full((B, L, L), 3.0)
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)

    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    ref_dimer = _colabdesign_ipsae_reference(
        pae_ev[:1].numpy().astype(np.float64),
        chain_idx[:1].numpy(),
        mask_eff[:1].numpy().astype(np.float64),
        15.0,
    )
    for k_ours, k_ref in (("avg_ipsae", "avg"), ("min_ipsae", "min"), ("max_ipsae", "max")):
        assert abs(out[k_ours].item() - ref_dimer[k_ref]) < 1e-5, (
            f"{k_ours} diluted by monomer: ours={out[k_ours].item()} ref={ref_dimer[k_ref]}"
        )


def test_bf16_pae_ev_yields_finite_fp32_output() -> None:
    B, L = 1, 4
    pae_ev_fp32 = torch.full((B, L, L), 3.0)
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)
    out_bf16 = ipsae_family(pae_ev_fp32.to(torch.bfloat16), chain_idx, mask_eff)
    out_fp32 = ipsae_family(pae_ev_fp32, chain_idx, mask_eff)
    for k in ("avg_ipsae", "min_ipsae", "max_ipsae"):
        assert torch.isfinite(out_bf16[k])
        assert out_bf16[k].dtype == torch.float32
        assert abs(out_bf16[k].item() - out_fp32[k].item()) < 5e-3, k


def test_returns_exactly_six_keys() -> None:
    B, L = 1, 4
    pae_ev = torch.full((B, L, L), 3.0)
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(B, L, L, dtype=torch.float32)
    out = ipsae_family(pae_ev, chain_idx, mask_eff)
    expected = {
        "avg_ipsae",
        "min_ipsae",
        "max_ipsae",
        "avg_ipsae_10",
        "min_ipsae_10",
        "max_ipsae_10",
    }
    assert set(out.keys()) == expected
