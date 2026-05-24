"""Integration: `PaeHead.compute_loss_and_metrics` emits interface metrics on val.

Covers three contracts:
1. Dimer val batch -> all 10 interface keys present (i_pae, i_ptm,
   i_ptm_energy, min_ipae, avg_ipsae, min_ipsae, max_ipsae,
   avg_ipsae_10, min_ipsae_10, max_ipsae_10), all finite, on
   the same device as logits.
2. Monomer val batch (no `chain_idx`) -> no interface keys, existing
   PAE keys (`pae_accuracy`, `pae_mae`, `pearson_r`, ...) still
   present, no raise.
3. Train stage -> no interface keys (val-only).
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


B, L_PER_CHAIN = 2, 6
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
NUM_PAE_BINS = 64
L_DIMER = 2 * L_PER_CHAIN

INTERFACE_KEYS = {
    "i_pae",
    "i_ptm",
    "i_ptm_energy",
    "min_ipae",
    "avg_ipsae",
    "min_ipsae",
    "max_ipsae",
    "avg_ipsae_10",
    "min_ipsae_10",
    "max_ipsae_10",
}


LATENT_DIM = 8


def _make_trunk() -> ConfidenceTrunk:
    return ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=True,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
        latent_dim=LATENT_DIM,
    )


def _make_head() -> PaeHead:
    return PaeHead(
        trunk=_make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
    )


def _make_dimer_batch_and_out() -> tuple[PaeHead, dict, dict, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    s = torch.randn(B, L_DIMER, TOKEN_DIM, generator=g)
    z = torch.randn(B, L_DIMER, L_DIMER, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(B, L_DIMER, dtype=torch.bool)
    cond = torch.randn(B, L_DIMER, DIM_COND, generator=g)
    local_latents = torch.randn(B, L_DIMER, LATENT_DIM, generator=g)

    head = _make_head().eval()
    with torch.no_grad():
        out = head(s, z, mask, cond, local_latents)
    mask_eff = (mask[:, None, :] & mask[:, :, None]).to(torch.float32)

    chain_idx = torch.zeros(B, L_DIMER, dtype=torch.long)
    chain_idx[:, L_PER_CHAIN:] = 1
    ca_coords = torch.randn(B, L_DIMER, 3, generator=g) * 3.0
    batch = {
        "pae_bin": torch.randint(0, NUM_PAE_BINS, (B, L_DIMER, L_DIMER), generator=g),
        "pae_residue_pair": torch.rand(B, L_DIMER, L_DIMER, generator=g) * 31.75,
        "chain_idx": chain_idx,
        "ca_coords": ca_coords,
    }
    return head, batch, out, mask_eff


def test_val_dimer_emits_all_ten_interface_keys() -> None:
    head, batch, out, mask_eff = _make_dimer_batch_and_out()
    _, log = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    missing = INTERFACE_KEYS - set(log.keys())
    assert not missing, f"missing interface keys: {missing}"
    for k in INTERFACE_KEYS:
        v = log[k]
        assert torch.isfinite(v).all(), f"{k} not finite: {v}"
        assert v.device == out["pae_logits"].device


def test_val_monomer_omits_interface_keys_keeps_existing() -> None:
    head, batch, out, mask_eff = _make_dimer_batch_and_out()
    batch.pop("chain_idx")
    _, log = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    assert INTERFACE_KEYS.isdisjoint(log.keys()), (
        f"interface keys present without chain_idx: {INTERFACE_KEYS & set(log.keys())}"
    )
    for k in ("pae_accuracy", "pae_mae", "pearson_r", "spearman_r"):
        assert k in log, f"missing baseline PAE val key: {k}"


def test_train_stage_omits_interface_keys() -> None:
    head, batch, out, mask_eff = _make_dimer_batch_and_out()
    _, log = head.compute_loss_and_metrics(out, batch, mask_eff, stage="train")
    assert INTERFACE_KEYS.isdisjoint(log.keys()), (
        f"interface keys leaked into train stage: {INTERFACE_KEYS & set(log.keys())}"
    )
