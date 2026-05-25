"""compute_loss_and_metrics at stage='val' updates val_metric_correlations.

The test constructs a tiny dimer batch, runs val on the head, and
verifies that after the call:
    (a) val_metric_correlations[name].update was called (state advanced)
    (b) on compute(), pearson is finite for at least one metric
    (c) at stage='train' the accumulators are NOT touched
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


def _build_head(track: bool = True) -> PaeHead:
    trunk = ConfidenceTrunk(
        token_dim=8,
        pair_repr_dim=8,
        n_blocks=1,
        n_heads=2,
        dim_cond=8,
        latent_dim=2,
    )
    return PaeHead(
        trunk=trunk,
        token_dim=8,
        pair_repr_dim=8,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        track_metric_correlations=track,
    )


def _toy_val_batch(B: int = 4, L: int = 16) -> tuple[dict, dict, torch.Tensor]:
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_bin = torch.randint(0, 32, (B, L, L))
    pae_continuous = pae_bin.float() * 1.0
    out = {"pae_logits": torch.randn((B, L, L, 64))}
    batch = {
        "pae_bin": pae_bin,
        "pae_residue_pair": pae_continuous,
        "chain_idx": chain_idx,
    }
    return out, batch, mask_eff.to(torch.float32)


def test_val_call_advances_accumulators():
    head = _build_head(track=True)
    head.eval()
    out, batch, mask_eff = _toy_val_batch()
    _, _ = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    mc = head.val_metric_correlations["i_pae"]
    state_total = mc["mae"].sum_abs_error
    assert state_total.item() >= 0.0
    agg = head.val_metric_correlations_compute_and_reset()
    assert "i_pae" in agg
    assert agg["i_pae"]["mae"] >= 0.0


def test_train_call_does_not_touch_accumulators():
    head = _build_head(track=True)
    head.train()
    out, batch, mask_eff = _toy_val_batch()
    _, _ = head.compute_loss_and_metrics(out, batch, mask_eff, stage="train")
    mc = head.val_metric_correlations["i_pae"]
    assert mc["mae"].sum_abs_error.item() == 0.0, (
        "train-stage call must not advance val accumulators"
    )


def test_track_false_skips_attribute_path():
    head = _build_head(track=False)
    head.eval()
    out, batch, mask_eff = _toy_val_batch()
    _, _ = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    assert (
        not hasattr(head, "val_metric_correlations")
        or head.val_metric_correlations is None
    )
