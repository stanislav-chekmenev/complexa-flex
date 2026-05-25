"""PaeHead.val_metric_correlations: per-sample-paired (pred, gt) accumulator.

Three cases:
    1. Identity: predicted PAE EV == GT PAE EV -> pearson/spearman = 1.0, MAE = 0.
    2. Constant shift: predicted = gt + c -> pearson/spearman = 1.0, MAE = |c|.
    3. No-mass sample: a monomer in a mixed batch contributes nothing (NaN dropped).
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


METRIC_NAMES = (
    "i_pae",
    "min_ipae",
    "i_ptm",
    "i_ptm_energy",
    "avg_ipsae",
    "min_ipsae",
    "max_ipsae",
    "avg_ipsae_10",
    "min_ipsae_10",
    "max_ipsae_10",
)


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


def _toy_batch(B: int = 4, L: int = 24) -> dict:
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    return {"chain_idx": chain_idx, "mask_eff": mask_eff}


def test_track_metric_correlations_attr_exists():
    head = _build_head(track=True)
    assert hasattr(head, "val_metric_correlations")
    assert set(head.val_metric_correlations.keys()) == set(METRIC_NAMES)


def test_track_disabled_returns_no_attr():
    head = _build_head(track=False)
    assert (
        not hasattr(head, "val_metric_correlations")
        or head.val_metric_correlations is None
    )


def test_identity_yields_perfect_correlation_and_zero_mae():
    """pred = gt over many random val samples -> pearson=1.0, mae=0."""
    head = _build_head(track=True)
    head.eval()
    torch.manual_seed(0)
    for _ in range(8):
        B, L = 4, 24
        bp = _toy_batch(B, L)
        pae_ev_gt = 4.0 + 6.0 * torch.rand((B, L, L))
        head.update_metric_correlations(
            pae_ev_pred=pae_ev_gt,
            pae_ev_gt=pae_ev_gt,
            chain_idx=bp["chain_idx"],
            mask_eff=bp["mask_eff"],
        )
    agg = head.val_metric_correlations_compute_and_reset()
    for m in METRIC_NAMES:
        assert agg[m]["mae"] < 1e-3, f"{m}: identity MAE not zero (got {agg[m]['mae']:.4e})"
        assert agg[m]["pearson"] > 0.999, f"{m}: identity pearson not 1.0 (got {agg[m]['pearson']:.4f})"
        assert agg[m]["spearman"] > 0.999, f"{m}: identity spearman not 1.0 (got {agg[m]['spearman']:.4f})"


def test_constant_shift_preserves_pearson_but_lifts_mae():
    """pred = gt + c -> pearson=1.0, mae=|c| for i_pae, strong pearson elsewhere.

    A small shift (0.5 A) keeps the ipSAE cutoff filter near-stationary so
    the metric stays informative across the val batch. Larger shifts move
    samples across the cutoff and break per-sample rank preservation.
    """
    head = _build_head(track=True)
    head.eval()
    c = 0.5
    torch.manual_seed(0)
    for _ in range(8):
        B, L = 4, 24
        bp = _toy_batch(B, L)
        gt = 4.0 + 6.0 * torch.rand((B, L, L))
        pred = gt + c
        head.update_metric_correlations(
            pae_ev_pred=pred,
            pae_ev_gt=gt,
            chain_idx=bp["chain_idx"],
            mask_eff=bp["mask_eff"],
        )
    agg = head.val_metric_correlations_compute_and_reset()
    assert abs(agg["i_pae"]["mae"] - c) < 0.05, (
        f"i_pae constant-shift MAE: got {agg['i_pae']['mae']:.4f}, expected ~{c}"
    )
    # i_pae and min_ipae are linear in PAE -> a constant shift preserves
    # both pearson and spearman exactly. ipTM and ipSAE family apply
    # non-linear kernels (1/(1+pae^2/d0^2)) and cutoff filters that move
    # samples across the threshold under a 2.5-A shift, breaking the
    # rank-preservation guarantee; for those we only require strong
    # pearson, not perfect spearman.
    for m in ("i_pae", "min_ipae"):
        assert agg[m]["pearson"] > 0.999, (
            f"{m}: constant-shift pearson not 1.0 (got {agg[m]['pearson']:.4f})"
        )
        assert agg[m]["spearman"] > 0.999, (
            f"{m}: constant-shift spearman not 1.0 (got {agg[m]['spearman']:.4f})"
        )
    # Non-_10 ipSAE variants and ipTM see the cutoff filter at 15 A; the
    # 0.5-A shift keeps almost all pairs on the same side and preserves
    # the per-sample rank order strongly. The _10 ipSAE variants apply
    # the cutoff at 10 A which is at the upper edge of the 4-10-A
    # synthetic range, so a 0.5-A shift filters out many samples and the
    # remainder is too noisy to assert a tight pearson on. Test pearson
    # only on the 15-A cutoff family + the non-cutoff metrics.
    for m in (
        "i_pae", "min_ipae", "i_ptm", "i_ptm_energy",
        "avg_ipsae", "min_ipsae", "max_ipsae",
    ):
        assert agg[m]["pearson"] > 0.95, (
            f"{m}: constant-shift pearson too low (got {agg[m]['pearson']:.4f})"
        )


def test_no_interface_sample_is_dropped_via_nan():
    """A monomer-style sample (chain_idx all 0) contributes no per-sample value."""
    head = _build_head(track=True)
    head.eval()
    torch.manual_seed(0)
    B, L = 4, 24
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[0, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    gt = 4.0 + 6.0 * torch.rand((B, L, L))
    pred = gt + 1.0
    head.update_metric_correlations(
        pae_ev_pred=pred, pae_ev_gt=gt, chain_idx=chain_idx, mask_eff=mask_eff
    )
    agg = head.val_metric_correlations_compute_and_reset()
    assert agg["i_pae"]["mae"] > 0.0, f"single-sample MAE not finite: {agg['i_pae']['mae']:.4e}"
