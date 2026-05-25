"""ConfidenceDistillationModule.on_validation_epoch_end logs per-head correlations.

Exercises the helper directly with a populated PaeHead and verifies the
30 expected log keys appear and the accumulators reset to zero.
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


def _head(track: bool = True) -> PaeHead:
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


def test_on_validation_epoch_end_logs_all_metrics():
    """After the helper runs the head's val_metric_correlations have been
    compute()'d, logged, and reset."""
    from proteinfoundation.confidence.lightning_module import (
        _log_metric_correlations_for_head,
    )

    head = _head(track=True)
    head.eval()
    B, L = 4, 16
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    gt = 4.0 + 6.0 * torch.rand((B, L, L))
    pred = gt + 0.5
    head.update_metric_correlations(
        pae_ev_pred=pred, pae_ev_gt=gt, chain_idx=chain_idx, mask_eff=mask_eff
    )

    logged: dict[str, float] = {}

    def log_fn(key, value, **kwargs):
        logged[key] = float(value)

    _log_metric_correlations_for_head(head, prefix="val/pae", log_fn=log_fn)

    expected_metrics = (
        "i_pae", "min_ipae", "i_ptm", "i_ptm_energy",
        "avg_ipsae", "min_ipsae", "max_ipsae",
        "avg_ipsae_10", "min_ipsae_10", "max_ipsae_10",
    )
    expected_stats = ("mae", "pearson", "spearman")
    for m in expected_metrics:
        for s in expected_stats:
            key = f"val/pae/{m}/{s}"
            assert key in logged, f"missing log key {key}; logged keys: {sorted(logged)[:5]}..."

    mc = head.val_metric_correlations["i_pae"]
    assert mc["mae"].sum_abs_error.item() == 0.0, "accumulator not reset after compute"


def test_log_function_skips_head_without_track():
    from proteinfoundation.confidence.lightning_module import (
        _log_metric_correlations_for_head,
    )

    head = _head(track=False)
    logged: dict[str, float] = {}
    _log_metric_correlations_for_head(
        head, prefix="val/pae", log_fn=lambda k, v, **kw: logged.update({k: float(v)})
    )
    assert logged == {}, "track_metric_correlations=False must produce no log entries"
