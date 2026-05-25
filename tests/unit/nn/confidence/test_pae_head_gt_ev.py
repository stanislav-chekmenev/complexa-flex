"""GT-PAE expected-value helper on PaeHead.

The CE term is trained against integer-A AFDB labels via the bin index;
the EV term reads the float-A continuous label. Both are inputs to the
metric-correlation pipeline. The helper that the pipeline uses (the
'GT side') must produce values consistent with `_labels_to_continuous`
called with the head's bin centers -- pinned here so a future refactor
of the bin convention can't silently shift the GT EV.
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


def _build_head() -> PaeHead:
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
    )


def test_pae_ev_from_labels_matches_bin_centers():
    head = _build_head()
    labels = torch.tensor([[[0, 1, 63], [10, 31, 63]]])
    ev = head._pae_ev_from_labels(labels)
    expected_centers = torch.tensor([0.25, 0.75, 31.75, 5.25, 15.75, 31.75]).reshape(1, 2, 3)
    assert torch.allclose(ev, expected_centers, atol=1e-6), (
        f"GT EV mismatch: got {ev.tolist()}, expected {expected_centers.tolist()}"
    )


def test_pae_ev_from_logits_one_hot_matches_centers():
    """One-hot logits over bin k must yield exactly centers[k]."""
    head = _build_head()
    B, L, K = 1, 4, 64
    logits = torch.full((B, L, L, K), -1e9)
    bin_idx = torch.randint(0, K, (B, L, L))
    logits.scatter_(-1, bin_idx[..., None], 1e9)
    ev_logits = head.pae_ev_from_logits(logits)
    ev_labels = head._pae_ev_from_labels(bin_idx)
    assert torch.allclose(ev_logits, ev_labels, atol=1e-4), (
        f"One-hot logits and label EV must agree: max |diff| = {(ev_logits - ev_labels).abs().max().item():.2e}"
    )
