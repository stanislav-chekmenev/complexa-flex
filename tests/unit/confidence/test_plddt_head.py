"""Contract tests for `PLDDTHead`.

Covers shape of `plddt_logits`, the bin-expected-value reduction, the
uniform-logits midpoint property, and that the call is mask-safe.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


B, N = 2, 9
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
LATENT_DIM = 8
NUM_BINS = 50


def _make_head(num_plddt_bins: int = NUM_BINS) -> PLDDTHead:
    trunk = ConfidenceTrunk(
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
    return PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=num_plddt_bins,
        bin_min=0.0,
        bin_max=100.0,
    ).eval()


def _make_inputs(b: int = B, n: int = N, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    local_latents = torch.randn(b, n, LATENT_DIM, generator=g)
    return s, z, mask, cond, local_latents


def test_forward_returns_plddt_logits() -> None:
    head = _make_head()
    s, z, mask, cond, ll = _make_inputs()
    out = head(s, z, mask, cond, ll)
    assert "plddt_logits" in out
    assert out["plddt_logits"].shape == (B, N, NUM_BINS)


def test_logits_to_expected_value_in_range() -> None:
    head = _make_head()
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(B, N, NUM_BINS, generator=g)
    ev = head.logits_to_expected_value(logits)
    assert ev.shape == (B, N)
    assert torch.all(ev >= 0.0)
    assert torch.all(ev <= 100.0)


def test_uniform_logits_yields_midpoint() -> None:
    head = _make_head()
    logits = torch.zeros(B, N, NUM_BINS)
    ev = head.logits_to_expected_value(logits)
    assert torch.allclose(ev, torch.full_like(ev, 50.0), atol=1e-4)


def test_mask_zero_safe() -> None:
    head = _make_head()
    s, z, mask, cond, ll = _make_inputs()
    mask[0, 6:] = False
    out = head(s, z, mask, cond, ll)
    logits = out["plddt_logits"]

    assert torch.isfinite(logits[mask]).all()
    assert torch.all(logits[0, 6:, :] == 0.0)


def test_bin_centers_are_buffer_and_correct() -> None:
    head = _make_head()
    assert "bin_centers" in dict(head.named_buffers())
    centers = head.bin_centers
    expected = torch.tensor([1.0 + 2.0 * i for i in range(NUM_BINS)])
    assert torch.allclose(centers, expected, atol=1e-5)


def test_logits_to_expected_value_tracks_dominant_bin() -> None:
    head = _make_head()
    logits = torch.zeros(B, N, NUM_BINS)
    logits[..., 30] = 1e3
    ev = head.logits_to_expected_value(logits)
    assert torch.allclose(ev, torch.full_like(ev, 61.0), atol=0.5)
