"""Contract tests for `d_in_token` / `d_in_pair_token` readout kwargs.

PR-1 quality-graft port: the adaptor module collapses the trunk's 776-dim `s`
and 256-dim `z` to 384/128 respectively, so the heads must accept a readout
width that may differ from the trunk's emitted `token_dim` / `pair_repr_dim`.

These tests pin:

- Default behavior: omitting `d_in_token` keeps `LayerNorm` + `Linear`
  readout at `token_dim` (same for `d_in_pair_token` and `pair_repr_dim`).
- Override behavior: passing the kwarg retargets BOTH the pre-readout
  `LayerNorm` and the readout `Linear` to that new width.
- The trunk-vs-head dim assertion in `BaseConfidenceHead.__init__` still
  applies — the kwarg only retargets the readout, not the trunk-coupling.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
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


def test_plddt_head_default_d_in_token_matches_token_dim() -> None:
    head = PLDDTHead(
        trunk=_make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=50,
    )
    assert head.d_in_token == TOKEN_DIM
    assert isinstance(head.logits_norm, nn.LayerNorm)
    assert head.logits_norm.normalized_shape == (TOKEN_DIM,)
    assert head.logits_linear.in_features == TOKEN_DIM
    assert head.logits_linear.out_features == 50


def test_plddt_head_d_in_token_override_retargets_readout() -> None:
    new_width = 384
    head = PLDDTHead(
        trunk=_make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=50,
        d_in_token=new_width,
    )
    assert head.d_in_token == new_width
    assert head.logits_norm.normalized_shape == (new_width,)
    assert head.logits_linear.in_features == new_width
    assert head.logits_linear.out_features == 50

    s_adapted = torch.randn(2, 7, new_width)
    out_pre_norm = head.logits_norm(s_adapted)
    logits = head.logits_linear(out_pre_norm)
    assert logits.shape == (2, 7, 50)


def test_pae_head_default_d_in_pair_token_matches_pair_repr_dim() -> None:
    head = PaeHead(
        trunk=_make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=64,
    )
    assert head.d_in_pair_token == PAIR_REPR_DIM
    assert head.logits_norm.normalized_shape == (PAIR_REPR_DIM,)
    assert head.logits_linear.in_features == PAIR_REPR_DIM
    assert head.logits_linear.out_features == 64


def test_pae_head_d_in_pair_token_override_retargets_readout() -> None:
    new_width = 128
    head = PaeHead(
        trunk=_make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=64,
        d_in_pair_token=new_width,
    )
    assert head.d_in_pair_token == new_width
    assert head.logits_norm.normalized_shape == (new_width,)
    assert head.logits_linear.in_features == new_width
    assert head.logits_linear.out_features == 64

    z_adapted = torch.randn(2, 7, 7, new_width)
    out_pre_norm = head.logits_norm(z_adapted)
    logits = head.logits_linear(out_pre_norm)
    assert logits.shape == (2, 7, 7, 64)


def test_plddt_head_default_forward_unchanged() -> None:
    """Default-kwargs construction must produce a head whose forward matches
    the pre-kwarg behavior: same readout module-list (LN(token_dim) +
    Linear(token_dim, bins))."""
    trunk = _make_trunk()
    head = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=50,
    ).eval()
    g = torch.Generator().manual_seed(0)
    s = torch.randn(2, 10, TOKEN_DIM, generator=g)
    z = torch.randn(2, 10, 10, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(2, 10, dtype=torch.bool)
    cond = torch.randn(2, 10, DIM_COND, generator=g)
    ll = torch.randn(2, 10, LATENT_DIM, generator=g)
    out = head(s, z, mask, cond, ll)
    assert out["plddt_logits"].shape == (2, 10, 50)


def test_pae_head_default_forward_unchanged() -> None:
    trunk = _make_trunk()
    head = PaeHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=64,
    ).eval()
    g = torch.Generator().manual_seed(0)
    s = torch.randn(2, 10, TOKEN_DIM, generator=g)
    z = torch.randn(2, 10, 10, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(2, 10, dtype=torch.bool)
    cond = torch.randn(2, 10, DIM_COND, generator=g)
    ll = torch.randn(2, 10, LATENT_DIM, generator=g)
    out = head(s, z, mask, cond, ll)
    assert out["pae_logits"].shape == (2, 10, 10, 64)
