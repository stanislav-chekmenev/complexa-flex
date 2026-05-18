"""Symmetrisation contract: trunk passes `z` through, heads opt in per-`_predict`.

The pre-PR-B trunk applied an explicit `(i, j) <-> (j, i)` projection on `z`
before returning. PR-B removes that line (CLAUDE.md option (b)); concrete
symmetric heads must symmetrise inside their own `_predict`, and asymmetric
heads (PaeHead, PR-B Slice 2) consume the directional `z` as-is.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


B, N = 2, 11
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
N_BLOCKS = 2
N_HEADS = 4
NUM_PLDDT_BINS = 50


def _make_trunk(**overrides) -> ConfidenceTrunk:
    kwargs = dict(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        dim_cond=DIM_COND,
        use_tri_mult=True,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
    )
    kwargs.update(overrides)
    return ConfidenceTrunk(**kwargs).eval()


def _make_inputs(b: int = B, n: int = N, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    return s, z, mask, cond


def test_trunk_does_not_symmetrise_z() -> None:
    trunk = _make_trunk()
    s, z, mask, cond = _make_inputs()
    _, z_out = trunk(s, z, mask, cond)
    assert not torch.allclose(z_out, z_out.transpose(-3, -2), atol=1e-3)


def test_plddt_predict_invariant_to_z_symmetrisation() -> None:
    """`PLDDTHead._predict` consumes `s` only; the `z` argument is dropped.

    The contract is exercised at the head's `_predict` boundary, not at
    `forward`, because the shared trunk does depend on `z` (pair-biased
    attention from `z` to `s`); only the head's own prediction MLP is the
    surface that the per-head symmetrisation refactor governs.
    """
    trunk = _make_trunk()
    head = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_PLDDT_BINS,
        bin_min=0.0,
        bin_max=100.0,
    ).eval()
    s, z, mask, _ = _make_inputs()

    out_direct = head._predict(s, z, mask)["plddt_logits"]
    out_transposed = head._predict(s, z.transpose(-3, -2), mask)["plddt_logits"]
    assert torch.allclose(out_direct, out_transposed, atol=1e-6)


def test_pae_predict_is_directional() -> None:
    from proteinfoundation.nn.confidence.pae_head import PaeHead

    trunk = _make_trunk()
    head = PaeHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
    ).eval()
    s, z, mask, cond = _make_inputs()

    out = head(s, z, mask, cond)
    logits = out["pae_logits"]
    assert not torch.allclose(logits, logits.transpose(-3, -2), atol=1e-3)
