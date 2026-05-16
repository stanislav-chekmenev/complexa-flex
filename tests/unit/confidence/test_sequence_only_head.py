"""Contract tests for `SequenceOnlyPLDDTHead`.

The control head is a diagnostic: same trunk surface as `PLDDTHead`, but
`z` is zeroed before the trunk call so the trunk's pair-bias attention sees
a constant pair representation. This isolates the sequence-only signal.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence import CONFIDENCE_HEAD_REGISTRY
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_sequence_only_head import (
    SequenceOnlyPLDDTHead,
)


B, N = 2, 37
TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
NUM_BINS = 50


def _make_head() -> SequenceOnlyPLDDTHead:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1_000_000,
    )
    return SequenceOnlyPLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    ).eval()


def _make_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, N, TOKEN_DIM, generator=g)
    z = torch.randn(B, N, N, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(B, N, dtype=torch.bool)
    cond = torch.randn(B, N, DIM_COND, generator=g)
    return s, z, mask, cond


def test_registry_contains_sequence_only_head() -> None:
    assert "plddt_sequence_only" in CONFIDENCE_HEAD_REGISTRY
    assert CONFIDENCE_HEAD_REGISTRY["plddt_sequence_only"] is SequenceOnlyPLDDTHead


def test_forward_returns_plddt_logits_with_expected_shape() -> None:
    head = _make_head()
    s, z, mask, cond = _make_inputs()
    out = head(s, z, mask, cond)
    assert "plddt_logits" in out
    assert out["plddt_logits"].shape == (B, N, NUM_BINS)


def test_output_is_z_independent() -> None:
    head = _make_head()
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    s = torch.randn(B, N, TOKEN_DIM, generator=g1)
    cond = torch.randn(B, N, DIM_COND, generator=g1)
    mask = torch.ones(B, N, dtype=torch.bool)

    z1 = torch.randn(B, N, N, PAIR_REPR_DIM, generator=g1)
    z2 = torch.randn(B, N, N, PAIR_REPR_DIM, generator=g2)

    out1 = head(s, z1, mask, cond)["plddt_logits"]
    out2 = head(s, z2, mask, cond)["plddt_logits"]
    assert torch.allclose(out1, out2, atol=1e-5)


def test_padded_positions_are_zeroed() -> None:
    head = _make_head()
    s, z, mask, cond = _make_inputs()
    mask[0, 20:] = False
    out = head(s, z, mask, cond)
    logits = out["plddt_logits"]
    assert torch.all(logits[0, 20:, :] == 0.0)
    assert torch.isfinite(logits[mask]).all()
