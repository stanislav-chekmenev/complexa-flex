"""Contract tests for `ConfidenceTrunk`.

The trunk consumes upstream `(s, z, mask, cond)` from a frozen
`LocalLatentsTransformer` (via `expose_intermediates`) and refines them with
N x `MultiheadAttnAndTransition` + `PairReprUpdate` blocks. Outputs are
mask-zeroed; `z` is symmetrised; both are LayerNormed.
"""

from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk


B, N = 2, 11
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
N_BLOCKS = 2
N_HEADS = 4


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
        expects_external_cond=True,
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


def test_forward_shapes_match_inputs() -> None:
    trunk = _make_trunk()
    s, z, mask, cond = _make_inputs()
    s_out, z_out = trunk(s, z, mask, cond)
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape


def test_mask_zeroes_padded_positions() -> None:
    trunk = _make_trunk()
    s, z, mask, cond = _make_inputs()
    mask[0, 8:] = False
    s_out, z_out = trunk(s, z, mask, cond)

    assert torch.all(s_out[0, 8:] == 0.0)
    assert torch.all(z_out[0, 8:, :, :] == 0.0)
    assert torch.all(z_out[0, :, 8:, :] == 0.0)


def test_z_is_symmetric_after_forward() -> None:
    trunk = _make_trunk()
    s, z, mask, cond = _make_inputs()
    _, z_out = trunk(s, z, mask, cond)
    assert torch.allclose(z_out, z_out.transpose(-3, -2), atol=1e-5)


def test_s_layernorm_output_sanity() -> None:
    trunk = _make_trunk()
    s, z, mask, cond = _make_inputs()
    s_out, _ = trunk(s, z, mask, cond)

    valid = s_out[mask]
    per_token_mean = valid.mean(dim=-1)
    per_token_std = valid.std(dim=-1)
    assert per_token_mean.abs().max() < 1e-4
    assert (per_token_std > 0.1).all()
    assert (per_token_std < 5.0).all()


def test_cond_required() -> None:
    trunk = _make_trunk()
    s, z, mask, _ = _make_inputs()
    with pytest.raises(TypeError):
        trunk(s, z, mask)
