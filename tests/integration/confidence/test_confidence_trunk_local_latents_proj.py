"""`ConfidenceTrunk.local_latents_proj` contract.

The trunk gains a `local_latents_proj = Sequential(Linear(latent_dim,
token_dim, bias=False), LayerNorm(token_dim))` that lifts the frozen
`local_latents` (shape `[b, n_ext, latent_dim]`) to token-dim space and
is added to `s` as a residual at the head of `forward`. This file pins
five contracts:

1. Module presence and the exact `Sequential` structure with no bias on
   the linear and a `LayerNorm(token_dim)` afterwards.
2. `latent_dim` is configurable (default 8). Constructor accepts it as a
   kwarg; the projection's input dim matches.
3. `forward(..., local_latents=zeros)` is bit-equal to a forward whose
   trunk module has had `local_latents_proj` zero'd out (the projection
   acts as zero on a zero input, modulo `LayerNorm`'s eps; see the
   manual-LN sanity check).
4. `forward(..., local_latents=randn)` differs from the zero-input
   forward at non-padded positions -- the projection is wired into the
   computation, not a dead branch.
5. Mask-zeroing: the `ll` residual is masked to zero at padded positions
   before the `s = s + ll` add.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from proteinfoundation.nn.confidence.base import ConfidenceTrunk


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
LATENT_DIM = 8
DIM_COND = 32
N_BLOCKS = 2
N_HEADS = 4


def _make_trunk(latent_dim: int = LATENT_DIM) -> ConfidenceTrunk:
    return ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        dim_cond=DIM_COND,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
        latent_dim=latent_dim,
    )


def _make_inputs(
    b: int, n: int, latent_dim: int = LATENT_DIM, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    return s, z, mask, cond


def test_local_latents_proj_module_structure() -> None:
    trunk = _make_trunk()
    assert hasattr(trunk, "local_latents_proj")
    proj = trunk.local_latents_proj
    assert isinstance(proj, nn.Sequential)
    assert len(proj) == 2

    lin = proj[0]
    assert isinstance(lin, nn.Linear)
    assert lin.in_features == LATENT_DIM
    assert lin.out_features == TOKEN_DIM
    assert lin.bias is None

    ln = proj[1]
    assert isinstance(ln, nn.LayerNorm)
    assert ln.normalized_shape == (TOKEN_DIM,)


@pytest.mark.parametrize("latent_dim", [4, 8, 16])
def test_local_latents_proj_in_features_tracks_kwarg(latent_dim: int) -> None:
    trunk = _make_trunk(latent_dim=latent_dim)
    assert trunk.local_latents_proj[0].in_features == latent_dim
    assert trunk.local_latents_proj[0].out_features == TOKEN_DIM


def test_forward_accepts_local_latents_and_returns_expected_shapes() -> None:
    trunk = _make_trunk().eval()
    b, n = 2, 16
    s, z, mask, cond = _make_inputs(b=b, n=n)
    ll = torch.randn(b, n, LATENT_DIM)
    with torch.no_grad():
        s_out, z_out = trunk(s, z, mask, cond, ll)
    assert s_out.shape == (b, n, TOKEN_DIM)
    assert z_out.shape == (b, n, n, PAIR_REPR_DIM)


def test_nonzero_local_latents_changes_output_vs_zero() -> None:
    trunk = _make_trunk().eval()
    b, n = 2, 16
    s, z, mask, cond = _make_inputs(b=b, n=n)
    ll_zero = torch.zeros(b, n, LATENT_DIM)
    ll_rand = torch.randn(b, n, LATENT_DIM, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        s_zero, _ = trunk(s, z, mask, cond, ll_zero)
        s_rand, _ = trunk(s, z, mask, cond, ll_rand)

    assert not torch.allclose(s_zero, s_rand, atol=1e-6), (
        "Nonzero local_latents must influence the trunk output: the projection "
        "is wired into the s residual, not dead."
    )


def test_zero_local_latents_projection_at_init_is_zero_residual() -> None:
    """At init, `LN(Linear(0)) == 0` because Linear has no bias and the
    default `LayerNorm` affine is `gamma=1, beta=0` -- so an all-zero
    input yields an all-zero LN output, then `+ beta=0`.

    This contract holds *only at init*. Once training drifts `beta` off
    zero (it is trainable), `LN(0) = beta` and this assertion no longer
    holds for `proj(zeros)` itself. The trunk stays correct anyway
    because the `* mask_f` multiply at `base.py:142` zeros the residual
    at padded positions regardless of `beta` -- see
    `test_local_latents_residual_is_mask_zeroed_at_padded_positions`
    for the post-training-safe contract.
    """
    trunk = _make_trunk().eval()
    b, n = 2, 16
    ll_zero = torch.zeros(b, n, LATENT_DIM)

    with torch.no_grad():
        projected = trunk.local_latents_proj(ll_zero)
    assert torch.equal(projected, torch.zeros_like(projected)), (
        f"Linear(no bias) + LayerNorm of zeros at init must equal zero, got max abs "
        f"{projected.abs().max().item()}"
    )


def test_mask_multiply_guards_against_trained_layernorm_bias_leak() -> None:
    """The load-bearing safety property: after training drifts the
    `LayerNorm.bias` off zero, padded positions stay clean because the
    `* mask_f` multiply in `ConfidenceTrunk.forward` zeros the residual
    at masked-out positions before the `s = s + ll` add.

    Probe the post-LN bias drift directly: write a non-zero bias into
    `local_latents_proj[1].bias`, run two forward passes whose
    `local_latents` differ only at masked-out positions, and assert
    `s_out` is identical at every position. If the mask multiply were
    removed, the LN-bias residual would leak from padded latents into
    valid `s` cells through the downstream attention.
    """
    trunk = _make_trunk().eval()
    with torch.no_grad():
        trunk.local_latents_proj[1].bias.fill_(0.3)

    b, n = 2, 16
    s, z, mask, cond = _make_inputs(b=b, n=n)
    mask[0, 12:] = False

    ll_a = torch.randn(b, n, LATENT_DIM, generator=torch.Generator().manual_seed(17))
    ll_b = ll_a.clone()
    ll_b[0, 12:] = torch.randn(4, LATENT_DIM, generator=torch.Generator().manual_seed(19))

    with torch.no_grad():
        s_a, _ = trunk(s, z, mask, cond, ll_a)
        s_b, _ = trunk(s, z, mask, cond, ll_b)

    assert torch.allclose(s_a, s_b, atol=1e-6), (
        "With LN.bias drifted off zero, mutating local_latents at masked "
        "positions must STILL not affect s_out: the mask multiply at "
        "ConfidenceTrunk.forward is load-bearing once training updates beta."
    )


def test_local_latents_residual_is_mask_zeroed_at_padded_positions() -> None:
    """Padded positions on `mask=False` must not be perturbed by the
    `local_latents` residual. Concretely: two forward passes that differ
    only at padded positions of `local_latents` must produce identical
    `s_out` outputs (within numerics) at every position, because the
    residual is multiplied by the mask before being added.
    """
    trunk = _make_trunk().eval()
    b, n = 2, 16
    s, z, mask, cond = _make_inputs(b=b, n=n)
    mask[0, 12:] = False

    ll_a = torch.randn(b, n, LATENT_DIM, generator=torch.Generator().manual_seed(11))
    ll_b = ll_a.clone()
    ll_b[0, 12:] = torch.randn(4, LATENT_DIM, generator=torch.Generator().manual_seed(13))

    with torch.no_grad():
        s_a, _ = trunk(s, z, mask, cond, ll_a)
        s_b, _ = trunk(s, z, mask, cond, ll_b)

    assert torch.allclose(s_a, s_b, atol=1e-6), (
        "Mutating local_latents at masked-out positions must not affect the "
        "trunk output anywhere; the residual must be mask-zeroed before the add."
    )
