"""`ConfidenceTrunk` Cα binned-distogram residual on `z`.

When constructed with `add_ca_distogram=True`, `ConfidenceTrunk.forward`
accepts a `ca_coords: [b, n, 3]` tensor (nm units) and adds a binned
one-hot distogram to `z` (no projection: `n_bins == pair_repr_dim`)
**before** the pair-biased attention stack. The recipe mirrors the
deleted quality-graft adaptor:

    pair_dists = ||ca[:, :, None] - ca[:, None, :]||_2   # [b, n, n]
    bin_limits = linspace(ca_pair_dist_min, ca_pair_dist_max, n_bins-1)
    bin_indices = bucketize(pair_dists, bin_limits)       # [b, n, n] in [0, n_bins-1]
    one_hot = F.one_hot(bin_indices, num_classes=n_bins).to(z.dtype)
    z = z + one_hot
    z = z * pair_mask                                     # re-zero padded (i, j)

This file pins:

1. Constructor surface: `add_ca_distogram`, `ca_pair_dist_min`,
   `ca_pair_dist_max` exist with sensible nm defaults; default is
   `False` (opt-in).
2. Shape correctness: the helper returns `[b, n, n, pair_repr_dim]` and
   is one-hot along the bin axis.
3. Bin assignment: a distance below `ca_pair_dist_min` lands in bin 0
   (left of `linspace`'s first edge); a distance above `ca_pair_dist_max`
   lands in bin `n_bins-1` (right of the last edge). Diagonal `(i, i)`
   is always 0.0 nm → bin 0.
4. Pad invariance: forward outputs at non-padded positions are
   bit-identical when `ca_coords` is mutated at padded positions only
   (mask-zeroing the distogram with `pair_mask` keeps padded
   contributions from leaking through `PairReprUpdate` → attention).
5. Default-off bit-identity: with `add_ca_distogram=False` (or unset),
   forward output is unchanged from the pre-feature trunk on the same
   `(s, z, mask, cond, local_latents)` tuple — `ca_coords` is ignored
   (and may be `None`).
6. `add_ca_distogram=True` with `ca_coords=None` is a loud failure
   (not silent fallback).
"""

from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
LATENT_DIM = 8
DIM_COND = 32
N_BLOCKS = 2
N_HEADS = 4


def _make_trunk(
    *,
    add_ca_distogram: bool = False,
    ca_pair_dist_min: float = 0.1,
    ca_pair_dist_max: float = 3.0,
    pair_repr_dim: int = PAIR_REPR_DIM,
) -> ConfidenceTrunk:
    return ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=pair_repr_dim,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        dim_cond=DIM_COND,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
        latent_dim=LATENT_DIM,
        add_ca_distogram=add_ca_distogram,
        ca_pair_dist_min=ca_pair_dist_min,
        ca_pair_dist_max=ca_pair_dist_max,
    )


def _make_inputs(
    b: int, n: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    ll = torch.randn(b, n, LATENT_DIM, generator=g)
    return s, z, mask, cond, ll


def test_constructor_kwargs_default_off() -> None:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        dim_cond=DIM_COND,
        latent_dim=LATENT_DIM,
    )
    assert trunk.add_ca_distogram is False
    assert trunk.ca_pair_dist_min == pytest.approx(0.1)
    assert trunk.ca_pair_dist_max == pytest.approx(3.0)


def test_binned_distogram_shape_and_one_hot() -> None:
    trunk = _make_trunk(add_ca_distogram=True)
    b, n = 2, 7
    ca = torch.randn(b, n, 3) * 1.5
    distogram = trunk._binned_ca_distogram(ca)
    assert distogram.shape == (b, n, n, PAIR_REPR_DIM)
    sums = distogram.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums)), (
        "Each (i, j) cell must one-hot sum to 1 across the bin axis."
    )
    assert torch.all((distogram == 0) | (distogram == 1)), (
        "Distogram entries must be exactly 0 or 1."
    )


def test_binned_distogram_assigns_extremes_to_endpoint_bins() -> None:
    trunk = _make_trunk(
        add_ca_distogram=True, ca_pair_dist_min=0.1, ca_pair_dist_max=3.0
    )
    n_bins = PAIR_REPR_DIM
    b, n = 1, 4
    ca = torch.zeros(b, n, 3)
    ca[0, 1] = torch.tensor([0.05, 0.0, 0.0])
    ca[0, 2] = torch.tensor([10.0, 0.0, 0.0])
    ca[0, 3] = torch.tensor([1.5, 0.0, 0.0])

    distogram = trunk._binned_ca_distogram(ca)
    bin_idx = distogram.argmax(dim=-1)

    assert bin_idx[0, 0, 0].item() == 0, "diagonal (i, i) distance is 0 → bin 0"
    assert bin_idx[0, 0, 1].item() == 0, "0.05 nm < ca_pair_dist_min → bin 0"
    assert bin_idx[0, 0, 2].item() == n_bins - 1, (
        "10 nm > ca_pair_dist_max → bin n_bins-1"
    )
    mid = bin_idx[0, 0, 3].item()
    assert 0 < mid < n_bins - 1, (
        f"1.5 nm (middle of [0.1, 3.0]) should land strictly between endpoint bins, got {mid}"
    )


def test_binned_distogram_is_symmetric_in_i_j() -> None:
    trunk = _make_trunk(add_ca_distogram=True)
    b, n = 2, 6
    ca = torch.randn(b, n, 3) * 1.5
    distogram = trunk._binned_ca_distogram(ca)
    assert torch.equal(distogram, distogram.transpose(1, 2)), (
        "Pair distances are symmetric, so the distogram must be too."
    )


def test_forward_off_is_bit_identical_to_baseline() -> None:
    """With `add_ca_distogram=False`, forward must be bit-identical to a
    trunk constructed without the kwarg (proving the feature is a true
    no-op when disabled and that `ca_coords=None` is the legal default).
    """
    trunk_off = _make_trunk(add_ca_distogram=False).eval()
    trunk_baseline = ConfidenceTrunk(
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
        latent_dim=LATENT_DIM,
    ).eval()
    trunk_off.load_state_dict(trunk_baseline.state_dict(), strict=False)

    b, n = 2, 8
    s, z, mask, cond, ll = _make_inputs(b=b, n=n, seed=5)

    with torch.no_grad():
        s_off, z_off = trunk_off(s, z, mask, cond, ll)
        s_base, z_base = trunk_baseline(s, z, mask, cond, ll)

    assert torch.equal(s_off, s_base)
    assert torch.equal(z_off, z_base)


def test_forward_on_changes_z_vs_off() -> None:
    trunk_off = _make_trunk(add_ca_distogram=False).eval()
    trunk_on = _make_trunk(add_ca_distogram=True).eval()
    trunk_on.load_state_dict(trunk_off.state_dict(), strict=False)

    b, n = 2, 8
    s, z, mask, cond, ll = _make_inputs(b=b, n=n, seed=11)
    ca = torch.randn(b, n, 3) * 1.5

    with torch.no_grad():
        _, z_off = trunk_off(s, z, mask, cond, ll)
        _, z_on = trunk_on(s, z, mask, cond, ll, ca_coords=ca)

    assert not torch.allclose(z_on, z_off, atol=1e-6), (
        "Enabling the distogram must change z output: the residual is wired in."
    )


def test_forward_on_with_none_ca_coords_raises() -> None:
    trunk = _make_trunk(add_ca_distogram=True).eval()
    b, n = 2, 8
    s, z, mask, cond, ll = _make_inputs(b=b, n=n)
    with pytest.raises((ValueError, TypeError, AssertionError)):
        with torch.no_grad():
            trunk(s, z, mask, cond, ll, ca_coords=None)


def test_distogram_is_mask_zeroed_at_padded_positions() -> None:
    """Two forwards differing only in `ca_coords` at padded positions must
    produce identical `(s, z)` at every position. The `* pair_mask` after
    the distogram add keeps padded (i, j) cells from leaking through
    `PairReprUpdate`'s outer products and triangle multiplication.
    """
    trunk = _make_trunk(add_ca_distogram=True).eval()
    b, n = 2, 16
    s, z, mask, cond, ll = _make_inputs(b=b, n=n, seed=21)
    mask[0, 12:] = False

    ca_a = torch.randn(b, n, 3, generator=torch.Generator().manual_seed(23)) * 1.5
    ca_b = ca_a.clone()
    ca_b[0, 12:] = torch.randn(4, 3, generator=torch.Generator().manual_seed(29)) * 5.0

    with torch.no_grad():
        s_a, z_a = trunk(s, z, mask, cond, ll, ca_coords=ca_a)
        s_b, z_b = trunk(s, z, mask, cond, ll, ca_coords=ca_b)

    assert torch.allclose(s_a, s_b, atol=1e-6), (
        "Mutating ca_coords at padded positions must not affect s anywhere."
    )
    assert torch.allclose(z_a, z_b, atol=1e-6), (
        "Mutating ca_coords at padded positions must not affect z anywhere."
    )


def test_constructor_rejects_pair_repr_dim_below_two() -> None:
    """A binned distogram with `n_bins < 2` has no `linspace(min, max, n_bins-1)`
    edges to split on. Pin a loud failure rather than silently bucketing
    everything into bin 0.
    """
    with pytest.raises((ValueError, AssertionError)):
        ConfidenceTrunk(
            token_dim=TOKEN_DIM,
            pair_repr_dim=1,
            n_blocks=N_BLOCKS,
            n_heads=N_HEADS,
            dim_cond=DIM_COND,
            latent_dim=LATENT_DIM,
            add_ca_distogram=True,
        )
