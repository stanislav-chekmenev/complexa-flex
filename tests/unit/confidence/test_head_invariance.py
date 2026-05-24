"""Padding-permutation invariance for the confidence head.

PR-3 verifies the one structural property the head owns at this layer:
permuting padded positions does not perturb the un-padded outputs.
Rotational invariance under coordinate rotations is a trunk property and
is exercised by the end-to-end tests in PR-5.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


B, N = 2, 8
N_VALID = 5
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
LATENT_DIM = 8


def _make_head() -> PLDDTHead:
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
        num_plddt_bins=50,
        bin_min=0.0,
        bin_max=100.0,
    ).eval()


def test_padded_position_permutation_invariance() -> None:
    head = _make_head()
    g = torch.Generator().manual_seed(0)
    s = torch.randn(B, N, TOKEN_DIM, generator=g)
    z = torch.randn(B, N, N, PAIR_REPR_DIM, generator=g)
    cond = torch.randn(B, N, DIM_COND, generator=g)
    local_latents = torch.randn(B, N, LATENT_DIM, generator=g)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :N_VALID] = True

    out_a = head(s, z, mask, cond, local_latents)["plddt_logits"]

    perm = torch.tensor([0, 1, 2, 3, 4, 7, 5, 6])
    s_perm = s[:, perm]
    z_perm = z[:, perm][:, :, perm]
    cond_perm = cond[:, perm]
    ll_perm = local_latents[:, perm]
    mask_perm = mask[:, perm]

    out_b = head(s_perm, z_perm, mask_perm, cond_perm, ll_perm)["plddt_logits"]

    assert torch.allclose(out_a[:, :N_VALID, :], out_b[:, :N_VALID, :], atol=1e-5)
