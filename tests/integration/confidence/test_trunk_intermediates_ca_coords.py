"""Trunk-hook contract: `ca_coords` exposed in `trunk_intermediates`.

`ca_coords` is exposed alongside `s`, `z`, `local_latents` as a utility
for any future ca_coords-consuming confidence head (e.g. one that needs
a Cα-Cα distogram). This test pins the contract that
`expose_intermediates` either exposes ALL of them or none — there is no
partial state.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
LATENT_DIM = 8
DIM_COND = 32
NLAYERS = 2
NHEADS = 4


def _trunk_kwargs() -> dict:
    return dict(
        nlayers=NLAYERS,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        nheads=NHEADS,
        parallel_mha_transition=False,
        strict_feats=False,
        feats_seq=["xt_bb_ca", "xt_local_latents"],
        feats_cond_seq=["time_emb_bb_ca", "time_emb_local_latents"],
        feats_pair_repr=["rel_seq_sep"],
        feats_pair_cond=["time_emb_bb_ca"],
        dim_cond=DIM_COND,
        idx_emb_dim=DIM_COND,
        t_emb_dim=DIM_COND,
        seq_sep_dim=63,
        xt_pair_dist_dim=30,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        x_sc_pair_dist_dim=30,
        x_sc_pair_dist_min=0.1,
        x_sc_pair_dist_max=3.0,
        update_pair_repr=False,
        update_pair_repr_every_n=3,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        latent_dim=LATENT_DIM,
        output_parameterization={"bb_ca": "v", "local_latents": "v"},
        concat_features={
            "enable_motif": False,
            "enable_target": False,
            "enable_ligand": False,
            "motif_pair_features": False,
            "target_pair_features": False,
            "ligand_pair_features": False,
        },
    )


def build_minimal_transformer(expose_intermediates: bool) -> LocalLatentsTransformer:
    mdl = LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = expose_intermediates
    return mdl


def build_minimal_input(B: int, n: int) -> dict:
    torch.manual_seed(0)
    return {
        "mask": torch.ones(B, n, dtype=torch.bool),
        "x_t": {
            "bb_ca": torch.randn(B, n, 3),
            "local_latents": torch.randn(B, n, LATENT_DIM),
        },
        "t": {
            "bb_ca": torch.full((B,), 0.5),
            "local_latents": torch.full((B,), 0.5),
        },
    }


def test_intermediates_expose_ca_coords_when_enabled():
    model = build_minimal_transformer(expose_intermediates=True)
    B, n = 2, 16
    fake_input = build_minimal_input(B=B, n=n)
    with torch.no_grad():
        out = model(fake_input)

    assert "trunk_intermediates" in out
    inter = out["trunk_intermediates"]
    assert "ca_coords" in inter
    assert inter["ca_coords"].shape[0] == inter["local_latents"].shape[0]
    assert inter["ca_coords"].shape[1] == inter["local_latents"].shape[1]
    assert inter["ca_coords"].shape[-1] == 3


def test_intermediates_absent_when_expose_intermediates_false():
    model = build_minimal_transformer(expose_intermediates=False)
    B, n = 2, 16
    fake_input = build_minimal_input(B=B, n=n)
    with torch.no_grad():
        out = model(fake_input)

    assert "trunk_intermediates" not in out
