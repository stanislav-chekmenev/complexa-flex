"""Inter-PR seam test: PR-1 `expose_intermediates` x PR-3 `PLDDTHead`.

Wires a tiny `LocalLatentsTransformer` (with `expose_intermediates=True`)
to a tiny `PLDDTHead` and verifies the head consumes the trunk's
intermediates, returns `[b, n_extended, num_bins]`, and the caller can
trim to `[:, :n_orig]`.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn import local_latents_transformer as v1_module
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
LATENT_DIM = 8
DIM_COND = 32
NLAYERS = 2
NHEADS = 4
NUM_BINS = 50


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


def _make_trunk_input(b: int, n: int) -> dict:
    torch.manual_seed(0)
    return {
        "mask": torch.ones(b, n, dtype=torch.bool),
        "x_t": {
            "bb_ca": torch.randn(b, n, 3),
            "local_latents": torch.randn(b, n, LATENT_DIM),
        },
        "t": {
            "bb_ca": torch.full((b,), 0.5),
            "local_latents": torch.full((b,), 0.5),
        },
    }


def _make_plddt_head() -> PLDDTHead:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=NHEADS,
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
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    ).eval()


def test_tiny_trunk_to_plddt_head_wires() -> None:
    mdl = v1_module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True

    b, n_orig = 2, 12
    inp = _make_trunk_input(b=b, n=n_orig)

    with torch.no_grad():
        out = mdl(inp)
    inter = out["trunk_intermediates"]

    s = inter["s"]
    z = inter["z"]
    mask = inter["mask"]
    local_latents = inter["local_latents"]
    n_extended = s.shape[1]
    assert n_extended == n_orig

    head = _make_plddt_head()
    g = torch.Generator().manual_seed(1)
    cond = torch.randn(b, n_extended, DIM_COND, generator=g)

    head_out = head(s, z, mask, cond, local_latents)
    logits = head_out["plddt_logits"]

    assert logits.shape == (b, n_extended, NUM_BINS)

    trimmed = logits[:, : inter["n_orig"], :]
    assert trimmed.shape == (b, n_orig, NUM_BINS)
    assert torch.isfinite(trimmed).all()
