"""Trunk `expose_intermediates` must include `local_latents` (pre-trim).

Local-latents fusion contract: the frozen `LocalLatentsTransformer` exposes
the **pre-trim** `local_latents` tensor of shape `[b, n_extended, latent_dim]`
in `trunk_intermediates`. The user-facing `nn_out["local_latents"]` is the
trimmed `[b, n_orig, latent_dim]` tensor; the intermediate is the un-trimmed
companion that the confidence-head's `ConfidenceTrunk` projection consumes.

Exercises both `local_latents_transformer.py` (v1) and
`local_latents_transformer_v2.py` (v2) via parametrisation, mirroring
`test_trunk_feature_extraction.py`.
"""

from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn import local_latents_transformer as v1_module
from proteinfoundation.nn import local_latents_transformer_v2 as v2_module


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


def _make_input(b: int, n: int) -> dict:
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


class _StubConcatFactory(torch.nn.Module):
    def __init__(self, token_dim: int, n_concat: int):
        super().__init__()
        self.token_dim = token_dim
        self.n_concat = n_concat

    def forward(self, batch, seq_repr, seq_mask):
        b = seq_repr.shape[0]
        device = seq_repr.device
        pad_repr = torch.zeros(
            b, self.n_concat, self.token_dim, device=device, dtype=seq_repr.dtype
        )
        pad_mask = torch.ones(b, self.n_concat, dtype=torch.bool, device=device)
        extended_repr = torch.cat([seq_repr, pad_repr], dim=1)
        extended_mask = torch.cat([seq_mask, pad_mask], dim=1)
        return extended_repr, extended_mask


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_local_latents_in_intermediates_shape_and_dtype(module):
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True

    b, n = 2, 16
    out = mdl(_make_input(b=b, n=n))
    inter = out["trunk_intermediates"]

    assert "local_latents" in inter
    ll = inter["local_latents"]
    assert ll.shape == (b, n, LATENT_DIM)
    assert ll.dtype == torch.float32


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_local_latents_mask_zeroed_on_padded_positions(module):
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True

    b, n = 2, 16
    inp = _make_input(b=b, n=n)
    inp["mask"][0, 12:] = False

    out = mdl(inp)
    ll = out["trunk_intermediates"]["local_latents"]
    assert torch.all(ll[0, 12:] == 0.0)


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_local_latents_is_pre_trim_extended_axis(module):
    """When concat features extend `n_orig -> n_extended`, the intermediate
    `local_latents` lives in the **extended** frame, not the trimmed one.
    """
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True
    mdl.use_concat = True
    mdl.concat_factory = _StubConcatFactory(token_dim=TOKEN_DIM, n_concat=5)
    mdl.use_advanced_pair = False

    b, n_orig = 2, 16
    inp = _make_input(b=b, n=n_orig)
    out = mdl(inp)
    inter = out["trunk_intermediates"]
    n_extended = n_orig + 5

    assert inter["n_orig"] == n_orig
    ll = inter["local_latents"]
    assert ll.shape == (b, n_extended, LATENT_DIM)

    trimmed_key, trimmed_val = next(iter(out["local_latents"].items()))
    del trimmed_key
    assert trimmed_val.shape == (b, n_orig, LATENT_DIM), (
        "nn_out['local_latents'] keeps its trimmed-to-n_orig contract"
    )
