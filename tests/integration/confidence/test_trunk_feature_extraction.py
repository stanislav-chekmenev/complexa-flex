"""Trunk `expose_intermediates` hook contract.

Covers PR-1 of the confidence-head distillation work. Verifies, for both v1 and
v2 of `LocalLatentsTransformer`:

- Default `expose_intermediates=False`: `forward(input)` has no
  `trunk_intermediates` key.
- After setting `mdl.expose_intermediates = True`: forward returns
  `nn_out["trunk_intermediates"]` carrying `s, z, mask, orig_mask, n_orig`
  with the shape contract documented in
  `docs/superpowers/plans/2026-05-16-pr1-trunk-internals-hook.md` Section 3.3.
- Sub-case where concat features extend `n_orig -> n_extended`: `s.shape[1] ==
  n_extended > n_orig` and `orig_mask.shape[1] == n_orig`. The concat-factory
  forward is stubbed so the test exercises only the trunk's bookkeeping of
  pre- vs post-concat masks.
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


def _make_input(b: int, n: int, device: torch.device = torch.device("cpu")) -> dict:
    torch.manual_seed(0)
    return {
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "x_t": {
            "bb_ca": torch.randn(b, n, 3, device=device),
            "local_latents": torch.randn(b, n, LATENT_DIM, device=device),
        },
        "t": {
            "bb_ca": torch.full((b,), 0.5, device=device),
            "local_latents": torch.full((b,), 0.5, device=device),
        },
    }


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_default_off_has_no_intermediates(module):
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    out = mdl(_make_input(b=2, n=16))
    assert "trunk_intermediates" not in out
    assert mdl.expose_intermediates is False


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_expose_intermediates_returns_shape_contract(module):
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True

    b, n = 2, 16
    inp = _make_input(b=b, n=n)
    out = mdl(inp)

    assert "trunk_intermediates" in out
    inter = out["trunk_intermediates"]

    assert set(inter.keys()) == {"s", "z", "mask", "orig_mask", "n_orig", "local_latents", "ca_coords"}

    assert inter["s"].shape == (b, n, TOKEN_DIM)
    assert inter["z"].shape == (b, n, n, PAIR_REPR_DIM)
    assert inter["mask"].shape == (b, n)
    assert inter["mask"].dtype == torch.bool
    assert inter["orig_mask"].shape == (b, n)
    assert inter["orig_mask"].dtype == torch.bool
    assert inter["n_orig"] == n
    assert isinstance(inter["n_orig"], int)
    assert inter["local_latents"].shape == (b, n, LATENT_DIM)
    assert inter["local_latents"].dtype == torch.float32
    assert inter["ca_coords"].shape == (b, n, 3)
    assert inter["ca_coords"].dtype == torch.float32

    assert torch.equal(inter["mask"], inp["mask"])
    assert torch.equal(inter["orig_mask"], inp["mask"])


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_intermediates_mask_zeroed_on_padded_positions(module):
    mdl = module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = True

    b, n = 2, 16
    inp = _make_input(b=b, n=n)
    inp["mask"][0, 12:] = False

    out = mdl(inp)
    inter = out["trunk_intermediates"]

    s = inter["s"]
    assert torch.all(s[0, 12:] == 0.0), "padded positions on s must be mask-zeroed"


class _StubConcatFactory(torch.nn.Module):
    def __init__(self, token_dim: int, n_concat: int):
        super().__init__()
        self.token_dim = token_dim
        self.n_concat = n_concat

    def forward(self, batch, seq_repr, seq_mask):
        b = seq_repr.shape[0]
        device = seq_repr.device
        pad_repr = torch.zeros(b, self.n_concat, self.token_dim, device=device, dtype=seq_repr.dtype)
        pad_mask = torch.ones(b, self.n_concat, dtype=torch.bool, device=device)
        extended_repr = torch.cat([seq_repr, pad_repr], dim=1)
        extended_mask = torch.cat([seq_mask, pad_mask], dim=1)
        return extended_repr, extended_mask


@pytest.mark.parametrize(
    "module",
    [pytest.param(v1_module, id="v1"), pytest.param(v2_module, id="v2")],
)
def test_intermediates_distinguishes_extended_and_orig_masks(module):
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
    assert inter["s"].shape == (b, n_extended, TOKEN_DIM)
    assert inter["z"].shape == (b, n_extended, n_extended, PAIR_REPR_DIM)
    assert inter["mask"].shape == (b, n_extended)
    assert inter["orig_mask"].shape == (b, n_orig)
    assert inter["s"].shape[1] > inter["orig_mask"].shape[1]
    assert inter["local_latents"].shape == (b, n_extended, LATENT_DIM)
