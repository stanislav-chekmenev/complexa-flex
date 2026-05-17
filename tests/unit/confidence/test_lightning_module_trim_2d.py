"""Contract test for `_trim_batch` 2D label support (PR-B Slice 2 follow-up).

The pre-Slice-2 `_trim_batch` only sliced axis 1, so a 2D label like
`pae_mask` of shape `(B, N_ext, N_ext)` was mistakenly trimmed to
`(B, n_orig, N_ext)` and survived into the loss path with a stale axis-2.
Slice 2's green phase extends `_trim_batch` to mirror `_trim_head_output`'s
square-pair branch — square 2D pair fields trim along both axis 1 and 2,
1D residue fields trim along axis 1, scalars / non-tensors pass through.

This test binds that contract; the red phase fails with a shape assertion
on `pae_mask`.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16


class _MinimalProteinaNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond_factory = nn.Identity()
        self.expose_intermediates = True

    def forward(self, batch: dict) -> dict:
        raise RuntimeError("trim test must not invoke proteina forward")


class _MinimalProteina(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = _MinimalProteinaNN()
        self.fm = SimpleNamespace()
        self.autoencoder = None
        self.cfg_exp = SimpleNamespace(product_flowmatcher=())


def _make_module() -> ConfidenceDistillationModule:
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
    )
    head = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=50,
        bin_min=0.0,
        bin_max=100.0,
    )
    return ConfidenceDistillationModule.from_components(
        head=head,
        proteina=_MinimalProteina(),
        cond_modalities=("bb_ca",),
    )


def test_trim_batch_trims_2d_pair_and_1d_residue_fields() -> None:
    mod = _make_module()
    b, n_ext, n_orig = 2, 8, 5
    pae_mask = torch.ones(b, n_ext, n_ext, dtype=torch.bool)
    plddt_mask = torch.ones(b, n_ext, dtype=torch.bool)
    coords = torch.zeros(b, n_ext, 37, 3)

    batch = {
        "pae_mask": pae_mask,
        "plddt_mask": plddt_mask,
        "coords": coords,
        "scalar_meta": "string-not-tensor",
        "n_orig": n_orig,
    }
    trimmed = mod._trim_batch(batch, n_orig=n_orig)

    assert trimmed["pae_mask"].shape == (b, n_orig, n_orig)
    assert trimmed["plddt_mask"].shape == (b, n_orig)
    assert trimmed["coords"].shape == (b, n_orig, 37, 3)
    assert trimmed.get("scalar_meta", "string-not-tensor") == "string-not-tensor"
