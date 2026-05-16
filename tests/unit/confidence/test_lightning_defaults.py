"""Defaults pinned on `ConfidenceDistillationModule`.

`smooth_l1_weight=0.1` and `label_smoothing=0.05` were chosen by the
generative-protein scientist in round-2 review; the `from_components`
path must not silently re-introduce dummy `trunk_ckpt_path` /
`autoencoder_ckpt_path` into the persisted hparams.
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
NUM_BINS = 50


class _StubCondFactory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feat_creators = []

    def forward(self, batch: dict) -> torch.Tensor:
        mask = batch["mask"]
        b, n = mask.shape
        return torch.zeros(b, n, DIM_COND, device=mask.device)


class _StubProteinaNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond_factory = _StubCondFactory()
        self.expose_intermediates = True


class _StubProteina(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = _StubProteinaNN()
        self.fm = SimpleNamespace()
        self.autoencoder = None
        self.cfg_exp = SimpleNamespace(product_flowmatcher=("bb_ca",))


def _make_head() -> PLDDTHead:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=1,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=False,
        dropout=0.0,
        update_pair_repr_every_n=1,
    )
    return PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )


def test_smooth_l1_weight_default_is_0_1() -> None:
    mod = ConfidenceDistillationModule.from_components(
        head=_make_head(),
        proteina=_StubProteina(),
    )
    assert mod.smooth_l1_weight == 0.1


def test_label_smoothing_default_is_0_05() -> None:
    mod = ConfidenceDistillationModule.from_components(
        head=_make_head(),
        proteina=_StubProteina(),
    )
    assert mod.label_smoothing == 0.05


def test_hparams_no_dummy_paths() -> None:
    mod = ConfidenceDistillationModule.from_components(
        head=_make_head(),
        proteina=_StubProteina(),
    )
    assert "trunk_ckpt_path" not in mod.hparams
    assert "autoencoder_ckpt_path" not in mod.hparams
