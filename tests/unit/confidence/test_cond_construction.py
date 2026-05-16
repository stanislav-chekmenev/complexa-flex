"""Cond-construction tests for the distillation sidecar.

`_compute_cond` forwards the sidecar-prepared batch (which already carries
`batch['t'][<modality>]` pinned to `trunk_eval_t`) through the trunk's
`cond_factory` under `torch.no_grad()` to produce `[b, n, dim_cond]`. It
must be shape-correct, finite, and deterministic.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16


class _FakeCondFactory(nn.Module):
    """Stand-in for `FeatureFactory(mode='seq')`.

    Reads `batch['t'][m]` per modality, returns a deterministic linear
    projection of the concatenated time scalars expanded along `n`.
    """

    def __init__(self, modalities: tuple[str, ...], dim_cond: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.linear = nn.Linear(len(modalities), dim_cond, bias=False)

    def forward(self, batch: dict) -> torch.Tensor:
        t_stack = torch.stack(
            [batch["t"][m] for m in self.modalities], dim=-1
        )
        mask = batch["mask"]
        b, n = mask.shape
        cond = self.linear(t_stack)
        cond = cond[:, None, :].expand(b, n, -1).contiguous()
        return cond * mask[..., None].to(cond.dtype)


class _FakeProteinaNN(nn.Module):
    def __init__(self, dim_cond: int) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True


class _FakeProteina(nn.Module):
    def __init__(self, dim_cond: int) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond)


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
    )
    return PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=50,
        bin_min=0.0,
        bin_max=100.0,
    )


def _make_module() -> ConfidenceDistillationModule:
    head = _make_head()
    fake_trunk = _FakeProteina(DIM_COND)
    mod = ConfidenceDistillationModule.from_components(
        head=head,
        proteina=fake_trunk,
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=0.99,
    )
    return mod.eval()


def _pinned_batch(b: int, n: int, trunk_eval_t: float = 0.99) -> dict:
    return {
        "mask": torch.ones(b, n, dtype=torch.bool),
        "t": {
            "bb_ca": torch.full((b,), trunk_eval_t),
            "local_latents": torch.full((b,), trunk_eval_t),
        },
    }


def test_compute_cond_shape_and_finite() -> None:
    mod = _make_module()
    b, n = 2, 11
    cond = mod._compute_cond(_pinned_batch(b, n))
    assert cond.shape == (b, n, DIM_COND)
    assert torch.isfinite(cond).all()


def test_compute_cond_deterministic() -> None:
    mod = _make_module()
    b, n = 3, 7
    batch = _pinned_batch(b, n)
    c1 = mod._compute_cond(batch)
    c2 = mod._compute_cond(batch)
    assert torch.equal(c1, c2)
