"""Integration test for `SequenceOnlyPLDDTHead` inside the distillation sidecar.

Exercises the same fake `Proteina` stub as the canonical one-step test,
but constructs a `SequenceOnlyPLDDTHead` and confirms (a) one training
step is finite and grad-attached, (b) only head parameters collect
gradients, and (c) `_predict` does not crash when `n_extended != n_orig`.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_sequence_only_head import (
    SequenceOnlyPLDDTHead,
)


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
NUM_BINS = 50
LATENT_DIM = 8
TRUNK_EVAL_T = 0.99


class _FakeCondFactory(nn.Module):
    def __init__(self, modalities: tuple[str, ...], dim_cond: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.linear = nn.Linear(len(modalities), dim_cond, bias=False)

    def forward(self, batch: dict) -> torch.Tensor:
        t_stack = torch.stack([batch["t"][m] for m in self.modalities], dim=-1)
        mask = batch["mask"]
        b, n = mask.shape
        cond = self.linear(t_stack)
        cond = cond[:, None, :].expand(b, n, -1).contiguous()
        return cond * mask[..., None].to(cond.dtype)


class _FakeProteinaNN(nn.Module):
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int, n_extra: int = 0) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.n_extra = n_extra
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        n_ext = n + self.n_extra
        orig_mask = mask
        mask_ext = torch.zeros(b, n_ext, dtype=torch.bool, device=mask.device)
        mask_ext[:, :n] = mask
        ones = torch.ones(b, n_ext, 1, device=mask.device)
        s = self.embed(ones) * mask_ext[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n_ext, n_ext, 1, device=mask.device)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask_ext[:, None, :] & mask_ext[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        local_latents = torch.zeros(
            b, n_ext, LATENT_DIM, device=mask.device, dtype=torch.float32
        )
        local_latents[:, :n] = batch["x_t"]["local_latents"] * mask[..., None].to(
            local_latents.dtype
        )
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask_ext,
                "orig_mask": orig_mask,
                "n_orig": int(n),
                "local_latents": local_latents,
            }
        }


class _FakeFM:
    def __init__(self) -> None:
        self.data_modes = ("bb_ca", "local_latents")

    def corrupt_batch(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        device = mask.device
        x_1 = batch["x_1"]
        x_0 = {
            "bb_ca": torch.randn(b, n, 3, device=device),
            "local_latents": torch.randn(b, n, LATENT_DIM, device=device),
        }
        t = {
            "bb_ca": torch.rand(b, device=device),
            "local_latents": torch.rand(b, device=device),
        }
        x_t = {
            m: t[m][:, None, None] * x_1[m] + (1.0 - t[m][:, None, None]) * x_0[m]
            for m in self.data_modes
        }
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        return batch

    def interpolate(
        self,
        x_0: dict[str, torch.Tensor],
        x_1: dict[str, torch.Tensor],
        t: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            m: t[m][:, None, None] * x_1[m] + (1.0 - t[m][:, None, None]) * x_0[m]
            for m in x_1
        }


class _FakeAutoEncoder:
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        self.latent_dim = latent_dim

    def encode(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        return {"z_latent": torch.zeros(b, n, self.latent_dim, device=mask.device)}


class _FakeProteina(nn.Module):
    def __init__(
        self,
        dim_cond: int,
        token_dim: int,
        pair_repr_dim: int,
        n_extra: int = 0,
    ) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim, n_extra=n_extra)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


def _make_head() -> SequenceOnlyPLDDTHead:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1_000_000,
        latent_dim=LATENT_DIM,
    )
    return SequenceOnlyPLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )


def _make_module(n_extra: int = 0) -> ConfidenceDistillationModule:
    head = _make_head()
    fake_trunk = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM, n_extra=n_extra)
    return ConfidenceDistillationModule.from_components(
        head=head,
        proteina=fake_trunk,
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=TRUNK_EVAL_T,
    )


def _make_batch(b: int = 2, n: int = 9) -> dict:
    torch.manual_seed(0)
    mask = torch.ones(b, n, dtype=torch.bool)
    plddt_mask = torch.ones(b, n, dtype=torch.bool)
    plddt_mask[0, n // 2 :] = False
    coords_nm = torch.zeros(b, n, 37, 3)
    return {
        "mask": mask,
        "coords": coords_nm,
        "coords_nm": coords_nm,
        "plddt_residue": torch.rand(b, n) * 100.0,
        "plddt_bin": torch.randint(0, NUM_BINS, (b, n)),
        "plddt_mask": plddt_mask,
    }


def test_one_step_loss_finite_with_sequence_only_head() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_only_head_params_get_grad_with_sequence_only_head() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    loss.backward()
    for p in mod.proteina.parameters():
        assert p.grad is None
    head_trainable = [p for p in mod.head.parameters() if p.requires_grad]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head_trainable)


def test_predict_zeros_padded_logits_under_partial_mask() -> None:
    """Under a partial `mask_ext` (some positions False), the head must

    return zero logits at those positions. This pins the masking contract
    of `_predict`; the previous name implied an `n_orig != n_ext`
    relationship the function does not actually consume (it only sees
    `mask_ext` and is agnostic to which positions were concat-padded).
    """
    head = _make_head()
    b, n_ext = 2, 12
    s = torch.randn(b, n_ext, TOKEN_DIM)
    z = torch.randn(b, n_ext, n_ext, PAIR_REPR_DIM)
    mask_ext = torch.ones(b, n_ext, dtype=torch.bool)
    mask_ext[:, 9:] = False
    out = head._predict(s, z, mask_ext)
    assert out["plddt_logits"].shape == (b, n_ext, NUM_BINS)
    assert torch.all(out["plddt_logits"][:, 9:] == 0.0)
