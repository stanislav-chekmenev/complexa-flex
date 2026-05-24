"""Concat-features contract: sidecar must pad `cond` to `n_ext` before head.

The trunk's `cond_factory` emits `[b, n_orig, dim_cond]` because complexa's
feature factory currently lives in the `n_orig` (binder-only) frame. The
trunk itself extends to `n_ext = n_orig + n_concat` (motif/target/ligand);
when the sidecar feeds `(s, z, mask_ext, cond)` to the head, `cond` and
`s/z/mask_ext` must agree on the second axis.

This test wires a fake `Proteina` whose intermediates report
`n_ext > n_orig`, captures the cond the head actually receives, and
asserts (a) `cond.shape[1] == n_ext`, (b) the appended slice is exactly
zero.
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
LATENT_DIM = 8
TRUNK_EVAL_T = 0.99
N_EXTRA = 5


class _FakeCondFactory(nn.Module):
    """Emits cond of shape `[b, n_orig, dim_cond]` (binder-only frame)."""

    def __init__(self, modalities: tuple[str, ...], dim_cond: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.linear = nn.Linear(len(modalities), dim_cond, bias=False)

    def forward(self, batch: dict) -> torch.Tensor:
        t_stack = torch.stack([batch["t"][m] for m in self.modalities], dim=-1)
        mask = batch["mask"]
        b, n_orig = mask.shape
        cond = self.linear(t_stack)
        cond = cond[:, None, :].expand(b, n_orig, -1).contiguous()
        return cond * mask[..., None].to(cond.dtype)


class _FakeProteinaNN(nn.Module):
    """Returns intermediates with `n_ext = n_orig + N_EXTRA`."""

    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n_orig = mask.shape
        n_ext = n_orig + N_EXTRA
        mask_ext = torch.zeros(b, n_ext, dtype=torch.bool, device=mask.device)
        mask_ext[:, :n_orig] = mask
        ones = torch.ones(b, n_ext, 1, device=mask.device)
        s = self.embed(ones) * mask_ext[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n_ext, n_ext, 1, device=mask.device)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask_ext[:, None, :] & mask_ext[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        local_latents = torch.zeros(
            b, n_ext, LATENT_DIM, device=mask.device, dtype=torch.float32
        )
        local_latents[:, :n_orig] = batch["x_t"]["local_latents"] * mask[..., None].to(
            local_latents.dtype
        )
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask_ext,
                "orig_mask": mask,
                "n_orig": int(n_orig),
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
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


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
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )


def _make_module() -> ConfidenceDistillationModule:
    head = _make_head()
    fake_trunk = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
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
    coords_nm = torch.zeros(b, n, 37, 3)
    return {
        "mask": mask,
        "coords": coords_nm,
        "coords_nm": coords_nm,
        "plddt_residue": torch.rand(b, n) * 100.0,
        "plddt_bin": torch.randint(0, NUM_BINS, (b, n)),
        "plddt_mask": plddt_mask,
    }


def test_cond_padded_to_n_ext_with_zero_tail() -> None:
    """Sidecar must zero-pad cond from `n_orig` to `n_ext` before the head."""
    mod = _make_module()
    batch = _make_batch()

    captured: dict[str, torch.Tensor] = {}
    real_forward = mod.head.forward

    def spy(s, z, mask, cond, local_latents, chain_id=None):
        captured["cond"] = cond.detach().clone()
        captured["mask"] = mask.detach().clone()
        captured["local_latents"] = local_latents.detach().clone()
        return real_forward(s, z, mask, cond, local_latents, chain_id=chain_id)

    mod.head.forward = spy

    mod._forward(batch)

    cond = captured["cond"]
    mask_ext = captured["mask"]
    b, n_orig = batch["mask"].shape
    n_ext = n_orig + N_EXTRA
    assert cond.shape == (b, n_ext, DIM_COND), (
        f"cond shape {tuple(cond.shape)} != (b, n_ext, dim_cond)=({b}, {n_ext}, {DIM_COND})"
    )
    assert mask_ext.shape == (b, n_ext)
    assert torch.all(cond[:, n_orig:, :] == 0.0), "cond tail must be zero-padded"


def test_one_step_works_with_concat_features() -> None:
    """End-to-end: training_step finishes with finite loss when n_ext > n_orig."""
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.requires_grad
