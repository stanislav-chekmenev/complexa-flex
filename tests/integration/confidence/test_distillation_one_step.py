"""One-step distillation tests: loss finite, only head gets grads, trunk frozen.

The trunk fake stub asserts the sidecar prepares the batch via
`add_clean_samples` + `fm.corrupt_batch` and then pins `t = trunk_eval_t`
before invoking the trunk; equivalently, that `batch["t"]`, `batch["x_t"]`,
and `batch["x_1"]` are present at trunk entry, with `t` pinned. Real
`Proteina.load_from_checkpoint` is GPU-only and exercised in PR-5 smoke.
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
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)
        self.last_batch: dict | None = None

    def forward(self, batch: dict) -> dict:
        assert "t" in batch, "trunk requires batch['t'] populated by sidecar"
        assert "x_t" in batch, "trunk requires batch['x_t'] populated by sidecar"
        for m in ("bb_ca", "local_latents"):
            assert m in batch["t"], f"batch['t'] missing modality {m}"
            assert torch.allclose(
                batch["t"][m],
                torch.full_like(batch["t"][m], TRUNK_EVAL_T),
            ), f"batch['t'][{m}] not pinned to trunk_eval_t={TRUNK_EVAL_T}"
            assert m in batch["x_t"], f"batch['x_t'] missing modality {m}"
        mask = batch["mask"]
        b, n = mask.shape
        for m in ("bb_ca", "local_latents"):
            assert batch["x_t"][m].shape[:2] == (b, n), (
                f"batch['x_t'][{m}].shape[:2] must be (b, n)=({b}, {n})"
            )
        self.last_batch = {
            "t": {m: batch["t"][m].detach().clone() for m in batch["t"]},
            "x_t": {m: batch["x_t"][m].detach().clone() for m in batch["x_t"]},
            "x_0": {m: batch["x_0"][m].detach().clone() for m in batch["x_0"]},
            "x_1": {m: batch["x_1"][m].detach().clone() for m in batch["x_1"]},
        }
        ones = torch.ones(b, n, 1, device=mask.device)
        s = self.embed(ones) * mask[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n, n, 1, device=mask.device)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask,
                "orig_mask": mask,
                "n_orig": int(n),
            }
        }


class _FakeFM:
    """Minimal `ProductSpaceFlowMatcher` stand-in.

    `corrupt_batch` populates `x_0`, `x_1`, `x_t`, `t` per modality with
    arbitrary values (random `t` to prove the sidecar overwrites it).
    `interpolate` is the canonical conditional-OT mix.
    """

    def __init__(self) -> None:
        self.data_modes = ("bb_ca", "local_latents")
        self.corrupt_calls = 0
        self.interpolate_calls = 0

    def corrupt_batch(self, batch: dict) -> dict:
        self.corrupt_calls += 1
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
        self.interpolate_calls += 1
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


def test_one_step_loss_finite() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_only_head_params_get_grad() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    loss.backward()
    for p in mod.proteina.parameters():
        assert p.grad is None, "frozen proteina trunk must not collect gradients"
    head_trainable = [p for p in mod.head.parameters() if p.requires_grad]
    assert len(head_trainable) > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head_trainable)


def test_trunk_frozen_grad_state() -> None:
    mod = _make_module()
    for p in mod.proteina.parameters():
        assert p.requires_grad is False
    assert mod.proteina.training is False
