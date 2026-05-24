"""PR-5 validation-logging extension contract.

Adds `val/ece_adaptive` on top of the PR-4 validation keys. The PR-4 set
must remain intact; the new key must appear when `validation_step` runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
NUM_BINS = 50
LATENT_DIM = 8
TRUNK_EVAL_T = 0.99

PR4_VAL_KEYS = {
    "val/plddt/loss",
    "val/plddt/loss_ce",
    "val/plddt/loss_smooth_l1",
    "val/plddt/loss_total",
    "val/plddt/ece",
    "val/plddt/plddt_accuracy",
    "val/plddt/plddt_mae",
    "val/plddt/pearson_r",
    "val/plddt/spearman_r",
    "val/plddt/mae_lt50",
    "val/plddt/mae_50_70",
    "val/plddt/mae_70_90",
    "val/plddt/mae_ge90",
}


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

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        ones = torch.ones(b, n, 1, device=mask.device)
        s = self.embed(ones) * mask[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n, n, 1, device=mask.device)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        local_latents = batch["x_t"]["local_latents"] * mask[..., None].to(
            batch["x_t"]["local_latents"].dtype
        )
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask,
                "orig_mask": mask,
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
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


class _DummyValDataset(Dataset):
    def __init__(self, n_items: int = 2, n_res: int = 8) -> None:
        torch.manual_seed(0)
        self.items = []
        for _ in range(n_items):
            self.items.append(
                {
                    "mask": torch.ones(n_res, dtype=torch.bool),
                    "coords": torch.zeros(n_res, 37, 3),
                    "coords_nm": torch.zeros(n_res, 37, 3),
                    "plddt_residue": torch.rand(n_res) * 100.0,
                    "plddt_bin": torch.randint(0, NUM_BINS, (n_res,)),
                    "plddt_mask": torch.ones(n_res, dtype=torch.bool),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def _collate(items: list[dict]) -> dict:
    return {k: torch.stack([item[k] for item in items], dim=0) for k in items[0]}


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


def _make_module(**kwargs) -> ConfidenceDistillationModule:
    head = _make_head()
    fake_trunk = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
    return ConfidenceDistillationModule.from_components(
        head=head,
        proteina=fake_trunk,
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=TRUNK_EVAL_T,
        **kwargs,
    )


def test_pr4_validation_keys_preserved_and_ece_adaptive_logged(tmp_path: Path) -> None:
    mod = _make_module()
    loader = DataLoader(_DummyValDataset(), batch_size=2, collate_fn=_collate)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        default_root_dir=str(tmp_path),
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=0,
        limit_val_batches=1,
    )
    trainer.validate(mod, dataloaders=loader)
    logged = set(trainer.callback_metrics.keys())
    missing = PR4_VAL_KEYS - logged
    assert not missing, f"PR-4 validation keys missing: {missing}"
    assert "val/plddt/ece_adaptive" in logged
