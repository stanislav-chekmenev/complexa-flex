"""Lightning 2.5.x checkpoint round-trip for `ConfidenceDistillationModule`.

Trains one step on a fake `Proteina` stub, saves the checkpoint, reloads via
`ConfidenceDistillationModule.load_from_checkpoint`, and asserts the head's
state dict matches bit-exactly. The frozen trunk is also serialised — the
spec accepts the disk cost (Risk 9). The reload path uses the same fake
trunk wiring as the original module.
"""

from __future__ import annotations

from pathlib import Path

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
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask,
                "orig_mask": mask,
                "n_orig": int(n),
            }
        }


class _FakeProteina(nn.Module):
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim)


class _DummyDataset(Dataset):
    def __init__(self, n_items: int = 2, n_res: int = 8) -> None:
        torch.manual_seed(0)
        self.items = []
        for _ in range(n_items):
            self.items.append(
                {
                    "mask": torch.ones(n_res, dtype=torch.bool),
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
        trunk_eval_t=0.99,
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    mod = _make_module()
    loader = DataLoader(_DummyDataset(), batch_size=2, collate_fn=_collate)
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(mod, train_dataloaders=loader)

    ckpt_path = tmp_path / "module.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    reload_head = _make_head()
    reload_fake = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
    reloaded = ConfidenceDistillationModule.load_from_checkpoint(
        str(ckpt_path),
        head=reload_head,
        proteina=reload_fake,
        cond_modalities=("bb_ca", "local_latents"),
    )

    src_state = mod.head.state_dict()
    dst_state = reloaded.head.state_dict()
    assert set(src_state.keys()) == set(dst_state.keys())
    for k, v in src_state.items():
        assert torch.equal(v, dst_state[k]), f"head parameter {k!r} drifted across ckpt round-trip"
