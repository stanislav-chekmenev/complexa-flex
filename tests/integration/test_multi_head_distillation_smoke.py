"""End-to-end smoke for `MultiHeadConfidence` distillation (PR-B Slice 3).

Two checks on a fake datamodule that emits a Teddymer-shaped batch with
both `plddt_*` and `pae_*` labels populated:

1. `fast_dev_run=True` with `MultiHeadConfidence({"plddt": PLDDTHead, "pae":
   PaeHead})` completes one train + one val step under Lightning and emits
   the per-head log keys the multi-head reviewer panel expects
   (`train/multi/plddt/...`, `train/multi/pae/...`, `train/multi/total`).
2. DDP gloo 2-rank single-step under `torch.distributed` with the wrapper
   (mirrors the single-head DDP smoke at
   `tests/integration/confidence/test_distillation_ddp_smoke.py` but
   exercises the multi-head dispatcher's interaction with
   `DistributedDataParallel` -- the framework reviewer's carry-over).

Marked `slow` so the suite default skips it. The fake datamodule avoids
AFDB/Teddymer-mirror coupling; the real Teddymer view is exercised in
the SLURM run, not in CI.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import lightning as pl
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.confidence.losses import MultiHeadLoss
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


pytestmark = pytest.mark.slow


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
LATENT_DIM = 8
NUM_PLDDT_BINS = 50
NUM_PAE_BINS = 64
TRUNK_EVAL_T = 0.99
B, L = 2, 12


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

    def interpolate(self, x_0, x_1, t, mask=None):
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
    def __init__(self) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


def _make_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    chain_idx = torch.cat(
        [torch.zeros(L // 2, dtype=torch.int64), torch.ones(L - L // 2, dtype=torch.int64)],
        dim=0,
    ).unsqueeze(0).expand(B, -1).contiguous()
    return {
        "mask": torch.ones(B, L, dtype=torch.bool),
        "coords": torch.zeros(B, L, 37, 3),
        "coords_nm": torch.zeros(B, L, 37, 3),
        "plddt_residue": torch.rand(B, L, generator=g) * 100.0,
        "plddt_bin": torch.randint(0, NUM_PLDDT_BINS, (B, L), generator=g),
        "plddt_mask": torch.ones(B, L, dtype=torch.bool),
        "pae_residue_pair": torch.rand(B, L, L, generator=g) * 32.0,
        "pae_bin": torch.randint(0, NUM_PAE_BINS, (B, L, L), generator=g),
        "pae_mask": torch.ones(B, L, L, dtype=torch.bool),
        "chain_idx": chain_idx,
    }


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict, length: int = 4) -> None:
        self._batch = batch
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        return {k: v[0] if torch.is_tensor(v) else v for k, v in self._batch.items()}


def _collate(samples: list[dict]) -> dict:
    out: dict = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


class _FakeDM(pl.LightningDataModule):
    def __init__(self, batch: dict) -> None:
        super().__init__()
        self._ds = _SingleBatchDataset(batch, length=4)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._ds, batch_size=B, collate_fn=_collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._ds, batch_size=B, collate_fn=_collate)


def _make_wrapper() -> MultiHeadConfidence:
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
    plddt = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_PLDDT_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )
    pae = PaeHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
    )
    return MultiHeadConfidence(
        trunk=trunk,
        children={"plddt": plddt, "pae": pae},
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
    )


def _make_module() -> ConfidenceDistillationModule:
    return ConfidenceDistillationModule.from_components(
        head=_make_wrapper(),
        proteina=_FakeProteina(),
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=TRUNK_EVAL_T,
        lr=5e-3,
    )


def test_multi_head_fast_dev_run_one_step_emits_per_head_metrics() -> None:
    pl.seed_everything(0, workers=True)
    mod = _make_module()
    dm = _FakeDM(_make_batch(seed=0))
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(mod, datamodule=dm)

    metrics = trainer.callback_metrics
    plddt_ce_keys = [
        k for k in metrics if "plddt" in k and "loss_ce" in k and k.startswith("train")
    ]
    plddt_sl1_keys = [
        k for k in metrics if "plddt" in k and "loss_smooth_l1" in k and k.startswith("train")
    ]
    pae_ce_keys = [
        k for k in metrics if "pae" in k and "loss_ce" in k and k.startswith("train")
    ]
    pae_sl1_keys = [
        k for k in metrics if "pae" in k and "loss_smooth_l1" in k and k.startswith("train")
    ]
    total_keys = [k for k in metrics if k.startswith("train") and "total" in k]
    assert plddt_ce_keys, f"no train/.../plddt/.../loss_ce metric found in {sorted(metrics.keys())}"
    assert plddt_sl1_keys, f"no train pLDDT smooth_l1 metric found in {sorted(metrics.keys())}"
    assert pae_ce_keys, f"no train PAE loss_ce metric found in {sorted(metrics.keys())}"
    assert pae_sl1_keys, f"no train PAE loss_smooth_l1 metric found in {sorted(metrics.keys())}"
    assert total_keys, f"no train aggregate 'total' metric found in {sorted(metrics.keys())}"

    for k in plddt_ce_keys + plddt_sl1_keys + pae_ce_keys + pae_sl1_keys + total_keys:
        assert torch.isfinite(metrics[k]), f"{k} is not finite"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(
    rank: int,
    world_size: int,
    port: int,
    out_queue: mp.Queue,
    loss_weights: dict[str, float] | None = None,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.manual_seed(42 + rank)

        module = _make_module()
        wrapper = module.head
        ddp_head = nn.parallel.DistributedDataParallel(
            wrapper,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )

        batch = _make_batch(seed=rank)
        with torch.no_grad():
            from proteinfoundation.utils.sample_utils import add_clean_samples
            batch = add_clean_samples(
                batch,
                module.proteina.cfg_exp.product_flowmatcher,
                module.proteina.autoencoder,
            )
            batch = module.proteina.fm.corrupt_batch(batch)
            b = batch["mask"].shape[0]
            t_pinned = {
                m: torch.full(
                    (b,),
                    module.trunk_eval_t,
                    dtype=batch["t"][m].dtype,
                )
                for m in batch["t"]
            }
            batch["t"] = t_pinned
            batch["x_t"] = module.proteina.fm.interpolate(
                x_0=batch["x_0"],
                x_1=batch["x_1"],
                t=t_pinned,
                mask=batch["mask"],
            )
            nn_out = module.proteina.nn(batch)
            cond = module.proteina.nn.cond_factory(batch)

        inter = nn_out["trunk_intermediates"]
        head_out = ddp_head(inter["s"], inter["z"], inter["mask"], cond)
        masks_by_head = {
            "plddt": batch["plddt_mask"].to(torch.float32),
            "pae": batch["pae_mask"].to(torch.float32),
        }
        weights = loss_weights if loss_weights is not None else {"plddt": 0.7, "pae": 0.7}
        loss_fn = MultiHeadLoss(weights)
        total, _ = loss_fn(head_out, wrapper.children_heads, batch, masks_by_head)
        total.backward()

        loss_finite = bool(torch.isfinite(total).item())
        head_trainable = [p for p in wrapper.parameters() if p.requires_grad]
        any_head_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in head_trainable
        )
        proteina_no_grad = all(p.grad is None for p in module.proteina.parameters())

        out_queue.put(
            {
                "rank": rank,
                "loss_finite": loss_finite,
                "any_head_grad": any_head_grad,
                "proteina_no_grad": proteina_no_grad,
                "error": None,
            }
        )
    except Exception as exc:
        out_queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(), reason="torch.distributed not available"
)
def test_multi_head_ddp_gloo_two_rank_one_step() -> None:
    world_size = 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_ddp_worker, args=(rank, world_size, port, out_queue)
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(world_size):
        results.append(out_queue.get(timeout=120))
    for p in processes:
        p.join(timeout=10)

    for r in results:
        assert r.get("error") is None, f"rank {r.get('rank')} crashed: {r['error']}"
        assert r["loss_finite"], f"loss not finite on rank {r['rank']}"
        assert r["any_head_grad"], f"no head grad on rank {r['rank']}"
        assert r["proteina_no_grad"], f"proteina has grad on rank {r['rank']}"


@pytest.mark.skipif(
    not dist.is_available(), reason="torch.distributed not available"
)
def test_multi_head_ddp_zero_weight_with_find_unused_parameters_true() -> None:
    """Documented escape hatch: zero-weighted child + DDP static_graph=True.

    When loss_weights[pae] == 0.0, the PaeHead's parameters enter the autograd
    graph during forward but receive no gradient (MultiHeadLoss short-circuits
    before invoking compute_loss_and_metrics). The warning emitted at
    construction documents two escape hatches: removing the child from the
    config, or relaxing DDP (find_unused_parameters_true / static_graph=True).
    This subtest exercises static_graph=True because the trunk module is
    currently rebound across the wrapper and its children (child.trunk =
    self.trunk), which would trip the "parameter marked ready twice" assertion
    under find_unused_parameters=True; static_graph=True is the working
    workaround until that rebinding is replaced (post-merge cleanup).
    """
    world_size = 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_ddp_worker,
            args=(rank, world_size, port, out_queue),
            kwargs={
                "loss_weights": {"plddt": 1.0, "pae": 0.0},
                "find_unused_parameters": False,
                "static_graph": True,
            },
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(world_size):
        results.append(out_queue.get(timeout=120))
    for p in processes:
        p.join(timeout=10)

    for r in results:
        assert r.get("error") is None, f"rank {r.get('rank')} crashed: {r['error']}"
        assert r["loss_finite"], f"loss not finite on rank {r['rank']}"
        assert r["any_head_grad"], f"no head grad on rank {r['rank']}"
        assert r["proteina_no_grad"], f"proteina has grad on rank {r['rank']}"
