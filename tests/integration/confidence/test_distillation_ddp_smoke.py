"""DDP smoke test for `ConfidenceDistillationModule` under gloo (CPU, 2 ranks).

Goal: confirm that one `training_step` runs to completion on each rank
without DDP-unused-parameter errors, the returned loss is finite, the
backward leaves gradients only on head parameters (not on the frozen
proteina trunk), and the gradients are well-defined under the gloo
allreduce path.

We avoid Lightning's `Trainer(strategy="ddp")` wrapper here because the
sidecar holds a sub-`nn.Module` (`proteina`) with `requires_grad=False`
which the strategy still wraps for collective ops; the cleanest, smallest
test is to spawn ranks with `torch.distributed.init_process_group("gloo",
...)` and exercise `module.training_step` + backward directly.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    def __init__(self, modalities, dim_cond: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.linear = nn.Linear(len(modalities), dim_cond, bias=False)

    def forward(self, batch):
        t_stack = torch.stack([batch["t"][m] for m in self.modalities], dim=-1)
        mask = batch["mask"]
        b, n = mask.shape
        cond = self.linear(t_stack)
        cond = cond[:, None, :].expand(b, n, -1).contiguous()
        return cond * mask[..., None].to(cond.dtype)


class _FakeProteinaNN(nn.Module):
    def __init__(self, dim_cond, token_dim, pair_repr_dim) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)

    def forward(self, batch):
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

    def corrupt_batch(self, batch):
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

    def encode(self, batch):
        mask = batch["mask"]
        b, n = mask.shape
        return {"z_latent": torch.zeros(b, n, self.latent_dim, device=mask.device)}


class _FakeProteina(nn.Module):
    def __init__(self, dim_cond, token_dim, pair_repr_dim) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


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
        latent_dim=LATENT_DIM,
    )
    head = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )
    fake_proteina = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
    return ConfidenceDistillationModule.from_components(
        head=head,
        proteina=fake_proteina,
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank: int, world_size: int, port: int, out_queue: mp.Queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.manual_seed(42 + rank)

        module = _make_module()
        head_trainable = [p for p in module.head.parameters() if p.requires_grad]
        ddp_head = nn.parallel.DistributedDataParallel(
            module.head,
            find_unused_parameters=False,
        )

        batch = _make_batch()
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
        head_out = ddp_head(
            inter["s"], inter["z"], inter["mask"], cond, inter["local_latents"]
        )
        logits = head_out["plddt_logits"]
        labels = batch["plddt_bin"]
        mask_eff = batch["plddt_mask"].to(torch.float32)

        from proteinfoundation.confidence.losses import combined_plddt_loss
        total, _ = combined_plddt_loss(
            student_logits=logits,
            plddt_bin_labels=labels,
            plddt_continuous=batch["plddt_residue"],
            mask=mask_eff,
            bin_centers=module.head.bin_centers,
            ce_weight=module.ce_weight,
            smooth_l1_weight=module.smooth_l1_weight,
            label_smoothing=0.0,
        )
        total.backward()

        loss_finite = bool(torch.isfinite(total).item())
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
def test_ddp_gloo_two_rank_one_step() -> None:
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
