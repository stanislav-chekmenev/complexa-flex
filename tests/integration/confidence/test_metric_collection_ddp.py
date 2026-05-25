"""MetricCollection state is DDP-aggregated at compute() time.

Spawn 2 cpu workers, feed half the per-sample tensor to each, and check
that compute() on rank 0 equals the single-rank reference computed over
the concatenated tensor.

Skipped if gloo is unavailable.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef, SpearmanCorrCoef


def _worker(rank: int, world_size: int, pred: torch.Tensor, gt: torch.Tensor, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    mc = MetricCollection(
        {
            "mae": MeanAbsoluteError(),
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        },
        compute_groups=False,
    )
    n_per = pred.numel() // world_size
    lo, hi = rank * n_per, (rank + 1) * n_per
    mc.update(pred[lo:hi], gt[lo:hi])
    out = mc.compute()
    if rank == 0:
        q.put({k: float(v) for k, v in out.items()})
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="gloo backend not available"
)
@pytest.mark.parametrize("world_size", [1, 2])
def test_metric_collection_distributed_matches_single_rank(world_size: int):
    torch.manual_seed(0)
    N = 64
    pred = torch.randn(N)
    gt = pred + 0.5 * torch.randn(N)

    mc_ref = MetricCollection(
        {
            "mae": MeanAbsoluteError(),
            "pearson": PearsonCorrCoef(),
            "spearman": SpearmanCorrCoef(),
        },
        compute_groups=False,
    )
    mc_ref.update(pred, gt)
    ref = {k: float(v) for k, v in mc_ref.compute().items()}

    if world_size == 1:
        for k, v in ref.items():
            assert v == v, f"NaN in reference {k}"
        return

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, pred, gt, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    rank0 = q.get(timeout=10)
    for k in ref:
        assert abs(rank0[k] - ref[k]) < 1e-4, (
            f"{k}: ddp={rank0[k]:.6f}, single-rank={ref[k]:.6f}"
        )
