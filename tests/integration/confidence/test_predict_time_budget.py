"""``PredictTimeBudget`` halts ``trainer.predict`` early (CPU-only, Lightning 2.5.x).

Lightning's prediction loop ignores ``trainer.should_stop``; the callback halts
by raising ``StopIteration`` from ``on_predict_batch_start`` once the wall-clock
budget is exceeded. This smoke confirms the mechanism actually stops predict
before the (large) dataloader is exhausted, and that already-completed batches
survive and are stamped with ``elapsed_gpu_hours``.
"""

from __future__ import annotations

import time

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY
from proteinfoundation.utils.predict_time_budget import PredictTimeBudget


class _SlowDataset(Dataset):
    def __init__(self, n: int = 1000) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([float(idx)])


class _SlowModule(L.LightningModule):
    def predict_step(self, batch, batch_idx):
        time.sleep(0.05)
        b = batch.shape[0]
        return {
            "coors": torch.zeros(b, 4, 37, 3),
            "residue_type": torch.zeros(b, 4, dtype=torch.long),
            "rewards": {TOTAL_REWARD_KEY: torch.zeros(b)},
        }


def test_time_budget_stops_predict_early() -> None:
    dataset = _SlowDataset(n=1000)
    loader = DataLoader(dataset, batch_size=1)
    module = _SlowModule()

    budget_hours = 0.2 / 3600.0  # 0.2 s
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[PredictTimeBudget(budget_hours=budget_hours)],
    )

    t0 = time.time()
    preds = trainer.predict(module, loader)
    elapsed = time.time() - t0

    assert preds is not None and 0 < len(preds) < 1000, "predict did not stop early"
    assert elapsed < 30.0, "budget cutoff did not bound wall-clock"
    # elapsed_gpu_hours stamped on each surviving batch
    assert "elapsed_gpu_hours" in preds[0]["rewards"]
    assert preds[0]["rewards"]["elapsed_gpu_hours"].shape[0] == 1


def test_time_budget_disabled_runs_all() -> None:
    dataset = _SlowDataset(n=5)
    loader = DataLoader(dataset, batch_size=1)
    module = _SlowModule()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[PredictTimeBudget(budget_hours=0)],
    )
    preds = trainer.predict(module, loader)
    assert len(preds) == 5
