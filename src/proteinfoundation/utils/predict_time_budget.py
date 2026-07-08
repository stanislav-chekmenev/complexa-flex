"""Wall-clock time budget for ``trainer.predict``.

Lightning's prediction loop does not honour ``trainer.should_stop`` (only the
fit/validation loops do, verified against Lightning 2.5.x
``loops/prediction_loop.py``). The loop only breaks on ``StopIteration`` raised
inside its ``try`` block (which wraps both ``next(data_fetcher)`` and the
``_predict_step`` call, incl. the ``on_predict_batch_*`` hooks). So the reliable
way to halt predict early is to raise ``StopIteration`` from a callback hook.

``on_predict_batch_end`` fires *before* the batch's prediction is appended to
the returned list, so raising there would discard the just-computed batch.
This callback instead checks the budget in ``on_predict_batch_start`` and raises
once it is exceeded: every completed batch is preserved, and no further batch is
started. It also stamps each finished prediction with ``elapsed_gpu_hours`` in
``on_predict_batch_end`` (mutating the prediction dict in place, before Lightning
moves it to CPU), so downstream saving can record per-sample discovery time.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from lightning.pytorch import Callback
from loguru import logger

from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY


class PredictTimeBudget(Callback):
    """Stop ``trainer.predict`` once ``budget_hours`` of wall-clock elapses.

    Args:
        budget_hours: Wall-clock budget in hours. ``None`` or ``0`` disables the
            cutoff (predict runs until the dataloader is exhausted).
    """

    def __init__(self, budget_hours: float | None = 16.0) -> None:
        super().__init__()
        self.budget_hours = None if not budget_hours else float(budget_hours)
        self.start_time = time.time()

    def _elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600.0

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        self.start_time = time.time()
        if self.budget_hours is not None:
            logger.info(f"[PredictTimeBudget] budget = {self.budget_hours:.3f} h")

    def on_predict_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.budget_hours is None:
            return
        elapsed = self._elapsed_hours()
        if elapsed >= self.budget_hours:
            logger.info(
                f"[PredictTimeBudget] budget reached ({elapsed:.3f} h >= "
                f"{self.budget_hours:.3f} h) before batch {batch_idx}; stopping predict."
            )
            raise StopIteration

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict) or "coors" not in outputs:
            return
        b = outputs["coors"].shape[0]
        device = outputs["coors"].device
        elapsed_h = torch.full((b,), self._elapsed_hours(), device=device, dtype=torch.float32)
        rewards = outputs.get("rewards")
        if isinstance(rewards, dict):
            rewards["elapsed_gpu_hours"] = elapsed_h
        else:
            outputs["rewards"] = {
                TOTAL_REWARD_KEY: torch.full((b,), float("nan"), device=device, dtype=torch.float32),
                "elapsed_gpu_hours": elapsed_h,
            }
