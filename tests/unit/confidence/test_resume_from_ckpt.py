"""End-to-end plumbing test for `cfg.resume_ckpt_path` in `train_confidence`.

The confidence-distill entry point must forward `cfg.resume_ckpt_path` to
`L.Trainer.fit(..., ckpt_path=...)` so a crashed or preempted run can resume
optimizer / LR-sched / step-counter state from a previously written
`ModelCheckpoint`. The contract:

* `cfg.resume_ckpt_path` set to a path string -> `trainer.fit(...)` receives
  `ckpt_path` equal to that string.
* `cfg.resume_ckpt_path` absent or `None` -> `trainer.fit(...)` receives
  `ckpt_path=None` (Lightning's default; starts from scratch).

These two cases are orthogonal to the WandB `resume_id` knob, which only
controls run-history continuity, not trainer state.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


def _minimal_cfg(**overrides: Any) -> Any:
    base = OmegaConf.create(
        {
            "run_name": "resume-test",
            "seed": 42,
            "resume_id": None,
            "resume_ckpt_path": None,
            "logging": {"log_wandb": False},
            "confidence": {"head": {}},
            "data": {"datamodule": {}},
            "training": {
                "trunk_ckpt_path": "/tmp/trunk.ckpt",
                "autoencoder_ckpt_path": "/tmp/ae.ckpt",
                "trunk_eval_t": 0.99,
                "opt": {
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999],
                    "warmup_steps": 500,
                    "min_lr": 5e-6,
                },
                "loss": {
                    "ce_weight": 0.9,
                    "smooth_l1_weight": 0.1,
                    "label_smoothing": 0.05,
                },
            },
            "trainer": {"_target_": "lightning.pytorch.Trainer"},
        }
    )
    for k, v in overrides.items():
        OmegaConf.update(base, k, v, merge=False)
    return base


def _run_main_with_mocks(cfg: Any) -> MagicMock:
    """Invoke `train_confidence.main.__wrapped__(cfg)` with everything heavy
    mocked, return the mock used in place of the constructed `Trainer`."""
    from proteinfoundation.confidence import train_confidence as tc

    fake_trainer = MagicMock(name="Trainer")
    fake_trainer.is_global_zero = True

    with (
        patch.object(tc, "build_confidence_head_from_cfg", return_value=MagicMock(name="head")),
        patch.object(tc, "ConfidenceDistillationModule", return_value=MagicMock(name="module")),
        patch("hydra.utils.instantiate", return_value=MagicMock(name="datamodule")),
        patch.object(tc, "_build_trainer", return_value=fake_trainer),
        patch.object(tc, "_build_wandb_logger", return_value=None),
    ):
        # `@hydra.main` wraps `main`; call the underlying function directly so
        # we can pass a synthetic cfg without spawning a Hydra job dir.
        tc.main.__wrapped__(cfg)

    return fake_trainer


def test_resume_ckpt_path_forwarded_to_trainer_fit() -> None:
    cfg = _minimal_cfg(resume_ckpt_path="/netscratch/ckpts/pae-ce-001-1.3797.ckpt")
    fake_trainer = _run_main_with_mocks(cfg)

    fake_trainer.fit.assert_called_once()
    _, kwargs = fake_trainer.fit.call_args
    assert kwargs.get("ckpt_path") == "/netscratch/ckpts/pae-ce-001-1.3797.ckpt", (
        "train_confidence.main must forward cfg.resume_ckpt_path to "
        "trainer.fit(..., ckpt_path=...)"
    )


def test_resume_ckpt_path_none_is_forwarded_as_none() -> None:
    cfg = _minimal_cfg(resume_ckpt_path=None)
    fake_trainer = _run_main_with_mocks(cfg)

    fake_trainer.fit.assert_called_once()
    _, kwargs = fake_trainer.fit.call_args
    assert kwargs.get("ckpt_path") is None, (
        "When cfg.resume_ckpt_path is None, trainer.fit must be called with "
        "ckpt_path=None (Lightning's fresh-start default)."
    )


def test_resume_ckpt_path_missing_is_treated_as_none() -> None:
    cfg = _minimal_cfg()
    # Simulate a legacy config that does not even define `resume_ckpt_path`.
    OmegaConf.set_struct(cfg, False)
    del cfg["resume_ckpt_path"]

    fake_trainer = _run_main_with_mocks(cfg)

    fake_trainer.fit.assert_called_once()
    _, kwargs = fake_trainer.fit.call_args
    assert kwargs.get("ckpt_path") is None, (
        "A composed config that pre-dates the resume_ckpt_path field must "
        "behave like resume_ckpt_path=None (fresh start, no AttributeError)."
    )
