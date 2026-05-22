"""Disabled-path tests for `_build_wandb_logger`.

Three short-circuit paths must each return `None`:
- `logging.log_wandb` is False;
- environment has `WANDB_MODE=disabled`;
- no `logging:` block in the composed config at all.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


def test_returns_none_when_log_wandb_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    from proteinfoundation.confidence.train_confidence import _build_wandb_logger

    cfg = OmegaConf.create(
        {
            "run_name": "ignored",
            "logging": {
                "log_wandb": False,
                "wandb_project": "confidence-distillation",
                "wandb_entity": None,
                "wandb_group": None,
                "wandb_tags": [],
            },
        }
    )

    assert _build_wandb_logger(cfg, wandb_id="dummy", wandb_name="dummy") is None


def test_returns_none_when_wandb_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    from proteinfoundation.confidence.train_confidence import _build_wandb_logger

    cfg = OmegaConf.create(
        {
            "run_name": "pae-distill-teddymer",
            "logging": {
                "log_wandb": True,
                "wandb_project": "confidence-distillation",
                "wandb_entity": None,
                "wandb_group": None,
                "wandb_tags": ["pae", "teddymer"],
            },
        }
    )

    assert _build_wandb_logger(cfg, wandb_id="dummy", wandb_name="dummy") is None


def test_returns_none_when_logging_block_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    from proteinfoundation.confidence.train_confidence import _build_wandb_logger

    cfg = OmegaConf.create({"run_name": "no-logging-block"})

    assert _build_wandb_logger(cfg, wandb_id="dummy", wandb_name="dummy") is None
