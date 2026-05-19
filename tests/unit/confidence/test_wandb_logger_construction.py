"""Construction-path test for `_build_wandb_logger`.

Patches `wandb.init` so no network or `~/.netrc` lookup happens; asserts
the returned `WandbLogger` carries the project, id, name, and tags taken
from the Hydra `logging.*` block.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


def test_build_wandb_logger_returns_wandb_logger_with_expected_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

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

    with patch("wandb.init"):
        wandb_logger = _build_wandb_logger(cfg)

    assert isinstance(wandb_logger, WandbLogger)
    assert wandb_logger._project == "confidence-distillation"
    assert wandb_logger._id == "pae-distill-teddymer"
    assert wandb_logger._name == "pae-distill-teddymer"
    assert wandb_logger._wandb_init["tags"] == ["pae", "teddymer"]
