"""WandB entity resolves from $WANDB_ENTITY at compose time.

`configs/logging/wandb.yaml` declares `wandb_entity:
${oc.env:WANDB_ENTITY,null}`. The env var is the source of truth in
production (set via the sourced `.env`); the null fallback keeps
existing unit tests that patch the env to absence green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def test_wandb_entity_resolves_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("DATA_PATH", "/tmp")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="confidence/distillation_teddymer_pae")
    assert cfg.logging.wandb_entity == "test-entity"


def test_wandb_entity_null_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("DATA_PATH", "/tmp")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="confidence/distillation_teddymer_pae")
    assert cfg.logging.wandb_entity is None
