"""Hydra compose tests for the confidence distillation entry points.

Pins:
- dead keys (`training_mode`, `loss_weight`, `freeze_trunk`, `confidence.enabled`)
  are dropped from the round-2 fix.
- `trainer.strategy` declares DDP behaviour for PR-6 multi-GPU readiness.
- `smooth_l1_weight=0.1` and `label_smoothing=0.05` match the scientist's
  round-2 recommendation.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose_distillation_cfg():
    import os

    os.environ.setdefault("DATA_PATH", "/tmp")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="confidence/distillation_swissprot")


def test_dead_keys_dropped() -> None:
    cfg = _compose_distillation_cfg()
    assert "training_mode" not in cfg.training
    assert "loss_weight" not in cfg.training
    assert "freeze_trunk" not in cfg.training
    confidence = cfg.get("confidence", {})
    assert "enabled" not in confidence


def test_ddp_strategy_declared() -> None:
    cfg = _compose_distillation_cfg()
    assert cfg.trainer.strategy == "ddp_find_unused_parameters_false"


def test_loss_defaults_round_two() -> None:
    cfg = _compose_distillation_cfg()
    assert cfg.training.loss.smooth_l1_weight == 0.1
    assert cfg.training.loss.label_smoothing == 0.05
