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


def test_ddp_strategy_block_instantiates_for_all_confidence_runs() -> None:
    import os

    import hydra
    from lightning.pytorch.strategies import DDPStrategy

    os.environ.setdefault("DATA_PATH", "/tmp")
    for config_name in _EXPECTED_TAGS:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name=config_name)
        assert (
            cfg.trainer.strategy._target_
            == "lightning.pytorch.strategies.DDPStrategy"
        ), config_name
        assert cfg.trainer.strategy.find_unused_parameters is True, config_name
        assert cfg.trainer.strategy.static_graph is True, config_name
        strategy = hydra.utils.instantiate(cfg.trainer.strategy)
        assert isinstance(strategy, DDPStrategy), config_name


def test_loss_defaults_round_two() -> None:
    cfg = _compose_distillation_cfg()
    assert cfg.training.loss.ce_weight == 0.9
    assert cfg.training.loss.smooth_l1_weight == 0.1
    assert cfg.training.loss.label_smoothing == 0.05


_EXPECTED_TAGS = {
    "confidence/distillation_swissprot": ["plddt", "swissprot"],
    "confidence/distillation_swissprot_control": ["plddt", "swissprot", "control"],
    "confidence/distillation_teddymer_pae": ["pae", "teddymer"],
    "confidence/distillation_teddymer_multihead": ["plddt", "pae", "teddymer", "multi"],
}


def test_logging_block_composed_for_all_confidence_runs() -> None:
    import os

    os.environ.setdefault("DATA_PATH", "/tmp")
    for config_name, expected_tags in _EXPECTED_TAGS.items():
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name=config_name)

        assert cfg.logging.log_wandb is True, config_name
        assert cfg.logging.wandb_project == "confidence-distillation", config_name
        assert list(cfg.logging.wandb_tags) == expected_tags, config_name
        assert cfg.run_name, f"{config_name}: run_name must be non-empty for wandb_id"
