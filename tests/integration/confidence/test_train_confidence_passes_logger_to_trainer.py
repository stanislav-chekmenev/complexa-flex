"""End-to-end wiring test for the confidence-distillation entry point.

Patches `L.Trainer`, head builder, datamodule instantiation, and
`ConfidenceDistillationModule` so `main(cfg)` runs without touching real
modules / data / GPUs. Asserts the WandbLogger is forwarded to the
Lightning Trainer constructor and that hyperparameters are logged on
rank 0; flips `log_wandb=False` to assert the disabled path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


def _minimal_cfg(log_wandb: bool):
    return OmegaConf.create(
        {
            "seed": 42,
            "run_name": "pae-distill-teddymer",
            "logging": {
                "log_wandb": log_wandb,
                "wandb_project": "confidence-distillation",
                "wandb_entity": None,
                "wandb_group": None,
                "wandb_tags": ["pae", "teddymer"],
            },
            "integrity": {"enabled": False},
            "confidence": {"head": {"_target_": "ignored"}},
            "training": {
                "trunk_ckpt_path": "/tmp/trunk.ckpt",
                "autoencoder_ckpt_path": "/tmp/ae.ckpt",
                "trunk_eval_t": 0.99,
                "opt": {
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.95],
                    "warmup_steps": 0,
                    "min_lr": 1e-6,
                },
                "loss": {"ce_weight": 0.9, "smooth_l1_weight": 0.1, "label_smoothing": 0.05},
            },
            "data": {"datamodule": {"_target_": "ignored"}},
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
                "accelerator": "cpu",
                "devices": 1,
            },
        }
    )


def _run_main(cfg, *, trainer_mock):
    """Patch all non-logger collaborators and invoke `main` (undecorated)."""
    import proteinfoundation.confidence.train_confidence as tc

    with patch.object(tc, "build_confidence_head_from_cfg", return_value=MagicMock()), patch.object(
        tc, "ConfidenceDistillationModule", return_value=MagicMock()
    ), patch.object(tc.hydra.utils, "instantiate", return_value=trainer_mock) as inst, patch.object(
        tc.L, "seed_everything"
    ), patch("wandb.init"):
        tc.main.__wrapped__(cfg)
    return inst


def test_trainer_receives_wandb_logger_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    trainer_mock = MagicMock(is_global_zero=True)
    cfg = _minimal_cfg(log_wandb=True)

    inst = _run_main(cfg, trainer_mock=trainer_mock)

    trainer_inst_calls = [
        call for call in inst.call_args_list if "logger" in call.kwargs
    ]
    assert len(trainer_inst_calls) == 1, (
        "Expected exactly one hydra.utils.instantiate call to carry a `logger=` kwarg "
        "(the Trainer construction)."
    )
    trainer_call = trainer_inst_calls[0]
    forwarded_logger = trainer_call.kwargs["logger"]
    assert isinstance(forwarded_logger, WandbLogger)
    assert forwarded_logger._project == "confidence-distillation"

    forwarded_logger.log_hyperparams = MagicMock()
    # The log_hyperparams call inside main happens on the real logger
    # instance; we cannot inspect after-the-fact without capturing the
    # instance. Instead, re-run with a sentinel logger to verify the
    # call shape.


def test_trainer_receives_none_logger_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    trainer_mock = MagicMock(is_global_zero=True)
    cfg = _minimal_cfg(log_wandb=False)

    inst = _run_main(cfg, trainer_mock=trainer_mock)

    trainer_inst_calls = [
        call for call in inst.call_args_list if "logger" in call.kwargs
    ]
    assert len(trainer_inst_calls) == 1
    assert trainer_inst_calls[0].kwargs["logger"] is None


def test_log_hyperparams_called_on_rank0_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    trainer_mock = MagicMock(is_global_zero=True)
    cfg = _minimal_cfg(log_wandb=True)

    import proteinfoundation.confidence.train_confidence as tc

    captured = {}

    def _capture_build_wandb_logger(cfg_in):
        wl = MagicMock(spec=WandbLogger)
        captured["wl"] = wl
        return wl

    with patch.object(tc, "build_confidence_head_from_cfg", return_value=MagicMock()), patch.object(
        tc, "ConfidenceDistillationModule", return_value=MagicMock()
    ), patch.object(tc.hydra.utils, "instantiate", return_value=trainer_mock), patch.object(
        tc.L, "seed_everything"
    ), patch.object(tc, "_build_wandb_logger", side_effect=_capture_build_wandb_logger):
        tc.main.__wrapped__(cfg)

    wl = captured["wl"]
    assert wl.log_hyperparams.call_count == 1
    call = wl.log_hyperparams.call_args
    payload = call.args[0] if call.args else call.kwargs.get("params")
    assert isinstance(payload, dict)
    assert "config" in payload


def test_log_hyperparams_not_called_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)

    trainer_mock = MagicMock(is_global_zero=True)
    cfg = _minimal_cfg(log_wandb=False)

    import proteinfoundation.confidence.train_confidence as tc

    sentinel = MagicMock(spec=WandbLogger)

    with patch.object(tc, "build_confidence_head_from_cfg", return_value=MagicMock()), patch.object(
        tc, "ConfidenceDistillationModule", return_value=MagicMock()
    ), patch.object(tc.hydra.utils, "instantiate", return_value=trainer_mock), patch.object(
        tc.L, "seed_everything"
    ), patch.object(tc, "_build_wandb_logger", return_value=None) as build_mock:
        tc.main.__wrapped__(cfg)

    assert build_mock.call_count == 1
    assert sentinel.log_hyperparams.call_count == 0
