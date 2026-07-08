"""Unit: the adaptive per-round pipeline config composes with the right contract.

Pins that search_binder_confhead_adaptive_pipeline.yaml:
  - keeps the confidence-head scorer enabled (head labels provisional successes),
  - disables the per-round wall-clock cutoff (the OUTER loop owns the 16h budget),
  - gates analyze on the FULL canonical AlphaProteo criteria (published 0.90
    pLDDT floor, so the per-round analyze CSV's success column matches the loop's
    success accounting and the binder-design literature), NOT the parent's
    scRMSD-only collapse. The head's 0.92 floor is a separate provisional gate.

Run: .venv/bin/python -m pytest tests/unit/search/test_adaptive_pipeline_config.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_CONFIGS_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@pytest.fixture
def cfg(monkeypatch):
    # oc.env resolvers in the composed tree need these present.
    for var in ("CONF_CKPT_PATH", "AF2_DIR", "CKPT_PATH", "DATA_PATH"):
        monkeypatch.setenv(var, "/tmp/placeholder")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS_DIR):
        composed = compose(config_name="search_binder_confhead_adaptive_pipeline")
    yield composed
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def test_confidence_scorer_enabled(cfg):
    assert cfg.generation.confidence_scorer.enabled is True


def test_ckpt_paths_are_absolute_from_env(cfg):
    # The adaptive driver runs each round with cwd=<round_dir>, so the parent
    # config's cwd-relative `./ckpts` would break. This config must resolve the
    # trunk + autoencoder ckpts from the sbatch-staged absolute $CKPT_PATH.
    assert cfg.ckpt_path == "/tmp/placeholder"
    assert cfg.autoencoder_ckpt_path == "/tmp/placeholder/complexa_ae.ckpt"


def test_per_round_time_budget_disabled(cfg):
    assert cfg.generation.time_budget_hours is None


def test_analyze_gate_is_canonical_alphaproteo_090_plddt(cfg):
    thr = cfg.aggregation.success_thresholds
    assert set(thr.keys()) == {"i_pAE", "pLDDT", "scRMSD_ca"}
    # Canonical AlphaProteo floor (0.90), NOT the head's 0.92 provisional gate.
    assert thr.pLDDT.threshold == 0.90
    assert thr.pLDDT.op == ">="
    assert thr.i_pAE.threshold == 7.0
    assert thr.i_pAE.scale == 31.0
    assert thr.scRMSD_ca.threshold == 1.5
    assert thr.scRMSD_ca.op == "<"
