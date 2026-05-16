"""Registry + Hydra factory tests for the confidence-head package.

Covers:
- the `@register_confidence_head` decorator populates the registry on import,
- `build_confidence_head_from_cfg` accepts the `_target_` style,
- `build_confidence_head_from_cfg` accepts the `name` style,
- unknown `name` raises `ValueError` listing the known heads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from proteinfoundation.nn.confidence import (
    CONFIDENCE_HEAD_REGISTRY,
    PLDDTHead,
    build_confidence_head_from_cfg,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose_plddt_head_cfg():
    with initialize_config_dir(config_dir=str(CONFIG_DIR / "nn" / "confidence"), version_base="1.3"):
        return compose(config_name="plddt_head")


def test_register_decorator_populates_registry() -> None:
    assert "plddt" in CONFIDENCE_HEAD_REGISTRY
    assert CONFIDENCE_HEAD_REGISTRY["plddt"] is PLDDTHead


def test_build_from_yaml_via_target() -> None:
    cfg = _compose_plddt_head_cfg()
    assert cfg.get("_target_") == "proteinfoundation.nn.confidence.plddt_head.PLDDTHead"
    head = build_confidence_head_from_cfg(cfg)
    assert isinstance(head, PLDDTHead)


def test_build_from_yaml_via_name() -> None:
    cfg = _compose_plddt_head_cfg()
    cfg_no_target = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    del cfg_no_target["_target_"]
    head = build_confidence_head_from_cfg(cfg_no_target)
    assert isinstance(head, PLDDTHead)


def test_unknown_name_raises() -> None:
    cfg = OmegaConf.create({"name": "iptm_v2"})
    with pytest.raises(ValueError) as excinfo:
        build_confidence_head_from_cfg(cfg)
    msg = str(excinfo.value)
    assert "iptm_v2" in msg
    for known in CONFIDENCE_HEAD_REGISTRY:
        assert known in msg
