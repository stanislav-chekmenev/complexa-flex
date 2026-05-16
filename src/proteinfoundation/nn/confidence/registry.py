"""Registry + Hydra factory for confidence heads.

Two construction styles are supported:

1. The cfg carries a `_target_` key -> `hydra.utils.instantiate(cfg)`.
2. The cfg carries a `name` key -> look up `CONFIDENCE_HEAD_REGISTRY[name]`
   then call the class with the remaining cfg fields as kwargs.

Unknown `name` raises `ValueError` listing every registered head.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from proteinfoundation.nn.confidence.base import BaseConfidenceHead


CONFIDENCE_HEAD_REGISTRY: dict[str, type] = {}


def register_confidence_head(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        if name in CONFIDENCE_HEAD_REGISTRY and CONFIDENCE_HEAD_REGISTRY[name] is not cls:
            raise ValueError(
                f"Confidence head name {name!r} already registered to "
                f"{CONFIDENCE_HEAD_REGISTRY[name].__name__}; cannot rebind to {cls.__name__}."
            )
        CONFIDENCE_HEAD_REGISTRY[name] = cls
        return cls

    return _wrap


def build_confidence_head_from_cfg(cfg: Any) -> "BaseConfidenceHead":
    """Instantiate a confidence head from a Hydra/OmegaConf cfg.

    `_target_` takes precedence over `name` when both are present.
    """
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    if isinstance(cfg, DictConfig) and "_target_" in cfg:
        cfg_for_target = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if "name" in cfg_for_target:
            del cfg_for_target["name"]
        return hydra.utils.instantiate(cfg_for_target)

    name = cfg.get("name", None) if isinstance(cfg, DictConfig) else None
    if name is None:
        raise ValueError(
            "Confidence-head cfg requires either `_target_` or `name`; got "
            f"keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else cfg}"
        )

    if name not in CONFIDENCE_HEAD_REGISTRY:
        known = sorted(CONFIDENCE_HEAD_REGISTRY.keys())
        raise ValueError(
            f"Unknown confidence head name {name!r}. Known heads: {known}"
        )

    cls = CONFIDENCE_HEAD_REGISTRY[name]
    kwargs = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k != "name"}
    if "trunk" in kwargs:
        kwargs["trunk"] = hydra.utils.instantiate(cfg.trunk)
    return cls(**kwargs)
