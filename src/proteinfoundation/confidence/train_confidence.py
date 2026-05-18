"""Hydra entry point for confidence-head distillation.

Composes the head + dataset + training config, instantiates a
`ConfidenceDistillationModule` (sidecar) and the dataset's
`StructureDataModule`, and runs `L.Trainer.fit(...)`.

Run via `python -m proteinfoundation.confidence.train_confidence` or, more
commonly, `srun python -m proteinfoundation.confidence.train_confidence
--config-name=confidence/distillation_swissprot ...`.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as L
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.registry import build_confidence_head_from_cfg


def _build_trainer(cfg: DictConfig) -> L.Trainer:
    return hydra.utils.instantiate(cfg.trainer, _convert_="partial")


@hydra.main(
    config_path="../../../configs",
    config_name="confidence/distillation_swissprot",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    logger.info("Composed config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))

    integrity_cfg = cfg.get("integrity", None)
    if integrity_cfg is not None and bool(integrity_cfg.get("enabled", False)):
        from proteinfoundation.datasets.teddymer.integrity import (
            verify_teddymer_blob_integrity,
        )
        verify_teddymer_blob_integrity(
            view_root=Path(integrity_cfg["view_root"]),
            snapshot_path=Path(integrity_cfg["snapshot_path"]),
        )

    head = build_confidence_head_from_cfg(cfg.confidence.head)

    training_cfg = cfg.training
    # Loss weights / label smoothing live on the head's Hydra block since
    # PR-B Slice 2 (see `configs/nn/confidence/{plddt,pae}_head.yaml`). We
    # still forward `cfg.training.loss.*` into the Lightning module so legacy
    # configs that override these values surface the DeprecationWarning
    # emitted in `ConfidenceDistillationModule.__init__`; the kwargs are
    # otherwise no-op on the module side.
    loss_cfg = training_cfg.get("loss", {})
    module = ConfidenceDistillationModule(
        head=head,
        trunk_ckpt_path=training_cfg.trunk_ckpt_path,
        autoencoder_ckpt_path=training_cfg.autoencoder_ckpt_path,
        trunk_eval_t=training_cfg.trunk_eval_t,
        lr=training_cfg.opt.lr,
        weight_decay=training_cfg.opt.weight_decay,
        betas=tuple(training_cfg.opt.betas),
        warmup_steps=training_cfg.opt.warmup_steps,
        min_lr=training_cfg.opt.min_lr,
        ce_weight=loss_cfg.get("ce_weight", 0.9),
        smooth_l1_weight=loss_cfg.get("smooth_l1_weight", 0.1),
        label_smoothing=loss_cfg.get("label_smoothing", 0.05),
    )

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)

    trainer = _build_trainer(cfg)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
