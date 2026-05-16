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

    head = build_confidence_head_from_cfg(cfg.confidence.head)

    training_cfg = cfg.training
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
        ce_weight=training_cfg.loss.ce_weight,
        smooth_l1_weight=training_cfg.loss.smooth_l1_weight,
        label_smoothing=training_cfg.loss.get("label_smoothing", 0.0),
    )

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)

    trainer = _build_trainer(cfg)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
