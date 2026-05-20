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
import secrets
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.registry import build_confidence_head_from_cfg


def _gate_loguru_to_rank0() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    if local_rank != 0 or node_rank != 0 or global_rank != 0:
        logger.remove()


_FORK_ID_ENV = "COMPLEXA_RUN_FORK_ID"


def _resolve_run_identity(cfg: DictConfig) -> tuple[str, str]:
    """Resolve `(wandb_id, wandb_name)` from the composed config.

    Default: fork — append a fresh 8-char suffix to `run_name` so each launch
    gets a distinct WandB run. The suffix must be identical across DDP ranks
    (each `srun` rank is a separate Python process); precedence:
      1. `COMPLEXA_RUN_FORK_ID` env var (explicit override; also lets a parent
         process mint once and broadcast),
      2. `SLURM_JOB_ID` (set identically across all ranks of the same job),
      3. `secrets.token_hex(4)` (single-rank dev fallback).
    Resume: if `cfg.resume_id` is set, use it verbatim as the WandB id (and
    keep `run_name` as the display name) — re-enters the existing run's
    history.
    """
    base_name = str(cfg.run_name)
    resume_id = cfg.get("resume_id", None)
    if resume_id:
        return str(resume_id), base_name

    suffix = os.environ.get(_FORK_ID_ENV)
    if not suffix:
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        suffix = slurm_job_id if slurm_job_id else secrets.token_hex(4)
    forked = f"{base_name}-{suffix}"
    return forked, forked


def _build_wandb_logger(cfg: DictConfig, *, wandb_id: str, wandb_name: str) -> Optional[WandbLogger]:
    log_cfg = cfg.get("logging", None)
    if log_cfg is None or not bool(log_cfg.get("log_wandb", False)):
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    return WandbLogger(
        project=log_cfg["wandb_project"],
        id=wandb_id,
        name=wandb_name,
        entity=log_cfg.get("wandb_entity", None),
        group=log_cfg.get("wandb_group", None),
        tags=list(log_cfg.get("wandb_tags", []) or []),
    )


def _build_trainer(cfg: DictConfig, *, logger: Optional[WandbLogger]) -> L.Trainer:
    return hydra.utils.instantiate(cfg.trainer, logger=logger, _convert_="partial")


@hydra.main(
    config_path="../../../configs",
    config_name="confidence/distillation_swissprot",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    _gate_loguru_to_rank0()

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

    wandb_id, wandb_name = _resolve_run_identity(cfg)
    wandb_logger = _build_wandb_logger(cfg, wandb_id=wandb_id, wandb_name=wandb_name)
    trainer = _build_trainer(cfg, logger=wandb_logger)

    if wandb_logger is not None and trainer.is_global_zero:
        wandb_logger.log_hyperparams(
            {"config": OmegaConf.to_container(cfg, resolve=True)}
        )

    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    if resume_ckpt_path is not None:
        logger.info("Resuming trainer state from {}", resume_ckpt_path)
    trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt_path)


if __name__ == "__main__":
    main()
