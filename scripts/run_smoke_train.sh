#!/usr/bin/env bash
# Single-GPU smoke run for PR-5 verification.
#
# Exit criterion: 100 steps complete, `val/plddt/loss_ce` is finite,
# `val/plddt/ece_adaptive` is logged, and a reliability-diagram artifact appears
# under the trainer log dir.
#
# Required env vars:
#   CKPT_DIR     directory containing complexa.ckpt + complexa_ae.ckpt
#                (defaults to "ckpts")
#   DATA_PATH    root of the AFDB CIF cache (used by the dataset config)
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-ckpts}" uv run python -m proteinfoundation.confidence.train_confidence \
  --config-name=confidence/distillation_swissprot \
  trainer.max_steps=100 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=2 \
  trainer.devices=1 \
  trainer.strategy=auto
