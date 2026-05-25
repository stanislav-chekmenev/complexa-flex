"""One-step smoke train of the qg-style multihead Teddymer distill.

Constructs the Lightning module + datamodule from the Hydra config,
runs trainer.fit for a single train step + one val pass on a 4-sample
micro-batch, asserts the headline log keys are present and finite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.registry import build_confidence_head_from_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not (REPO_ROOT / "ckpts" / "complexa.ckpt").exists(),
    reason="complexa.ckpt not staged for smoke",
)
def test_qg_multihead_one_step_smoke(tmp_path):
    os.environ["RUN_DIR"] = str(tmp_path / "run")
    os.environ["WANDB_MODE"] = "disabled"
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS_DIR)):
        cfg = compose(
            config_name="confidence/distillation_teddymer_qg_multihead",
            overrides=[
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=1",
                "trainer.val_check_interval=1",
                "data.datamodule.batch_size=2",
                "integrity.enabled=false",
            ],
        )

    head = build_confidence_head_from_cfg(cfg.confidence.head)
    module = ConfidenceDistillationModule(
        head=head,
        trunk_ckpt_path=str(REPO_ROOT / "ckpts" / "complexa.ckpt"),
        autoencoder_ckpt_path=str(REPO_ROOT / "ckpts" / "complexa_ae.ckpt"),
        trunk_eval_t=cfg.training.trunk_eval_t,
        lr=cfg.training.opt.lr,
        weight_decay=cfg.training.opt.weight_decay,
        betas=tuple(cfg.training.opt.betas),
        warmup_steps=cfg.training.opt.warmup_steps,
        min_lr=cfg.training.opt.min_lr,
    )
    if not Path("/netscratch/schekmenev/teddymer_v1_blob").exists():
        pytest.skip("Teddymer staged blob unavailable")

    import hydra

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
    trainer.fit(module, datamodule=datamodule)

    logged = trainer.callback_metrics
    for key in (
        "val/multi/total",
        "val/plddt/total",
        "val/pae/total",
    ):
        assert key in logged, f"Missing log key {key!r}; got {sorted(logged.keys())}"
        assert torch.isfinite(logged[key]).all()
