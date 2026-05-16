"""Sidecar Lightning module for confidence-head distillation.

`ConfidenceDistillationModule` owns:

- a frozen `Proteina` (loaded once at construction time from
  `trunk_ckpt_path` + `autoencoder_ckpt_path`, with
  `proteina.nn.expose_intermediates = True`, every parameter
  `requires_grad_(False)`, and `.eval()`),
- a trainable `BaseConfidenceHead` (the PR-3 head).

Forward path per batch:

1. Build `cond` once by stamping `batch['t'][<modality>] = trunk_eval_t`
   for every modality the trunk's `cond_factory` consumes, then forwarding
   `cond_factory(batch)`. This keeps the head in the AdaLN regime the
   trunk was trained for.
2. Under `torch.no_grad()`, call `proteina.nn(batch)` to get
   `trunk_intermediates = {s, z, mask, orig_mask, n_orig}`.
3. Call `head(s, z, mask, cond)` and slice the logits / labels to
   `orig_mask & batch['plddt_mask']`.
4. `combined_plddt_loss` with `ce_weight / smooth_l1_weight`.

Frozen-trunk caveat: Lightning calls `model.train()` at the start of every
epoch. The defensive hooks `on_train_epoch_start` / `on_train_batch_start`
re-assert `.eval()` on `self.proteina` to guard against dropout and any
batched-norm-like layers re-entering training mode.

The full frozen trunk is currently persisted into the Lightning checkpoint
(spec §9 Risk 9 — accepted disk cost for PR-4).
"""

from __future__ import annotations

from typing import Iterable

import lightning as L
import torch
from loguru import logger
from torch import nn

from proteinfoundation.confidence.losses import combined_plddt_loss, masked_plddt_cross_entropy
from proteinfoundation.confidence.metrics import (
    _labels_to_continuous,
    _logits_to_continuous,
    pearson_r,
    plddt_accuracy,
    plddt_mae,
    plddt_mae_stratified,
    spearman_r,
)
from proteinfoundation.nn.confidence.base import BaseConfidenceHead
from proteinfoundation.utils.sample_utils import add_clean_samples


def _autodetect_cond_modalities(cond_factory: nn.Module) -> tuple[str, ...]:
    """Read `data_mode_use` off the time-embedding feature creators.

    Falls back to `("bb_ca",)` when no time-embedded feature is present —
    a `cond_factory` with only non-time features still needs `batch['t']`
    set for downstream code paths that consume it (callers can override).
    """
    feat_creators = getattr(cond_factory, "feat_creators", None)
    if feat_creators is None:
        return ("bb_ca",)
    modalities: list[str] = []
    seen: set[str] = set()
    for creator in feat_creators:
        m = getattr(creator, "data_mode_use", None)
        if m is not None and m not in seen:
            seen.add(m)
            modalities.append(m)
    if not modalities:
        return ("bb_ca",)
    return tuple(modalities)


class ConfidenceDistillationModule(L.LightningModule):
    def __init__(
        self,
        head: BaseConfidenceHead,
        trunk_ckpt_path: str,
        autoencoder_ckpt_path: str,
        trunk_eval_t: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 5e-6,
        ce_weight: float = 0.7,
        smooth_l1_weight: float = 0.3,
        label_smoothing: float = 0.0,
        cond_modalities: tuple[str, ...] | None = None,
        proteina: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["head", "proteina"],
        )
        self.head = head
        if proteina is None:
            self.proteina = self._load_frozen_trunk(
                trunk_ckpt_path=trunk_ckpt_path,
                autoencoder_ckpt_path=autoencoder_ckpt_path,
            )
        else:
            self.proteina = proteina
        self._freeze_trunk()

        if cond_modalities is None:
            cond_modalities = _autodetect_cond_modalities(self.proteina.nn.cond_factory)
        self._cond_modalities = tuple(cond_modalities)

        self.trunk_eval_t = float(trunk_eval_t)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = (float(betas[0]), float(betas[1]))
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.ce_weight = float(ce_weight)
        self.smooth_l1_weight = float(smooth_l1_weight)
        self.label_smoothing = float(label_smoothing)

    @classmethod
    def from_components(
        cls,
        head: BaseConfidenceHead,
        proteina: nn.Module,
        cond_modalities: Iterable[str] | None = None,
        trunk_eval_t: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 5e-6,
        ce_weight: float = 0.7,
        smooth_l1_weight: float = 0.3,
        label_smoothing: float = 0.0,
    ) -> "ConfidenceDistillationModule":
        """Construct without loading from disk. Used by tests and PR-5 smoke."""
        return cls(
            head=head,
            trunk_ckpt_path="",
            autoencoder_ckpt_path="",
            trunk_eval_t=trunk_eval_t,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            ce_weight=ce_weight,
            smooth_l1_weight=smooth_l1_weight,
            label_smoothing=label_smoothing,
            cond_modalities=tuple(cond_modalities) if cond_modalities is not None else None,
            proteina=proteina,
        )

    def _load_frozen_trunk(
        self,
        trunk_ckpt_path: str,
        autoencoder_ckpt_path: str,
    ) -> nn.Module:
        from proteinfoundation.proteina import Proteina

        proteina = Proteina.load_from_checkpoint(
            trunk_ckpt_path,
            autoencoder_ckpt_path=autoencoder_ckpt_path,
        )
        proteina.nn.expose_intermediates = True
        return proteina

    def _freeze_trunk(self) -> None:
        self.proteina.requires_grad_(False)
        self.proteina.eval()
        if hasattr(self.proteina, "nn"):
            self.proteina.nn.expose_intermediates = True

    def _compute_cond(self, batch: dict) -> torch.Tensor:
        """Trunk frozen; cond computed under no_grad on the sidecar-prepared batch.

        The caller is responsible for stamping `batch['t']` to
        `trunk_eval_t` *before* invoking this; this keeps the head in the
        AdaLN regime the trunk was trained for.
        """
        with torch.no_grad():
            cond = self.proteina.nn.cond_factory(batch)
        expected_dim = getattr(self.head.trunk, "dim_cond", None)
        if expected_dim is not None and cond.shape[-1] != expected_dim:
            raise ValueError(
                f"cond_factory output dim {cond.shape[-1]} does not match "
                f"head.trunk.dim_cond {expected_dim}"
            )
        return cond

    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batch = add_clean_samples(
                batch,
                self.proteina.cfg_exp.product_flowmatcher,
                getattr(self.proteina, "autoencoder", None),
            )
            batch = self.proteina.fm.corrupt_batch(batch)
            b = batch["mask"].shape[0]
            device = batch["mask"].device
            t_pinned = {
                m: torch.full(
                    (b,),
                    self.trunk_eval_t,
                    device=device,
                    dtype=batch["t"][m].dtype,
                )
                for m in batch["t"]
            }
            batch["t"] = t_pinned
            batch["x_t"] = self.proteina.fm.interpolate(
                x_0=batch["x_0"],
                x_1=batch["x_1"],
                t=t_pinned,
                mask=batch["mask"],
            )

            self.proteina.nn.expose_intermediates = True
            nn_out = self.proteina.nn(batch)

        cond = self._compute_cond(batch)
        inter = nn_out["trunk_intermediates"]
        s = inter["s"]
        z = inter["z"]
        mask_ext = inter["mask"]
        orig_mask = inter["orig_mask"]
        n_orig = int(inter["n_orig"])

        head_out = self.head(s, z, mask_ext, cond)
        logits = head_out["plddt_logits"][:, :n_orig, :]

        plddt_mask = batch["plddt_mask"]
        if plddt_mask.dtype != torch.bool:
            plddt_mask = plddt_mask.bool()
        if orig_mask.dtype != torch.bool:
            orig_mask = orig_mask.bool()
        mask_eff = (orig_mask & plddt_mask).to(torch.float32)

        labels_bin = batch["plddt_bin"][:, :n_orig]
        labels_cont = batch["plddt_residue"][:, :n_orig]

        return {
            "logits": logits,
            "mask_eff": mask_eff,
            "labels_bin": labels_bin,
            "labels_cont": labels_cont,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._forward(batch)
        total, parts = combined_plddt_loss(
            student_logits=out["logits"],
            plddt_bin_labels=out["labels_bin"],
            plddt_continuous=out["labels_cont"],
            mask=out["mask_eff"],
            bin_centers=self.head.bin_centers,
            ce_weight=self.ce_weight,
            smooth_l1_weight=self.smooth_l1_weight,
            label_smoothing=self.label_smoothing,
        )
        b = out["logits"].shape[0]
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True, batch_size=b, sync_dist=True)
        self.log("train/loss_ce", parts["loss_ce"], on_step=True, on_epoch=True, batch_size=b, sync_dist=True)
        self.log("train/loss_smooth_l1", parts["loss_smooth_l1"], on_step=True, on_epoch=True, batch_size=b, sync_dist=True)
        return total

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._forward(batch)
        logits = out["logits"]
        mask_eff = out["mask_eff"]
        labels_bin = out["labels_bin"]
        labels_cont = out["labels_cont"]
        centers = self.head.bin_centers

        loss_ce = masked_plddt_cross_entropy(
            logits, labels_bin, mask_eff, label_smoothing=0.0
        )

        acc = plddt_accuracy(logits, labels_bin, mask_eff)
        mae = plddt_mae(logits, labels_bin, mask_eff, centers)
        pred_cont = _logits_to_continuous(logits, centers)
        target_cont = _labels_to_continuous(labels_bin, centers)
        pr = pearson_r(pred_cont, target_cont, mask_eff)
        sr = spearman_r(pred_cont, target_cont, mask_eff)
        strat = plddt_mae_stratified(logits, labels_bin, mask_eff, centers)

        b = logits.shape[0]
        self.log("val/loss", loss_ce, prog_bar=True, batch_size=b, sync_dist=True)
        self.log("val/plddt_accuracy", acc, prog_bar=True, batch_size=b, sync_dist=True)
        self.log("val/plddt_mae", mae, batch_size=b, sync_dist=True)
        self.log("val/pearson_r", pr, batch_size=b, sync_dist=True)
        self.log("val/spearman_r", sr, batch_size=b, sync_dist=True)
        for k, v in strat.items():
            self.log(f"val/{k}", v, batch_size=b, sync_dist=True)
        return loss_ce

    def configure_optimizers(self):
        trainable = [p for p in self.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None) if self.trainer is not None else None
        if total_steps is None or total_steps <= 0:
            return 1.0
        decay_steps = total_steps - self.warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - self.warmup_steps) / decay_steps
        min_factor = self.min_lr / self.lr if self.lr > 0 else 0.0
        return max(1.0 - progress * (1.0 - min_factor), min_factor)

    def on_train_epoch_start(self) -> None:
        self.proteina.eval()
        self.head.train()

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        if self.proteina.training:
            self.proteina.eval()
            logger.warning(
                f"[on_train_batch_start epoch={self.current_epoch} batch={batch_idx}] "
                "Had to re-enforce eval() on frozen Proteina trunk."
            )

    def on_validation_epoch_start(self) -> None:
        self.proteina.eval()
        self.head.eval()
