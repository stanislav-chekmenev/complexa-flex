"""Sidecar Lightning module for confidence-head distillation.

`ConfidenceDistillationModule` owns:

- a frozen `Proteina` (loaded once at construction time from
  `trunk_ckpt_path` + `autoencoder_ckpt_path`, with
  `proteina.nn.expose_intermediates = True`, every parameter
  `requires_grad_(False)`, and `.eval()`),
- a trainable `BaseConfidenceHead` (the PR-3 head).

Forward path per batch (all pre-head steps under `torch.no_grad()`):

1. `add_clean_samples` populates `batch['x_1']` per modality.
2. `fm.corrupt_batch` populates `x_0`, `x_1`, `x_t`, `t` per modality.
3. `t` is overwritten to `trunk_eval_t` and `x_t` is recomputed via
   `fm.interpolate` so the trunk sees a deterministic on-distribution
   regime. This keeps the head in the AdaLN regime the trunk was
   trained for.
4. `proteina.nn(batch)` returns `trunk_intermediates = {s, z, mask,
   orig_mask, n_orig}`.
5. `_compute_cond(batch)` runs the trunk's `cond_factory` on the same
   `t`-pinned batch (still under no_grad), producing `[b, n, dim_cond]`.
6. `head(s, z, mask, cond)` (trainable); logits sliced to
   `orig_mask & batch['plddt_mask']`.
7. `combined_plddt_loss` with `ce_weight / smooth_l1_weight`.

Frozen-trunk caveat: Lightning calls `model.train()` at the start of every
epoch. The defensive hooks `on_train_epoch_start` / `on_train_batch_start`
re-assert `.eval()` on `self.proteina` to guard against dropout and any
batched-norm-like layers re-entering training mode.

The full frozen trunk is currently persisted into the Lightning checkpoint
(spec §9 Risk 9 — accepted disk cost for PR-4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torch
from loguru import logger
from torch import nn

from proteinfoundation.confidence.losses import combined_plddt_loss, masked_plddt_cross_entropy
from proteinfoundation.confidence.metrics import (
    _labels_to_continuous,
    _logits_to_continuous,
    expected_calibration_error,
    expected_calibration_error_adaptive,
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
    _log_table_warned: bool = False

    def __init__(
        self,
        head: BaseConfidenceHead,
        trunk_ckpt_path: str = "",
        autoencoder_ckpt_path: str = "",
        trunk_eval_t: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 5e-6,
        ce_weight: float = 0.7,
        smooth_l1_weight: float = 0.1,
        label_smoothing: float = 0.05,
        num_bins_ece_adaptive: int = 15,
        reliability_diagram_every_n_epochs: int = 1,
        reliability_diagram_num_bins: int = 10,
        cond_modalities: tuple[str, ...] | None = None,
        proteina: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["head", "proteina", "trunk_ckpt_path", "autoencoder_ckpt_path"],
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
        self.num_bins_ece_adaptive = int(num_bins_ece_adaptive)
        self.reliability_diagram_every_n_epochs = int(reliability_diagram_every_n_epochs)
        self.reliability_diagram_num_bins = int(reliability_diagram_num_bins)
        # intentionally not registered as buffers to avoid state_dict pollution
        self._rd_conf_sum: torch.Tensor | None = None
        self._rd_correct_sum: torch.Tensor | None = None
        self._rd_count: torch.Tensor | None = None

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
        smooth_l1_weight: float = 0.1,
        label_smoothing: float = 0.05,
        num_bins_ece_adaptive: int = 15,
        reliability_diagram_every_n_epochs: int = 1,
        reliability_diagram_num_bins: int = 10,
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
            num_bins_ece_adaptive=num_bins_ece_adaptive,
            reliability_diagram_every_n_epochs=reliability_diagram_every_n_epochs,
            reliability_diagram_num_bins=reliability_diagram_num_bins,
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

    def _pad_cond_to_n_ext(
        self, cond: torch.Tensor, mask_ext: torch.Tensor
    ) -> torch.Tensor:
        """Pad cond from `(b, n_orig, dim_cond)` to `(b, n_ext, dim_cond)`.

        The trunk's `cond_factory` emits cond in the `n_orig` (binder)
        frame; the head expects cond aligned with the extended mask
        produced by concat-features (motif/target/ligand). Pad the tail
        with zero — the head's AdaLN at those positions only affects
        masked-out tokens, so the value is irrelevant downstream.
        """
        n_orig = cond.shape[1]
        n_ext = mask_ext.shape[1]
        if n_orig == n_ext:
            return cond
        if n_orig > n_ext:
            raise ValueError(
                f"cond has more residues ({n_orig}) than mask_ext ({n_ext}); "
                "cond_factory should never extend beyond the trunk's extended mask."
            )
        pad = cond.new_zeros((cond.shape[0], n_ext - n_orig, cond.shape[2]))
        return torch.cat([cond, pad], dim=1)

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

        cond = self._pad_cond_to_n_ext(cond, mask_ext)
        assert cond.shape[1] == mask_ext.shape[1], (
            f"cond axis-1 {cond.shape[1]} must match mask_ext axis-1 {mask_ext.shape[1]} after padding"
        )

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

        total, parts = combined_plddt_loss(
            student_logits=logits,
            plddt_bin_labels=labels_bin,
            plddt_continuous=labels_cont,
            mask=mask_eff,
            bin_centers=centers,
            ce_weight=self.ce_weight,
            smooth_l1_weight=self.smooth_l1_weight,
            label_smoothing=0.0,
        )
        loss_ce = parts["loss_ce"]
        loss_smooth_l1 = parts["loss_smooth_l1"]

        acc = plddt_accuracy(logits, labels_bin, mask_eff)
        mae = plddt_mae(logits, labels_bin, mask_eff, centers)
        pred_cont = _logits_to_continuous(logits, centers)
        target_cont = _labels_to_continuous(labels_bin, centers)
        pr = pearson_r(pred_cont, target_cont, mask_eff)
        sr = spearman_r(pred_cont, target_cont, mask_eff)
        strat = plddt_mae_stratified(logits, labels_bin, mask_eff, centers)
        ece = expected_calibration_error(logits, labels_bin, mask_eff)
        ece_adaptive = expected_calibration_error_adaptive(
            logits, labels_bin, mask_eff, num_bins_ece=self.num_bins_ece_adaptive
        )

        b = logits.shape[0]
        self.log("val/loss", loss_ce, prog_bar=True, batch_size=b, sync_dist=True)
        self.log("val/loss_ce", loss_ce, batch_size=b, sync_dist=True)
        self.log("val/loss_smooth_l1", loss_smooth_l1, batch_size=b, sync_dist=True)
        self.log("val/loss_total", total, batch_size=b, sync_dist=True)
        self.log("val/ece", ece, batch_size=b, sync_dist=True)
        self.log("val/ece_adaptive", ece_adaptive, batch_size=b, sync_dist=True)
        self.log("val/plddt_accuracy", acc, prog_bar=True, batch_size=b, sync_dist=True)
        self.log("val/plddt_mae", mae, batch_size=b, sync_dist=True)
        self.log("val/pearson_r", pr, batch_size=b, sync_dist=True)
        self.log("val/spearman_r", sr, batch_size=b, sync_dist=True)
        for k, v in strat.items():
            self.log(f"val/{k}", v, batch_size=b, sync_dist=True)

        self._accumulate_reliability(logits.detach(), labels_bin.detach(), mask_eff.detach())
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
        num_bins = self.reliability_diagram_num_bins
        device = self.device
        self._rd_conf_sum = torch.zeros(num_bins, dtype=torch.float32, device=device)
        self._rd_correct_sum = torch.zeros(num_bins, dtype=torch.float32, device=device)
        self._rd_count = torch.zeros(num_bins, dtype=torch.float32, device=device)

    def _accumulate_reliability(
        self,
        logits: torch.Tensor,
        labels_bin: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if self._rd_count is None:
            return
        num_bins = self.reliability_diagram_num_bins
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        mask_f = mask.to(torch.float32)
        correct = (pred == labels_bin).to(torch.float32) * mask_f

        edges = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
        for i in range(num_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == num_bins - 1:
                in_bin = (conf >= lo) & (conf <= hi)
            else:
                in_bin = (conf >= lo) & (conf < hi)
            in_bin_f = in_bin.to(torch.float32) * mask_f
            n_b = in_bin_f.sum()
            if n_b.item() == 0.0:
                continue
            self._rd_conf_sum[i] = self._rd_conf_sum[i] + (conf * in_bin_f).sum()
            self._rd_correct_sum[i] = self._rd_correct_sum[i] + (correct * in_bin_f).sum()
            self._rd_count[i] = self._rd_count[i] + n_b

    def on_validation_epoch_end(self) -> None:
        if self._rd_count is None:
            return
        every = max(self.reliability_diagram_every_n_epochs, 1)
        if (self.current_epoch + 1) % every != 0:
            return

        conf_sum = self._rd_conf_sum
        correct_sum = self._rd_correct_sum
        count = self._rd_count
        if self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            strategy = self.trainer.strategy
            conf_sum = strategy.reduce(conf_sum, reduce_op="sum")
            correct_sum = strategy.reduce(correct_sum, reduce_op="sum")
            count = strategy.reduce(count, reduce_op="sum")

        diagram = torch.zeros(
            (self.reliability_diagram_num_bins, 3), dtype=torch.float32, device=count.device
        )
        nonempty = count > 0
        diagram[nonempty, 0] = conf_sum[nonempty] / count[nonempty]
        diagram[nonempty, 1] = correct_sum[nonempty] / count[nonempty]
        diagram[nonempty, 2] = count[nonempty]
        self._emit_reliability_diagram(diagram.cpu())

    def _emit_reliability_diagram(self, diagram: torch.Tensor) -> None:
        if self.trainer is not None and self.trainer.global_rank != 0:
            return
        epoch = int(self.current_epoch)
        lightning_logger = getattr(self, "logger", None)
        log_table = getattr(lightning_logger, "log_table", None) if lightning_logger else None
        if callable(log_table):
            try:
                log_table(
                    key=f"val/reliability_epoch_{epoch}",
                    columns=["conf", "acc", "count"],
                    data=diagram.tolist(),
                )
                return
            except Exception as exc:
                if not type(self)._log_table_warned:
                    logger.warning(
                        f"log_table unavailable, falling back to .npy: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    type(self)._log_table_warned = True
        log_dir = self.trainer.log_dir if self.trainer is not None else "."
        out_dir = Path(log_dir) if log_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"reliability_epoch_{epoch}.npy"
        np.save(out_path, diagram.numpy())
