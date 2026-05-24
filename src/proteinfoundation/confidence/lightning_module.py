"""Sidecar Lightning module for confidence-head distillation.

`ConfidenceDistillationModule` owns:

- a frozen `Proteina` (loaded once at construction time from
  `trunk_ckpt_path` + `autoencoder_ckpt_path`, with
  `proteina.nn.expose_intermediates = True`, every parameter
  `requires_grad_(False)`, and `.eval()`),
- a trainable `BaseConfidenceHead` (the PR-3 head; pLDDT today, PAE / multi
  via PR-B).

Forward path per batch (all pre-head steps under `torch.no_grad()`):

1. `add_clean_samples` populates `batch['x_1']` per modality.
2. `fm.corrupt_batch` populates `x_0`, `x_1`, `x_t`, `t` per modality.
3. `t` is overwritten to `trunk_eval_t` and `x_t` is recomputed via
   `fm.interpolate` so the trunk sees a deterministic on-distribution
   regime. This keeps the head in the AdaLN regime the trunk was
   trained for.
4. `proteina.nn(batch)` returns `trunk_intermediates = {s, z, mask,
   orig_mask, n_orig, local_latents}`. `local_latents` is the pre-trim
   `[b, n_extended, latent_dim]` tensor from the frozen trunk; the head's
   shared `ConfidenceTrunk` projects and adds it to `s` as a mask-zeroed
   residual before its pair-biased attention stack.
5. `_compute_cond(batch)` runs the trunk's `cond_factory` on the same
   `t`-pinned batch (still under no_grad), producing `[b, n, dim_cond]`.
6. `head(s, z, mask, cond, local_latents)` (trainable); outputs trimmed
   to `n_orig` along the residue axis.
7. The head's own `compute_loss_and_metrics(out, batch_trimmed, mask_eff,
   stage)` returns `(total, log_dict)`; the module prefixes the log keys
   with `{train,val}/{head.output_name_root}/`.

Frozen-trunk caveat: Lightning calls `model.train()` at the start of every
epoch. The defensive hooks `on_train_epoch_start` / `on_train_batch_start`
re-assert `.eval()` on `self.proteina` to guard against dropout and any
batched-norm-like layers re-entering training mode.

The full frozen trunk is currently persisted into the Lightning checkpoint
(spec §9 Risk 9 — accepted disk cost for PR-4).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torch
from loguru import logger
from torch import nn

from proteinfoundation.nn.confidence.base import BaseConfidenceHead
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
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


def _cosine_warmup_factor(
    step: int,
    warmup_steps: int,
    total_steps: int | None,
    min_factor: float,
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    if total_steps is None or total_steps <= 0:
        return 1.0
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return 1.0
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


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
        ce_weight: float = 0.9,
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
        if (
            self.ce_weight != 0.9
            or self.smooth_l1_weight != 0.1
            or self.label_smoothing != 0.05
        ):
            warnings.warn(
                "ce_weight/smooth_l1_weight/label_smoothing on "
                "ConfidenceDistillationModule are no-op since PR-B Slice 1; "
                "set head.{ce_weight,ev_weight,label_smoothing} via the Hydra "
                "`confidence.head.loss` block instead.",
                DeprecationWarning,
                stacklevel=2,
            )
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
        ce_weight: float = 0.9,
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

    def _trim_batch(self, batch: dict, n_orig: int) -> dict[str, torch.Tensor]:
        trimmed: dict = {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                trimmed[k] = v
                continue
            if (
                v.ndim >= 3
                and v.shape[1] == v.shape[2]
                and v.shape[1] >= n_orig
            ):
                trimmed[k] = v[:, :n_orig, :n_orig]
            elif v.ndim >= 2 and v.shape[1] >= n_orig:
                trimmed[k] = v[:, :n_orig]
            else:
                trimmed[k] = v
        return trimmed

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
        local_latents = inter["local_latents"]

        cond = self._pad_cond_to_n_ext(cond, mask_ext)
        assert cond.shape[1] == mask_ext.shape[1], (
            f"cond axis-1 {cond.shape[1]} must match mask_ext axis-1 {mask_ext.shape[1]} after padding"
        )

        head_out_raw = self.head(s, z, mask_ext, cond, local_latents)
        head_out = self._trim_head_output(head_out_raw, n_orig)

        if orig_mask.dtype != torch.bool:
            orig_mask = orig_mask.bool()

        batch_trimmed = self._trim_batch(batch, n_orig)

        if isinstance(self.head, MultiHeadConfidence):
            masks_by_head: dict[str, torch.Tensor] = {}
            for name, child in self.head.children_heads.items():
                child_root = child.output_name_root
                masks_by_head[name] = self._build_mask_eff(
                    batch_trimmed, orig_mask, f"{child_root}_mask"
                )
            return {
                "head_out": head_out,
                "masks_by_head": masks_by_head,
                "batch_trimmed": batch_trimmed,
            }

        mask_field = f"{self.head.output_name_root}_mask"
        mask_eff = self._build_mask_eff(batch_trimmed, orig_mask, mask_field)
        return {
            "head_out": head_out,
            "mask_eff": mask_eff,
            "batch_trimmed": batch_trimmed,
        }

    def _build_mask_eff(
        self,
        batch_trimmed: dict[str, torch.Tensor],
        orig_mask: torch.Tensor,
        mask_field: str,
    ) -> torch.Tensor:
        label_mask = batch_trimmed[mask_field]
        if label_mask.dtype != torch.bool:
            label_mask = label_mask.bool()
        if label_mask.ndim == orig_mask.ndim:
            assert label_mask.shape[1] == orig_mask.shape[1], (
                f"{mask_field} axis-1 ({label_mask.shape[1]}) must equal orig_mask "
                f"axis-1 ({orig_mask.shape[1]}); the dataset should align the label "
                f"mask with the binder/trunk residue frame before reaching the sidecar."
            )
            mask_eff_bool = orig_mask & label_mask
        elif label_mask.ndim == orig_mask.ndim + 1:
            assert (
                label_mask.shape[1] == label_mask.shape[2] == orig_mask.shape[1]
            ), (
                f"{mask_field} is a pair mask with shape {tuple(label_mask.shape)} "
                f"but orig_mask axis-1 is {orig_mask.shape[1]}; expected a square "
                f"(B, L, L) aligned with the binder/trunk frame."
            )
            pair_orig = orig_mask[:, :, None] & orig_mask[:, None, :]
            mask_eff_bool = pair_orig & label_mask
        else:
            raise ValueError(
                f"{mask_field}.ndim={label_mask.ndim} not compatible with "
                f"orig_mask.ndim={orig_mask.ndim}"
            )
        return mask_eff_bool.to(torch.float32)

    def _trim_head_output(
        self, head_out: dict, n_orig: int
    ) -> dict:
        trimmed: dict = {}
        for k, v in head_out.items():
            if isinstance(v, dict):
                trimmed[k] = self._trim_head_output(v, n_orig)
                continue
            if not torch.is_tensor(v):
                trimmed[k] = v
                continue
            if v.ndim >= 3 and v.shape[1] == v.shape[2]:
                trimmed[k] = v[:, :n_orig, :n_orig]
            elif v.ndim >= 2:
                trimmed[k] = v[:, :n_orig]
            else:
                trimmed[k] = v
        return trimmed

    def _log_head_metrics(
        self,
        stage: str,
        log_dict: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        prefix = f"{stage}/{self.head.output_name_root}"
        on_step = stage == "train"
        for key, value in log_dict.items():
            prog_bar = key in ("loss", "plddt_accuracy", "total") and stage in (
                "train",
                "val",
            )
            self.log(
                f"{prefix}/{key}",
                value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=prog_bar,
                batch_size=batch_size,
                sync_dist=True,
            )

    def _first_logits_tensor(self, head_out: dict) -> torch.Tensor:
        for v in head_out.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, dict):
                for inner in v.values():
                    if torch.is_tensor(inner):
                        return inner
        raise RuntimeError("could not locate any logits tensor in head_out")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._forward(batch)
        if isinstance(self.head, MultiHeadConfidence):
            total, log_dict = self.head.compute_multi_loss_and_metrics(
                out["head_out"],
                out["batch_trimmed"],
                out["masks_by_head"],
                stage="train",
            )
            log_dict = dict(log_dict)
            log_dict["total"] = total
        else:
            total, log_dict = self.head.compute_loss_and_metrics(
                out["head_out"], out["batch_trimmed"], out["mask_eff"], stage="train"
            )
        first_tensor = self._first_logits_tensor(out["head_out"])
        b = first_tensor.shape[0]
        self._log_head_metrics("train", log_dict, b)
        return total

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._forward(batch)
        if isinstance(self.head, MultiHeadConfidence):
            total, log_dict = self.head.compute_multi_loss_and_metrics(
                out["head_out"],
                out["batch_trimmed"],
                out["masks_by_head"],
                stage="val",
            )
            log_dict = dict(log_dict)
            log_dict["total"] = total
        else:
            total, log_dict = self.head.compute_loss_and_metrics(
                out["head_out"], out["batch_trimmed"], out["mask_eff"], stage="val"
            )
        first_tensor = self._first_logits_tensor(out["head_out"])
        b = first_tensor.shape[0]
        self._log_head_metrics("val", log_dict, b)

        if not isinstance(self.head, MultiHeadConfidence):
            diag_key = getattr(self.head, "reliability_diagram_logits_key", None)
            if diag_key is not None and diag_key in out["head_out"]:
                labels_bin = out["batch_trimmed"].get(
                    f"{self.head.output_name_root}_bin"
                )
                if labels_bin is not None:
                    self._accumulate_reliability(
                        out["head_out"][diag_key].detach(),
                        labels_bin.detach(),
                        out["mask_eff"].detach(),
                    )
        return total

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
        total_steps = (
            getattr(self.trainer, "estimated_stepping_batches", None)
            if self.trainer is not None
            else None
        )
        min_factor = self.min_lr / self.lr if self.lr > 0 else 0.0
        return _cosine_warmup_factor(
            step=step,
            warmup_steps=self.warmup_steps,
            total_steps=total_steps,
            min_factor=min_factor,
        )

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
