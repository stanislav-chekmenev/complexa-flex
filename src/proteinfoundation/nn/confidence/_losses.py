"""Pure-function loss helpers for confidence heads.

Lives under `nn.confidence` so concrete heads can depend on it without
inverting the layering against the sidecar `proteinfoundation.confidence`
package. The sidecar `proteinfoundation.confidence.losses` re-exports
these names for back-compat with callers that already imported them
under that path.

Includes:
    pLDDT family (single-axis residue labels) — `plddt_to_bin`,
      `masked_plddt_cross_entropy`, `masked_smooth_l1_on_expected_value`,
      `combined_plddt_loss`.
    PAE family (two-axis residue-pair labels) — `pae_to_bin`,
      `masked_pae_cross_entropy`, `masked_smooth_l1_on_pae_expected_value`,
      `combined_pae_loss`.

All masked-reduce operations apply `sum(loss * mask) / mask.sum().clamp_min(1)`
so empty / all-False masks yield a finite zero, never a NaN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from proteinfoundation.nn.confidence.base import StageLiteral

if TYPE_CHECKING:
    from torch import nn

    from proteinfoundation.nn.confidence.base import BaseConfidenceHead


__all__ = [
    "plddt_to_bin",
    "pae_to_bin",
    "_mask_reduce",
    "masked_plddt_cross_entropy",
    "masked_smooth_l1_on_expected_value",
    "combined_plddt_loss",
    "masked_pae_cross_entropy",
    "masked_smooth_l1_on_pae_expected_value",
    "combined_pae_loss",
    "MultiHeadLoss",
]


def plddt_to_bin(
    plddt: torch.Tensor | float,
    bin_width: float = 2.0,
    num_bins: int = 50,
) -> torch.Tensor:
    """Discretise per-residue AF2 pLDDT to integer bin indices.

    AF2 pLDDT is on the `[0, 100]` scale. The default `(num_bins=50,
    bin_width=2.0)` partitions that range into 50 disjoint bins of width 2.
    Inputs above the top edge clamp to `num_bins - 1`; inputs below zero
    clamp to `0`.
    """
    if not torch.is_tensor(plddt):
        plddt = torch.tensor(plddt, dtype=torch.float32)
    return (plddt / bin_width).floor().clamp(0, num_bins - 1).to(torch.int64)


def pae_to_bin(
    pae: torch.Tensor | float,
    bin_width: float = 0.5,
    num_bins: int = 64,
) -> torch.Tensor:
    """Discretise predicted aligned error onto integer bin indices.

    AF2 PAE values live in `[0, 31.75]` Angstrom; default
    `(num_bins=64, bin_width=0.5)` partitions that range into 64 disjoint
    bins of width 0.5 Angstrom. Inputs above the top edge clamp to
    `num_bins - 1`; inputs below zero clamp to `0`.
    """
    if not torch.is_tensor(pae):
        pae = torch.tensor(pae, dtype=torch.float32)
    return (pae / bin_width).floor().clamp(0, num_bins - 1).to(torch.int64)


def _mask_reduce(per_residue: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(per_residue.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (per_residue * mask_f).sum() / denom


def masked_plddt_cross_entropy(
    student_logits: torch.Tensor,
    plddt_bin_labels: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked per-residue cross-entropy over pLDDT bins."""
    b, n, k = student_logits.shape
    per_residue = F.cross_entropy(
        student_logits.reshape(b * n, k),
        plddt_bin_labels.reshape(b * n),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(b, n)
    return _mask_reduce(per_residue, mask)


def masked_smooth_l1_on_expected_value(
    student_logits: torch.Tensor,
    plddt_continuous: torch.Tensor,
    mask: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 between softmax-weighted bin-center EV and continuous pLDDT."""
    centers = bin_centers.to(student_logits.device, torch.float32)
    probs = torch.softmax(student_logits.float(), dim=-1)
    ev = (probs * centers).sum(dim=-1)
    per_residue = F.smooth_l1_loss(ev, plddt_continuous.float(), reduction="none")
    return _mask_reduce(per_residue, mask)


def combined_plddt_loss(
    student_logits: torch.Tensor,
    plddt_bin_labels: torch.Tensor,
    plddt_continuous: torch.Tensor,
    mask: torch.Tensor,
    bin_centers: torch.Tensor,
    ce_weight: float = 0.9,
    smooth_l1_weight: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`ce_weight * CE + smooth_l1_weight * SmoothL1_on_EV`."""
    loss_ce = masked_plddt_cross_entropy(
        student_logits, plddt_bin_labels, mask, label_smoothing=label_smoothing
    )
    loss_smooth_l1 = masked_smooth_l1_on_expected_value(
        student_logits, plddt_continuous, mask, bin_centers
    )
    total = ce_weight * loss_ce + smooth_l1_weight * loss_smooth_l1
    return total, {"loss_ce": loss_ce, "loss_smooth_l1": loss_smooth_l1}


def masked_pae_cross_entropy(
    student_logits: torch.Tensor,
    pae_bin_labels: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked per-pair cross-entropy over PAE bins.

    Args:
        student_logits: `[b, n, n, num_bins]`. Upcast to fp32 internally.
        pae_bin_labels: `[b, n, n]` int64 bin indices.
        mask: `[b, n, n]` bool or float. Caller is responsible for
            AND-ing `pair_orig_mask & pae_mask`.
        label_smoothing: forwarded to `F.cross_entropy`.

    Returns:
        Scalar loss in fp32.
    """
    logits_f = student_logits.float()
    b, n1, n2, k = logits_f.shape
    per_cell = F.cross_entropy(
        logits_f.reshape(b * n1 * n2, k),
        pae_bin_labels.reshape(b * n1 * n2),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(b, n1, n2)
    return _mask_reduce(per_cell, mask)


def masked_smooth_l1_on_pae_expected_value(
    student_logits: torch.Tensor,
    pae_continuous: torch.Tensor,
    mask: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 between softmax-weighted bin-center EV and continuous PAE.

    Args:
        student_logits: `[b, n, n, num_bins]`, any dtype (upcast to fp32).
        pae_continuous: `[b, n, n]` continuous PAE in Angstrom.
        mask: `[b, n, n]`.
        bin_centers: `[num_bins]`.

    Returns:
        Scalar loss in fp32.
    """
    centers = bin_centers.to(student_logits.device, torch.float32)
    probs = torch.softmax(student_logits.float(), dim=-1)
    ev = (probs * centers).sum(dim=-1)
    per_cell = F.smooth_l1_loss(ev, pae_continuous.float(), reduction="none")
    return _mask_reduce(per_cell, mask)


def combined_pae_loss(
    student_logits: torch.Tensor,
    pae_bin_labels: torch.Tensor,
    pae_continuous: torch.Tensor,
    mask: torch.Tensor,
    bin_centers: torch.Tensor,
    ce_weight: float = 0.9,
    smooth_l1_weight: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`ce_weight * masked_CE + smooth_l1_weight * SmoothL1_on_EV` for PAE."""
    loss_ce = masked_pae_cross_entropy(
        student_logits, pae_bin_labels, mask, label_smoothing=label_smoothing
    )
    loss_smooth_l1 = masked_smooth_l1_on_pae_expected_value(
        student_logits, pae_continuous, mask, bin_centers
    )
    total = ce_weight * loss_ce + smooth_l1_weight * loss_smooth_l1
    return total, {"loss_ce": loss_ce, "loss_smooth_l1": loss_smooth_l1}


class MultiHeadLoss:
    """Across-head aggregator for `MultiHeadConfidence`.

    Each child's within-head ce/ev recipe is invoked by its own
    `compute_loss_and_metrics`; this class sums across heads with
    per-head scalar weights and returns a flat log dict prefixed by the
    child's dict key.

    Two short-circuits drop a child's contribution to exactly zero with
    no contaminated gradient and no NaN-from-empty-mean:

    - `weights[name] == 0.0` skips the child's loss call entirely. The
      child's `_predict` may still have run inside the wrapper's forward
      (the wrapper does not know in advance which weights are zero), but
      since `total` never depends on the child's logits the autograd
      graph never reaches the child's parameters.
    - `masks_by_head[name].sum() == 0` (all-False) skips the loss for that
      child. The masked-reduce helpers already clamp the denominator at
      1, but skipping here saves the compute and keeps the log dict
      clean of pseudo-zero metrics from a head whose batch carries no
      labels for that quantity.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = {k: float(v) for k, v in weights.items()}

    def __call__(
        self,
        multi_out: dict[str, dict[str, torch.Tensor]],
        heads: "dict[str, BaseConfidenceHead] | nn.ModuleDict",
        batch: dict[str, torch.Tensor],
        masks_by_head: dict[str, torch.Tensor],
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(iter(masks_by_head.values())).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        log_dict: dict[str, torch.Tensor] = {}
        for name, head in heads.items():
            w = self.weights.get(name, 1.0)
            if w == 0.0:
                continue
            mask_eff = masks_by_head[name]
            if float(mask_eff.sum().item()) == 0.0:
                continue
            l_head, l_log = head.compute_loss_and_metrics(
                multi_out[name], batch, mask_eff, stage=stage
            )
            total = total + w * l_head
            log_dict[f"{name}/total"] = l_head
            for k, v in l_log.items():
                log_dict[f"{name}/{k}"] = v
        return total, log_dict
