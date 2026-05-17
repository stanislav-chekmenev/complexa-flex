"""Confidence-head losses and helpers.

`plddt_to_bin` discretises continuous AF2 pLDDT onto integer bin indices
(shipped in PR-2). PR-4 adds the masked cross-entropy loss, the smooth-L1
loss on the bin-expected value, and a `combined_plddt_loss` that wraps both
under a shared mask reduction `sum(loss * mask) / mask.sum().clamp_min(1)`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def plddt_to_bin(
    plddt: torch.Tensor | float,
    bin_width: float = 2.0,
    num_bins: int = 50,
) -> torch.Tensor:
    """Discretise per-residue AF2 pLDDT to integer bin indices.

    AF2 pLDDT is on the `[0, 100]` scale. The default `(num_bins=50,
    bin_width=2.0)` partitions that range into 50 disjoint bins of width 2,
    matching the spec for the confidence head's cross-entropy target.

    Inputs above the top edge clamp to `num_bins - 1`; inputs below zero
    clamp to `0`.

    Args:
        plddt: Continuous pLDDT values, any shape. Accepts a Python scalar.
        bin_width: Width of each bin on the pLDDT scale.
        num_bins: Number of bins. Output is clamped to `[0, num_bins - 1]`.

    Returns:
        `torch.int64` tensor of the same shape as `plddt`.
    """
    if not torch.is_tensor(plddt):
        plddt = torch.tensor(plddt, dtype=torch.float32)
    return (plddt / bin_width).floor().clamp(0, num_bins - 1).to(torch.int64)


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
    """Masked per-residue cross-entropy over pLDDT bins.

    Reduces by `sum(loss * mask) / mask.sum().clamp_min(1)`. Empty masks
    yield a zero scalar (no division by zero, no NaN).

    Args:
        student_logits: `[b, n, num_bins]`.
        plddt_bin_labels: `[b, n]` int64 bin indices.
        mask: `[b, n]` bool or float. Combined `orig_mask & plddt_mask` is
            the caller's responsibility.
        label_smoothing: passed through to `F.cross_entropy`.

    Returns:
        Scalar loss.
    """
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
    """Smooth-L1 between the softmax-weighted bin-center expected value and
    the continuous pLDDT target.

    The softmax and the dot-product with `bin_centers` are computed in fp32
    regardless of the input dtype (bf16 logits remain numerically stable).

    Args:
        student_logits: `[b, n, num_bins]`, any dtype.
        plddt_continuous: `[b, n]`, in `[0, bin_max]`.
        mask: `[b, n]`.
        bin_centers: `[num_bins]`, broadcast onto the logits' device.

    Returns:
        Scalar loss (fp32).
    """
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
    """`ce_weight * CE + smooth_l1_weight * SmoothL1_on_EV`.

    Returns the scalar total and a dict with the unscaled component losses
    so they can be logged independently.
    """
    loss_ce = masked_plddt_cross_entropy(
        student_logits, plddt_bin_labels, mask, label_smoothing=label_smoothing
    )
    loss_smooth_l1 = masked_smooth_l1_on_expected_value(
        student_logits, plddt_continuous, mask, bin_centers
    )
    total = ce_weight * loss_ce + smooth_l1_weight * loss_smooth_l1
    return total, {"loss_ce": loss_ce, "loss_smooth_l1": loss_smooth_l1}
