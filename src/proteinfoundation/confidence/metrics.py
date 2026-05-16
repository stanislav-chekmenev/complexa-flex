"""Validation metrics for pLDDT distillation.

Scale-agnostic: every metric that needs a continuous pLDDT value takes
`bin_centers` explicitly. The AF2 `[0, 100]` scale is encoded in the head's
`bin_centers` buffer, not hard-coded here.

Per-protein Pearson and Spearman correlations are averaged across the
batch with a per-protein guard for `numel < 3` and zero-variance series.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _logits_to_continuous(logits: Tensor, bin_centers: Tensor) -> Tensor:
    """Softmax-weighted bin-center mean. fp32 internally."""
    centers = bin_centers.to(logits.device, torch.float32)
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * centers).sum(dim=-1)


def _labels_to_continuous(labels: Tensor, bin_centers: Tensor) -> Tensor:
    centers = bin_centers.to(labels.device, torch.float32)
    return centers[labels]


def plddt_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Masked top-1 bin accuracy averaged over valid residues."""
    preds = logits.argmax(dim=-1)
    mask_f = mask.to(torch.float32)
    correct = (preds == labels).to(torch.float32) * mask_f
    return correct.sum() / mask_f.sum().clamp_min(1.0)


def plddt_mae(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
) -> Tensor:
    """Masked MAE between expected pLDDT and label-bin-center pLDDT.

    `bin_centers` carries the scale (e.g. `[1, 3, ..., 99]` for AF2
    `[0, 100]`), so the returned MAE is in the *same* units.
    """
    pred_cont = _logits_to_continuous(logits, bin_centers)
    target_cont = _labels_to_continuous(labels, bin_centers)
    mask_f = mask.to(torch.float32)
    ae = (pred_cont - target_cont).abs() * mask_f
    return ae.sum() / mask_f.sum().clamp_min(1.0)


def pearson_r(pred_cont: Tensor, target_cont: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Pearson r averaged across the batch."""
    batch_size = pred_cont.shape[0]
    rs: list[Tensor] = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred_cont[i][m]
        t = target_cont[i][m]
        if p.numel() < 3:
            continue
        p_c = p - p.mean()
        t_c = t - t.mean()
        num = (p_c * t_c).sum()
        den = (p_c.pow(2).sum() * t_c.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred_cont.device)
    return torch.stack(rs).mean()


def _rank(x: Tensor) -> Tensor:
    sorted_indices = x.argsort()
    ranks = torch.empty_like(x)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
    return ranks


def spearman_r(pred_cont: Tensor, target_cont: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Spearman rank correlation averaged across the batch."""
    batch_size = pred_cont.shape[0]
    rs: list[Tensor] = []
    for i in range(batch_size):
        m = mask[i].bool()
        p = pred_cont[i][m]
        t = target_cont[i][m]
        if p.numel() < 3:
            continue
        p_r = _rank(p)
        t_r = _rank(t)
        p_c = p_r - p_r.mean()
        t_c = t_r - t_r.mean()
        num = (p_c * t_c).sum()
        den = (p_c.pow(2).sum() * t_c.pow(2).sum()).sqrt()
        if den < 1e-8:
            continue
        rs.append(num / den)
    if not rs:
        return torch.tensor(0.0, device=pred_cont.device)
    return torch.stack(rs).mean()


def plddt_mae_stratified(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
    buckets: tuple[tuple[float, float], ...] = (
        (0.0, 50.0),
        (50.0, 70.0),
        (70.0, 90.0),
        (90.0, 100.0),
    ),
    bucket_names: tuple[str, ...] = (
        "mae_lt50",
        "mae_50_70",
        "mae_70_90",
        "mae_ge90",
    ),
) -> dict[str, Tensor]:
    """MAE per target-pLDDT bucket. Empty buckets yield finite `0.0`."""
    if len(buckets) != len(bucket_names):
        raise ValueError("buckets and bucket_names must have the same length")
    pred_cont = _logits_to_continuous(logits, bin_centers)
    target_cont = _labels_to_continuous(labels, bin_centers)
    mask_f = mask.to(torch.float32)
    ae = (pred_cont - target_cont).abs()

    out: dict[str, Tensor] = {}
    last_idx = len(buckets) - 1
    for i, ((lo, hi), name) in enumerate(zip(buckets, bucket_names)):
        if i == last_idx:
            bucket_mask = (target_cont >= lo) & (target_cont <= hi)
        else:
            bucket_mask = (target_cont >= lo) & (target_cont < hi)
        active = bucket_mask.to(torch.float32) * mask_f
        denom = active.sum().clamp_min(1.0)
        out[name] = (ae * active).sum() / denom
    return out
