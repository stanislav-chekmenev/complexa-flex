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


def expected_calibration_error(
    logits: Tensor,
    labels_bin: Tensor,
    mask: Tensor,
    num_bins_ece: int = 10,
) -> Tensor:
    """Top-1-probability vs accuracy ECE with equal-width buckets in `[0, 1]`.

    Definition: `ECE = sum_b (n_b / n_total) * |acc_b - conf_b|`, where the
    sum is over `num_bins_ece` equal-width confidence buckets, `conf_b` is
    the mean top-1 probability in bucket `b`, `acc_b` is the fraction of
    correct top-1 predictions, and `n_b` is the count of valid (masked)
    residues falling in bucket `b`. Empty buckets are skipped (weight 0).

    Returns a scalar `Tensor` in `[0, 1]`.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    mask_f = mask.to(torch.float32)
    correct = (pred == labels_bin).to(torch.float32) * mask_f
    n_total = mask_f.sum().clamp_min(1.0)

    edges = torch.linspace(0.0, 1.0, num_bins_ece + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i in range(num_bins_ece):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins_ece - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        in_bin_f = in_bin.to(torch.float32) * mask_f
        n_b = in_bin_f.sum()
        if n_b.item() == 0.0:
            continue
        acc_b = (correct * in_bin_f).sum() / n_b
        conf_b = (conf * in_bin_f).sum() / n_b
        ece = ece + (n_b / n_total) * (acc_b - conf_b).abs()
    return ece


def expected_calibration_error_adaptive(
    logits: Tensor,
    labels_bin: Tensor,
    mask: Tensor,
    num_bins_ece: int = 15,
) -> Tensor:
    """Top-1 vs accuracy ECE with **equal-mass** confidence buckets.

    Bucket boundaries are top-1-probability quantiles taken on the masked
    residue distribution; this avoids the equal-width buckets' pathology
    where 99 % of probability mass lives in the top bucket. Ties on
    identical top-1 confidences are absorbed by the highest bucket via the
    closed upper edge of the final bin (`<=` rather than `<`).

    Returns a 0-dim fp32 tensor in `[0, 1]`; all-masked input returns 0.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    mask_b = mask.bool()
    mask_f = mask.to(torch.float32)
    correct = (pred == labels_bin).to(torch.float32) * mask_f
    n_total = mask_f.sum()
    if n_total.item() == 0.0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    conf_valid = conf[mask_b]
    q = torch.linspace(0.0, 1.0, num_bins_ece + 1, device=logits.device)
    edges = torch.quantile(conf_valid, q)
    edges[0] = 0.0
    edges[-1] = 1.0

    ece = torch.zeros((), device=logits.device, dtype=torch.float32)
    for i in range(num_bins_ece):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins_ece - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        in_bin_f = in_bin.to(torch.float32) * mask_f
        n_b = in_bin_f.sum()
        if n_b.item() == 0.0:
            continue
        acc_b = (correct * in_bin_f).sum() / n_b
        conf_b = (conf * in_bin_f).sum() / n_b
        ece = ece + (n_b / n_total) * (acc_b - conf_b).abs()
    return ece.to(torch.float32)


def reliability_diagram(
    logits: Tensor,
    labels_bin: Tensor,
    mask: Tensor,
    num_bins_ece: int = 10,
) -> Tensor:
    """Per-bucket `(conf, acc, count)` table for a top-1-probability reliability plot.

    Equal-width buckets over `[0, 1]` (same partition as
    `expected_calibration_error`). Returns an `[num_bins_ece, 3]` fp32
    tensor where row `b` is `(mean_conf_b, mean_acc_b, count_b)` and
    empty buckets are `(0, 0, 0)`.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    mask_f = mask.to(torch.float32)
    correct = (pred == labels_bin).to(torch.float32) * mask_f

    edges = torch.linspace(0.0, 1.0, num_bins_ece + 1, device=logits.device)
    out = torch.zeros((num_bins_ece, 3), device=logits.device, dtype=torch.float32)
    for i in range(num_bins_ece):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins_ece - 1:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf >= lo) & (conf < hi)
        in_bin_f = in_bin.to(torch.float32) * mask_f
        n_b = in_bin_f.sum()
        if n_b.item() == 0.0:
            continue
        conf_b = (conf * in_bin_f).sum() / n_b
        acc_b = (correct * in_bin_f).sum() / n_b
        out[i, 0] = conf_b
        out[i, 1] = acc_b
        out[i, 2] = n_b
    return out


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
