"""Pure-function metric helpers for confidence heads.

Lives under `nn.confidence` for the same layering reason as `_losses.py`.
The sidecar `proteinfoundation.confidence.metrics` re-exports these names
for back-compat with callers that already imported them under that path.

Includes:
    Shared shape-agnostic primitives — `_logits_to_continuous`,
      `_labels_to_continuous`, `pearson_r`, `spearman_r`,
      `expected_calibration_error`, `expected_calibration_error_adaptive`.
    pLDDT-specific — `plddt_accuracy`, `plddt_mae`, `plddt_mae_stratified`.
    PAE-specific — `pae_accuracy`, `pae_mae`, `pae_mae_stratified_by_value`,
      `pae_mae_stratified_by_distance`, `pae_ece`, `pae_ece_adaptive`.

PAE metrics consume a `[b, n, n]` mask and (where applicable) `[b, n, 3]`
CA coordinates. The metric API mirrors the pLDDT family — bin centers
are passed in by the head, not hard-coded.
"""

from __future__ import annotations

import torch
from torch import Tensor


__all__ = [
    "_logits_to_continuous",
    "_labels_to_continuous",
    "pearson_r",
    "spearman_r",
    "expected_calibration_error",
    "expected_calibration_error_adaptive",
    "plddt_accuracy",
    "plddt_mae",
    "plddt_mae_stratified",
    "pae_accuracy",
    "pae_mae",
    "pae_mae_stratified_by_value",
    "pae_mae_stratified_by_distance",
    "pae_ece",
    "pae_ece_adaptive",
    "interface_pair_mask",
    "i_pae",
    "min_ipae",
    "iptm_from_logits",
    "iptm_energy_from_logits",
    "ipsae_family",
]


def _logits_to_continuous(logits: Tensor, bin_centers: Tensor) -> Tensor:
    """Softmax-weighted bin-center mean. fp32 internally."""
    centers = bin_centers.to(logits.device, torch.float32)
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * centers).sum(dim=-1)


def _labels_to_continuous(labels: Tensor, bin_centers: Tensor) -> Tensor:
    centers = bin_centers.to(labels.device, torch.float32)
    return centers[labels]


def plddt_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
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
    pred_cont = _logits_to_continuous(logits, bin_centers)
    target_cont = _labels_to_continuous(labels, bin_centers)
    mask_f = mask.to(torch.float32)
    ae = (pred_cont - target_cont).abs() * mask_f
    return ae.sum() / mask_f.sum().clamp_min(1.0)


def pearson_r(pred_cont: Tensor, target_cont: Tensor, mask: Tensor) -> Tensor:
    """Per-protein Pearson r averaged across the batch.

    The implementation iterates per-protein over `pred_cont.shape[0]` and
    `mask`-selects valid entries; this works for both `(B, L)` (pLDDT) and
    `(B, L, L)` (PAE) shapes — the masked select flattens automatically.
    """
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
    """Equal-width top-1-confidence ECE in `[0, 1]`."""
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
    """Equal-mass top-1-confidence ECE in `[0, 1]`. Empty mask -> 0."""
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


def pae_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    """Masked top-1 bin accuracy over `(B, L, L)` pair positions."""
    preds = logits.argmax(dim=-1)
    mask_f = mask.to(torch.float32)
    correct = (preds == labels).to(torch.float32) * mask_f
    return correct.sum() / mask_f.sum().clamp_min(1.0)


def pae_mae(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
) -> Tensor:
    """Masked MAE between EV and label-bin-center over pair positions."""
    pred_cont = _logits_to_continuous(logits, bin_centers)
    target_cont = _labels_to_continuous(labels, bin_centers)
    mask_f = mask.to(torch.float32)
    ae = (pred_cont - target_cont).abs() * mask_f
    return ae.sum() / mask_f.sum().clamp_min(1.0)


def pae_mae_stratified_by_value(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
    buckets: tuple[tuple[float, float], ...] = ((0.0, 5.0), (5.0, 15.0), (15.0, 32.0)),
    bucket_names: tuple[str, ...] = ("v_lt5", "v_5_15", "v_ge15"),
) -> dict[str, Tensor]:
    """MAE per target-PAE-value bucket. Empty buckets yield finite `0.0`."""
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


def pae_mae_stratified_by_distance(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
    ca_coords: Tensor,
    dist_thresholds: tuple[float, float] = (8.0, 16.0),
) -> dict[str, Tensor]:
    """MAE per CA-distance bucket on the residue-pair grid.

    `ca_coords` has shape `(B, L, 3)`. Distance `d_ij = ||CA_i - CA_j||_2`
    forms the `(B, L, L)` distance map; three buckets are
    `d < t0`, `t0 <= d < t1`, `d >= t1`.
    """
    if len(dist_thresholds) != 2:
        raise ValueError(f"dist_thresholds must be 2 values, got {dist_thresholds}")
    t0, t1 = float(dist_thresholds[0]), float(dist_thresholds[1])

    pred_cont = _logits_to_continuous(logits, bin_centers)
    target_cont = _labels_to_continuous(labels, bin_centers)
    # Exclude diagonal — PAE_ii is trivially 0 and pulls the d_lt8 bucket toward 0.
    L = pred_cont.shape[-1]
    off_diag = ~torch.eye(L, dtype=torch.bool, device=mask.device)
    mask_f = mask.to(torch.float32) * off_diag.to(torch.float32)
    ae = (pred_cont - target_cont).abs()

    diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]
    d_ij = diff.pow(2).sum(dim=-1).clamp_min(0.0).sqrt()

    bucket_specs: tuple[tuple[str, Tensor], ...] = (
        ("d_lt8", d_ij < t0),
        ("d_8_16", (d_ij >= t0) & (d_ij < t1)),
        ("d_ge16", d_ij >= t1),
    )
    out: dict[str, Tensor] = {}
    for name, bucket_mask in bucket_specs:
        active = bucket_mask.to(torch.float32) * mask_f
        denom = active.sum().clamp_min(1.0)
        out[name] = (ae * active).sum() / denom
    return out


def interface_pair_mask(chain_idx: Tensor, mask_eff: Tensor) -> Tensor:
    """Pairs that cross a chain boundary AND are valid under `mask_eff`.

    Args:
        chain_idx: `[B, L]` integer chain identifiers (monomers carry a
            single id, dimers carry two, etc.).
        mask_eff: `[B, L, L]` already-AND-ed pair validity (float or bool).

    Returns:
        `[B, L, L]` bool. True iff `chain_idx[..., i] != chain_idx[..., j]`
        AND `mask_eff[..., i, j]` is true.
    """
    cross_chain = chain_idx[..., None, :] != chain_idx[..., :, None]
    return cross_chain & mask_eff.bool()


def i_pae(
    pae_ev: Tensor,
    interface_mask: Tensor,
    *,
    reduce: str = "batch_mean",
) -> Tensor:
    """Mean PAE expected-value over the cross-chain interface.

    Args:
        pae_ev: `[B, L, L]` softmax-weighted bin-center mean of the
            student's PAE logits (continuous EV in Angstroms).
        interface_mask: `[B, L, L]` cross-chain validity mask.
        reduce: `"batch_mean"` (default, scalar) or `"per_sample"`
            (returns `[B]` with NaN at samples-without-mass).

    Returns:
        Either a scalar mean over samples that carry at least one
        interface pair, or a `[B]` per-sample tensor with NaN sentinels
        on no-mass samples.
    """
    mask_f = interface_mask.to(torch.float32)
    sample_denom = mask_f.sum(dim=(-2, -1))
    sample_has_mass = sample_denom > 0
    per_sample = (pae_ev.float() * mask_f).sum(dim=(-2, -1)) / sample_denom.clamp_min(1.0)
    if reduce == "per_sample":
        return torch.where(sample_has_mass, per_sample, torch.full_like(per_sample, float("nan")))
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample = torch.where(sample_has_mass, per_sample, torch.zeros_like(per_sample))
    n_valid_samples = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample.sum() / n_valid_samples).to(torch.float32)


def min_ipae(
    pae_ev: Tensor,
    interface_mask: Tensor,
    *,
    reduce: str = "batch_mean",
) -> Tensor:
    """Min over interface rows of the per-row EV mean.

    Args:
        pae_ev: `[B, L, L]` continuous PAE expected value.
        interface_mask: `[B, L, L]` cross-chain validity mask.
        reduce: `"batch_mean"` (default, scalar) or `"per_sample"`
            (returns `[B]` with NaN at samples-without-mass).

    Returns:
        Either a scalar mean over samples that carry at least one
        interface row, or a `[B]` per-sample tensor with NaN sentinels.
        Rows with no interface mass are excluded via sentinel
        substitution before the row-wise min.
    """
    mask_f = interface_mask.to(torch.float32)
    row_denom = mask_f.sum(dim=-1)
    row_has_mass = row_denom > 0
    per_row = (pae_ev.float() * mask_f).sum(dim=-1) / row_denom.clamp_min(1.0)
    per_row_for_min = torch.where(
        row_has_mass, per_row, torch.full_like(per_row, float("inf"))
    )
    sample_has_any_row = row_has_mass.any(dim=-1)
    per_sample_min = per_row_for_min.min(dim=-1).values
    if reduce == "per_sample":
        return torch.where(
            sample_has_any_row, per_sample_min, torch.full_like(per_sample_min, float("nan"))
        )
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample_min = torch.where(
        sample_has_any_row, per_sample_min, torch.zeros_like(per_sample_min)
    )
    n_valid_samples = sample_has_any_row.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample_min.sum() / n_valid_samples).to(torch.float32)


def _per_row_tm_score(
    logits: Tensor,
    row_mask: Tensor,
    n_valid: Tensor,
    bin_centers: Tensor,
    *,
    d0_clip_min: int,
) -> Tensor:
    """Per-row TM-score-style aggregate using per-sample `d0`.

    `n_valid` is `[B]`; `d0` therefore varies per sample. The broadcast
    pattern is `d0[:, None]` for the K axis (per-sample weighting) and
    `w[:, None, None, :]` to multiply against `[B, L, L, K]` softmax
    probabilities. The non-obviousness is exactly that broadcast: a
    naive `bin_centers / d0` (both 1D) would collapse the batch axis.
    """
    centers = bin_centers.to(logits.device, torch.float32)
    n_eff = n_valid.to(torch.float32).clamp_min(float(d0_clip_min))
    d0 = 1.24 * (n_eff - 15.0).clamp_min(0.0).pow(1.0 / 3.0) - 1.8
    w = 1.0 / (1.0 + (centers[None, :] / d0[:, None]).pow(2))
    probs = torch.softmax(logits.float(), dim=-1)
    score_ij = (probs * w[:, None, None, :]).sum(dim=-1)
    row_mask_f = row_mask.to(torch.float32)
    row_denom = row_mask_f.sum(dim=-1).clamp_min(1.0)
    return (score_ij * row_mask_f).sum(dim=-1) / row_denom


def _per_sample_n_valid(mask_eff: Tensor) -> Tensor:
    """Residue count per sample inferred from the pair mask.

    A residue is "valid" if it participates in any valid pair; for the
    standard `mask_eff = m[:, None] & m[:, :, None]` construction this
    is identical to `m.sum(-1)`.
    """
    mb = mask_eff.bool()
    assert torch.equal(mb, mb.transpose(-1, -2)), (
        "mask_eff must be symmetric in the standard residue-pair construction"
    )
    return mb.any(dim=-1).sum(dim=-1)


def iptm_from_logits(
    logits: Tensor,
    mask_eff: Tensor,
    interface_mask: Tensor,
    bin_centers: Tensor,
    *,
    d0_clip_min: int = 19,
    reduce: str = "batch_mean",
) -> Tensor:
    """ipTM = mean over samples of (max over interface rows of per-row TM score)."""
    n_valid = _per_sample_n_valid(mask_eff)
    per_row = _per_row_tm_score(
        logits, interface_mask, n_valid, bin_centers, d0_clip_min=d0_clip_min
    )
    row_has_mass = interface_mask.to(torch.float32).sum(dim=-1) > 0
    per_row_masked = torch.where(
        row_has_mass, per_row, torch.full_like(per_row, float("-inf"))
    )
    sample_has_any_row = row_has_mass.any(dim=-1)
    per_sample_max = per_row_masked.max(dim=-1).values
    if reduce == "per_sample":
        return torch.where(
            sample_has_any_row, per_sample_max, torch.full_like(per_sample_max, float("nan"))
        )
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample_max = torch.where(
        sample_has_any_row, per_sample_max, torch.zeros_like(per_sample_max)
    )
    n_valid_samples = sample_has_any_row.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample_max.sum() / n_valid_samples).to(torch.float32)


def iptm_energy_from_logits(
    logits: Tensor,
    mask_eff: Tensor,
    interface_mask: Tensor,
    bin_centers: Tensor,
    *,
    tm_lambda: float = 1.0,
    d0_clip_min: int = 19,
    reduce: str = "batch_mean",
) -> Tensor:
    """Log-sum-exp energy variant of ipTM. Returns per-batch mean energy.

    For each pair, `energy_ij = -logsumexp(logits_ij + lambda * log w)`.
    Averaged over the interface mask per-sample, then across samples
    that have at least one interface pair.
    """
    centers = bin_centers.to(logits.device, torch.float32)
    n_valid = _per_sample_n_valid(mask_eff)
    n_eff = n_valid.to(torch.float32).clamp_min(float(d0_clip_min))
    d0 = 1.24 * (n_eff - 15.0).clamp_min(0.0).pow(1.0 / 3.0) - 1.8
    w = 1.0 / (1.0 + (centers[None, :] / d0[:, None]).pow(2))
    log_w = w.log()
    weighted_logits = logits.float() + tm_lambda * log_w[:, None, None, :]
    pos_energy = -torch.logsumexp(weighted_logits, dim=-1)
    mask_f = interface_mask.to(torch.float32)
    sample_denom = mask_f.sum(dim=(-2, -1))
    per_sample = (pos_energy * mask_f).sum(dim=(-2, -1)) / sample_denom.clamp_min(1.0)
    sample_has_mass = sample_denom > 0
    if reduce == "per_sample":
        return torch.where(sample_has_mass, per_sample, torch.full_like(per_sample, float("nan")))
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample = torch.where(sample_has_mass, per_sample, torch.zeros_like(per_sample))
    n_valid_samples = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample.sum() / n_valid_samples).to(torch.float32)


def _calc_d0_colabdesign(L: Tensor) -> Tensor:
    """colabdesign-compatible per-row d0 from filtered count.

    Mirrors `calc_d0` at community_models/colabdesign/af/loss.py:288-291:
        L_eff = clamp_min(L, 27)
        d0    = 1.24 * (L_eff - 15) ** (1/3) - 1.8
        d0    = clamp_min(d0, 1.0)
    """
    L_eff = L.to(torch.float32).clamp_min(27.0)
    d0 = 1.24 * (L_eff - 15.0).pow(1.0 / 3.0) - 1.8
    return d0.clamp_min(1.0)


def _ipsae_directional(
    pae_ev: Tensor,
    src_id: Tensor,
    tgt_id: Tensor,
    pae_cutoff: float,
) -> tuple[Tensor, Tensor]:
    """One direction of the colabdesign ipSAE kernel.

    Returns `(per_sample, has_mass)` where:
        per_sample: `[B]` directional ipSAE score (mean over filtered
            row max — colabdesign takes `mean_tm.max()` per sample).
        has_mass: `[B]` bool, True iff this direction has any
            cutoff-passing inter-chain pair.

    Filter `pae_mask_2d = src[:, :, None] * tgt[:, None, :] * (pae < cutoff)`
    matches colabdesign's `mask_1b[:, None] * mask_1d[None, :] * (p < cutoff)`.
    """
    pae = pae_ev.float()
    src_f = src_id.to(torch.float32)
    tgt_f = tgt_id.to(torch.float32)
    pae_mask_2d = src_f[:, :, None] * tgt_f[:, None, :] * (pae < pae_cutoff).to(torch.float32)
    row_count = pae_mask_2d.sum(dim=-1)
    d0 = _calc_d0_colabdesign(row_count)
    tm_term = 1.0 / (1.0 + pae.pow(2) / d0[..., None].pow(2))
    mean_tm = (pae_mask_2d * tm_term).sum(dim=-1) / (row_count + 1e-8)
    per_sample = mean_tm.max(dim=-1).values
    has_mass = pae_mask_2d.sum(dim=(-2, -1)) > 0
    return per_sample, has_mass


def ipsae_family(
    pae_ev: Tensor,
    chain_idx: Tensor,
    mask_eff: Tensor,
    *,
    pae_cutoffs: tuple[float, float] = (15.0, 10.0),
    reduce: str = "batch_mean",
) -> dict[str, Tensor]:
    """colabdesign-compatible ipSAE family.

    Mirrors `get_ipsae_loss` at community_models/colabdesign/af/loss.py:314-339.
    For each PAE cutoff (base = 15.0 A for AF2, _10 = 10.0 A for AF3/Boltz):
        1. Build the cutoff-filtered inter-chain mask in both directions
           (B->A: tgt rows x src cols; A->B: swapped).
        2. Per row, count filtered columns, compute `d0` via
           `_calc_d0_colabdesign` (L>=27 then d0>=1.0 clip).
        3. TM kernel: `tm = 1 / (1 + pae^2 / d0^2)`, weighted-mean per
           row over filtered columns with `+ 1e-8` additive denominator
           (colabdesign convention, not `clamp_min`).
        4. Per direction, take `mean_tm.max()` over rows; aggregate the
           two directions per sample as `{min, max, avg}`.
        5. Batch reduction averages over samples with mass (i.e. samples
           with at least one cutoff-passing inter-chain pair in either
           direction); empty samples contribute zero and are excluded
           from the denominator.

    Dimer convention: Teddymer dimers always have `chain_idx in {0, 1}`.
    Binder := chain 0, target := chain 1. The metric is symmetric in
    the chain assignment because the {min, max, avg} aggregation over
    the two directions is symmetric — swapping 0<->1 leaves the output
    unchanged.

    Args:
        pae_ev: `[B, L, L]` student continuous PAE expected value.
        chain_idx: `[B, L]` integer chain identifiers.
        mask_eff: `[B, L, L]` already-AND-ed pair validity mask. Combined
            with the chain split to drop padded residues from the chain
            indicator vectors.
        pae_cutoffs: `(base, _10)` Angstroms. AF2 default is 15.0;
            AF3/Boltz use 10.0 (colabdesign comment).

    Returns:
        `{avg_ipsae, min_ipsae, max_ipsae, avg_ipsae_10, min_ipsae_10,
         max_ipsae_10}`, each scalar.
    """
    if reduce not in ("batch_mean", "per_sample"):
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    res_valid = mask_eff.bool().any(dim=-1).to(torch.float32)
    binder_id = (chain_idx == 0).to(torch.float32) * res_valid
    target_id = (chain_idx == 1).to(torch.float32) * res_valid

    out: dict[str, Tensor] = {}
    base_cutoff, ten_cutoff = float(pae_cutoffs[0]), float(pae_cutoffs[1])
    for cutoff, suffix in ((base_cutoff, ""), (ten_cutoff, "_10")):
        ipsae_ba, mass_ba = _ipsae_directional(pae_ev, target_id, binder_id, cutoff)
        ipsae_ab, mass_ab = _ipsae_directional(pae_ev, binder_id, target_id, cutoff)
        sample_has_mass = mass_ba | mass_ab

        min_per = torch.minimum(ipsae_ab, ipsae_ba)
        max_per = torch.maximum(ipsae_ab, ipsae_ba)
        avg_per = 0.5 * (ipsae_ab + ipsae_ba)

        if reduce == "per_sample":
            nan = torch.full_like(avg_per, float("nan"))
            out[f"avg_ipsae{suffix}"] = torch.where(sample_has_mass, avg_per, nan)
            out[f"min_ipsae{suffix}"] = torch.where(sample_has_mass, min_per, nan)
            out[f"max_ipsae{suffix}"] = torch.where(sample_has_mass, max_per, nan)
            continue

        zeros = torch.zeros_like(avg_per)
        min_per = torch.where(sample_has_mass, min_per, zeros)
        max_per = torch.where(sample_has_mass, max_per, zeros)
        avg_per = torch.where(sample_has_mass, avg_per, zeros)

        n_valid = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
        out[f"avg_ipsae{suffix}"] = (avg_per.sum() / n_valid).to(torch.float32)
        out[f"min_ipsae{suffix}"] = (min_per.sum() / n_valid).to(torch.float32)
        out[f"max_ipsae{suffix}"] = (max_per.sum() / n_valid).to(torch.float32)
    return out


def pae_ece(
    logits: Tensor,
    labels_bin: Tensor,
    mask: Tensor,
    num_bins_ece: int = 10,
) -> Tensor:
    """Pair-level equal-width ECE. Thin wrapper over the shared implementation."""
    return expected_calibration_error(logits, labels_bin, mask, num_bins_ece=num_bins_ece)


def pae_ece_adaptive(
    logits: Tensor,
    labels_bin: Tensor,
    mask: Tensor,
    num_bins_ece: int = 15,
) -> Tensor:
    """Pair-level equal-mass ECE. Thin wrapper over the shared implementation."""
    return expected_calibration_error_adaptive(logits, labels_bin, mask, num_bins_ece=num_bins_ece)
