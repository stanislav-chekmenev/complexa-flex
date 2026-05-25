"""Directional PAE confidence head (PR-B Slice 2).

AF2-style directional predicted aligned error: `PAE(i, j)` is the
error in residue `j`'s position when the structure is aligned on
residue `i`'s frame. `PAE(i, j) != PAE(j, i)` in general; the head
therefore consumes the trunk-refined pair representation `z`
directionally, without the `(i, j) <-> (j, i)` symmetrisation that
symmetric pair heads (PDE, ipLDDT, ipTM) apply inside their own
`_predict`.

Bin convention: AF2 PAE standard — `num_bins=64`, `bin_width=0.5` A,
bin centers `[0.25, 0.75, ..., 31.75]` covering `[0, 32)` A.

AFDB integer-A label bias (read once, save your future self the
debugging hour): AFDB v4 PAE values are stored on disk as integer
Angstroms in `[0, 31]`. After binning with `bin_width=0.5`, an
integer-A label `v` lands in bin `2 * v` whose center is
`2 * v + 0.25` — i.e. the EV of a one-hot label is biased
`+0.25` A high relative to the true integer-A value. The SmoothL1
EV term reads `pae_continuous` (the float-A label) and cancels this
bias at training time; the CE term ignores it (the bin index is
exact). At the default `0.9 * CE + 0.1 * SmoothL1` split, the
stationary-point EV is `0.9 * (k + 0.25) + 0.1 * k = k + 0.225`,
so the residual model bias on validation MAE is approximately
`+0.225` A (not the much smaller `+0.025` A; the smaller estimate
ignored the dominant CE contribution), still well within the
noise floor of the AFDB labels themselves.
"""

from __future__ import annotations

import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef, SpearmanCorrCoef

from proteinfoundation.nn.confidence._losses import combined_pae_loss
from proteinfoundation.nn.confidence._metrics import (
    _labels_to_continuous,
    _logits_to_continuous,
    i_pae,
    interface_pair_mask,
    ipsae_family,
    iptm_energy_from_logits,
    iptm_from_logits,
    min_ipae,
    pae_accuracy,
    pae_ece,
    pae_ece_adaptive,
    pae_mae,
    pae_mae_stratified_by_distance,
    pae_mae_stratified_by_value,
    pearson_r,
    spearman_r,
)
from proteinfoundation.nn.confidence.base import (
    BaseConfidenceHead,
    ConfidenceTrunk,
    StageLiteral,
)
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("pae")
class PaeHead(BaseConfidenceHead):
    output_keys: tuple[str, ...] = ("pae_logits",)
    output_name_root: str = "pae"
    expected_trunk_eval_t: float = 0.99

    METRIC_CORRELATION_NAMES: tuple[str, ...] = (
        "i_pae",
        "min_ipae",
        "i_ptm",
        "i_ptm_energy",
        "avg_ipsae",
        "min_ipsae",
        "max_ipsae",
        "avg_ipsae_10",
        "min_ipsae_10",
        "max_ipsae_10",
    )

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_pae_bins: int = 64,
        bin_min: float = 0.0,
        bin_max: float = 32.0,
        ce_weight: float = 0.9,
        ev_weight: float = 0.1,
        label_smoothing: float = 0.0,
        num_bins_ece_adaptive: int = 15,
        d_in_pair_token: int | None = None,
        track_metric_correlations: bool = True,
    ) -> None:
        super().__init__(trunk=trunk, token_dim=token_dim, pair_repr_dim=pair_repr_dim)
        if num_pae_bins <= 0:
            raise ValueError(f"num_pae_bins must be positive, got {num_pae_bins}")
        if bin_max <= bin_min:
            raise ValueError(f"bin_max ({bin_max}) must exceed bin_min ({bin_min})")

        self.num_pae_bins = int(num_pae_bins)
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.ce_weight = float(ce_weight)
        self.ev_weight = float(ev_weight)
        self.label_smoothing = float(label_smoothing)
        self.num_bins_ece_adaptive = int(num_bins_ece_adaptive)

        self.d_in_pair_token = d_in_pair_token if d_in_pair_token is not None else pair_repr_dim
        self.logits_norm = nn.LayerNorm(self.d_in_pair_token)
        self.logits_linear = nn.Linear(self.d_in_pair_token, self.num_pae_bins)

        bin_width = (self.bin_max - self.bin_min) / self.num_pae_bins
        centers = torch.tensor(
            [self.bin_min + bin_width * (i + 0.5) for i in range(self.num_pae_bins)],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", centers, persistent=False)

        self.track_metric_correlations = bool(track_metric_correlations)
        if self.track_metric_correlations:
            self.val_metric_correlations = nn.ModuleDict(
                {
                    name: MetricCollection(
                        {
                            "mae": MeanAbsoluteError(),
                            "pearson": PearsonCorrCoef(),
                            "spearman": SpearmanCorrCoef(),
                        },
                        compute_groups=False,
                    )
                    for name in self.METRIC_CORRELATION_NAMES
                }
            )

    def _predict(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del s
        logits = self.logits_linear(self.logits_norm(z))
        pair_mask = (mask[:, None, :] & mask[:, :, None]).to(logits.dtype)
        logits = logits * pair_mask[..., None]
        return {"pae_logits": logits}

    def pae_ev_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax-weighted bin-center mean of the student's PAE logits.

        fp32 internally regardless of input dtype. Returns `(B, L, L)`
        in `[bin_min, bin_max)`.
        """
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def _pae_ev_from_labels(self, pae_bin: torch.Tensor) -> torch.Tensor:
        """Bin-center lookup for AFDB integer-A bin labels.

        Mirrors `_labels_to_continuous(pae_bin, self.bin_centers)` but lives
        on the head so callers (including the metric-correlation pipeline)
        use a single canonical source of GT EV without leaking the head's
        bin convention into the Lightning module.
        """
        return self.bin_centers.to(pae_bin.device, torch.float32)[pae_bin]

    def logits_to_expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Back-compat alias for `pae_ev_from_logits`. Do not use in new code."""
        return self.pae_ev_from_logits(logits)

    def update_metric_correlations(
        self,
        pae_ev_pred: torch.Tensor,
        pae_ev_gt: torch.Tensor,
        chain_idx: torch.Tensor,
        mask_eff: torch.Tensor,
    ) -> None:
        """Update the per-sample (pred, gt) metric-collection accumulators.

        Called from `compute_loss_and_metrics` at val stage. The GT and
        predicted EVs are computed once in fp32 by the caller; this method
        owns the per-sample metric extraction and the torchmetrics
        accumulator updates.
        """
        if not self.track_metric_correlations:
            return

        inter = interface_pair_mask(chain_idx, mask_eff)
        per_pred: dict[str, torch.Tensor] = {}
        per_gt: dict[str, torch.Tensor] = {}
        per_pred["i_pae"] = i_pae(pae_ev_pred, inter, reduce="per_sample")
        per_gt["i_pae"] = i_pae(pae_ev_gt, inter, reduce="per_sample")
        per_pred["min_ipae"] = min_ipae(pae_ev_pred, inter, reduce="per_sample")
        per_gt["min_ipae"] = min_ipae(pae_ev_gt, inter, reduce="per_sample")

        # iptm / iptm_energy need bin logits; synthesise sharp one-hot logits
        # from the EVs so the GT side flows through the same kernel and
        # inherits the per-sample d0(L) shape and the LSE energy reduction.
        # Magnitudes are chosen large enough to make softmax effectively
        # one-hot (within 1e-13) but small enough that the LSE energy stays
        # in fp32-representable territory (extreme +/-1e9 collapses energy
        # variance below fp32 resolution and breaks Pearson on i_ptm_energy).
        pred_bin = torch.bucketize(pae_ev_pred, self.bin_centers[:-1])
        gt_bin = torch.bucketize(pae_ev_gt, self.bin_centers[:-1])
        K = self.num_pae_bins
        pred_logits = torch.full(
            (*pred_bin.shape, K), -30.0, device=pred_bin.device, dtype=torch.float32
        )
        pred_logits.scatter_(-1, pred_bin[..., None], 30.0)
        gt_logits = torch.full(
            (*gt_bin.shape, K), -30.0, device=gt_bin.device, dtype=torch.float32
        )
        gt_logits.scatter_(-1, gt_bin[..., None], 30.0)
        per_pred["i_ptm"] = iptm_from_logits(
            pred_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_gt["i_ptm"] = iptm_from_logits(
            gt_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_pred["i_ptm_energy"] = iptm_energy_from_logits(
            pred_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_gt["i_ptm_energy"] = iptm_energy_from_logits(
            gt_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )

        ipsae_pred = ipsae_family(pae_ev_pred, chain_idx, mask_eff, reduce="per_sample")
        ipsae_gt = ipsae_family(pae_ev_gt, chain_idx, mask_eff, reduce="per_sample")
        for k in ipsae_pred:
            per_pred[k] = ipsae_pred[k]
            per_gt[k] = ipsae_gt[k]

        for name, mc in self.val_metric_correlations.items():
            pred_v = per_pred[name]
            gt_v = per_gt[name]
            keep = ~(torch.isnan(pred_v) | torch.isnan(gt_v))
            if not bool(keep.any()):
                continue
            mc.update(pred_v[keep].float(), gt_v[keep].float())

    def val_metric_correlations_compute_and_reset(self) -> dict[str, dict[str, float]]:
        """Compute and reset every collection. Empty collections yield NaN."""
        out: dict[str, dict[str, float]] = {}
        if not self.track_metric_correlations:
            return out
        for name, mc in self.val_metric_correlations.items():
            try:
                vals = mc.compute()
                out[name] = {k: float(v) for k, v in vals.items()}
            except Exception:
                out[name] = {
                    "mae": float("nan"),
                    "pearson": float("nan"),
                    "spearman": float("nan"),
                }
            mc.reset()
        return out

    def compute_loss_and_metrics(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask_eff: torch.Tensor,
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Explicit fp32 upcast: do not rely on autocast in mixed precision.
        # CE on bf16 logits is numerically unstable near sharp predictions;
        # the SmoothL1-on-EV path also softmaxes in fp32 internally.
        logits = out["pae_logits"].float()
        labels_bin = batch["pae_bin"]
        labels_cont = batch["pae_residue_pair"]
        centers = self.bin_centers

        chain_idx = batch.get("chain_idx")
        if chain_idx is not None and torch.is_tensor(chain_idx):
            assert chain_idx.shape[-1] == logits.shape[-3], (
                f"chain_idx axis-1 ({chain_idx.shape[-1]}) must equal the head's "
                f"residue axis ({logits.shape[-3]}); the dataset should align "
                f"chain_idx with the binder/trunk residue frame before reaching "
                f"the head."
            )

        ls = self.label_smoothing if stage == "train" else 0.0
        total, parts = combined_pae_loss(
            student_logits=logits,
            pae_bin_labels=labels_bin,
            pae_continuous=labels_cont,
            mask=mask_eff,
            bin_centers=centers,
            ce_weight=self.ce_weight,
            smooth_l1_weight=self.ev_weight,
            label_smoothing=ls,
        )
        log_dict: dict[str, torch.Tensor] = {
            "loss": total,
            "loss_ce": parts["loss_ce"],
            "loss_smooth_l1": parts["loss_smooth_l1"],
        }
        if stage != "train":
            log_dict["loss_total"] = total
            log_dict["pae_accuracy"] = pae_accuracy(logits, labels_bin, mask_eff)
            log_dict["pae_mae"] = pae_mae(logits, labels_bin, mask_eff, centers)
            pred_cont = _logits_to_continuous(logits, centers)
            target_cont = _labels_to_continuous(labels_bin, centers)
            log_dict["pearson_r"] = pearson_r(pred_cont, target_cont, mask_eff)
            log_dict["spearman_r"] = spearman_r(pred_cont, target_cont, mask_eff)
            log_dict["ece"] = pae_ece(logits, labels_bin, mask_eff)
            log_dict["ece_adaptive"] = pae_ece_adaptive(
                logits, labels_bin, mask_eff, num_bins_ece=self.num_bins_ece_adaptive
            )
            log_dict.update(
                pae_mae_stratified_by_value(logits, labels_bin, mask_eff, centers)
            )
            ca_coords = self._extract_ca_coords(batch)
            if ca_coords is not None:
                log_dict.update(
                    pae_mae_stratified_by_distance(
                        logits, labels_bin, mask_eff, centers, ca_coords
                    )
                )
            chain_idx = batch.get("chain_idx")
            if chain_idx is not None and torch.is_tensor(chain_idx):
                inter = interface_pair_mask(chain_idx, mask_eff)
                log_dict["i_pae"] = i_pae(pred_cont, inter)
                log_dict["min_ipae"] = min_ipae(pred_cont, inter)
                log_dict["i_ptm"] = iptm_from_logits(logits, mask_eff, inter, centers)
                log_dict["i_ptm_energy"] = iptm_energy_from_logits(
                    logits, mask_eff, inter, centers
                )
                log_dict.update(ipsae_family(pred_cont, chain_idx, mask_eff))
        return total, log_dict

    @staticmethod
    def _extract_ca_coords(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Surface CA coords for distance-stratified MAE.

        Order of preference: explicit `ca_coords`; the OpenFold-style
        atom37 `coords` tensor (CA is index 1); the nm-scaled `coords_nm`
        tensor. Returns `None` if no source is present.
        """
        if "ca_coords" in batch and torch.is_tensor(batch["ca_coords"]):
            return batch["ca_coords"]
        for key in ("coords", "coords_nm"):
            v = batch.get(key)
            if torch.is_tensor(v) and v.ndim == 4 and v.shape[-2] >= 2 and v.shape[-1] == 3:
                return v[..., 1, :]
        return None
