"""Per-residue pLDDT confidence head.

Final `LayerNorm` over the trunk-refined sequence representation followed
by a `Linear(token_dim, num_plddt_bins)` projection. The companion
`logits_to_expected_value` reduces a logits tensor to the
softmax-weighted bin-center mean in fp32 -- the quantity downstream
filters and SMC reward models consume.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.nn.confidence._losses import combined_plddt_loss
from proteinfoundation.nn.confidence._metrics import (
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
from proteinfoundation.nn.confidence.base import (
    BaseConfidenceHead,
    ConfidenceTrunk,
    StageLiteral,
)
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("plddt")
class PLDDTHead(BaseConfidenceHead):
    output_keys: tuple[str, ...] = ("plddt_logits",)
    output_name_root: str = "plddt"

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_plddt_bins: int = 50,
        bin_min: float = 0.0,
        bin_max: float = 100.0,
        ce_weight: float = 0.9,
        ev_weight: float = 0.1,
        label_smoothing: float = 0.0,
        num_bins_ece_adaptive: int = 15,
    ) -> None:
        super().__init__(trunk=trunk, token_dim=token_dim, pair_repr_dim=pair_repr_dim)
        if num_plddt_bins <= 0:
            raise ValueError(f"num_plddt_bins must be positive, got {num_plddt_bins}")
        if bin_max <= bin_min:
            raise ValueError(f"bin_max ({bin_max}) must exceed bin_min ({bin_min})")

        self.num_plddt_bins = num_plddt_bins
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.ce_weight = float(ce_weight)
        self.ev_weight = float(ev_weight)
        self.label_smoothing = float(label_smoothing)
        self.num_bins_ece_adaptive = int(num_bins_ece_adaptive)

        self.logits_norm = nn.LayerNorm(token_dim)
        self.logits_linear = nn.Linear(token_dim, num_plddt_bins)

        bin_width = (self.bin_max - self.bin_min) / num_plddt_bins
        centers = torch.tensor(
            [self.bin_min + bin_width * (i + 0.5) for i in range(num_plddt_bins)],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", centers, persistent=False)

    def _predict(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del z
        logits = self.logits_linear(self.logits_norm(s))
        logits = logits * mask[..., None]
        return {"plddt_logits": logits}

    def logits_to_expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax-weighted bin-center mean. fp32 internally."""
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def compute_loss_and_metrics(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask_eff: torch.Tensor,
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = out["plddt_logits"]
        labels_bin = batch["plddt_bin"]
        labels_cont = batch["plddt_residue"]
        centers = self.bin_centers

        ls = self.label_smoothing if stage == "train" else 0.0
        total, parts = combined_plddt_loss(
            student_logits=logits,
            plddt_bin_labels=labels_bin,
            plddt_continuous=labels_cont,
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
            log_dict["plddt_accuracy"] = plddt_accuracy(logits, labels_bin, mask_eff)
            log_dict["plddt_mae"] = plddt_mae(logits, labels_bin, mask_eff, centers)
            pred_cont = _logits_to_continuous(logits, centers)
            target_cont = _labels_to_continuous(labels_bin, centers)
            log_dict["pearson_r"] = pearson_r(pred_cont, target_cont, mask_eff)
            log_dict["spearman_r"] = spearman_r(pred_cont, target_cont, mask_eff)
            log_dict["ece"] = expected_calibration_error(logits, labels_bin, mask_eff)
            log_dict["ece_adaptive"] = expected_calibration_error_adaptive(
                logits, labels_bin, mask_eff, num_bins_ece=self.num_bins_ece_adaptive
            )
            log_dict.update(plddt_mae_stratified(logits, labels_bin, mask_eff, centers))
        return total, log_dict
