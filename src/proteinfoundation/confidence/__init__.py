"""Confidence-distillation package.

PR-2 shipped `plddt_to_bin`. PR-4 adds the masked losses, validation
metrics, and the sidecar Lightning module used to distil AF2 pLDDT into
the trainable `PLDDTHead` produced by PR-3.
"""

from __future__ import annotations

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.confidence.losses import (
    combined_plddt_loss,
    masked_plddt_cross_entropy,
    masked_smooth_l1_on_expected_value,
    plddt_to_bin,
)
from proteinfoundation.confidence.metrics import (
    expected_calibration_error,
    expected_calibration_error_adaptive,
    pearson_r,
    plddt_accuracy,
    plddt_mae,
    plddt_mae_stratified,
    reliability_diagram,
    spearman_r,
)

__all__ = [
    "ConfidenceDistillationModule",
    "combined_plddt_loss",
    "masked_plddt_cross_entropy",
    "masked_smooth_l1_on_expected_value",
    "plddt_to_bin",
    "expected_calibration_error",
    "expected_calibration_error_adaptive",
    "pearson_r",
    "plddt_accuracy",
    "plddt_mae",
    "plddt_mae_stratified",
    "reliability_diagram",
    "spearman_r",
]
