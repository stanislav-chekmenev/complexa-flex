"""Back-compat re-exports for confidence metric helpers.

The implementations live under `proteinfoundation.nn.confidence._metrics`
since PR-B Slice 2 (concrete heads need to depend on them without
inverting the package layering against the sidecar). Existing callers
that import these names from `proteinfoundation.confidence.metrics` keep
working unchanged.
"""

from __future__ import annotations

from proteinfoundation.nn.confidence._metrics import (
    _labels_to_continuous,
    _logits_to_continuous,
    expected_calibration_error,
    expected_calibration_error_adaptive,
    pae_accuracy,
    pae_ece,
    pae_ece_adaptive,
    pae_mae,
    pae_mae_stratified_by_distance,
    pae_mae_stratified_by_value,
    pearson_r,
    plddt_accuracy,
    plddt_mae,
    plddt_mae_stratified,
    spearman_r,
)


__all__ = [
    "_labels_to_continuous",
    "_logits_to_continuous",
    "expected_calibration_error",
    "expected_calibration_error_adaptive",
    "pae_accuracy",
    "pae_ece",
    "pae_ece_adaptive",
    "pae_mae",
    "pae_mae_stratified_by_distance",
    "pae_mae_stratified_by_value",
    "pearson_r",
    "plddt_accuracy",
    "plddt_mae",
    "plddt_mae_stratified",
    "spearman_r",
]
