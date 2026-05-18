"""Back-compat re-exports for confidence loss helpers.

The implementations live under `proteinfoundation.nn.confidence._losses`
since PR-B Slice 2 (concrete heads need to depend on them without
inverting the package layering against the sidecar). Existing callers
that import these names from `proteinfoundation.confidence.losses` keep
working unchanged.
"""

from __future__ import annotations

from proteinfoundation.nn.confidence._losses import (
    MultiHeadLoss,
    _mask_reduce,
    combined_pae_loss,
    combined_plddt_loss,
    masked_pae_cross_entropy,
    masked_plddt_cross_entropy,
    masked_smooth_l1_on_expected_value,
    masked_smooth_l1_on_pae_expected_value,
    pae_to_bin,
    plddt_to_bin,
)


__all__ = [
    "MultiHeadLoss",
    "_mask_reduce",
    "combined_pae_loss",
    "combined_plddt_loss",
    "masked_pae_cross_entropy",
    "masked_plddt_cross_entropy",
    "masked_smooth_l1_on_expected_value",
    "masked_smooth_l1_on_pae_expected_value",
    "pae_to_bin",
    "plddt_to_bin",
]
