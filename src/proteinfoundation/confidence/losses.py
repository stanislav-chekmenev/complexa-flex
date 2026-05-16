"""Confidence-head losses and helpers.

PR-2 ships only `plddt_to_bin`. Cross-entropy and SmoothL1 losses on the
expected bin value arrive in PR-4 alongside the head's training loop.
"""

from __future__ import annotations

import torch


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
