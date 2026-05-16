"""Input projections from trunk dims to head dims.

When `in_dim == out_dim`, both modules act as a parameter-free identity so
the default complexa-native dim contract (768, 256) incurs zero overhead.
PR-4's sidecar uses these when the head's dims diverge from the trunk's
(e.g. a 384/128 Boltz-style head over a 768/256 trunk).
"""

from __future__ import annotations

import torch
from torch import nn


class SeqProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.identity = in_dim == out_dim
        if not self.identity:
            self.norm = nn.LayerNorm(in_dim)
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return s * mask[..., None]
        return self.proj(self.norm(s)) * mask[..., None]


class PairProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.identity = in_dim == out_dim
        if not self.identity:
            self.norm = nn.LayerNorm(in_dim)
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None]
        if self.identity:
            return z * pair_mask
        return self.proj(self.norm(z)) * pair_mask
