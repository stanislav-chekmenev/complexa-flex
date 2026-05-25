"""Quality-graft-style stack of Boltz-1 PairformerLayer's.

Thin wrapper that owns N stacked `PairformerLayer` instances (from the
vendored Boltz-1 slice) and re-applies the residue mask after each
layer. Used as the trainable backbone of `MultiHeadConfidence` together
with the upstream `AdaptorModule`.

The layers are randomly initialised — no Boltz-1 checkpoint loading
(per design spec section 3 non-goals).

Constructor exposes `s_dim` / `z_dim` even though the vendored
`PairformerLayer` uses `token_s` / `token_z`; the rename is purely at
the call site.
"""

from __future__ import annotations

import torch
from torch import nn

from community_models.boltz.model.modules.pairformer import PairformerLayer


class QgPairformerStack(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                PairformerLayer(token_s=s_dim, token_z=z_dim, num_heads=num_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.to(s.dtype) if mask.dtype != s.dtype else mask
        pair_mask = mask_f[:, :, None] * mask_f[:, None, :]
        for layer in self.layers:
            s, z = layer(s=s, z=z, mask=mask_f, pair_mask=pair_mask)
            s = s * mask_f[..., None]
            z = z * pair_mask[..., None]
        return s, z
