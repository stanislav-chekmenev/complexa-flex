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
from torch.utils.checkpoint import checkpoint

from community_models.boltz.model.modules.pairformer import PairformerLayer


class QgPairformerStack(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
        chunk_size_tri_attn: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.num_heads = num_heads
        # Chunk the L-dimension of triangle-attention softmax to cap the
        # O(L^3 * h) per-layer peak — at L>=500 in bf16 a single
        # `tri_att_end` softmax materialises >2 GiB (job 57622 OOM site).
        # None = boltz default (no chunking in training).
        self.chunk_size_tri_attn = chunk_size_tri_attn
        # Reentrant gradient checkpointing wraps each PairformerLayer's
        # forward, dropping per-layer activations and recomputing them on
        # backward. Needed at L>=500 with n_layers>=6 because residual z
        # streams accumulate ~O(n_layers * L^2 * z_dim) activations even
        # after triangle-attention chunking (job 57623 OOM site).
        # `find_unused_parameters=True` + `static_graph=True` on the
        # DDPStrategy already handles the reentrant-ckpt DDP interaction
        # (CLAUDE.md confidence-distill subsystem note).
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                PairformerLayer(token_s=s_dim, token_z=z_dim, num_heads=num_heads)
                for _ in range(n_layers)
            ]
        )

    def _layer_call(
        self,
        layer: PairformerLayer,
        s: torch.Tensor,
        z: torch.Tensor,
        mask_f: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer(
            s=s,
            z=z,
            mask=mask_f,
            pair_mask=pair_mask,
            chunk_size_tri_attn=self.chunk_size_tri_attn,
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
            if self.gradient_checkpointing and self.training and s.requires_grad:
                s, z = checkpoint(
                    self._layer_call,
                    layer,
                    s,
                    z,
                    mask_f,
                    pair_mask,
                    use_reentrant=True,
                )
            else:
                s, z = self._layer_call(layer, s, z, mask_f, pair_mask)
            s = s * mask_f[..., None]
            z = z * pair_mask[..., None]
        return s, z
