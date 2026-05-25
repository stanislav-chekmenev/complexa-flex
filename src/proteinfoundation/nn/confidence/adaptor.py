"""Quality-graft-style adaptor.

Adapts complexa trunk intermediates `(trunk_seqs[B,n,768],
trunk_pair[B,n,n,256], local_latents[B,n,8], ca_coords[B,n,3])` into
Boltz-1 confidence-head input dims `(s=384, z=128)`.

Architecture mirrors `quality_graft.models.adaptor.AdaptorModule` with
two simplifications:

- `source_mode` and the decoder-fusion path are dropped; complexa has
  no decoder analogue in the confidence-distill sidecar.
- `AttentionPairBias` is imported from the vendored Boltz-1 slice at
  `community_models.boltz.model.layers.attention`.

LayerNorm-bias-leak invariant. Every projection into a masked sequence
representation is followed by `* mask[..., None]` (for s) or
`* mask[:, :, None, None] * mask[:, None, :, None]` (for z) AFTER the
LayerNorm — so drifted `LN.bias != 0` cannot contaminate valid
positions through padded ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from community_models.boltz.model.layers.attention import AttentionPairBias


class AdaptorAttentionBlock(nn.Module):
    """Pair-biased self-attention block in the Boltz-1 dim space.

    `AttentionPairBias.forward(s, z, mask)` returns the post-attention
    `s` (not a delta), so we wrap it in an explicit residual add to
    match the upstream `quality_graft.models.adaptor.AdaptorAttentionBlock`
    semantics (`s = s + self.attn(s=s, z=z, mask=mask)`). The block does
    not update `z` via attention; only the residual `z_mlp` does.

    All output projections (`attn.proj_o`, `s_mlp[1]`, `z_mlp[1]`) are
    zero-initialised so the block starts as a near-identity residual.
    """

    def __init__(
        self,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
    ) -> None:
        super().__init__()

        self.attn = AttentionPairBias(
            c_s=s_dim,
            c_z=z_dim,
            num_heads=num_heads,
            initial_norm=True,
        )
        self.silu = nn.SiLU()

        self.s_mlp = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim, bias=False),
        )
        self.z_mlp = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, z_dim, bias=False),
        )

        nn.init.zeros_(self.s_mlp[1].weight)
        nn.init.zeros_(self.z_mlp[1].weight)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = s + self.attn(s=s, z=z, mask=mask)
        s = s + self.silu(self.s_mlp(s))
        z = z + self.silu(self.z_mlp(z))

        s = s * mask[..., None]
        z = z * mask[:, :, None, None] * mask[:, None, :, None]
        return s, z


class AdaptorModule(nn.Module):
    """Adapt complexa trunk intermediates to Boltz-1 confidence-head dims.

    Output: `(s[B,n,target_s_dim], z[B,n,n,target_z_dim])`. Cα coordinates
    enter via a one-hot pairwise distogram added to `z` after the pair
    projection.
    """

    def __init__(
        self,
        trunk_dim: int = 768,
        pair_dim: int = 256,
        latent_dim: int = 8,
        target_s_dim: int = 384,
        target_z_dim: int = 128,
        n_attn_layers: int = 1,
        num_heads: int = 16,
        ca_pair_dist_min: float = 0.1,
        ca_pair_dist_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_attn_layers = n_attn_layers
        self.ca_pair_dist_min = ca_pair_dist_min
        self.ca_pair_dist_max = ca_pair_dist_max

        single_input_dim = trunk_dim + latent_dim
        self.single_proj = nn.Sequential(
            nn.LayerNorm(single_input_dim),
            nn.Linear(single_input_dim, target_s_dim, bias=False),
        )
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, target_z_dim, bias=False),
        )

        if n_attn_layers > 0:
            self.attn_blocks = nn.ModuleList(
                [
                    AdaptorAttentionBlock(
                        s_dim=target_s_dim,
                        z_dim=target_z_dim,
                        num_heads=num_heads,
                    )
                    for _ in range(n_attn_layers)
                ]
            )

    def _binned_ca_distogram(self, ca_coords: torch.Tensor) -> torch.Tensor:
        pair_dists = torch.norm(
            ca_coords[:, :, None, :] - ca_coords[:, None, :, :],
            dim=-1,
        )
        n_bins = self.pair_proj[1].out_features
        bin_limits = torch.linspace(
            self.ca_pair_dist_min,
            self.ca_pair_dist_max,
            n_bins - 1,
            device=ca_coords.device,
            dtype=ca_coords.dtype,
        )
        bin_indices = torch.bucketize(pair_dists, bin_limits)
        return F.one_hot(bin_indices, num_classes=n_bins).to(dtype=ca_coords.dtype)

    def forward(
        self,
        trunk_seqs: torch.Tensor,
        trunk_pair: torch.Tensor,
        local_latents: torch.Tensor,
        ca_coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(
                trunk_seqs.shape[:2],
                dtype=trunk_seqs.dtype,
                device=trunk_seqs.device,
            )

        single_in = torch.cat([trunk_seqs, local_latents], dim=-1)
        s = self.single_proj(single_in)
        s = s * mask[..., None]

        z = self.pair_proj(trunk_pair)
        z = z + self._binned_ca_distogram(ca_coords)
        z = z * mask[:, :, None, None] * mask[:, None, :, None]

        if self.n_attn_layers > 0:
            for block in self.attn_blocks:
                s, z = block(s, z, mask)

        return s, z
