"""Base classes for the light-weight confidence-head trunk and heads.

`ConfidenceTrunk` is a stack of complexa-native pair-biased attention blocks
(`MultiheadAttnAndTransition` + interleaved `PairReprUpdate`) that runs on
top of the frozen `LocalLatentsTransformer` intermediates. It is shared by
every concrete confidence head (pLDDT today; ipTM / ipAE / ipLDDT later).

`BaseConfidenceHead` is the abstract surface every head shares: it owns the
trunk, delegates prediction-head MLPs to subclasses via `_predict`, and
reserves `compute_loss` for PR-4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate


class ConfidenceTrunk(nn.Module):
    """Shared pair-biased trunk for the confidence heads.

    Built from `n_blocks` of `MultiheadAttnAndTransition` interleaved with
    `n_blocks - 1` `PairReprUpdate` layers (gated by
    `update_pair_repr_every_n` so block `i` runs a pair update when
    `i % update_pair_repr_every_n == 0`). Final `LayerNorm` is applied to
    both `s` and `z`; `z` is symmetrised; both outputs are mask-zeroed.

    `cond` is **mandatory** (no `Optional`). `MultiheadAttnAndTransition`
    calls AdaLN, which has no fallback path when `cond` is missing.
    The sidecar in PR-4 embeds `t = 0.99` through the trunk's
    `FeatureFactory` time embedder and feeds the result here.
    """

    def __init__(
        self,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 16,
        dim_cond: int = 256,
        use_tri_mult: bool = True,
        use_tri_attn: bool = False,
        use_qkln: bool = True,
        dropout: float = 0.1,
        update_pair_repr_every_n: int = 1,
        expects_external_cond: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dim_cond = dim_cond
        self.update_pair_repr_every_n = update_pair_repr_every_n
        self.expects_external_cond = expects_external_cond

        self.transformer_layers = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=token_dim,
                    dim_pair=pair_repr_dim,
                    nheads=n_heads,
                    dim_cond=dim_cond,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=use_qkln,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.pair_update_layers = nn.ModuleList(
            [
                (
                    PairReprUpdate(
                        token_dim=token_dim,
                        pair_dim=pair_repr_dim,
                        use_tri_mult=use_tri_mult,
                        use_tri_attn=use_tri_attn,
                        dropout=dropout,
                    )
                    if i % update_pair_repr_every_n == 0
                    else None
                )
                for i in range(n_blocks - 1)
            ]
        )

        self.s_layer_norm = nn.LayerNorm(token_dim)
        self.z_layer_norm = nn.LayerNorm(pair_repr_dim)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk.

        Args:
            s: `[b, n, token_dim]` sequence representation.
            z: `[b, n, n, pair_repr_dim]` pair representation.
            mask: `[b, n]` boolean mask.
            cond: `[b, n, dim_cond]` AdaLN conditioning. Mandatory.

        Returns:
            `(s_refined, z_refined)`. Both mask-zeroed; `z_refined`
            symmetrised; both LayerNormed.
        """
        mask_f = mask[..., None]
        s = s * mask_f
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None]
        z = z * pair_mask

        for i in range(self.n_blocks):
            s = self.transformer_layers[i](s, z, cond, mask)
            if i < self.n_blocks - 1 and self.pair_update_layers[i] is not None:
                z = self.pair_update_layers[i](s, z, mask)

        s = self.s_layer_norm(s) * mask_f
        z = self.z_layer_norm(z) * pair_mask
        z = (z + z.transpose(-3, -2)) / 2.0
        z = z * pair_mask

        return s, z


class BaseConfidenceHead(nn.Module, ABC):
    """Abstract base for confidence heads.

    Subclasses override `_predict` to produce their head-specific outputs.
    `compute_loss` is reserved for PR-4.
    """

    output_keys: tuple[str, ...] = ()

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
    ) -> None:
        super().__init__()
        if trunk.token_dim != token_dim:
            raise ValueError(
                f"trunk.token_dim ({trunk.token_dim}) != head token_dim ({token_dim})"
            )
        if trunk.pair_repr_dim != pair_repr_dim:
            raise ValueError(
                f"trunk.pair_repr_dim ({trunk.pair_repr_dim}) != head pair_repr_dim ({pair_repr_dim})"
            )
        self.trunk = trunk
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim

    @abstractmethod
    def _predict(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Subclass-specific prediction head."""

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
        chain_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run trunk then `_predict`.

        `chain_id` is reserved for future ipTM / ipAE / ipLDDT subclasses;
        unused by `PLDDTHead`. Defaults to zeros (monomer) inside the
        forward when `None`.

        The head does **not** trim concat-features; the caller decides
        whether to slice `[:, :n_orig]` using `orig_mask`.
        """
        del chain_id

        s_ref, z_ref = self.trunk(s, z, mask, cond)
        return self._predict(s_ref, z_ref, mask)

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("compute_loss is implemented in PR-4")
