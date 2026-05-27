"""Base classes for the light-weight confidence-head trunk and heads.

`ConfidenceTrunk` is a stack of complexa-native pair-biased attention blocks
(`MultiheadAttnAndTransition` + interleaved `PairReprUpdate`) that runs on
top of the frozen `LocalLatentsTransformer` intermediates. It is shared by
every concrete confidence head (pLDDT today; ipTM / ipAE / ipLDDT later).

`BaseConfidenceHead` is the abstract surface every head shares: it owns the
trunk, delegates prediction-head MLPs to subclasses via `_predict`, and
exposes `compute_loss_and_metrics` so the sidecar Lightning module stays
head-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate

StageLiteral = Literal["train", "val", "test", "predict"]


class ConfidenceTrunk(nn.Module):
    """Shared pair-biased trunk for the confidence heads.

    Built from `n_blocks` of `MultiheadAttnAndTransition` interleaved with
    `n_blocks - 1` `PairReprUpdate` layers (gated by
    `update_pair_repr_every_n` so block `i` runs a pair update when
    `i % update_pair_repr_every_n == 0`). Final `LayerNorm` is applied to
    both `s` and `z`; both outputs are mask-zeroed. `z` is passed through
    directionally; symmetric heads (PDE, ipLDDT, ipTM) opt in by
    symmetrising inside their own `_predict`.

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
        latent_dim: int = 8,
        add_ca_distogram: bool = False,
        ca_pair_dist_min: float = 0.1,
        ca_pair_dist_max: float = 3.0,
    ) -> None:
        super().__init__()
        if add_ca_distogram and pair_repr_dim < 2:
            raise ValueError(
                f"add_ca_distogram=True requires pair_repr_dim >= 2 (got "
                f"{pair_repr_dim}); the distogram is one-hot over pair_repr_dim "
                f"bins and needs at least one interior edge."
            )
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dim_cond = dim_cond
        self.update_pair_repr_every_n = update_pair_repr_every_n
        self.latent_dim = latent_dim
        self.add_ca_distogram = add_ca_distogram
        self.ca_pair_dist_min = ca_pair_dist_min
        self.ca_pair_dist_max = ca_pair_dist_max

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

        self.local_latents_proj = nn.Sequential(
            nn.Linear(latent_dim, token_dim, bias=False),
            nn.LayerNorm(token_dim),
        )

    def _binned_ca_distogram(self, ca_coords: torch.Tensor) -> torch.Tensor:
        """Symmetric one-hot Cα distogram over `pair_repr_dim` bins.

        Mirrors the deleted quality-graft adaptor verbatim: split the
        interval `[ca_pair_dist_min, ca_pair_dist_max]` into `n_bins - 1`
        equal interior edges; distances below the first edge land in
        bin 0, distances above the last edge land in bin `n_bins - 1`.

        Args:
            ca_coords: `[b, n, 3]` Cα coordinates in nm.

        Returns:
            `[b, n, n, pair_repr_dim]` one-hot tensor, cast to
            `ca_coords.dtype`.
        """
        pair_dists = torch.norm(
            ca_coords[:, :, None, :] - ca_coords[:, None, :, :],
            dim=-1,
        )
        n_bins = self.pair_repr_dim
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
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
        local_latents: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk.

        Args:
            s: `[b, n, token_dim]` sequence representation.
            z: `[b, n, n, pair_repr_dim]` pair representation.
            mask: `[b, n]` boolean mask.
            cond: `[b, n, dim_cond]` AdaLN conditioning. Mandatory.
            local_latents: `[b, n, latent_dim]` pre-trim latents from the
                frozen complexa trunk; aligned with `s`/`z`/`mask` on the
                same `n_extended` axis. Projected to `token_dim` and added
                as a mask-zeroed residual to `s` before the pair-biased
                attention stack.
            ca_coords: `[b, n, 3]` Cα coordinates in nm. Required when
                `add_ca_distogram=True`; a one-hot binned distogram is
                added directly to `z` (no projection, `n_bins ==
                pair_repr_dim`) and re-mask-zeroed before the attention
                stack. Ignored otherwise.

        Returns:
            `(s_refined, z_refined)`. Both mask-zeroed and LayerNormed;
            `z_refined` is passed through directionally so asymmetric
            heads (PaeHead) see un-projected pair features.
        """
        mask_f = mask[..., None]
        s = s * mask_f
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None]
        z = z * pair_mask

        # The `* mask_f` is load-bearing once training drifts `LayerNorm.bias`
        # off zero: LN of a (mask-zeroed) zero input returns the trained bias,
        # not zero, so the post-LN mask multiply is what keeps padded positions
        # from leaking the LN bias into valid cells through downstream attention.
        ll = self.local_latents_proj(local_latents) * mask_f
        s = s + ll

        if self.add_ca_distogram:
            if ca_coords is None:
                raise ValueError(
                    "ConfidenceTrunk: add_ca_distogram=True but ca_coords is None; "
                    "the caller must supply [b, n, 3] Cα coords in nm."
                )
            z = z + self._binned_ca_distogram(ca_coords)
            z = z * pair_mask

        for i in range(self.n_blocks):
            s = self.transformer_layers[i](s, z, cond, mask)
            if i < self.n_blocks - 1 and self.pair_update_layers[i] is not None:
                z = self.pair_update_layers[i](s, z, mask)

        s = self.s_layer_norm(s) * mask_f
        z = self.z_layer_norm(z) * pair_mask

        return s, z


class BaseConfidenceHead(nn.Module, ABC):
    """Abstract base for confidence heads.

    Subclasses override `_predict` to produce their head-specific outputs
    and `compute_loss_and_metrics` to produce the per-step `(loss, log_dict)`
    the sidecar Lightning module consumes head-agnostically.
    """

    output_keys: tuple[str, ...] = ()
    output_name_root: str = ""
    expected_trunk_eval_t: float = 0.99

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
        local_latents: torch.Tensor,
        chain_id: torch.Tensor | None = None,
        ca_coords: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run trunk then `_predict`.

        `local_latents` is the pre-trim `[b, n, latent_dim]` tensor from the
        frozen complexa trunk's intermediates; the shared `ConfidenceTrunk`
        projects and adds it to `s` as a mask-zeroed residual before its own
        pair-biased attention stack.

        `chain_id` is accepted but currently ignored; the seat is reserved
        for future ipTM / ipAE / ipLDDT subclasses that need per-residue
        chain identity. `PLDDTHead` does not consume it.

        `ca_coords` is forwarded to the trunk when its
        `add_ca_distogram=True`; otherwise ignored.

        The head does **not** trim concat-features; the caller decides
        whether to slice `[:, :n_orig]` using `orig_mask`.
        """
        del chain_id

        s_ref, z_ref = self.trunk(s, z, mask, cond, local_latents, ca_coords=ca_coords)
        return self._predict(s_ref, z_ref, mask)

    def compute_loss_and_metrics(
        self,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask_eff: torch.Tensor,
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return `(total_loss, log_dict)` for one optimisation step.

        Args:
            out: head's forward return (e.g. `{"plddt_logits": ...}`).
            batch: trimmed-to-`n_orig` batch with the fields this head needs.
            mask_eff: already-AND-ed effective mask (`orig_mask & *_mask`),
                float dtype.
            stage: `"train"` or `"val"`. Heads may branch behaviour
                (e.g. drop `label_smoothing` and emit validation metrics
                during `"val"`).

        Returns:
            `(total, log_dict)`. The log dict keys are unprefixed; the
            sidecar prepends `train/{output_name_root}/` or
            `val/{output_name_root}/`.
        """
        raise NotImplementedError
