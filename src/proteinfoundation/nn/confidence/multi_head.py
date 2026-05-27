"""Multi-head confidence wrapper.

`MultiHeadConfidence` is itself a registered `BaseConfidenceHead`. It owns
the shared `ConfidenceTrunk`, holds an `nn.ModuleDict` of child heads, and
runs the trunk **exactly once** per wrapper forward, dispatching the same
refined `(s, z, mask)` to every child's `_predict`. Joint training is
opt-in via this wrapper; the per-head registry (`PLDDTHead`, `PaeHead`,
...) remains the primary path for single-head distillation.

Trunk-once invariant. Every child carries a `trunk` attribute from its
own construction (the abstract `BaseConfidenceHead.__init__` requires it),
but inside `MultiHeadConfidence` we **rebind** every `child.trunk` to the
wrapper's single trunk instance and then dispatch through each child's
`_predict` directly (bypassing each child's `.forward`, which would
otherwise re-invoke its own `self.trunk(...)`). This keeps the trunk
forward count at exactly 1 per wrapper forward, regardless of how many
children consume the refined `(s, z)`.

Two layers of loss weighting, kept structurally separate (see
`CLAUDE.md`'s Multi-head paragraph):

- **Within-head** (`ce_weight` / `ev_weight` / `label_smoothing`) lives on
  each child's constructor and is consumed by the child's own
  `compute_loss_and_metrics`. Unchanged from single-head usage.
- **Across-head** `loss_weights[name]` multiplies each child's total
  before summing into the wrapper aggregate. `loss_weights[name] == 0.0`
  is a hard short-circuit: the child's loss path is skipped entirely
  (saves compute and guarantees zero gradient through that child).

`expected_trunk_eval_t` parity is asserted at construction. A child that
needs a different `t` cannot live under this wrapper; spin up a separate
sidecar instead.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from proteinfoundation.nn.confidence.base import (
    BaseConfidenceHead,
    ConfidenceTrunk,
    StageLiteral,
)
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("multi_head")
class MultiHeadConfidence(BaseConfidenceHead):
    output_keys: tuple[str, ...] = ()
    output_name_root: str = "multi"
    expected_trunk_eval_t: float = 0.99

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        children: dict[str, BaseConfidenceHead],
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        """Construct the multi-head wrapper.

        A child with `loss_weights[name] == 0.0` is still forward-traversed
        (its `_predict` runs); only its loss term is skipped. Under DDP with
        `find_unused_parameters=False`, prefer removing the child from the
        config over zeroing its weight.
        """
        super().__init__(trunk=trunk, token_dim=token_dim, pair_repr_dim=pair_repr_dim)

        for name, child in children.items():
            if not child.output_name_root:
                raise ValueError(
                    f"MultiHeadConfidence: child head {name!r} has empty "
                    f"output_name_root; every child must declare a non-empty "
                    f"output_name_root for log-key prefixing."
                )
            if child.expected_trunk_eval_t != self.expected_trunk_eval_t:
                raise ValueError(
                    f"MultiHeadConfidence: child head {name!r} has "
                    f"expected_trunk_eval_t={child.expected_trunk_eval_t} which "
                    f"!= wrapper's {self.expected_trunk_eval_t}. Use a separate "
                    f"sidecar for heads needing a different t."
                )

        for child in children.values():
            child.trunk = self.trunk

        self.children_heads = nn.ModuleDict(children)
        self.output_keys = tuple(
            key for child in children.values() for key in child.output_keys
        )
        self.loss_weights = (
            {k: float(v) for k, v in loss_weights.items()}
            if loss_weights is not None
            else {name: 1.0 for name in children}
        )

        zero_weighted = [n for n, w in self.loss_weights.items() if w == 0.0]
        if zero_weighted:
            warnings.warn(
                f"MultiHeadConfidence: loss_weights[{zero_weighted}] == 0.0 disables "
                f"those heads' gradient paths. Under DDP with "
                f"find_unused_parameters=False this will raise at backward (each "
                f"child's _predict still runs in forward, so its parameters enter "
                f"the autograd graph). To run with a zero-weighted child:\n"
                f"  (a) remove the child from `children:` in the Hydra config, OR\n"
                f"  (b) set the trainer strategy to ddp_find_unused_parameters_true, "
                f"or wrap with DDP(static_graph=True) when the per-batch graph is fixed.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _predict(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: head._predict(s, z, mask)
            for name, head in self.children_heads.items()
        }

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
        local_latents: torch.Tensor,
        chain_id: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run the trunk once and dispatch refined `(s, z, mask)` to every child.

        Children's `_predict` must not mutate the shared `s` or `z` in place
        - the wrapper runs every child against the same `(s, z)` reference
        from a single trunk forward.
        """
        del chain_id
        s_ref, z_ref = self.trunk(s, z, mask, cond, local_latents)
        return {
            name: head._predict(s_ref, z_ref, mask)
            for name, head in self.children_heads.items()
        }

    def compute_loss_and_metrics(
        self,
        out: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
        mask_eff: torch.Tensor,
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "MultiHeadConfidence does not own a single scalar loss; the sidecar "
            "Lightning module invokes `compute_multi_loss_and_metrics` with a "
            "per-head `masks_by_head` dict instead."
        )

    def compute_multi_loss_and_metrics(
        self,
        multi_out: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
        masks_by_head: dict[str, torch.Tensor],
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from proteinfoundation.nn.confidence._losses import MultiHeadLoss

        loss_fn = MultiHeadLoss(self.loss_weights)
        return loss_fn(
            multi_out=multi_out,
            heads=self.children_heads,
            batch=batch,
            masks_by_head=masks_by_head,
            stage=stage,
        )
