"""Multi-head confidence wrapper (quality-graft-style backbone).

`MultiHeadConfidence` owns an `AdaptorModule` (projects complexa trunk
intermediates into Boltz-1 dims `(s=384, z=128)` and consumes the Cα
distogram and 8-dim local latents) and a `QgPairformerStack` (4 Boltz-1
PairformerLayer's with triangular attention). It holds an `nn.ModuleDict`
of child heads, runs the backbone exactly once per forward, and
dispatches the same refined `(s, z, mask)` to every child's `_predict`.
Joint training is opt-in via this wrapper; the per-head registry
(`PLDDTHead`, `PaeHead`, ...) remains the primary path for single-head
distillation through `ConfidenceTrunk`.

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

This wrapper is deliberately NOT a subclass of `BaseConfidenceHead`. The
base class' `__init__` requires a `ConfidenceTrunk`, which the qg-style
backbone no longer uses. The sidecar Lightning module dispatches via
`isinstance(self.head, MultiHeadConfidence)`, not by abstract surface.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from proteinfoundation.nn.confidence.adaptor import AdaptorModule
from proteinfoundation.nn.confidence.base import BaseConfidenceHead, StageLiteral
from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("multi_head")
class MultiHeadConfidence(nn.Module):
    output_keys: tuple[str, ...] = ()
    output_name_root: str = "multi"
    expected_trunk_eval_t: float = 0.99

    def __init__(
        self,
        adaptor: AdaptorModule,
        backbone: QgPairformerStack,
        children: dict[str, BaseConfidenceHead],
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

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

        self.adaptor = adaptor
        self.backbone = backbone
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

    def forward(
        self,
        trunk_seqs: torch.Tensor,
        trunk_pair: torch.Tensor,
        local_latents: torch.Tensor,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        # Boltz-1 pairformer math wants a float mask (multiplied inside softmax
        # biases / triangle ops); the heads' `_predict` use boolean & for
        # pair-mask construction. Keep both views separate.
        mask_f = mask if mask.is_floating_point() else mask.to(trunk_seqs.dtype)
        mask_bool = mask.bool() if not mask.dtype == torch.bool else mask

        s, z = self.adaptor(
            trunk_seqs=trunk_seqs,
            trunk_pair=trunk_pair,
            local_latents=local_latents,
            ca_coords=ca_coords,
            mask=mask_f,
        )
        s, z = self.backbone(s, z, mask_f)
        return {
            name: head._predict(s, z, mask_bool)
            for name, head in self.children_heads.items()
        }

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
