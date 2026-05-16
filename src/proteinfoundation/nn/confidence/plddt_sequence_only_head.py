"""Sequence-only diagnostic pLDDT head.

`SequenceOnlyPLDDTHead` is a control variant of `PLDDTHead` that zeros the
pair representation `z` before invoking the shared `ConfidenceTrunk`. The
trunk's pair-bias attention thus consumes a constant `z`, isolating
whatever signal the head can recover from the sequence representation
alone. Spec Risk 3 (head shortcuts to residue type) is monitored by
comparing Spearman r against the full `PLDDTHead`.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.nn.confidence.base import BaseConfidenceHead, ConfidenceTrunk
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("plddt_sequence_only")
class SequenceOnlyPLDDTHead(BaseConfidenceHead):
    output_keys: tuple[str, ...] = ("plddt_logits",)

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_plddt_bins: int = 50,
        bin_min: float = 0.0,
        bin_max: float = 100.0,
    ) -> None:
        super().__init__(trunk=trunk, token_dim=token_dim, pair_repr_dim=pair_repr_dim)
        if num_plddt_bins <= 0:
            raise ValueError(f"num_plddt_bins must be positive, got {num_plddt_bins}")
        if bin_max <= bin_min:
            raise ValueError(f"bin_max ({bin_max}) must exceed bin_min ({bin_min})")

        self.num_plddt_bins = num_plddt_bins
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)

        self.logits_norm = nn.LayerNorm(token_dim)
        self.logits_linear = nn.Linear(token_dim, num_plddt_bins)

        bin_width = (self.bin_max - self.bin_min) / num_plddt_bins
        centers = torch.tensor(
            [self.bin_min + bin_width * (i + 0.5) for i in range(num_plddt_bins)],
            dtype=torch.float32,
        )
        self.register_buffer("bin_centers", centers, persistent=False)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        cond: torch.Tensor,
        chain_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del chain_id
        z_zeroed = torch.zeros_like(z)
        s_ref, z_ref = self.trunk(s, z_zeroed, mask, cond)
        return self._predict(s_ref, z_ref, mask)

    def _predict(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del z
        logits = self.logits_linear(self.logits_norm(s))
        logits = logits * mask[..., None]
        return {"plddt_logits": logits}

    def logits_to_expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)
