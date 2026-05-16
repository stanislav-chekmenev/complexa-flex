"""Per-residue pLDDT confidence head.

Final `LayerNorm` over the trunk-refined sequence representation followed
by a `Linear(token_dim, num_plddt_bins)` projection. The companion
`logits_to_expected_value` reduces a logits tensor to the
softmax-weighted bin-center mean in fp32 -- the quantity downstream
filters and SMC reward models consume.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.nn.confidence.base import BaseConfidenceHead, ConfidenceTrunk
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("plddt")
class PLDDTHead(BaseConfidenceHead):
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
        """Softmax-weighted bin-center mean. fp32 internally."""
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        return (probs * self.bin_centers.to(probs.dtype)).sum(dim=-1)
