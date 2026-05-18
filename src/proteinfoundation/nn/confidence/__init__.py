"""Confidence-head architecture package.

Importing this package triggers registration of every concrete head via the
`@register_confidence_head` decorator side effect (e.g. `PLDDTHead`).
"""

from __future__ import annotations

from proteinfoundation.nn.confidence.base import BaseConfidenceHead, ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.nn.confidence.plddt_sequence_only_head import (
    SequenceOnlyPLDDTHead,
)
from proteinfoundation.nn.confidence.projections import PairProjection, SeqProjection
from proteinfoundation.nn.confidence.registry import (
    CONFIDENCE_HEAD_REGISTRY,
    build_confidence_head_from_cfg,
    register_confidence_head,
)

__all__ = [
    "BaseConfidenceHead",
    "ConfidenceTrunk",
    "MultiHeadConfidence",
    "PaeHead",
    "PLDDTHead",
    "SequenceOnlyPLDDTHead",
    "PairProjection",
    "SeqProjection",
    "CONFIDENCE_HEAD_REGISTRY",
    "build_confidence_head_from_cfg",
    "register_confidence_head",
]
