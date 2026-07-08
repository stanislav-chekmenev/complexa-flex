"""Unit: the provisional pLDDT gate is tightened from 0.90 to 0.92.

The confidence head runs overconfident vs AF2-multimer (its complex pLDDT
reads high), so the best-of-N provisional gate raises the pLDDT acceptance
floor to 0.92 while ipAE < 7 A is unchanged. A sample whose complex pLDDT
lands in (0.90, 0.92] was accepted under the old 0.90 gate and must now be
REJECTED provisionally.

CPU-only, no checkpoint. Run:
  .venv/bin/python -m pytest tests/unit/confidence/test_inference_scorer_plddt_gate_092.py
"""

from __future__ import annotations

import torch

from proteinfoundation.confidence.inference_scorer import (
    SUCCESS_PLDDT_01,
    ConfidenceHeadScorer,
)


def test_success_plddt_threshold_is_0p92() -> None:
    assert SUCCESS_PLDDT_01 == 0.92


def test_plddt_between_090_and_092_now_rejected() -> None:
    """complex_plddt in (0.90, 0.92] passes the old gate but fails the new one.

    Two chains -> a real interface; ipAE fixed low so pLDDT is the sole
    deciding gate. Sample 0 at 0.905 (would pass at 0.90) must now fail;
    sample 1 at 0.93 must still pass.
    """
    chain_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    pae_ev = torch.full((2, 2, 2), 1.0)  # 1 A << 7 A, ipAE always passes
    plddt_ev = torch.tensor([[90.5, 90.5], [93.0, 93.0]])  # 0.905 vs 0.93

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert abs(float(out["complex_plddt"][0]) - 0.905) < 1e-6
    assert bool(out["provisional_success"][0]) is False  # rejected under 0.92
    assert bool(out["provisional_success"][1]) is True


def test_plddt_gate_bites_strictly_above_092() -> None:
    """The gate is `> 0.92`: exactly 0.92 fails, just above 0.92 passes."""
    chain_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    pae_ev = torch.full((2, 2, 2), 1.0)
    plddt_ev = torch.tensor([[92.0, 92.0], [92.5, 92.5]])  # 0.92 vs 0.925

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert bool(out["provisional_success"][0]) is False  # 0.92 not > 0.92
    assert bool(out["provisional_success"][1]) is True
