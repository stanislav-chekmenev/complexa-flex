"""Confidence-scorer wiring into ``Proteina.predict_step`` (CPU-only).

Exercises the search-loop plumbing added for best-of-N confidence scoring
without loading a trunk, an AF2 reward, or a checkpoint:

- ``_confidence_scorer_cfg`` gates on ``enabled``.
- ``_score_finals_with_confidence`` returns a ``compute_reward_from_samples``
  shaped dict whose ``total_reward == -ipae``, with NaN ipae sunk to a very
  low (finite) reward so ``filter.py``'s ``dropna`` keeps the row.
- ``predict_step`` with a confidence scorer configured attaches
  ``confidence_ipae`` / ``confidence_complex_plddt`` / ``provisional_success``
  to the output ``rewards`` dict and never touches the AF2 reward model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from proteinfoundation.proteina import Proteina
from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY


class _StubScorer:
    def __init__(self, ipae: torch.Tensor, plddt: torch.Tensor, success: torch.Tensor) -> None:
        self._out = {"ipae": ipae, "complex_plddt": plddt, "provisional_success": success}
        self.calls = 0

    def score_native(self, complex_prots: dict) -> dict:
        self.calls += 1
        return self._out


def _bare_proteina() -> Proteina:
    model = Proteina.__new__(Proteina)
    model._confidence_scorer = None
    return model


def _final_prots(b: int = 3, n: int = 8) -> dict:
    coors = torch.randn(b, n, 37, 3)
    return {
        "coors": coors,
        "residue_type": torch.randint(0, 20, (b, n)),
        "chain_index": torch.zeros(b, n, dtype=torch.long),
        "mask": torch.ones(b, n, dtype=torch.bool),
    }


def test_confidence_scorer_cfg_gate() -> None:
    model = _bare_proteina()
    model.inf_cfg = SimpleNamespace(confidence_scorer=None)
    assert model._confidence_scorer_cfg() is None

    model.inf_cfg = SimpleNamespace(
        confidence_scorer=SimpleNamespace(enabled=False, get=lambda k, d=None: False)
    )
    assert model._confidence_scorer_cfg() is None


def test_score_finals_total_reward_is_neg_ipae_with_nan_sink() -> None:
    model = _bare_proteina()
    ipae = torch.tensor([5.0, float("nan"), 12.0])
    plddt = torch.tensor([0.95, 0.5, 0.8])
    success = torch.tensor([True, False, False])
    model._confidence_scorer = _StubScorer(ipae, plddt, success)

    out = model._score_finals_with_confidence(_final_prots(b=3))

    assert set(out) >= {
        TOTAL_REWARD_KEY,
        "confidence_ipae",
        "confidence_complex_plddt",
        "provisional_success",
    }
    tr = out[TOTAL_REWARD_KEY]
    assert torch.isfinite(tr).all(), "NaN ipae must not leak into total_reward"
    assert tr[0].item() == -5.0
    assert tr[2].item() == -12.0
    assert tr[1].item() <= -1e8, "NaN ipae must sink to a very low reward"
    # confidence_ipae keeps the raw (NaN-preserving) value
    assert torch.isnan(out["confidence_ipae"][1])


def test_predict_step_uses_confidence_and_skips_af2(monkeypatch) -> None:
    model = _bare_proteina()

    finals = _final_prots(b=2)
    model._get_search_instance = lambda: SimpleNamespace(
        search=lambda batch: {"final": finals, "lookahead": None}
    )
    model._refinement_enabled = lambda: False

    ipae = torch.tensor([4.0, 9.0])
    stub = _StubScorer(ipae, torch.tensor([0.92, 0.7]), torch.tensor([True, False]))
    model._confidence_scorer = stub
    model.inf_cfg = SimpleNamespace(
        confidence_scorer=SimpleNamespace(enabled=True, get=lambda k, d=None: True)
    )

    # Any AF2 reward attempt must blow up if reached.
    import proteinfoundation.proteina as proteina_mod

    def _boom(*a, **k):
        raise AssertionError("AF2 reward model must not be initialized under confidence scorer")

    monkeypatch.setattr(proteina_mod, "initialize_reward_model", _boom)
    monkeypatch.setattr(proteina_mod, "compute_reward_from_samples", _boom)

    batch = {"mask": torch.ones(2, 8, dtype=torch.bool)}
    out = model.predict_step(batch, batch_idx=0)

    assert stub.calls == 1
    rewards = out["rewards"]
    assert torch.allclose(rewards[TOTAL_REWARD_KEY], -ipae)
    assert torch.allclose(rewards["confidence_ipae"], ipae)
    assert "confidence_complex_plddt" in rewards
    assert "provisional_success" in rewards
