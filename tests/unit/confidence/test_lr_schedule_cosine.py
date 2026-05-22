"""Cosine warmup LR schedule for confidence-head distillation.

The helper `_cosine_warmup_factor` is the load-bearing piece — the
Lightning module's `_lr_lambda` just plugs `self.warmup_steps`,
`self.trainer.estimated_stepping_batches`, and `self.min_lr / self.lr`
into it. Test the helper directly; bind it through the module once to
keep the integration honest.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import lightning as L
import pytest

from proteinfoundation.confidence.lightning_module import (
    _cosine_warmup_factor,
)


def _make_bare_module():
    """Construct a `ConfidenceDistillationModule` shell without loading the trunk.

    `__init__` would pull the frozen Proteina from disk; we only need the
    attrs that `_lr_lambda` reads, so initialise the `LightningModule`
    base in-place to satisfy descriptor access (`self.trainer` etc.) and
    set the four attrs the lambda touches.
    """
    from proteinfoundation.confidence.lightning_module import (
        ConfidenceDistillationModule,
    )

    mod = ConfidenceDistillationModule.__new__(ConfidenceDistillationModule)
    L.LightningModule.__init__(mod)
    return mod


WARMUP = 500
TOTAL = 10_000
LR = 1e-4
MIN_LR = 5e-6
MIN_FACTOR = MIN_LR / LR


def test_step_zero_returns_zero() -> None:
    assert _cosine_warmup_factor(0, WARMUP, TOTAL, MIN_FACTOR) == 0.0


def test_linear_ramp_during_warmup() -> None:
    half = WARMUP // 2
    assert _cosine_warmup_factor(half, WARMUP, TOTAL, MIN_FACTOR) == pytest.approx(
        half / WARMUP
    )


def test_factor_one_at_warmup_end() -> None:
    assert _cosine_warmup_factor(WARMUP, WARMUP, TOTAL, MIN_FACTOR) == pytest.approx(1.0)


def test_factor_min_at_total_steps() -> None:
    assert _cosine_warmup_factor(TOTAL, WARMUP, TOTAL, MIN_FACTOR) == pytest.approx(
        MIN_FACTOR, abs=1e-12
    )


def test_midpoint_is_cosine_not_linear() -> None:
    mid = WARMUP + (TOTAL - WARMUP) // 2
    expected = MIN_FACTOR + 0.5 * (1.0 - MIN_FACTOR)
    got = _cosine_warmup_factor(mid, WARMUP, TOTAL, MIN_FACTOR)
    assert got == pytest.approx(expected, abs=1e-6)


def test_monotonically_non_increasing_in_decay_phase() -> None:
    steps = list(range(WARMUP, TOTAL + 1, max((TOTAL - WARMUP) // 50, 1)))
    vals = [_cosine_warmup_factor(s, WARMUP, TOTAL, MIN_FACTOR) for s in steps]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_clamped_beyond_total_steps() -> None:
    assert _cosine_warmup_factor(TOTAL + 1000, WARMUP, TOTAL, MIN_FACTOR) == pytest.approx(
        MIN_FACTOR, abs=1e-12
    )


def test_unknown_total_steps_holds_at_one() -> None:
    assert _cosine_warmup_factor(WARMUP + 1, WARMUP, None, MIN_FACTOR) == 1.0
    assert _cosine_warmup_factor(WARMUP + 1, WARMUP, 0, MIN_FACTOR) == 1.0


def test_total_steps_le_warmup_holds_at_one() -> None:
    assert _cosine_warmup_factor(WARMUP + 1, WARMUP, WARMUP, MIN_FACTOR) == 1.0
    assert _cosine_warmup_factor(WARMUP + 1, WARMUP, WARMUP - 10, MIN_FACTOR) == 1.0


def test_zero_min_factor_decays_to_zero() -> None:
    assert _cosine_warmup_factor(TOTAL, WARMUP, TOTAL, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_lightning_module_binding_uses_cosine() -> None:
    """`_lr_lambda` reads `self.{warmup_steps,lr,min_lr,trainer.estimated_stepping_batches}`.

    Bypass `__init__` (which loads a trunk) and bind only those attrs to
    confirm the method delegates to the cosine helper end-to-end.
    """
    mod = _make_bare_module()
    mod.warmup_steps = WARMUP
    mod.lr = LR
    mod.min_lr = MIN_LR
    mod._trainer = SimpleNamespace(estimated_stepping_batches=TOTAL)

    mid = WARMUP + (TOTAL - WARMUP) // 2
    expected_mid = MIN_FACTOR + 0.5 * (1.0 - MIN_FACTOR)
    assert mod._lr_lambda(0) == 0.0
    assert mod._lr_lambda(WARMUP) == pytest.approx(1.0)
    assert mod._lr_lambda(mid) == pytest.approx(expected_mid, abs=1e-6)
    assert mod._lr_lambda(TOTAL) == pytest.approx(MIN_FACTOR, abs=1e-12)


def test_lightning_module_zero_lr_guard() -> None:
    mod = _make_bare_module()
    mod.warmup_steps = WARMUP
    mod.lr = 0.0
    mod.min_lr = MIN_LR
    mod._trainer = SimpleNamespace(estimated_stepping_batches=TOTAL)

    assert mod._lr_lambda(TOTAL) == pytest.approx(0.0, abs=1e-12)


def test_unknown_estimated_steps_holds_at_one_after_warmup() -> None:
    """If `trainer.estimated_stepping_batches` is unknown, the schedule
    stays at the warmup-end factor of 1.0 (no decay)."""
    mod = _make_bare_module()
    mod.warmup_steps = WARMUP
    mod.lr = LR
    mod.min_lr = MIN_LR
    mod._trainer = SimpleNamespace(estimated_stepping_batches=None)

    assert mod._lr_lambda(WARMUP + 100) == 1.0


def test_cosine_shape_passes_through_half_decay() -> None:
    """Sanity: at quarter-decay the cosine value should exceed the linear value."""
    quarter = WARMUP + (TOTAL - WARMUP) // 4
    linear_val = 1.0 - 0.25 * (1.0 - MIN_FACTOR)
    cosine_val = _cosine_warmup_factor(quarter, WARMUP, TOTAL, MIN_FACTOR)
    expected_cosine = MIN_FACTOR + (1.0 - MIN_FACTOR) * 0.5 * (1.0 + math.cos(math.pi * 0.25))
    assert cosine_val == pytest.approx(expected_cosine, abs=1e-6)
    assert cosine_val > linear_val
