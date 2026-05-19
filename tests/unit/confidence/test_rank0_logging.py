"""Rank-0-only loguru gating for the confidence-distillation entry point.

Loguru handler state is process-global, so an autouse snapshot/restore
fixture is required to keep these two cases independent of each other
and of any other test that runs in the same session.
"""

from __future__ import annotations

import pytest
from loguru import logger as _loguru_logger


@pytest.fixture(autouse=True)
def _snapshot_loguru_handlers():
    core = _loguru_logger._core
    snapshot = dict(core.handlers)
    try:
        yield
    finally:
        _loguru_logger.remove()
        for handler_id, handler in snapshot.items():
            core.handlers[handler_id] = handler


def test_non_rank0_silences_loguru(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("NODE_RANK", "0")

    from proteinfoundation.confidence.train_confidence import _gate_loguru_to_rank0

    assert len(_loguru_logger._core.handlers) > 0
    _gate_loguru_to_rank0()
    assert _loguru_logger._core.handlers == {}


def test_rank0_keeps_loguru(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("NODE_RANK", "0")

    from proteinfoundation.confidence.train_confidence import _gate_loguru_to_rank0

    before = dict(_loguru_logger._core.handlers)
    _gate_loguru_to_rank0()
    after = dict(_loguru_logger._core.handlers)
    assert after == before
    assert len(after) > 0
