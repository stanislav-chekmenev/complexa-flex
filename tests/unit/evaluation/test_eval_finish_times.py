"""Unit test for per-sample eval-finish-time instrumentation in run_binder_evaluation.

The plot's GPU-hours x-axis needs the AF2 refold stage, not just generation time. The
binder refold happens inside compute_binder_metrics (a single batched call), so the honest
per-sample offset is the total binder-eval wall clock, written uniformly into ``eval_finish_s``
(seconds from eval start). This test pins the pure attach helper.

CPU-only. Run: .venv/bin/python -m pytest tests/unit/evaluation/test_eval_finish_times.py
"""

from __future__ import annotations

import pandas as pd

from proteinfoundation.evaluate import _attach_eval_finish_times


def test_attach_uniform_eval_finish_seconds():
    df = pd.DataFrame({"pdb_path": ["a", "b", "c"]})
    out = _attach_eval_finish_times(df, elapsed_s=12.5)
    assert "eval_finish_s" in out.columns
    assert list(out["eval_finish_s"]) == [12.5, 12.5, 12.5]


def test_attach_empty_df_is_noop():
    df = pd.DataFrame({"pdb_path": []})
    out = _attach_eval_finish_times(df, elapsed_s=5.0)
    assert "eval_finish_s" in out.columns
    assert len(out) == 0
