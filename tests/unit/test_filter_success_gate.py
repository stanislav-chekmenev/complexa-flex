"""Unit: `filter.select_top_samples` success-gate population fix.

The confhead best-of-N pipeline replaces AF2 in the search loop: the head
labels every sample with a boolean `provisional_success` during generation.
The FINAL confidence-head filter must select the SUCCESS subset, not the
top-N-by-reward pool, so the downstream move-to-`filtered_out_samples/` step
relocates every non-success dir and `evaluate` only refolds successes.

Contract:
- WHEN a `provisional_success` column is present, the kept set is restricted
  to rows with `provisional_success > 0.5` BEFORE the top-N cap; the cap and
  reward-threshold act as secondary limits AMONG successes only, ranked by
  `total_reward` descending.
- WHEN the column is ABSENT (legacy AF2-reward pipeline), behaviour is
  byte-identical to the pre-fix top-N-by-reward selection (regression guard).
"""

from __future__ import annotations

import pandas as pd

from proteinfoundation.filter import select_top_samples


def _legacy_top_n(df: pd.DataFrame, total_samples: int, reward_threshold=None) -> pd.DataFrame:
    """Mirror of the pre-fix `main()` selection.

    The historical code applied the reward threshold and `.head()` cap ONLY
    inside `if len(combined_rewards) > total_samples:`; when there were no more
    rows than the cap, it kept the sorted frame untouched (threshold included).
    """
    top = df.sort_values("total_reward", ascending=False)
    if len(top) > total_samples:
        if reward_threshold is not None:
            top = top[top["total_reward"] >= reward_threshold]
        top = top.head(total_samples)
    return top


def test_gate_keeps_only_successes_ranked_within() -> None:
    df = pd.DataFrame(
        {
            "pdb_path": [f"/x/job_{i}/sample.pdb" for i in range(6)],
            "total_reward": [-1.0, -9.0, -2.0, -8.0, -3.0, -7.0],
            "provisional_success": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    out = select_top_samples(df, total_samples=1000, reward_threshold=None)

    # Only the three successes survive, ranked by total_reward descending.
    assert list(out["provisional_success"]) == [1.0, 1.0, 1.0]
    assert list(out["total_reward"]) == [-1.0, -2.0, -3.0]


def test_gate_caps_among_successes_only() -> None:
    df = pd.DataFrame(
        {
            "pdb_path": [f"/x/job_{i}/sample.pdb" for i in range(5)],
            "total_reward": [-1.0, -2.0, -3.0, -4.0, -0.5],
            "provisional_success": [1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    # 4 successes, cap of 2 -> the two best-reward successes, NOT the -0.5 fail
    # (which would top the raw sort).
    out = select_top_samples(df, total_samples=2, reward_threshold=None)

    assert len(out) == 2
    assert list(out["total_reward"]) == [-1.0, -2.0]
    assert (out["provisional_success"] > 0.5).all()


def test_gate_with_reward_threshold_secondary() -> None:
    df = pd.DataFrame(
        {
            "pdb_path": [f"/x/job_{i}/sample.pdb" for i in range(4)],
            "total_reward": [-1.0, -5.0, -2.0, -1.5],
            "provisional_success": [1.0, 1.0, 0.0, 1.0],
        }
    )
    # reward_threshold -3.0 drops the -5.0 success; the -2.0 fail is gone anyway.
    out = select_top_samples(df, total_samples=1000, reward_threshold=-3.0)

    assert list(out["total_reward"]) == [-1.0, -1.5]
    assert (out["provisional_success"] > 0.5).all()


def test_no_successes_yields_empty() -> None:
    df = pd.DataFrame(
        {
            "pdb_path": ["/x/job_0/sample.pdb", "/x/job_1/sample.pdb"],
            "total_reward": [-1.0, -2.0],
            "provisional_success": [0.0, 0.0],
        }
    )
    out = select_top_samples(df, total_samples=1000, reward_threshold=None)
    assert len(out) == 0


def test_legacy_path_without_column_is_byte_identical() -> None:
    """No provisional_success column -> pure top-N-by-reward (regression guard)."""
    df = pd.DataFrame(
        {
            "pdb_path": [f"/x/job_{i}/sample.pdb" for i in range(5)],
            "total_reward": [-1.0, -9.0, -2.0, -8.0, -3.0],
        }
    )
    # Last case: reward_threshold set AND len <= total_samples -> the pre-fix
    # guard did NOT apply the threshold, so all rows are kept. This is the
    # corner the earlier mirror got wrong (threshold applied unconditionally).
    for total_samples, thresh in [(3, None), (1000, None), (2, -8.5), (1000, -3.0)]:
        out = select_top_samples(df, total_samples=total_samples, reward_threshold=thresh)
        expected = _legacy_top_n(df, total_samples, thresh)
        pd.testing.assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))
