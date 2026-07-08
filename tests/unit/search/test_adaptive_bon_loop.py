"""Unit: adaptive best-of-N outer-loop control logic (no GPU).

The loop drives rounds of the generate->filter->evaluate->analyze pipeline until
it accumulates >= target_successes UNIQUE AF2-confirmed successes across all
rounds, or the cumulative wall-clock (all rounds' generate+evaluate) reaches the
budget. Round 1 samples `round1_nsamples`; every later round samples
`round_nsamples`; each round uses a distinct seed and its own output dir.

These tests inject the per-round pipeline call and the per-round success
harvest, so the control logic runs on CPU with no model or checkpoint.

Run: .venv/bin/python -m pytest tests/unit/search/test_adaptive_bon_loop.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from proteinfoundation.search.adaptive_bon_loop import (
    AdaptiveLoopConfig,
    RoundResult,
    _build_design_command,
    harvest_round_successes,
    run_adaptive_loop,
)


def _cfg(**kw) -> AdaptiveLoopConfig:
    base = dict(
        config_path="configs/x.yaml",
        task_name="02_PDL1",
        run_name="test",
        output_root=Path("/tmp/adaptive_test"),
        round1_nsamples=3000,
        round_nsamples=1000,
        target_successes=100,
        time_budget_hours=16.0,
        base_seed=5,
    )
    base.update(kw)
    return AdaptiveLoopConfig(**base)


def test_stops_when_target_successes_reached() -> None:
    """Loop halts as soon as the deduped accumulator crosses the target."""
    # Round k yields 40 fresh unique successes: 40, 80, 120 -> stop after round 3.
    calls: list[dict] = []

    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        calls.append({"round": round_index, "nsamples": nsamples, "seed": seed})
        ids = {f"r{round_index}_s{i}" for i in range(40)}
        return RoundResult(unique_success_ids=ids, wall_clock_hours=1.0)

    summary = run_adaptive_loop(_cfg(target_successes=100), run_round=run_round)

    assert summary.stop_reason == "target_reached"
    assert summary.total_unique_successes >= 100
    assert summary.n_rounds == 3
    assert len(calls) == 3


def test_stops_on_time_budget() -> None:
    """Loop halts when cumulative wall-clock reaches the budget, even below target."""
    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        # Only 1 unique success/round; each round burns 6h -> 3 rounds hit 18h >= 16h.
        return RoundResult(
            unique_success_ids={f"r{round_index}_only"},
            wall_clock_hours=6.0,
        )

    summary = run_adaptive_loop(
        _cfg(target_successes=100, time_budget_hours=16.0), run_round=run_round
    )

    assert summary.stop_reason == "time_budget"
    assert summary.total_wall_clock_hours >= 16.0
    assert summary.total_unique_successes < 100
    # Round 1 (6h) + round 2 (12h) both under budget; round 3 crosses it.
    assert summary.n_rounds == 3


def test_round1_uses_3000_then_1000() -> None:
    """Round 1 samples round1_nsamples; every subsequent round samples round_nsamples."""
    seen: list[int] = []

    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        seen.append(nsamples)
        return RoundResult(unique_success_ids=set(), wall_clock_hours=1.0)

    # No successes ever -> budget stops it; give a small budget for a few rounds.
    run_adaptive_loop(
        _cfg(round1_nsamples=3000, round_nsamples=1000, time_budget_hours=3.5),
        run_round=run_round,
    )

    assert seen[0] == 3000
    assert all(n == 1000 for n in seen[1:])
    assert len(seen) >= 2


def test_seeds_differ_per_round() -> None:
    """Every round gets a distinct seed so successive batches differ."""
    seeds: list[int] = []

    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        seeds.append(seed)
        return RoundResult(unique_success_ids=set(), wall_clock_hours=1.0)

    run_adaptive_loop(_cfg(base_seed=5, time_budget_hours=4.5), run_round=run_round)

    assert len(seeds) == len(set(seeds))
    assert seeds[0] == 5


def test_accumulation_is_deduplicated_across_rounds() -> None:
    """Successes seen in multiple rounds count once; only NEW ids increment the total."""
    # Round 1: {a, b, c}; round 2: {b, c, d} (b,c repeat) -> union {a,b,c,d} = 4.
    round_ids = [
        {"a", "b", "c"},
        {"b", "c", "d"},
        {"d", "e"},  # d repeats -> only e is new -> union {a,b,c,d,e} = 5
    ]

    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        return RoundResult(
            unique_success_ids=round_ids[round_index - 1],
            wall_clock_hours=1.0,
        )

    # target 5 -> reached exactly after round 3 (union grows 3 -> 4 -> 5).
    summary = run_adaptive_loop(
        _cfg(target_successes=5, time_budget_hours=100.0), run_round=run_round
    )

    assert summary.total_unique_successes == 5
    assert summary.n_rounds == 3
    assert summary.per_round_new_counts == [3, 1, 1]


def test_out_dir_is_round_indexed_and_distinct() -> None:
    """Each round is handed its own round-indexed output dir under output_root."""
    out_dirs: list[Path] = []

    def run_round(*, round_index, nsamples, seed, out_dir, cfg):
        out_dirs.append(out_dir)
        return RoundResult(unique_success_ids=set(), wall_clock_hours=1.0)

    root = Path("/tmp/adaptive_rounds_test")
    run_adaptive_loop(
        _cfg(output_root=root, time_budget_hours=3.5), run_round=run_round
    )

    assert len(out_dirs) == len(set(out_dirs))
    for d in out_dirs:
        assert root in d.parents or d.parent == root


def test_design_command_sets_nsamples_seed_and_task() -> None:
    """The per-round design command overrides nsamples, seed, task and run_name."""
    cfg = _cfg(task_name="33_TrkA", extra_overrides=["++generation.time_budget_hours=16"])
    cmd = _build_design_command(cfg, nsamples=3000, seed=7, run_name="rn_round01")

    assert "design" in cmd
    assert cfg.config_path in cmd
    assert "++generation.task_name=33_TrkA" in cmd
    assert "++generation.dataloader.dataset.nres.nsamples=3000" in cmd
    assert "++seed=7" in cmd
    assert "++run_name=rn_round01" in cmd
    assert "++generation.time_budget_hours=16" in cmd


def _write_round_csvs(round_dir: Path) -> None:
    """One round's analyze + sidecar + cluster CSVs: 2 confirmed (clusters A,B), 1 AF2-fail."""
    round_dir.mkdir(parents=True, exist_ok=True)

    def _pdb(tag: str) -> str:
        d = f"job_1_n_80_id_0_{tag}"
        return f"{round_dir}/inference/run/{d}/{d}_binder.pdb"

    def _row(tag: str, ipae: float, plddt: float, scrmsd: float) -> dict:
        return {
            "pdb_path": _pdb(tag),
            "self_complex_i_pAE": ipae,
            "self_complex_i_pAE_all": [ipae],
            "self_complex_pLDDT": plddt,
            "self_complex_pLDDT_all": [plddt],
            "self_binder_scRMSD_ca": scrmsd,
            "self_binder_scRMSD_ca_all": [scrmsd],
        }

    pd.DataFrame(
        [
            _row("bon_orig0_r0", ipae=0.10, plddt=0.95, scrmsd=1.0),  # confirmed A
            _row("bon_orig1_r0", ipae=0.12, plddt=0.93, scrmsd=1.1),  # confirmed B
            # pLDDT in the (0.90, 0.92] band: PASSES the canonical 0.90 AF2 gate
            # (this pins that the AF2 gate is 0.90, NOT the head's 0.92 provisional gate).
            _row("bon_orig2_r0", ipae=0.10, plddt=0.905, scrmsd=1.0),  # confirmed C (0.905 >= 0.90)
            # pLDDT below 0.90: AF2-rejected even though provisional.
            _row("bon_orig3_r0", ipae=0.10, plddt=0.88, scrmsd=1.0),  # AF2 FAIL under 0.90
        ]
    ).to_csv(round_dir / "binder_results_cfg_self_1.csv", index=False)

    pd.DataFrame(
        {
            "metadata_tag": ["bon_orig0_r0", "bon_orig1_r0", "bon_orig2_r0", "bon_orig3_r0"],
            "confidence_ipae": [3.0, 3.5, 3.0, 3.0],
            "confidence_complex_plddt": [0.95, 0.93, 0.905, 0.88],
            "provisional_success": [True, True, True, True],
            "elapsed_gpu_hours": [0.5, 1.0, 1.5, 2.0],
        }
    ).to_csv(round_dir / "confidence_scores_1.csv", index=False)

    pd.DataFrame(
        {
            "cluster_index": [0, 1, 2],
            "sample_index": [0, 1, 2],
            "path_name": [_pdb("bon_orig0_r0"), _pdb("bon_orig1_r0"), _pdb("bon_orig2_r0")],
        }
    ).to_csv(round_dir / "cluster_assignments_cfg.csv", index=False)


def test_harvest_applies_canonical_090_gate_and_namespaces_clusters(tmp_path: Path) -> None:
    """harvest_round_successes uses the canonical 0.90 AF2 gate and dedups by round-cluster.

    The 0.905-pLDDT sample PASSES (0.905 >= 0.90) -> three confirmed successes,
    each its own round-namespaced cluster id. The 0.88-pLDDT sample is AF2-rejected
    (< 0.90) even though provisional. This locks the AF2 success gate at the
    published 0.90 floor, distinct from the head's 0.92 provisional gate.
    """
    round_dir = tmp_path / "round_01"
    _write_round_csvs(round_dir)

    ids = harvest_round_successes(round_dir, round_index=1)

    assert len(ids) == 3
    assert all(round_dir.name in i for i in ids)


def test_harvest_missing_csv_returns_empty(tmp_path: Path) -> None:
    assert harvest_round_successes(tmp_path / "round_99", round_index=99) == set()
