"""Adaptive best-of-N outer loop for confidence-head binder search.

Drives repeated rounds of the ``generate -> filter -> evaluate -> analyze``
design pipeline until it accumulates a target number of UNIQUE AF2-confirmed
successes across all rounds, or the cumulative wall-clock (generation +
evaluation, summed over every round) reaches a budget.

Round 1 samples ``round1_nsamples`` (default 3000); every later round samples
``round_nsamples`` (default 1000) with a distinct seed so successive batches
differ. Each round writes into its own round-indexed working directory
(``<output_root>/round_XX/``) so rounds do not overwrite each other, and the
per-round analyze CSV + confidence sidecar remain available for the multi-round
plot (:mod:`script_utils.plot.plot_successes_vs_gpu_hours_multiround`).

Success accounting uses the canonical AlphaProteo AF2 gate (Bennett et al.):
a candidate counts only if AF2 confirms ``(complex i_pAE*31 <= 7) & (complex
pLDDT >= 0.90) & (binder scRMSD_ca < 1.5)`` — the PUBLISHED 0.90 pLDDT floor, so
the reported success count stays comparable to the binder-design literature. The
tightened 0.92 pLDDT floor is the confidence HEAD's provisional gate only (it
decides which candidates get refolded; see
:data:`proteinfoundation.confidence.inference_scorer.SUCCESS_PLDDT_01`), NOT the
AF2 success gate. Uniqueness dedups across rounds by Foldseek cluster id when a
``cluster_assignments_*.csv`` is present, else by sample dir basename. The
canonical ``DEFAULT_PROTEIN_BINDER_THRESHOLDS`` (0.90, a community-parity
constant) is used verbatim for AF2 confirmation and is never mutated.

The control loop (:func:`run_adaptive_loop`) takes the per-round runner as an
injected callable, so it is CPU-testable with no model or checkpoint. The
concrete runner (:func:`run_round_subprocess`) shells out to the CLI design
pipeline; shelling out (rather than importing the stage entrypoints) isolates
each round's Hydra/Lightning global state.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from proteinfoundation.result_analysis.binder_analysis_utils import (
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
)

# The AF2 success gate is the canonical AlphaProteo rule (published 0.90 pLDDT
# floor), used verbatim from the community-parity constant so success counts stay
# comparable to the binder-design literature. The head's tightened 0.92 provisional
# gate lives in inference_scorer.SUCCESS_PLDDT_01 and is a DIFFERENT gate (it only
# decides which candidates get refolded, not which AF2 refolds count as successes).
ADAPTIVE_AF2_SUCCESS_THRESHOLDS = DEFAULT_PROTEIN_BINDER_THRESHOLDS


@dataclass
class AdaptiveLoopConfig:
    """Static configuration for the adaptive outer loop."""

    config_path: str
    task_name: str
    run_name: str
    output_root: Path
    round1_nsamples: int = 3000
    round_nsamples: int = 1000
    target_successes: int = 100
    time_budget_hours: float = 16.0
    base_seed: int = 5
    # Forwarded verbatim to the CLI design command (extra ++key=value overrides).
    extra_overrides: list[str] = field(default_factory=list)
    # Interpreter used to launch the pipeline subprocess (default: this one).
    python_executable: str = sys.executable


@dataclass
class RoundResult:
    """Outcome of a single round.

    Attributes:
        unique_success_ids: Set of within-round unique-success identifiers
            (Foldseek cluster id or sample basename). The loop unions these
            across rounds to dedup successes that recur between rounds.
        wall_clock_hours: Wall-clock spent this round (generate + evaluate),
            added to the cumulative budget.
    """

    unique_success_ids: set[str]
    wall_clock_hours: float


@dataclass
class AdaptiveLoopSummary:
    """Result of the whole loop."""

    stop_reason: str  # "target_reached" | "time_budget"
    n_rounds: int
    total_unique_successes: int
    total_wall_clock_hours: float
    per_round_new_counts: list[int]
    per_round_wall_clock_hours: list[float]
    round_dirs: list[Path]


RunRoundFn = Callable[..., RoundResult]


def _round_dir(output_root: Path, round_index: int) -> Path:
    return output_root / f"round_{round_index:02d}"


def run_adaptive_loop(
    cfg: AdaptiveLoopConfig,
    run_round: RunRoundFn,
) -> AdaptiveLoopSummary:
    """Run rounds until the target is met or the time budget is exhausted.

    Args:
        cfg: Loop configuration.
        run_round: Callable invoked per round with keyword args
            ``round_index`` (1-based), ``nsamples``, ``seed``, ``out_dir``,
            ``cfg``; returns a :class:`RoundResult`. Injected so the control
            logic is testable without a GPU.

    Returns:
        An :class:`AdaptiveLoopSummary`. The loop stops as soon as the deduped
        accumulator reaches ``cfg.target_successes`` (``target_reached``), or as
        soon as cumulative wall-clock reaches ``cfg.time_budget_hours``
        (``time_budget``). A round already in flight is always allowed to finish
        and be counted before the budget check trips.
    """
    accumulated: set[str] = set()
    per_round_new_counts: list[int] = []
    per_round_wall_clock_hours: list[float] = []
    round_dirs: list[Path] = []
    total_wall_clock = 0.0
    round_index = 0

    while True:
        round_index += 1
        nsamples = cfg.round1_nsamples if round_index == 1 else cfg.round_nsamples
        seed = cfg.base_seed + (round_index - 1)
        out_dir = _round_dir(cfg.output_root, round_index)
        round_dirs.append(out_dir)

        result = run_round(
            round_index=round_index,
            nsamples=nsamples,
            seed=seed,
            out_dir=out_dir,
            cfg=cfg,
        )

        before = len(accumulated)
        accumulated |= result.unique_success_ids
        per_round_new_counts.append(len(accumulated) - before)
        per_round_wall_clock_hours.append(result.wall_clock_hours)
        total_wall_clock += result.wall_clock_hours

        if len(accumulated) >= cfg.target_successes:
            stop_reason = "target_reached"
            break
        if total_wall_clock >= cfg.time_budget_hours:
            stop_reason = "time_budget"
            break

    return AdaptiveLoopSummary(
        stop_reason=stop_reason,
        n_rounds=round_index,
        total_unique_successes=len(accumulated),
        total_wall_clock_hours=total_wall_clock,
        per_round_new_counts=per_round_new_counts,
        per_round_wall_clock_hours=per_round_wall_clock_hours,
        round_dirs=round_dirs,
    )


# ---------------------------------------------------------------------------
# Concrete per-round runner (shells out to the CLI design pipeline)
# ---------------------------------------------------------------------------
def _build_design_command(
    cfg: AdaptiveLoopConfig,
    nsamples: int,
    seed: int,
    run_name: str,
) -> list[str]:
    cmd = [
        cfg.python_executable,
        "-m",
        "proteinfoundation.cli.cli_runner",
        "design",
        cfg.config_path,
        f"++generation.task_name={cfg.task_name}",
        f"++generation.dataloader.dataset.nres.nsamples={nsamples}",
        f"++seed={seed}",
        f"++run_name={run_name}",
    ]
    cmd.extend(cfg.extra_overrides)
    return cmd


def _latest_matching(root: Path, pattern: str) -> Path | None:
    """Most-recently-modified file under ``root`` matching a glob ``pattern``.

    Returns a single file. The shipped adaptive config runs one generation/eval
    job per round (``gen_njobs=1``/``eval_njobs=1``), so exactly one shard exists.
    If a multi-GPU config produces multiple shards, only the newest is read and
    the rest are silently ignored, which UNDERCOUNTS successes — warn loudly so
    the operator either concatenates shards upstream or keeps njobs=1 for the loop.
    """
    candidates = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        import warnings

        warnings.warn(
            f"{len(candidates)} files match '{pattern}' under {root}; reading only the newest "
            f"({candidates[0].name}). Multi-shard (njobs>1) rounds will undercount successes — "
            "run the adaptive loop with gen_njobs=1/eval_njobs=1 or merge shards first.",
            RuntimeWarning,
            stacklevel=2,
        )
    return candidates[0] if candidates else None


def _load_cluster_lookup(round_dir: Path) -> dict[str, str]:
    """Map sample pdb basename -> namespaced Foldseek cluster id for a round.

    Reads any ``cluster_assignments_*.csv`` (diversity_foldseek format:
    ``cluster_index,sample_index,path_name``) under the round dir. Cluster ids
    are namespaced by file so identical indices from different subsets or rounds
    never collide.
    """
    lookup: dict[str, str] = {}
    for f in sorted(round_dir.rglob("cluster_assignments*.csv")):
        try:
            adf = pd.read_csv(f)
        except Exception:
            continue
        if not {"cluster_index", "path_name"}.issubset(adf.columns):
            continue
        for _, arow in adf.iterrows():
            base = os.path.basename(str(arow["path_name"]))
            lookup[base] = f"round={round_dir.name}:{f.stem}:{arow['cluster_index']}"
    return lookup


def harvest_round_successes(round_dir: Path, round_index: int) -> set[str]:
    """Read a round's analyze + sidecar CSVs and return unique-success ids.

    A success passes the canonical 0.90-pLDDT AlphaProteo AF2 gate AND was a provisional
    success in the sidecar. Uniqueness is by Foldseek cluster id when a
    ``cluster_assignments_*.csv`` is present, else the sample dir basename
    (namespaced by round so cross-round basenames never collide). Returns an
    empty set if either CSV is missing (a round that produced no successes).
    """
    # Import here so the CPU-only control-logic tests need no heavy deps.
    from script_utils.plot.plot_successes_vs_gpu_hours import (  # noqa: E402
        extract_metadata_tag,
        join_analyze_and_sidecar,
    )

    analyze_csv = _latest_matching(round_dir, "binder_results_*.csv")
    sidecar_csv = _latest_matching(round_dir, "confidence_scores_*.csv")
    if analyze_csv is None or sidecar_csv is None:
        return set()

    analyze_df = pd.read_csv(analyze_csv)
    sidecar_df = pd.read_csv(sidecar_csv)
    joined = join_analyze_and_sidecar(analyze_df, sidecar_df)
    if "provisional_success" not in joined.columns:
        return set()

    prov = joined[joined["provisional_success"].astype(bool)].copy()
    if prov.empty:
        return set()

    confirmed = prov[af2_confirmed_mask(prov)].copy()
    if confirmed.empty:
        return set()

    cluster_lookup = _load_cluster_lookup(round_dir)
    ids: set[str] = set()
    for pdb_path in confirmed["pdb_path"].astype(str):
        base = os.path.basename(pdb_path)
        if base in cluster_lookup:
            ids.add(cluster_lookup[base])
        else:
            tag = extract_metadata_tag(pdb_path) or base
            ids.add(f"round={round_dir.name}:{tag}")
    return ids


def af2_confirmed_mask(df: pd.DataFrame) -> pd.Series:
    """AF2-confirmed mask under the canonical AlphaProteo gate (0.90 pLDDT floor)."""
    from proteinfoundation.result_analysis.binder_analysis import (  # noqa: E402
        filter_by_success_thresholds,
    )

    if df.empty:
        return pd.Series([], dtype=bool)
    confirmed = filter_by_success_thresholds(
        df,
        seq_type="self",
        success_thresholds=ADAPTIVE_AF2_SUCCESS_THRESHOLDS,
        save_json=False,
    )
    return df.index.isin(confirmed.index)


def run_round_subprocess(
    *,
    round_index: int,
    nsamples: int,
    seed: int,
    out_dir: Path,
    cfg: AdaptiveLoopConfig,
) -> RoundResult:
    """Run one design round in its own working dir and harvest its successes.

    Runs the CLI design pipeline with ``cwd=out_dir`` so the pipeline's
    cwd-relative ``./inference`` and ``./evaluation_results`` trees land under
    the round dir. Wall-clock of the whole design invocation (generate through
    analyze) is charged to the round budget.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.run_name}_round{round_index:02d}"
    cmd = _build_design_command(cfg, nsamples=nsamples, seed=seed, run_name=run_name)

    env = os.environ.copy()
    # The pipeline resolves configs relative to the project root; run the child
    # with cwd=out_dir but keep the config path absolute so Hydra finds it.
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(out_dir), env=env)
    wall_clock_hours = (time.perf_counter() - start) / 3600.0

    ids = harvest_round_successes(out_dir, round_index)
    return RoundResult(unique_success_ids=ids, wall_clock_hours=wall_clock_hours)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive best-of-N binder search: generate/evaluate rounds "
        "until N unique AF2-confirmed successes or a wall-clock budget.",
    )
    p.add_argument("--config-path", required=True, help="Design pipeline YAML (absolute path).")
    p.add_argument("--task-name", required=True, help="Target task name, e.g. 02_PDL1.")
    p.add_argument("--run-name", required=True, help="Run name prefix for output dirs.")
    p.add_argument("--output-root", required=True, type=Path, help="Root dir for per-round subdirs.")
    p.add_argument("--round1-nsamples", type=int, default=3000)
    p.add_argument("--round-nsamples", type=int, default=1000)
    p.add_argument("--target-successes", type=int, default=100)
    p.add_argument("--time-budget-hours", type=float, default=16.0)
    p.add_argument("--base-seed", type=int, default=5)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        dest="extra_overrides",
        help="Extra Hydra ++key=value override, applied to every round (repeatable).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> AdaptiveLoopSummary:
    args = _parse_args(argv)
    cfg = AdaptiveLoopConfig(
        config_path=args.config_path,
        task_name=args.task_name,
        run_name=args.run_name,
        output_root=args.output_root,
        round1_nsamples=args.round1_nsamples,
        round_nsamples=args.round_nsamples,
        target_successes=args.target_successes,
        time_budget_hours=args.time_budget_hours,
        base_seed=args.base_seed,
        extra_overrides=list(args.extra_overrides),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = run_adaptive_loop(cfg, run_round=run_round_subprocess)

    print(f"[adaptive-bon] stop_reason:            {summary.stop_reason}")
    print(f"[adaptive-bon] rounds:                 {summary.n_rounds}")
    print(f"[adaptive-bon] total unique successes: {summary.total_unique_successes}")
    print(f"[adaptive-bon] total wall-clock (h):   {summary.total_wall_clock_hours:.3f}")
    print(f"[adaptive-bon] per-round new:          {summary.per_round_new_counts}")
    print(f"[adaptive-bon] per-round wall-clock:   {summary.per_round_wall_clock_hours}")
    _write_summary(summary, args.output_root)
    return summary


def _write_summary(summary: AdaptiveLoopSummary, output_root: Path) -> None:
    rows = [
        {
            "round_index": i + 1,
            "round_dir": str(summary.round_dirs[i]),
            "new_unique_successes": summary.per_round_new_counts[i],
            "wall_clock_hours": summary.per_round_wall_clock_hours[i],
        }
        for i in range(summary.n_rounds)
    ]
    df = pd.DataFrame(rows)
    df["cumulative_unique_successes"] = df["new_unique_successes"].cumsum()
    df["cumulative_wall_clock_hours"] = df["wall_clock_hours"].cumsum()
    out = output_root / "adaptive_loop_summary.csv"
    df.to_csv(out, index=False)
    print(f"[adaptive-bon] summary -> {out}")


if __name__ == "__main__":
    main()
