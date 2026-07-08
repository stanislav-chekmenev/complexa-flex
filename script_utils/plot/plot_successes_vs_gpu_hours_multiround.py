#!/usr/bin/env python3
"""Multi-round cumulative unique AF2-confirmed successes vs GPU hours.

The adaptive best-of-N loop runs several rounds of generate->evaluate->analyze,
each in its own round dir. This script stitches every round's per-sample
timeline onto ONE monotonic cumulative-GPU-hours axis: round k's within-round
elapsed times are offset by the summed wall-clock of rounds 1..k-1, so a round
that starts after 2.9 h of prior work has its first success plotted at 2.9 h,
not at 0. Successes are globally deduped across rounds by (round-namespaced)
Foldseek cluster id, and the resulting cumulative-unique curve is a monotone
step. A horizontal line marks the target success count; vertical dashed lines
mark round boundaries.

Per-round join / gate / cluster logic is reused from the single-round script
(``plot_successes_vs_gpu_hours``) so the two stay in lockstep. Two success curves
are plotted, identical to the single-round figure: the headline AlphaProteo AF2
gate (published 0.90 pLDDT floor, the rule the adaptive loop counts against) and
the head+scRMSD gate (the head's provisional_success interface metrics AND AF2
binder scRMSD_ca<1.5). Both are offset/accumulated across rounds the same way.

Usage:
    python script_utils/plot/plot_successes_vs_gpu_hours_multiround.py \
        --round-dir  inference/.../round_01 --round-wall-clock-hours 2.9 \
        --round-dir  inference/.../round_02 --round-wall-clock-hours 1.0 \
        --out-png    inference/.../successes_vs_gpu_hours_multiround.png \
        --target-name 02_PDL1 --target-successes 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from proteinfoundation.search.adaptive_bon_loop import (  # noqa: E402
    _load_cluster_lookup,
)
from script_utils.plot.plot_successes_vs_gpu_hours import (  # noqa: E402
    _GATE_COLORS,
    _GATE_LABELS,
    _GATES,
    _HEADLINE_GATE,
    af2_confirmed_mask,
    extract_metadata_tag,
    join_analyze_and_sidecar,
)


def _latest(round_dir: Path, pattern: str) -> Path | None:
    files = sorted(round_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def round_provisional_points(round_dir: Path) -> pd.DataFrame:
    """Per-provisional-success (within-round time, cluster id, per-gate pass bool).

    Joins the round's analyze + sidecar CSVs, keeps provisional successes, and
    tags each with both success gates' pass/fail (``pass_{gate}`` boolean columns)
    from the single-round module. Cluster ids are namespaced by round dir name so
    identical Foldseek indices in different rounds do not merge. Returns an empty
    frame (columns ``time_within_round``, ``cluster_id``, ``pass_{gate}``...) if the
    round produced no provisional successes or is missing a CSV.
    """
    gate_cols = [f"pass_{g}" for g in _GATES]
    empty = pd.DataFrame(columns=["time_within_round", "cluster_id", *gate_cols])
    analyze_csv = _latest(round_dir, "binder_results_*.csv")
    sidecar_csv = _latest(round_dir, "confidence_scores_*.csv")
    if analyze_csv is None or sidecar_csv is None:
        return empty

    analyze_df = pd.read_csv(analyze_csv)
    sidecar_df = pd.read_csv(sidecar_csv)
    joined = join_analyze_and_sidecar(analyze_df, sidecar_df)
    if "provisional_success" not in joined.columns:
        return empty
    prov = joined[joined["provisional_success"].astype(bool)].copy()
    if prov.empty:
        return empty

    # Resolve cluster ids the SAME way the adaptive loop's accumulator does
    # (round-namespaced Foldseek cluster if available, else metadata_tag /
    # basename) so the plotted curve matches the loop's dedup exactly.
    cluster_lookup = _load_cluster_lookup(round_dir)

    def _cluster_of(pdb_path: str) -> str:
        base = os.path.basename(pdb_path)
        if base in cluster_lookup:
            return cluster_lookup[base]
        tag = extract_metadata_tag(pdb_path) or base
        return f"round={round_dir.name}:{tag}"

    cluster_ids = [_cluster_of(str(p)) for p in prov["pdb_path"]]
    out = pd.DataFrame(
        {
            "time_within_round": prov["elapsed_gpu_hours"].astype(float).to_numpy(),
            "cluster_id": np.asarray(cluster_ids, dtype=object),
        }
    )
    for gate in _GATES:
        out[f"pass_{gate}"] = np.asarray(af2_confirmed_mask(prov, gate=gate))
    return out


def round_boundaries(wall_clocks: list[float]) -> list[float]:
    """Cumulative wall-clock at the END of each round (round-boundary x positions)."""
    return list(np.cumsum(np.asarray(wall_clocks, dtype=float)))


def _cumulative_unique(times: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Walk (time, cluster) in time order, counting first-seen clusters cumulatively."""
    order = np.argsort(times, kind="stable")
    times = times[order]
    clusters = clusters[order]
    seen: set = set()
    xs: list[float] = []
    ys: list[int] = []
    for t, c in zip(times, clusters):
        seen.add(c)
        xs.append(float(t))
        ys.append(len(seen))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def build_multiround_curves(
    round_dirs: list[Path],
    wall_clocks: list[float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Offset each round's per-sample times and build one global curve per gate.

    Round k's within-round times are shifted by the summed wall-clock of rounds
    1..k-1. Provisional successes from all rounds are merged; for each success
    gate the passing rows are sorted by absolute GPU hours and walked in time
    order incrementing the running count each time a NEW (round-namespaced)
    cluster id is first seen. Returns {gate: (xs, ys)} with each curve monotone
    non-decreasing; endpoint = number of distinct clusters passing that gate
    across all rounds.
    """
    if len(round_dirs) != len(wall_clocks):
        raise ValueError("round_dirs and wall_clocks must have equal length")

    offsets = [0.0] + round_boundaries(wall_clocks)[:-1]
    frames = []
    for round_dir, offset in zip(round_dirs, offsets):
        pts = round_provisional_points(round_dir)
        if pts.empty:
            continue
        pts = pts.copy()
        pts["abs_time"] = pts["time_within_round"] + offset
        frames.append(pts)

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not frames:
        for gate in _GATES:
            curves[gate] = (np.array([], dtype=float), np.array([], dtype=int))
        return curves

    allpts = pd.concat(frames, ignore_index=True)
    for gate in _GATES:
        g = allpts[allpts[f"pass_{gate}"].astype(bool)]
        curves[gate] = _cumulative_unique(
            g["abs_time"].astype(float).to_numpy(),
            g["cluster_id"].to_numpy(dtype=object),
        )
    return curves


def plot_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    boundaries: list[float],
    out_png: Path,
    target_name: str,
    target_successes: int,
) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    any_points = any(xs.size for xs, _ in curves.values())
    if not any_points:
        ax.text(0.5, 0.5, "No confirmed successes", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, max(boundaries) if boundaries else 1)
        ax.set_ylim(0, target_successes)
    else:
        for gate, (xs, ys) in curves.items():
            color = _GATE_COLORS.get(gate, None)
            label = _GATE_LABELS.get(gate, gate)
            headline = gate == _HEADLINE_GATE
            alpha = 1.0 if headline else 0.85
            linewidth = 2.2 if headline else 1.8
            if xs.size == 0:
                ax.step([0.0], [0], where="post", color=color, linewidth=linewidth,
                        alpha=alpha, label=f"{label} (0)")
                continue
            xs_plot = np.concatenate([[0.0], xs])
            ys_plot = np.concatenate([[0], ys])
            ax.step(xs_plot, ys_plot, where="post", color=color, linewidth=linewidth,
                    alpha=alpha, label=label)
            ax.scatter(xs, ys, s=18 if headline else 14, color=color, alpha=alpha, zorder=3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    ax.axhline(y=target_successes, color="#d62728", linestyle="-", linewidth=1.2,
               label=f"target = {target_successes}")
    for i, b in enumerate(boundaries):
        ax.axvline(x=b, color="0.5", linestyle="--", linewidth=0.9,
                   label="round boundary" if i == 0 else None)

    ax.set_xlabel("Cumulative GPU hours (all rounds)")
    ax.set_ylabel("Unique confirmed successes")
    ax.set_title(f"Adaptive best-of-N confidence search: {target_name}")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-round cumulative unique AF2-confirmed successes vs GPU hours.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--round-dir", action="append", required=True, type=Path, dest="round_dirs",
                   help="A round output dir (repeatable, in round order).")
    p.add_argument("--round-wall-clock-hours", action="append", required=True, type=float,
                   dest="wall_clocks", help="Wall-clock (h) for the matching --round-dir (repeatable).")
    p.add_argument("--out-png", required=True, type=Path)
    p.add_argument("--target-name", required=True, type=str)
    p.add_argument("--target-successes", type=int, default=100)
    args = p.parse_args(argv)
    if len(args.round_dirs) != len(args.wall_clocks):
        p.error("number of --round-dir must equal number of --round-wall-clock-hours")
    return args


def run(args: argparse.Namespace) -> None:
    curves = build_multiround_curves(args.round_dirs, args.wall_clocks)
    boundaries = round_boundaries(args.wall_clocks)
    plot_curves(curves, boundaries, args.out_png, args.target_name, args.target_successes)

    series_csv = args.out_png.with_suffix(".series.csv")
    series_frames = [
        pd.DataFrame(
            {"gate": gate, "elapsed_gpu_hours": xs, "cumulative_unique_successes": ys}
        )
        for gate, (xs, ys) in curves.items()
    ]
    pd.concat(series_frames, axis=0, ignore_index=True).to_csv(series_csv, index=False)

    print(f"[multiround] rounds: {len(args.round_dirs)}")
    for gate, (xs, ys) in curves.items():
        endpoint = int(ys[-1]) if ys.size else 0
        print(f"[multiround] gate={gate}: points={xs.size} unique endpoint={endpoint}")
    print(f"[multiround] round boundaries (h): {[round(b, 3) for b in boundaries]}")
    print(f"[multiround] curve -> {series_csv}")
    print(f"[multiround] plot  -> {args.out_png}")


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
