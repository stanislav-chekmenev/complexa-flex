#!/usr/bin/env python3
"""
Cumulative unique AF2-confirmed successes vs elapsed GPU hours for a confidence-head
best-of-N binder search (paper Fig. 21 style), plus a confidence/AF2 parity report.

The confidence head labels provisional successes inside the search loop (no AF2). The
`evaluate` stage then refolds each candidate with AF2 (ColabDesign) to produce the true
AlphaProteo metrics, and `analyze` Foldseek-clusters the successful subset. This script
joins the analyze CSV (AF2 refold metrics + Foldseek cluster id) to the confidence sidecar
CSV (per-candidate confidence scores + elapsed GPU hours) on ``metadata_tag`` and:

  1. keeps provisional successes (sidecar), then AF2-confirmed successes (all three
     AlphaProteo criteria on the AF2 columns);
  2. orders confirmed rows by elapsed GPU hours and walks in time order, incrementing a
     running unique count each time a NEW Foldseek cluster id is first seen;
  3. plots the cumulative-unique-successes step curve (x = GPU hours, y = # unique
     successes) and dumps the plotted series to CSV;
  4. reports Spearman correlation between the head's confidence_ipae and the AF2 complex
     ipAE (and, if present, between the concat-reuse ipae_reuse and AF2 ipAE).

Usage:
    python script_utils/plot/plot_successes_vs_gpu_hours.py \
        --analyze-csv  inference/.../binder_results_..._<job>.csv \
        --sidecar-csv  inference/.../confidence_scores_<job>.csv \
        --out-png      inference/.../successes_vs_gpu_hours.png \
        --target-name  02_PDL1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root for imports (reuse the repo's success-threshold logic).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from proteinfoundation.result_analysis.binder_analysis import filter_by_success_thresholds  # noqa: E402
from proteinfoundation.result_analysis.binder_analysis_utils import (  # noqa: E402
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
)

# metadata_tag format is `bon_orig{s}_r{r}` (see generate.py save_predictions). It is also
# embedded in the per-sample dir basename `job_{job_id}_n_{n}_id_{k}_{metadata_tag}`, so we
# recover it from the analyze CSV's pdb_path by matching this pattern directly (robust to the
# underscores inside the tag itself).
_METADATA_TAG_RE = re.compile(r"(bon_orig\d+_r\d+)")

# The best-of-N search evaluates a single `self` sequence per candidate.
_SEQ_TYPE = "self"

# AF2 refold columns in the analyze CSV that we key on. build_column_name() emits
# `{seq_type}_{column_prefix}_{metric}_all` (list variant, one entry per redesign);
# filter_by_success_thresholds consumes those directly. The paper's AlphaProteo rule
# (DEFAULT_PROTEIN_BINDER_THRESHOLDS) checks:
#   self_complex_i_pAE_all      * 31 <= 7   (binder_analysis_utils.py:75)
#   self_complex_pLDDT_all      >= 0.9      (binder_analysis_utils.py:82)
#   self_binder_scRMSD_ca_all   <  1.5      (binder_analysis_utils.py:88)
_AF2_IPAE_COL = f"{_SEQ_TYPE}_complex_i_pAE"
_AF2_IPAE_LIST_COL = f"{_AF2_IPAE_COL}_all"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cumulative unique AF2-confirmed successes vs GPU hours + confidence parity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--analyze-csv", type=Path, required=True, help="Analyze binder_results_*.csv")
    parser.add_argument("--sidecar-csv", type=Path, required=True, help="confidence_scores_*.csv sidecar")
    parser.add_argument("--out-png", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--target-name", type=str, required=True, help="Target name for the plot title/legend")
    parser.add_argument(
        "--tm-thresholds",
        type=float,
        nargs="+",
        default=[0.5],
        help="Foldseek TM thresholds annotated on the plot (default: 0.5). Cluster ids are read as-is.",
    )
    parser.add_argument(
        "--cluster-col",
        type=str,
        default=None,
        help=(
            "Name of a per-sample Foldseek cluster-id column in the analyze CSV. If omitted, the "
            "script auto-discovers cluster_assignments_*.csv next to the analyze CSV and joins on pdb_path."
        ),
    )
    return parser.parse_args(argv)


def extract_metadata_tag(pdb_path: str | float) -> str | None:
    """Recover the `bon_orig{s}_r{r}` metadata_tag from a sample pdb_path / dir basename."""
    if not isinstance(pdb_path, str):
        return None
    match = _METADATA_TAG_RE.search(pdb_path)
    return match.group(1) if match else None


def _cluster_id_column(pdb_path: str | float) -> str | None:
    """Fallback cluster id: the sample dir basename (each unmatched sample is its own cluster)."""
    if not isinstance(pdb_path, str):
        return None
    return os.path.basename(os.path.dirname(pdb_path)) or os.path.basename(pdb_path)


def load_foldseek_clusters(analyze_csv: Path, cluster_col: str | None, df: pd.DataFrame) -> pd.Series:
    """Return a per-row Foldseek cluster id Series aligned to ``df``.

    Resolution order:
      1. an explicit per-sample cluster-id column in the analyze CSV (``--cluster-col``);
      2. auto-discovered ``cluster_assignments_*.csv`` files (columns
         ``cluster_index,sample_index,path_name``) written by diversity_foldseek, joined on
         the complex pdb_path (basename match, tolerant of relocated run dirs);
      3. fallback: the sample dir basename (one cluster per sample) so the curve still forms.
    """
    if cluster_col is not None:
        if cluster_col not in df.columns:
            raise ValueError(f"--cluster-col '{cluster_col}' not found in analyze CSV columns")
        return df[cluster_col].astype("object")

    assignment_files = sorted(analyze_csv.parent.rglob("cluster_assignments*.csv"))
    basename_to_cluster: dict[str, str] = {}
    for f in assignment_files:
        try:
            adf = pd.read_csv(f)
        except Exception:
            continue
        if not {"cluster_index", "path_name"}.issubset(adf.columns):
            continue
        # Namespace cluster indices by file so identical indices from different subsets do not collide.
        for _, arow in adf.iterrows():
            base = os.path.basename(str(arow["path_name"]))
            basename_to_cluster[base] = f"{f.stem}:{arow['cluster_index']}"

    if basename_to_cluster:
        def _lookup(pdb_path: str | float) -> str | None:
            if not isinstance(pdb_path, str):
                return None
            base = os.path.basename(pdb_path)
            if base in basename_to_cluster:
                return basename_to_cluster[base]
            return _cluster_id_column(pdb_path)

        return df["pdb_path"].map(_lookup)

    return df["pdb_path"].map(_cluster_id_column)


def join_analyze_and_sidecar(analyze_df: pd.DataFrame, sidecar_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join analyze rows to the sidecar on metadata_tag parsed from pdb_path."""
    if "pdb_path" not in analyze_df.columns:
        raise ValueError("analyze CSV must contain a 'pdb_path' column")
    if "metadata_tag" not in sidecar_df.columns:
        raise ValueError("sidecar CSV must contain a 'metadata_tag' column")

    analyze_df = analyze_df.copy()
    analyze_df["metadata_tag"] = analyze_df["pdb_path"].map(extract_metadata_tag)
    analyze_df = analyze_df[analyze_df["metadata_tag"].notna()]

    joined = pd.merge(analyze_df, sidecar_df, on="metadata_tag", how="inner", suffixes=("", "_sidecar"))
    return joined


def af2_confirmed_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows passing all three AlphaProteo AF2 criteria.

    Reuses the repo's filter_by_success_thresholds (which reads the `_all` list columns) so
    the threshold semantics (i_pAE*31<=7, pLDDT>=0.9, scRMSD_ca<1.5) stay in one place.
    """
    if df.empty:
        return pd.Series([], dtype=bool)
    confirmed = filter_by_success_thresholds(
        df,
        seq_type=_SEQ_TYPE,
        success_thresholds=DEFAULT_PROTEIN_BINDER_THRESHOLDS,
        save_json=False,
    )
    return df.index.isin(confirmed.index)


def cumulative_unique_curve(
    df_confirmed: pd.DataFrame,
    cluster_ids: pd.Series,
    time_col: str = "elapsed_gpu_hours",
) -> tuple[np.ndarray, np.ndarray]:
    """Order confirmed rows by GPU hours and count first-seen Foldseek clusters cumulatively.

    Returns (x, y): x = elapsed GPU hours, y = running count of unique clusters. Monotone
    non-decreasing by construction; endpoint = number of distinct clusters among confirmed rows.
    """
    if df_confirmed.empty:
        return np.array([], dtype=float), np.array([], dtype=int)

    order = df_confirmed[time_col].astype(float).argsort(kind="stable")
    times = df_confirmed[time_col].astype(float).to_numpy()[order]
    clusters = cluster_ids.to_numpy(dtype=object)[order]

    seen: set = set()
    xs: list[float] = []
    ys: list[int] = []
    for t, c in zip(times, clusters):
        seen.add(c)
        xs.append(float(t))
        ys.append(len(seen))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def compute_parity(joined: pd.DataFrame) -> dict[str, object]:
    """Spearman correlation of head confidence_ipae (and ipae_reuse if present) vs AF2 ipAE.

    Uses the analyze best-sample AF2 ipAE column (`self_complex_i_pAE`); falls back to the min
    of the `_all` list if the scalar column is absent. NaNs on either side are dropped pairwise.
    """
    report: dict[str, object] = {"n_joined": int(len(joined))}

    af2 = _af2_ipae_series(joined)
    if af2 is None:
        report["error"] = f"AF2 ipAE column '{_AF2_IPAE_COL}' (or list variant) not found in analyze CSV"
        return report

    def _spearman(head_col: str) -> dict[str, object] | None:
        if head_col not in joined.columns:
            return None
        head = pd.to_numeric(joined[head_col], errors="coerce")
        pair = pd.DataFrame({"head": head, "af2": af2}).dropna()
        if len(pair) < 2:
            return {"n": int(len(pair)), "spearman": None, "note": "too few paired non-NaN points"}
        rho, pval = spearmanr(pair["head"], pair["af2"])
        return {"n": int(len(pair)), "spearman": float(rho), "pvalue": float(pval)}

    native = _spearman("confidence_ipae")
    if native is not None:
        report["confidence_ipae_vs_af2_ipae"] = native

    reuse = _spearman("ipae_reuse")
    if reuse is not None:
        report["ipae_reuse_vs_af2_ipae"] = reuse

    return report


def _af2_ipae_series(joined: pd.DataFrame) -> pd.Series | None:
    """Best-sample AF2 complex ipAE as a numeric Series, or None if unavailable."""
    if _AF2_IPAE_COL in joined.columns:
        return pd.to_numeric(joined[_AF2_IPAE_COL], errors="coerce")
    if _AF2_IPAE_LIST_COL in joined.columns:
        def _min_of_list(v: object) -> float:
            vals = _coerce_list(v)
            return min(vals) if vals else np.nan

        return joined[_AF2_IPAE_LIST_COL].map(_min_of_list)
    return None


def _coerce_list(v: object) -> list[float]:
    """Coerce a cell that may hold a python list or a stringified list into floats."""
    if isinstance(v, (list, tuple)):
        seq = v
    elif isinstance(v, str):
        try:
            import ast

            seq = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    else:
        return []
    out = []
    for x in seq if isinstance(seq, (list, tuple)) else []:
        try:
            fx = float(x)
        except (TypeError, ValueError):
            continue
        if not (np.isnan(fx) or np.isinf(fx)):
            out.append(fx)
    return out


def plot_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    out_png: Path,
    target_name: str,
    tm_thresholds: list[float],
) -> None:
    """Fig-21-style step curve: x = GPU hours, y = # unique successes, single target curve."""
    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    if xs.size == 0:
        ax.text(
            0.5,
            0.5,
            "No AF2-confirmed successes",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Anchor the step curve at the origin so the y-axis starts from zero coverage.
        xs_plot = np.concatenate([[0.0], xs])
        ys_plot = np.concatenate([[0], ys])
        ax.step(xs_plot, ys_plot, where="post", color="#1f77b4", linewidth=2.0, label=target_name)
        ax.scatter(xs, ys, s=18, color="#1f77b4", zorder=3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.legend(loc="lower right", frameon=True)

    tm_str = ", ".join(f"{t:g}" for t in tm_thresholds)
    ax.set_xlabel("Elapsed GPU hours")
    ax.set_ylabel("Unique AF2-confirmed successes")
    ax.set_title(f"Best-of-N confidence search: {target_name}\n(Foldseek TM threshold: {tm_str})")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    """End-to-end: join, filter, build curve, plot, parity. Returns the parity report dict."""
    analyze_df = pd.read_csv(args.analyze_csv)
    sidecar_df = pd.read_csv(args.sidecar_csv)

    joined = join_analyze_and_sidecar(analyze_df, sidecar_df)

    parity = compute_parity(joined)

    if "provisional_success" not in joined.columns:
        raise ValueError("sidecar CSV must contain a 'provisional_success' column")
    prov = joined[joined["provisional_success"].astype(bool)].copy()

    confirmed_mask = af2_confirmed_mask(prov)
    confirmed = prov[confirmed_mask].copy()

    cluster_ids = load_foldseek_clusters(args.analyze_csv, args.cluster_col, confirmed)
    xs, ys = cumulative_unique_curve(confirmed, cluster_ids)

    plot_curve(xs, ys, args.out_png, args.target_name, args.tm_thresholds)

    series_csv = args.out_png.with_suffix(".series.csv")
    pd.DataFrame({"elapsed_gpu_hours": xs, "cumulative_unique_successes": ys}).to_csv(series_csv, index=False)

    parity_path = args.out_png.with_suffix(".parity.json")
    with open(parity_path, "w") as f:
        json.dump(parity, f, indent=2)

    print(f"[plot] joined candidates: {len(joined)}")
    print(f"[plot] provisional successes: {len(prov)}")
    print(f"[plot] AF2-confirmed successes: {len(confirmed)}")
    print(f"[plot] unique clusters (endpoint): {int(ys[-1]) if ys.size else 0}")
    print(f"[plot] curve -> {series_csv}")
    print(f"[plot] plot  -> {args.out_png}")
    print(f"[parity] {json.dumps(parity, indent=2)}")
    print(f"[parity] -> {parity_path}")

    return parity


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
