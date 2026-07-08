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

from scipy.stats import pearsonr  # noqa: E402

from proteinfoundation.result_analysis.binder_analysis import filter_by_success_thresholds  # noqa: E402
from proteinfoundation.result_analysis.binder_analysis_utils import (  # noqa: E402
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
)

# metadata_tag has two on-disk shapes:
#   NEW    `bon_it{i}_orig{s}_r{r}`  — iteration-unique (one row per candidate);
#   LEGACY `bon_orig{s}_r{r}`        — collides across best-of-N iterations.
# Both are embedded in the per-sample dir basename `job_{job_id}_n_{n}_id_{k}_{tag}`, so we
# recover the tag from the analyze CSV's pdb_path (robust to the underscores in the tag).
_METADATA_TAG_RE = re.compile(r"(bon_(?:it\d+_)?orig\d+_r\d+)")

# `id_{k}` from the same dir basename; used to order colliding legacy rows for the positional join.
_ID_K_RE = re.compile(r"_id_(\d+)_")

# head+scRMSD gate: the confidence head already vetted ipAE/pLDDT in the search loop
# (its provisional_success bool = head ipAE<7 A AND head complex pLDDT>0.92), so this
# gate trusts the head's two interface metrics and asks AF2 only to confirm the fold
# geometry (binder scRMSD_ca < 1.5). The mask is provisional_success AND that single AF2
# threshold — NOT a pure AF2-threshold filter.
_SCRMSD_THRESHOLD = {"scRMSD_ca": DEFAULT_PROTEIN_BINDER_THRESHOLDS["scRMSD_ca"]}

# AF2 stores i_pAE normalised by 31; multiply back to Angstroms for apples-to-apples parity
# against the head's raw-Angstrom confidence_ipae.
_AF2_IPAE_ANGSTROM_SCALE = 31.0

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
    """Recover the metadata_tag (new or legacy shape) from a sample pdb_path / dir basename."""
    if not isinstance(pdb_path, str):
        return None
    match = _METADATA_TAG_RE.search(pdb_path)
    return match.group(1) if match else None


def extract_id_k(pdb_path: str | float) -> int | None:
    """Recover the per-sample `id_{k}` index from a sample pdb_path / dir basename."""
    if not isinstance(pdb_path, str):
        return None
    match = _ID_K_RE.search(pdb_path)
    return int(match.group(1)) if match else None


def _cluster_id_column(pdb_path: str | float) -> str | None:
    """Fallback cluster id: the sample dir basename (each unmatched sample is its own cluster)."""
    if not isinstance(pdb_path, str):
        return None
    return os.path.basename(os.path.dirname(pdb_path)) or os.path.basename(pdb_path)


def _discover_foldseek_assignments(analyze_csv: Path) -> dict[str, str]:
    """Basename -> namespaced cluster id from any cluster_assignments*.csv next to analyze_csv."""
    basename_to_cluster: dict[str, str] = {}
    for f in sorted(analyze_csv.parent.rglob("cluster_assignments*.csv")):
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
    return basename_to_cluster


def foldseek_clustering_available(analyze_csv: Path, cluster_col: str | None) -> bool:
    """True iff real Foldseek clustering will be used (explicit column or discovered CSV).

    When False the caller falls back to per-sample-dir ids (each sample its own cluster), so the
    plot must be annotated honestly rather than implying a TM threshold that was never applied.
    """
    if cluster_col is not None:
        return True
    return bool(_discover_foldseek_assignments(analyze_csv))


def load_foldseek_clusters(analyze_csv: Path, cluster_col: str | None, df: pd.DataFrame) -> pd.Series:
    """Return a per-row Foldseek cluster id Series aligned to ``df``.

    Resolution order:
      1. an explicit per-sample cluster-id column in the analyze CSV (``--cluster-col``);
      2. auto-discovered ``cluster_assignments_*.csv`` files (columns
         ``cluster_index,sample_index,path_name``) written by diversity_foldseek, joined on
         the complex pdb_path (basename match, tolerant of relocated run dirs);
      3. fallback: the sample dir basename (one cluster per sample) so the curve still forms.

    Use :func:`foldseek_clustering_available` to know which path was taken for annotation.
    """
    if cluster_col is not None:
        if cluster_col not in df.columns:
            raise ValueError(f"--cluster-col '{cluster_col}' not found in analyze CSV columns")
        return df[cluster_col].astype("object")

    basename_to_cluster = _discover_foldseek_assignments(analyze_csv)
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
    """Join analyze rows to the sidecar on metadata_tag parsed from pdb_path.

    Two modes, auto-detected by whether any tag repeats in the sidecar:

    - NEW runs (iteration-unique tags `bon_it{i}_orig{s}_r{r}`): a plain 1:1 inner
      merge on metadata_tag — every tag identifies exactly one candidate.
    - LEGACY runs (colliding tags `bon_orig{s}_r{r}` re-emitted every best-of-N
      iteration): a plain merge would fan out (N analyze x M sidecar per tag). Within
      each tag bucket the per-tag counts of analyze rows and provisional-success
      sidecar rows are identical, so we do a POSITIONAL within-bucket join: the k-th
      provisional-success sidecar row (ordered by elapsed_gpu_hours) maps to the k-th
      analyze row (ordered by the id_k parsed from the pdb_path dir basename). Both
      orderings are stable, so the pairing is deterministic.

      LIMITATION (legacy path only): id_k is a per-LENGTH counter (generate.py
      save_predictions: `samples_per_length[n]`), while the legacy tag keys only on
      the within-batch sample index. So a single legacy bucket can mix multiple
      lengths n, id_k is not unique within the bucket, and the sidecar carries no
      length column to disambiguate. The k-th-by-id_k / k-th-by-time pairing is then
      only exact when a bucket is single-length; across multi-length legacy buckets
      it is a best-effort reconstruction that may mispair a sidecar row to the wrong
      physical sample of the same length. This affects ONLY re-plots of pre-fix runs;
      NEW runs emit iteration-unique tags and take the exact 1:1 branch above. The
      aggregate cumulative-successes curve is robust to this (it only shuffles which
      near-identical timestamp a success lands on within a narrow bucket); the
      per-sample parity on legacy data is not, which is one more reason the legacy
      parity number is not trustworthy (see compute_parity's range-restriction caveat).
    """
    if "pdb_path" not in analyze_df.columns:
        raise ValueError("analyze CSV must contain a 'pdb_path' column")
    if "metadata_tag" not in sidecar_df.columns:
        raise ValueError("sidecar CSV must contain a 'metadata_tag' column")

    analyze_df = analyze_df.copy()
    analyze_df["metadata_tag"] = analyze_df["pdb_path"].map(extract_metadata_tag)
    analyze_df = analyze_df[analyze_df["metadata_tag"].notna()]

    legacy = bool(sidecar_df["metadata_tag"].duplicated().any())
    if not legacy:
        return pd.merge(analyze_df, sidecar_df, on="metadata_tag", how="inner", suffixes=("", "_sidecar"))

    return _positional_join(analyze_df, sidecar_df)


def _positional_join(analyze_df: pd.DataFrame, sidecar_df: pd.DataFrame) -> pd.DataFrame:
    """Within-bucket positional join for legacy colliding tags (see join_analyze_and_sidecar)."""
    analyze_df = analyze_df.copy()
    analyze_df["_id_k"] = analyze_df["pdb_path"].map(extract_id_k)

    if "provisional_success" in sidecar_df.columns:
        prov = sidecar_df[sidecar_df["provisional_success"].astype(bool)].copy()
    else:
        prov = sidecar_df.copy()

    time_col = "elapsed_gpu_hours" if "elapsed_gpu_hours" in prov.columns else None
    analyze_suffix_cols = set(analyze_df.columns)

    paired: list[pd.DataFrame] = []
    for tag, s_bucket in prov.groupby("metadata_tag", sort=False):
        a_bucket = analyze_df[analyze_df["metadata_tag"] == tag]
        if a_bucket.empty:
            continue
        a_ordered = a_bucket.sort_values("_id_k", kind="stable").reset_index(drop=True)
        s_ordered = (
            s_bucket.sort_values(time_col, kind="stable") if time_col else s_bucket
        ).reset_index(drop=True)
        k = min(len(a_ordered), len(s_ordered))
        a_k = a_ordered.iloc[:k].reset_index(drop=True)
        s_k = s_ordered.iloc[:k].reset_index(drop=True)
        # Drop overlapping columns from the sidecar side (analyze wins) to mirror
        # the suffixes=("", "_sidecar") of the 1:1 merge without duplicate columns.
        s_k = s_k.drop(columns=[c for c in s_k.columns if c in analyze_suffix_cols and c != "metadata_tag"])
        s_k = s_k.drop(columns=["metadata_tag"])
        paired.append(pd.concat([a_k, s_k], axis=1))

    if not paired:
        return analyze_df.iloc[0:0].copy()
    joined = pd.concat(paired, axis=0, ignore_index=True)
    return joined.drop(columns=["_id_k"], errors="ignore")


# Both success curves. The gate name maps to how its mask is computed in
# af2_confirmed_mask (alphaproteo -> pure AF2-threshold filter; head_plus_scrmsd
# -> the sidecar provisional_success bool AND the single AF2 scRMSD threshold).
_GATES = ("head_plus_scrmsd", "alphaproteo")

# The AlphaProteo gate is the HEADLINE success gate: the published rule
# (Bennett et al.) evaluated on AF2's independent refold. The head+scRMSD gate is
# the head's own two interface metrics (ipAE<7 A, complex pLDDT>0.92, folded into
# provisional_success) confirmed by AF2's fold geometry (scRMSD_ca<1.5) — a
# legitimate, cheaper success curve, shown alongside the headline.
_GATE_LABELS = {
    "alphaproteo": "AF2 AlphaProteo success (headline)",
    "head_plus_scrmsd": "head ipAE+pLDDT + AF2 scRMSD",
}

# The headline success gate: the one whose endpoint is the reported yield.
_HEADLINE_GATE = "alphaproteo"


def af2_confirmed_mask(df: pd.DataFrame, gate: str = "alphaproteo") -> pd.Series:
    """Boolean mask of rows passing the requested success gate.

    Two gates:

    - ``alphaproteo``: full paper rule on AF2's refold (i_pAE*31<=7, pLDDT>=0.9,
      scRMSD_ca<1.5), via the repo's ``filter_by_success_thresholds`` (which reads
      the ``_all`` list columns), so the threshold semantics stay in one place;
    - ``head_plus_scrmsd``: the sidecar ``provisional_success`` bool (head ipAE<7 A
      AND head complex pLDDT>0.92, computed by the confidence scorer during search)
      AND the single AF2 scRMSD_ca<1.5 threshold. NOT a pure AF2-threshold filter —
      the head supplies the two interface metrics, AF2 supplies only the fold
      geometry. Reads ``provisional_success`` explicitly so it is correct even on an
      unfiltered df; ``run`` pre-filters to provisional successes, within which the
      bool is all-True and the gate reduces to the AF2 scRMSD<1.5 subset.
    """
    if gate not in _GATES:
        raise ValueError(f"unknown gate '{gate}'; expected one of {sorted(_GATES)}")
    if df.empty:
        return pd.Series([], dtype=bool)
    if gate == "head_plus_scrmsd":
        if "provisional_success" not in df.columns:
            raise ValueError("head_plus_scrmsd gate requires a 'provisional_success' column")
        prov = df["provisional_success"].astype(bool)
        confirmed = filter_by_success_thresholds(
            df, seq_type=_SEQ_TYPE, success_thresholds=_SCRMSD_THRESHOLD, save_json=False
        )
        scrmsd_ok = df.index.isin(confirmed.index)
        return prov & pd.Series(scrmsd_ok, index=df.index)
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
    eval_time_col: str = "eval_finish_s",
) -> tuple[np.ndarray, np.ndarray]:
    """Order confirmed rows by GPU hours and count first-seen Foldseek clusters cumulatively.

    Returns (x, y): x = elapsed GPU hours, y = running count of unique clusters. Monotone
    non-decreasing by construction; endpoint = number of distinct clusters among confirmed rows.

    The true wall-clock cost of a confirmed sample is generation time + AF2 refold time. When
    ``eval_time_col`` (seconds from eval start) is present, x = generation ``time_col`` +
    ``eval_time_col``/3600. Legacy runs without the column degrade to generation-only x.
    """
    if df_confirmed.empty:
        return np.array([], dtype=float), np.array([], dtype=int)

    gen_hours = df_confirmed[time_col].astype(float).to_numpy()
    if eval_time_col in df_confirmed.columns:
        eval_hours = pd.to_numeric(df_confirmed[eval_time_col], errors="coerce").to_numpy() / 3600.0
        eval_hours = np.nan_to_num(eval_hours, nan=0.0)
        total_hours = gen_hours + eval_hours
    else:
        total_hours = gen_hours

    order = np.argsort(total_hours, kind="stable")
    times = total_hours[order]
    clusters = cluster_ids.to_numpy(dtype=object)[order]

    seen: set = set()
    xs: list[float] = []
    ys: list[int] = []
    for t, c in zip(times, clusters):
        seen.add(c)
        xs.append(float(t))
        ys.append(len(seen))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


# The parity join here is the SELECTED provisional-success set: the head only
# lets low-confidence_ipae candidates through, so this population is truncated to
# a narrow slice of the head's predictor range. Correlation attenuates toward 0
# under range restriction (Thorndike case II) REGARDLESS of the true full-range
# association, so a near-zero here is uninformative about whether the head tracks
# AF2 — in either direction. The report carries this caveat, and downstream
# display MUST NOT present the coefficient as head-quality evidence. A valid
# parity needs AF2 scores on the FULL candidate pool (head-accepted AND -rejected)
# stratified across the head's confidence range — a separate follow-up experiment.
_PARITY_RANGE_RESTRICTION_CAVEAT = (
    "Computed on the head's self-selected provisional-success set (confidence_ipae truncated near "
    "the acceptance gate); range-restricted and NOT valid evidence of head-vs-AF2 agreement. A "
    "full-pool stratified parity (AF2 on head-accepted and head-rejected candidates) is the "
    "correct measurement and is deferred to a follow-up."
)


def compute_parity(joined: pd.DataFrame) -> dict[str, object]:
    """Spearman + Pearson of head confidence_ipae (and ipae_reuse if present) vs AF2 ipAE.

    The head's confidence_ipae is in raw Angstroms; AF2 stores i_pAE normalised by 31, so we
    rescale AF2 back to Angstroms (``* 31``) before reporting for an apples-to-apples display.
    (Both Spearman and Pearson are invariant to the constant scale, so this is interpretability
    only, not a correlation change.) Uses the analyze best-sample AF2 ipAE column
    (`self_complex_i_pAE`); falls back to the min of the `_all` list if absent. NaNs are dropped
    pairwise. Runs on the collapsed 1:1 / positional join, so ``n`` is the candidate count.

    The returned coefficients are range-restricted (see ``_PARITY_RANGE_RESTRICTION_CAVEAT``,
    surfaced in the report under ``caveat``) and must not be shown as head-quality evidence.
    """
    report: dict[str, object] = {
        "n_joined": int(len(joined)),
        "caveat": _PARITY_RANGE_RESTRICTION_CAVEAT,
        "valid_head_quality_evidence": False,
    }

    af2 = _af2_ipae_series(joined)
    if af2 is None:
        report["error"] = f"AF2 ipAE column '{_AF2_IPAE_COL}' (or list variant) not found in analyze CSV"
        return report

    af2 = af2 * _AF2_IPAE_ANGSTROM_SCALE
    report["af2_ipae_units"] = "angstrom"
    if af2.notna().any():
        report["af2_ipae_max_angstrom"] = float(af2.max())

    def _corr(head_col: str) -> dict[str, object] | None:
        if head_col not in joined.columns:
            return None
        head = pd.to_numeric(joined[head_col], errors="coerce")
        pair = pd.DataFrame({"head": head, "af2": af2}).dropna()
        if len(pair) < 2:
            return {"n": int(len(pair)), "spearman": None, "pearson": None, "note": "too few paired non-NaN points"}
        rho, rho_p = spearmanr(pair["head"], pair["af2"])
        r, r_p = pearsonr(pair["head"], pair["af2"])
        return {
            "n": int(len(pair)),
            "spearman": float(rho),
            "spearman_pvalue": float(rho_p),
            "pearson": float(r),
            "pearson_pvalue": float(r_p),
        }

    native = _corr("confidence_ipae")
    if native is not None:
        report["confidence_ipae_vs_af2_ipae"] = native

    reuse = _corr("ipae_reuse")
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


_GATE_COLORS = {
    "head_plus_scrmsd": "#2ca02c",
    "alphaproteo": "#1f77b4",
}


def _foldseek_annotation(foldseek_applied: bool, tm_thresholds: list[float]) -> str:
    if foldseek_applied:
        tm_str = ", ".join(f"{t:g}" for t in tm_thresholds)
        return f"Foldseek TM threshold: {tm_str}"
    return "no Foldseek clustering (per-sample)"


def plot_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_png: Path,
    target_name: str,
    tm_thresholds: list[float],
    foldseek_applied: bool,
) -> None:
    """Fig-21-style step curves: x = GPU hours, y = # unique successes, one curve per gate."""
    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    any_points = any(xs.size for xs, _ in curves.values())
    if not any_points:
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
        for gate, (xs, ys) in curves.items():
            color = _GATE_COLORS.get(gate, None)
            label = _GATE_LABELS.get(gate, gate)
            # Both are legitimate success curves. The headline (AlphaProteo) gate
            # is the prominent solid line; the head+scRMSD gate is a second solid
            # line in its own colour, visually distinguishable but not faded.
            headline = gate == _HEADLINE_GATE
            linestyle = "-"
            alpha = 1.0 if headline else 0.85
            linewidth = 2.2 if headline else 1.8
            if xs.size == 0:
                # Still register the label so the legend documents both gates.
                ax.step(
                    [0.0], [0], where="post", color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, label=f"{label} (0)",
                )
                continue
            xs_plot = np.concatenate([[0.0], xs])
            ys_plot = np.concatenate([[0], ys])
            ax.step(
                xs_plot, ys_plot, where="post", color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, label=label,
            )
            ax.scatter(xs, ys, s=18 if headline else 14, color=color, alpha=alpha, zorder=3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.legend(loc="lower right", frameon=True)

    ax.set_xlabel("Elapsed GPU hours (generation + AF2 refold)")
    ax.set_ylabel("Unique AF2-confirmed successes")
    ax.set_title(
        f"Best-of-N confidence search: {target_name}\n({_foldseek_annotation(foldseek_applied, tm_thresholds)})"
    )
    # Honest footnote: the head-vs-AF2 parity coefficient is range-restricted and
    # deliberately not shown as a number on this figure.
    fig.text(
        0.5,
        0.005,
        "head-vs-AF2 ipAE parity not estimable on the selected set (range restriction); full-pool parity pending",
        ha="center",
        va="bottom",
        fontsize=6.5,
        style="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

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

    foldseek_applied = foldseek_clustering_available(args.analyze_csv, args.cluster_col)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    series_frames: list[pd.DataFrame] = []
    endpoints: dict[str, int] = {}
    for gate in _GATES:
        confirmed = prov[af2_confirmed_mask(prov, gate=gate)].copy()
        cluster_ids = load_foldseek_clusters(args.analyze_csv, args.cluster_col, confirmed)
        xs, ys = cumulative_unique_curve(confirmed, cluster_ids)
        curves[gate] = (xs, ys)
        endpoints[gate] = int(ys[-1]) if ys.size else 0
        series_frames.append(
            pd.DataFrame(
                {
                    "gate": gate,
                    "elapsed_gpu_hours": xs,
                    "cumulative_unique_successes": ys,
                }
            )
        )

    plot_curves(curves, args.out_png, args.target_name, args.tm_thresholds, foldseek_applied)

    series_csv = args.out_png.with_suffix(".series.csv")
    pd.concat(series_frames, axis=0, ignore_index=True).to_csv(series_csv, index=False)

    parity_path = args.out_png.with_suffix(".parity.json")
    with open(parity_path, "w") as f:
        json.dump(parity, f, indent=2)

    print(f"[plot] joined candidates: {len(joined)}")
    print(f"[plot] provisional successes: {len(prov)}")
    print(f"[plot] Foldseek clustering applied: {foldseek_applied}")
    for gate in _GATES:
        print(f"[plot] gate={gate}: unique endpoint = {endpoints[gate]}")
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
