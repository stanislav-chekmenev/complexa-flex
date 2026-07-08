"""Bug-fix tests for plot_successes_vs_gpu_hours: collision-safe join, two gates,
Angstrom parity, per-sample eval-time offset, and honest Foldseek annotation.

These reproduce the production collision (a metadata_tag repeated across best-of-N
iterations) that the original synthetic-tag tests never exercised, and pin the
two-mode join (1:1 for iteration-unique NEW tags, positional within-bucket for
LEGACY colliding tags).

CPU-only. Run: .venv/bin/python -m pytest tests/unit/plot/test_plot_join_and_gates.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _PROJECT_ROOT / "script_utils" / "plot" / "plot_successes_vs_gpu_hours.py"

_spec = importlib.util.spec_from_file_location("plot_successes_vs_gpu_hours", _SCRIPT)
plot_mod = importlib.util.module_from_spec(_spec)
sys.modules["plot_successes_vs_gpu_hours"] = plot_mod
_spec.loader.exec_module(plot_mod)


def _pdb_path(job_id: int, n: int, k: int, tag: str) -> str:
    dir_name = f"job_{job_id}_n_{n}_id_{k}_{tag}"
    return f"/tmp/inference/run/{dir_name}/{dir_name}_binder.pdb"


def _analyze_row(tag: str, k: int, ipae: float, plddt: float, scrmsd: float) -> dict:
    return {
        "pdb_path": _pdb_path(job_id=42, n=80, k=k, tag=tag),
        "self_complex_i_pAE": ipae,
        "self_complex_i_pAE_all": [ipae],
        "self_complex_pLDDT": plddt,
        "self_complex_pLDDT_all": [plddt],
        "self_binder_scRMSD_ca": scrmsd,
        "self_binder_scRMSD_ca_all": [scrmsd],
    }


# --------------------------------------------------------------------------- #
# NEW-format regex + tag parse
# --------------------------------------------------------------------------- #

def test_new_format_tag_parse():
    tag = plot_mod.extract_metadata_tag(_pdb_path(42, 80, 3, "bon_it7_orig2_r1"))
    assert tag == "bon_it7_orig2_r1"


def test_legacy_format_tag_still_parses():
    tag = plot_mod.extract_metadata_tag(_pdb_path(42, 80, 0, "bon_orig2_r1"))
    assert tag == "bon_orig2_r1"


def test_id_k_parse():
    assert plot_mod.extract_id_k(_pdb_path(42, 80, 17, "bon_orig2_r1")) == 17
    assert plot_mod.extract_id_k("/tmp/no_id_here/x.pdb") is None


# --------------------------------------------------------------------------- #
# Bug 2(a)+(b): colliding legacy sidecar must NOT fan out
# --------------------------------------------------------------------------- #

def _legacy_colliding_frames():
    """Two analyze rows and two sidecar rows sharing ONE legacy tag across 2 iterations.

    analyze:   id_0 (good), id_1 (bad ipae)     -> both tag bon_orig0_r0
    sidecar:   two provisional-success rows, distinct elapsed_gpu_hours, same tag.
    Positional: k-th analyze (by id_k) <-> k-th sidecar (by elapsed hours).
    """
    analyze_df = pd.DataFrame(
        [
            _analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0),
            _analyze_row("bon_orig0_r0", k=1, ipae=0.90, plddt=0.95, scrmsd=1.0),
        ]
    )
    sidecar_df = pd.DataFrame(
        {
            "metadata_tag": ["bon_orig0_r0", "bon_orig0_r0"],
            "confidence_ipae": [3.1, 6.0],
            "provisional_success": [True, True],
            "elapsed_gpu_hours": [0.5, 1.2],
        }
    )
    return analyze_df, sidecar_df


def test_colliding_sidecar_does_not_fan_out():
    analyze_df, sidecar_df = _legacy_colliding_frames()
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)
    # Naive merge would give 2 analyze x 2 sidecar = 4 rows. Correct join is 2.
    assert len(joined) == len(analyze_df) == 2


def test_positional_join_pairs_by_rank():
    analyze_df, sidecar_df = _legacy_colliding_frames()
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)
    joined = joined.sort_values("elapsed_gpu_hours").reset_index(drop=True)
    # earliest sidecar (0.5h, conf 3.1) <-> id_0 analyze (ipae 0.10)
    assert joined.loc[0, "elapsed_gpu_hours"] == pytest.approx(0.5)
    assert joined.loc[0, "self_complex_i_pAE"] == pytest.approx(0.10)
    # later sidecar (1.2h, conf 6.0) <-> id_1 analyze (ipae 0.90)
    assert joined.loc[1, "elapsed_gpu_hours"] == pytest.approx(1.2)
    assert joined.loc[1, "self_complex_i_pAE"] == pytest.approx(0.90)


def test_new_format_uses_one_to_one_merge():
    analyze_df = pd.DataFrame(
        [
            _analyze_row("bon_it0_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0),
            _analyze_row("bon_it1_orig0_r0", k=1, ipae=0.12, plddt=0.93, scrmsd=1.1),
        ]
    )
    sidecar_df = pd.DataFrame(
        {
            "metadata_tag": ["bon_it0_orig0_r0", "bon_it1_orig0_r0"],
            "confidence_ipae": [3.1, 3.7],
            "provisional_success": [True, True],
            "elapsed_gpu_hours": [0.5, 1.2],
        }
    )
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)
    assert len(joined) == 2
    assert set(joined["metadata_tag"]) == {"bon_it0_orig0_r0", "bon_it1_orig0_r0"}


# --------------------------------------------------------------------------- #
# Bug 2(d): parity n is the join count (not fanned), Angstrom rescale, Pearson
# --------------------------------------------------------------------------- #

def test_parity_n_from_collapsed_join_and_angstrom_and_pearson():
    analyze_df, sidecar_df = _legacy_colliding_frames()
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)
    report = plot_mod.compute_parity(joined)
    native = report["confidence_ipae_vs_af2_ipae"]
    # n is 2 (collapsed), NOT 4 (fanned).
    assert native["n"] == 2
    assert "spearman" in native and native["spearman"] is not None
    assert "pearson" in native and native["pearson"] is not None
    # AF2 ipAE reported in Angstroms: max stored value 0.90 -> ~27.9 A after *31.
    assert report["af2_ipae_units"] == "angstrom"
    assert report["af2_ipae_max_angstrom"] == pytest.approx(0.90 * 31.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Bug 2(c): both gates
# --------------------------------------------------------------------------- #

def _head_scrmsd_row(tag: str, prov: bool, af2_scrmsd: float) -> pd.DataFrame:
    """One joined-style row: sidecar provisional_success + AF2 scRMSD columns.

    The head has already vetted ipAE/pLDDT, so the AF2 ipAE/pLDDT here are set to
    values that would FAIL AlphaProteo, to prove the head+scRMSD gate does NOT
    read them (it trusts the head bool + the single AF2 scRMSD threshold).
    """
    row = _analyze_row(tag, k=0, ipae=0.90, plddt=0.5, scrmsd=af2_scrmsd)
    row["provisional_success"] = prov
    return pd.DataFrame([row])


def test_head_plus_scrmsd_gate_is_prov_and_af2_scrmsd():
    # (a) prov=True + AF2 scRMSD<1.5 -> in curve2.
    a = _head_scrmsd_row("bon_orig0_r0", prov=True, af2_scrmsd=1.0)
    assert bool(np.asarray(plot_mod.af2_confirmed_mask(a, gate="head_plus_scrmsd"))[0])
    # (b) prov=True + AF2 scRMSD>=1.5 -> NOT in curve2.
    b = _head_scrmsd_row("bon_orig1_r0", prov=True, af2_scrmsd=2.0)
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(b, gate="head_plus_scrmsd"))[0])
    # (c) prov=False + AF2 scRMSD<1.5 -> NOT in curve2 (prov gate fails).
    c = _head_scrmsd_row("bon_orig2_r0", prov=False, af2_scrmsd=1.0)
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(c, gate="head_plus_scrmsd"))[0])


def test_head_plus_scrmsd_endpoint_counts_only_case_a():
    df = pd.concat(
        [
            _head_scrmsd_row("bon_orig0_r0", prov=True, af2_scrmsd=1.0),   # (a) counts
            _head_scrmsd_row("bon_orig1_r0", prov=True, af2_scrmsd=2.0),   # (b) excluded
            _head_scrmsd_row("bon_orig2_r0", prov=False, af2_scrmsd=1.0),  # (c) excluded
        ],
        ignore_index=True,
    )
    mask = plot_mod.af2_confirmed_mask(df, gate="head_plus_scrmsd")
    assert int(np.asarray(mask).sum()) == 1
    assert bool(np.asarray(mask)[0])


def test_head_plus_scrmsd_gate_rejects_bad_scrmsd():
    df = _head_scrmsd_row("bon_orig0_r0", prov=True, af2_scrmsd=5.0)
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(df, gate="head_plus_scrmsd"))[0])


def test_alphaproteo_gate_unchanged():
    # Full AlphaProteo rule: ipAE*31<=7, pLDDT>=0.9, scRMSD<1.5. All pass -> confirmed.
    good = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0)])
    assert bool(np.asarray(plot_mod.af2_confirmed_mask(good, gate="alphaproteo"))[0])
    # Any single criterion failing rejects.
    bad_ipae = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.90, plddt=0.95, scrmsd=1.0)])
    bad_plddt = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.50, scrmsd=1.0)])
    bad_scrmsd = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=5.0)])
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(bad_ipae, gate="alphaproteo"))[0])
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(bad_plddt, gate="alphaproteo"))[0])
    assert not bool(np.asarray(plot_mod.af2_confirmed_mask(bad_scrmsd, gate="alphaproteo"))[0])


def test_scrmsd_only_gate_removed():
    df = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0)])
    with pytest.raises(ValueError):
        plot_mod.af2_confirmed_mask(df, gate="scrmsd_only")


# --------------------------------------------------------------------------- #
# Bug 2(e): honest Foldseek annotation
# --------------------------------------------------------------------------- #

def test_foldseek_fallback_flagged(tmp_path):
    # No cluster_assignments*.csv anywhere -> per-sample fallback -> not available.
    analyze_csv = tmp_path / "binder_results_cfg_self_42.csv"
    df = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0)])
    df.to_csv(analyze_csv, index=False)
    assert plot_mod.foldseek_clustering_available(analyze_csv, None) is False
    assert "no Foldseek clustering" in plot_mod._foldseek_annotation(False, [0.5])


def test_foldseek_applied_flagged(tmp_path):
    analyze_csv = tmp_path / "binder_results_cfg_self_42.csv"
    df = pd.DataFrame([_analyze_row("bon_orig0_r0", k=0, ipae=0.10, plddt=0.95, scrmsd=1.0)])
    df.to_csv(analyze_csv, index=False)
    cluster_df = pd.DataFrame(
        {
            "cluster_index": [0],
            "sample_index": [1],
            "path_name": [df.loc[0, "pdb_path"]],
        }
    )
    cluster_df.to_csv(tmp_path / "cluster_assignments_x.csv", index=False)
    assert plot_mod.foldseek_clustering_available(analyze_csv, None) is True
    assert "TM threshold" in plot_mod._foldseek_annotation(True, [0.5])


# --------------------------------------------------------------------------- #
# Bug 3: per-sample eval-time offset added to the GPU-hours axis
# --------------------------------------------------------------------------- #

def test_eval_time_offset_added_when_column_present():
    df = pd.DataFrame(
        {
            "elapsed_gpu_hours": [1.0, 2.0],
            "eval_finish_s": [3600.0, 7200.0],  # 1h and 2h eval offsets
        }
    )
    cluster_ids = pd.Series(["a", "b"])
    xs, _ = plot_mod.cumulative_unique_curve(df, cluster_ids)
    # x = gen hours + eval_finish_s/3600 -> [2.0, 4.0]
    assert list(xs) == pytest.approx([2.0, 4.0])


def test_eval_time_offset_absent_degrades_to_generation_only():
    df = pd.DataFrame({"elapsed_gpu_hours": [1.0, 2.0]})
    cluster_ids = pd.Series(["a", "b"])
    xs, _ = plot_mod.cumulative_unique_curve(df, cluster_ids)
    assert list(xs) == pytest.approx([1.0, 2.0])
