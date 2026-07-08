"""Unit tests for the best-of-N successes-vs-GPU-hours plot + parity script.

Feeds small synthetic analyze + sidecar CSVs and asserts:
  (a) the cumulative-unique curve is monotone non-decreasing and its endpoint equals the
      number of distinct Foldseek clusters among AF2-confirmed provisional successes;
  (b) a provisional-success candidate that AF2-fails is excluded;
  (c) parity returns a finite Spearman for a monotone synthetic relationship.

CPU-only. Run: .venv/bin/python -m pytest tests/unit/plot/test_plot_successes_vs_gpu_hours.py
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
    """Mirror generate.save_predictions naming: job_{job}_n_{n}_id_{k}_{tag}/...pdb."""
    dir_name = f"job_{job_id}_n_{n}_id_{k}_{tag}"
    return f"/tmp/inference/run/{dir_name}/{dir_name}_binder.pdb"


def _make_analyze_row(tag: str, ipae: float, plddt: float, scrmsd: float) -> dict:
    """One analyze CSV row with the `self_*_all` list columns filter_by_success_thresholds reads."""
    return {
        "pdb_path": _pdb_path(job_id=42, n=80, k=0, tag=tag),
        "self_complex_i_pAE": ipae,
        "self_complex_i_pAE_all": [ipae],
        "self_complex_pLDDT": plddt,
        "self_complex_pLDDT_all": [plddt],
        "self_binder_scRMSD_ca": scrmsd,
        "self_binder_scRMSD_ca_all": [scrmsd],
    }


@pytest.fixture
def synthetic_csvs(tmp_path: Path):
    """Build synthetic analyze + sidecar CSVs and a cluster_assignments CSV.

    Rows (tag -> semantics):
      r0: prov success, AF2 pass, cluster A
      r1: prov success, AF2 pass, cluster A  (duplicate cluster -> no new unique)
      r2: prov success, AF2 pass, cluster B  (new unique)
      r3: prov success, AF2 FAIL (bad ipAE) -> must be excluded
      r4: prov FAIL (dropped before AF2)    -> must be excluded
    Distinct clusters among AF2-confirmed provisional successes: {A, B} -> endpoint 2.
    """
    # AF2 pass thresholds: ipae*31 <= 7 (ipae <= 0.2258), plddt >= 0.9, scrmsd < 1.5.
    analyze_rows = [
        _make_analyze_row("bon_orig0_r0", ipae=0.10, plddt=0.95, scrmsd=1.0),
        _make_analyze_row("bon_orig1_r0", ipae=0.12, plddt=0.93, scrmsd=1.1),
        _make_analyze_row("bon_orig2_r0", ipae=0.15, plddt=0.92, scrmsd=1.2),
        _make_analyze_row("bon_orig3_r0", ipae=0.90, plddt=0.95, scrmsd=1.0),  # AF2 fail (ipAE)
        _make_analyze_row("bon_orig4_r0", ipae=0.10, plddt=0.95, scrmsd=1.0),  # prov fail
    ]
    analyze_df = pd.DataFrame(analyze_rows)
    analyze_csv = tmp_path / "binder_results_cfg_self_42.csv"
    analyze_df.to_csv(analyze_csv, index=False)

    # Sidecar: monotone confidence_ipae tracking AF2 ipae; elapsed hours in scrambled order to
    # exercise the time-ordering; ipae_reuse present but noisier.
    sidecar_df = pd.DataFrame(
        {
            "metadata_tag": ["bon_orig0_r0", "bon_orig1_r0", "bon_orig2_r0", "bon_orig3_r0", "bon_orig4_r0"],
            "confidence_ipae": [3.1, 3.7, 4.6, 27.9, 3.1],
            "ipae_reuse": [5.0, 5.5, 6.0, 20.0, 5.0],
            "confidence_complex_plddt": [0.95, 0.93, 0.92, 0.95, 0.95],
            "provisional_success": [True, True, True, True, False],
            "elapsed_gpu_hours": [2.0, 0.5, 3.0, 1.0, 4.0],
        }
    )
    sidecar_csv = tmp_path / "confidence_scores_42.csv"
    sidecar_df.to_csv(sidecar_csv, index=False)

    # Foldseek cluster assignments (diversity_foldseek format): r0,r1 -> cluster 0; r2 -> cluster 1.
    cluster_df = pd.DataFrame(
        {
            "cluster_index": [0, 0, 1],
            "sample_index": [1, 2, 3],
            "path_name": [
                _pdb_path(42, 80, 0, "bon_orig0_r0"),
                _pdb_path(42, 80, 0, "bon_orig1_r0"),
                _pdb_path(42, 80, 0, "bon_orig2_r0"),
            ],
        }
    )
    cluster_df.to_csv(tmp_path / "cluster_assignments_inf_cfg.csv", index=False)

    return analyze_csv, sidecar_csv, tmp_path


def test_metadata_tag_parse():
    tag = plot_mod.extract_metadata_tag(_pdb_path(42, 80, 0, "bon_orig7_r3"))
    assert tag == "bon_orig7_r3"
    assert plot_mod.extract_metadata_tag(float("nan")) is None
    assert plot_mod.extract_metadata_tag("/tmp/no_tag_here/x.pdb") is None


def test_curve_monotone_and_endpoint(synthetic_csvs):
    analyze_csv, sidecar_csv, tmp_path = synthetic_csvs
    out_png = tmp_path / "out.png"
    args = plot_mod.parse_args(
        [
            "--analyze-csv",
            str(analyze_csv),
            "--sidecar-csv",
            str(sidecar_csv),
            "--out-png",
            str(out_png),
            "--target-name",
            "02_PDL1",
        ]
    )
    plot_mod.run(args)

    series_all = pd.read_csv(out_png.with_suffix(".series.csv"))
    # Two gates are emitted: the headline AlphaProteo gate and the head+scRMSD gate.
    assert set(series_all["gate"]) == {"head_plus_scrmsd", "alphaproteo"}
    series = series_all[series_all["gate"] == "alphaproteo"]
    ys = series["cumulative_unique_successes"].to_numpy()
    xs = series["elapsed_gpu_hours"].to_numpy()

    # (a) monotone non-decreasing.
    assert np.all(np.diff(ys) >= 0)
    # time-ordered ascending.
    assert np.all(np.diff(xs) >= 0)
    # (a) endpoint == distinct clusters among AF2-confirmed provisional successes ({A, B} = 2).
    assert ys[-1] == 2
    # three AF2-confirmed provisional successes contribute three curve points.
    assert len(ys) == 3
    assert out_png.exists()

    # head+scRMSD gate = provisional_success AND AF2 scRMSD<1.5. r0,r1,r2 (clusters
    # A,B) plus r3 (prov=True, AF2 ipAE fails but scRMSD=1.0<1.5, its own cluster)
    # -> {A, B, r3} = 3 unique; r4 (prov=False) is dropped before AF2.
    hs = series_all[series_all["gate"] == "head_plus_scrmsd"]
    hs_ys = hs["cumulative_unique_successes"].to_numpy()
    assert np.all(np.diff(hs_ys) >= 0)
    assert hs_ys[-1] == 3


def test_af2_failing_provisional_excluded(synthetic_csvs):
    analyze_csv, sidecar_csv, tmp_path = synthetic_csvs
    analyze_df = pd.read_csv(analyze_csv)
    sidecar_df = pd.read_csv(sidecar_csv)
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)

    # Provisional successes: r0..r3 (r4 excluded by provisional_success == False).
    prov = joined[joined["provisional_success"].astype(bool)].copy()
    assert set(prov["metadata_tag"]) == {"bon_orig0_r0", "bon_orig1_r0", "bon_orig2_r0", "bon_orig3_r0"}

    confirmed_mask = plot_mod.af2_confirmed_mask(prov)
    confirmed = prov[confirmed_mask]
    # r3 AF2-fails (ipAE) and must be excluded; r4 already dropped as prov-fail.
    assert set(confirmed["metadata_tag"]) == {"bon_orig0_r0", "bon_orig1_r0", "bon_orig2_r0"}
    assert "bon_orig3_r0" not in set(confirmed["metadata_tag"])
    assert "bon_orig4_r0" not in set(confirmed["metadata_tag"])


def test_parity_finite_spearman(synthetic_csvs):
    analyze_csv, sidecar_csv, _ = synthetic_csvs
    analyze_df = pd.read_csv(analyze_csv)
    sidecar_df = pd.read_csv(sidecar_csv)
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, sidecar_df)

    report = plot_mod.compute_parity(joined)
    # (c) finite Spearman for the monotone confidence_ipae vs AF2 ipae relationship.
    native = report["confidence_ipae_vs_af2_ipae"]
    assert native["spearman"] is not None
    assert np.isfinite(native["spearman"])
    # confidence_ipae was built monotone in AF2 ipae across all 5 joined rows -> rho == 1.
    assert native["spearman"] == pytest.approx(1.0)
    assert native["n"] == 5

    # ipae_reuse present -> its parity line is emitted and finite.
    reuse = report["ipae_reuse_vs_af2_ipae"]
    assert reuse["spearman"] is not None
    assert np.isfinite(reuse["spearman"])


def test_missing_ipae_reuse_skipped(synthetic_csvs):
    analyze_csv, sidecar_csv, tmp_path = synthetic_csvs
    sidecar_df = pd.read_csv(sidecar_csv).drop(columns=["ipae_reuse"])
    sidecar_no_reuse = tmp_path / "confidence_scores_noreuse.csv"
    sidecar_df.to_csv(sidecar_no_reuse, index=False)

    analyze_df = pd.read_csv(analyze_csv)
    joined = plot_mod.join_analyze_and_sidecar(analyze_df, pd.read_csv(sidecar_no_reuse))
    report = plot_mod.compute_parity(joined)
    assert "confidence_ipae_vs_af2_ipae" in report
    assert "ipae_reuse_vs_af2_ipae" not in report


def test_no_successes_empty_plot(tmp_path: Path):
    # All provisional successes AF2-fail -> empty curve, empty-note plot, no crash.
    analyze_df = pd.DataFrame([_make_analyze_row("bon_orig0_r0", ipae=0.9, plddt=0.5, scrmsd=5.0)])
    analyze_csv = tmp_path / "binder_results_cfg_self_1.csv"
    analyze_df.to_csv(analyze_csv, index=False)
    sidecar_df = pd.DataFrame(
        {
            "metadata_tag": ["bon_orig0_r0"],
            "confidence_ipae": [3.0],
            "confidence_complex_plddt": [0.5],
            "provisional_success": [True],
            "elapsed_gpu_hours": [1.0],
        }
    )
    sidecar_csv = tmp_path / "confidence_scores_1.csv"
    sidecar_df.to_csv(sidecar_csv, index=False)

    out_png = tmp_path / "empty.png"
    args = plot_mod.parse_args(
        [
            "--analyze-csv",
            str(analyze_csv),
            "--sidecar-csv",
            str(sidecar_csv),
            "--out-png",
            str(out_png),
            "--target-name",
            "29_BHRF1",
        ]
    )
    plot_mod.run(args)
    assert out_png.exists()
    series = pd.read_csv(out_png.with_suffix(".series.csv"))
    assert len(series) == 0
