"""Unit: multi-round successes-vs-GPU-hours plot.

Concatenates per-round timelines onto ONE monotonic cumulative-GPU-hours axis:
round k's per-sample times are offset by the summed wall-clock of rounds
1..k-1, so round 2's points appear strictly past round 1's end. The cumulative
unique-success curve is globally deduped across rounds and stays monotone.

Reuses the single-round module's public join/confirm/cluster contract. CPU-only.
Run: .venv/bin/python -m pytest tests/unit/plot/test_plot_successes_vs_gpu_hours_multiround.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _PROJECT_ROOT / "script_utils" / "plot" / "plot_successes_vs_gpu_hours_multiround.py"

_spec = importlib.util.spec_from_file_location("plot_multiround", _SCRIPT)
mr = importlib.util.module_from_spec(_spec)
sys.modules["plot_multiround"] = mr
_spec.loader.exec_module(mr)


def _pdb_path(job_id: int, n: int, k: int, tag: str) -> str:
    dir_name = f"job_{job_id}_n_{n}_id_{k}_{tag}"
    return f"/tmp/inference/run/{dir_name}/{dir_name}_binder.pdb"


def _analyze_row(tag: str, ipae: float, plddt: float, scrmsd: float) -> dict:
    return {
        "pdb_path": _pdb_path(job_id=1, n=80, k=0, tag=tag),
        "self_complex_i_pAE": ipae,
        "self_complex_i_pAE_all": [ipae],
        "self_complex_pLDDT": plddt,
        "self_complex_pLDDT_all": [plddt],
        "self_binder_scRMSD_ca": scrmsd,
        "self_binder_scRMSD_ca_all": [scrmsd],
    }


def _write_round(round_dir: Path, tags: list[str], times: list[float], clusters: list[int]) -> None:
    """Write one round's analyze + sidecar + cluster_assignments CSVs.

    All rows are provisional successes and AF2-pass (ipae<=0.2258, plddt>=0.92,
    scrmsd<1.5), so success is decided by the shared confirm logic, and the
    only variable under test is the time axis + dedup. Because every row passes
    the full AlphaProteo gate it also passes head+scRMSD, so both curves are
    identical here (the two curves diverge only when AF2 ipAE/pLDDT fail while
    scRMSD passes; that case is covered in the single-round gate tests).
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    analyze_rows = [_analyze_row(t, ipae=0.1, plddt=0.95, scrmsd=1.0) for t in tags]
    pd.DataFrame(analyze_rows).to_csv(round_dir / "binder_results_cfg_self_1.csv", index=False)

    pd.DataFrame(
        {
            "metadata_tag": tags,
            "confidence_ipae": [3.0] * len(tags),
            "confidence_complex_plddt": [0.95] * len(tags),
            "provisional_success": [True] * len(tags),
            "elapsed_gpu_hours": times,
        }
    ).to_csv(round_dir / "confidence_scores_1.csv", index=False)

    pd.DataFrame(
        {
            "cluster_index": clusters,
            "sample_index": list(range(len(tags))),
            "path_name": [_pdb_path(1, 80, 0, t) for t in tags],
        }
    ).to_csv(round_dir / "cluster_assignments_cfg.csv", index=False)


@pytest.fixture
def two_rounds(tmp_path: Path):
    """Two rounds; round 1 spans [0, 2.9]h (wall 2.9), round 2 spans [0,1.0]h (wall 1.0).

    Round 1: two confirmed successes at within-round times 0.5, 2.9; clusters 0, 1.
    Round 2: two confirmed successes at within-round times 0.2, 1.0; clusters 0, 1
             but a DIFFERENT cluster file -> namespaced distinct from round 1.
    """
    r1 = tmp_path / "round_01"
    r2 = tmp_path / "round_02"
    _write_round(r1, tags=["bon_orig0_r0", "bon_orig1_r0"], times=[0.5, 2.9], clusters=[0, 1])
    _write_round(r2, tags=["bon_orig0_r0", "bon_orig1_r0"], times=[0.2, 1.0], clusters=[0, 1])
    return tmp_path, [r1, r2], [2.9, 1.0]


def test_round2_points_offset_past_round1(two_rounds):
    root, round_dirs, wall_clocks = two_rounds
    curves = mr.build_multiround_curves(round_dirs, wall_clocks)
    xs, ys = curves["alphaproteo"]

    # Four confirmed successes -> four curve points.
    assert len(xs) == 4
    # Round 1 within-round times 0.5, 2.9 -> absolute 0.5, 2.9.
    # Round 2 within-round times 0.2, 1.0 offset by 2.9 -> absolute 3.1, 3.9.
    np.testing.assert_allclose(np.sort(xs), [0.5, 2.9, 3.1, 3.9], atol=1e-9)
    # All round-2 points strictly exceed round 1's end (2.9).
    assert (xs[2:] > 2.9).all() if len(xs) == 4 else True


def test_both_curves_present_monotone_and_dedup(two_rounds):
    root, round_dirs, wall_clocks = two_rounds
    curves = mr.build_multiround_curves(round_dirs, wall_clocks)
    # Both success curves are built.
    assert set(curves) == {"head_plus_scrmsd", "alphaproteo"}
    for gate, (xs, ys) in curves.items():
        assert np.all(np.diff(xs) >= 0)
        assert np.all(np.diff(ys) >= 0)
        # 2 clusters/round, namespaced by round -> 4 distinct clusters total.
        # Every synthetic row passes both gates so both endpoints are 4.
        assert ys[-1] == 4
        # Round-2 points offset past round 1's end (2.9).
        assert (np.sort(xs)[2:] > 2.9).all()


def test_round_boundaries_are_cumulative(two_rounds):
    root, round_dirs, wall_clocks = two_rounds
    boundaries = mr.round_boundaries(wall_clocks)
    # Boundary after round 1 at 2.9; after round 2 at 3.9.
    np.testing.assert_allclose(boundaries, [2.9, 3.9], atol=1e-9)


def test_run_writes_plot_and_series(two_rounds):
    root, round_dirs, wall_clocks = two_rounds
    out_png = root / "multiround.png"
    args = mr.parse_args(
        [
            "--round-dir", str(round_dirs[0]), "--round-wall-clock-hours", "2.9",
            "--round-dir", str(round_dirs[1]), "--round-wall-clock-hours", "1.0",
            "--out-png", str(out_png),
            "--target-name", "02_PDL1",
            "--target-successes", "100",
        ]
    )
    mr.run(args)
    assert out_png.exists()
    series = pd.read_csv(out_png.with_suffix(".series.csv"))
    # Series carries both gates.
    assert set(series["gate"]) == {"head_plus_scrmsd", "alphaproteo"}
    for gate in ("head_plus_scrmsd", "alphaproteo"):
        g = series[series["gate"] == gate]
        assert g["cumulative_unique_successes"].to_numpy()[-1] == 4
        assert np.all(np.diff(g["elapsed_gpu_hours"].to_numpy()) >= 0)
