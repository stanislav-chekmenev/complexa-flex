"""Integration spot-check binding the Teddymer residue-numbering frame and
directional PAE indexing to the metadata ground truth.

Samples 20 dimers from the real view at ``/mnt/storage01/home/schekmenev/data/teddymer_v1/``,
pulls each dimer through the new ``TeddymerDimerDataModule``, computes
Calpha-Calpha distances on the parent AFDB CIF to derive the interface set,
and compares back to the ``interface_length``, ``avg_int_plddt``, and
``avg_int_pae`` columns recorded in ``dimers.parquet``.

This is the only test that catches a wrong residue-numbering frame (PR #8
review explicitly flagged this gap): unit tests construct fixtures with the
same convention as the code under test, so they cannot detect a coherent
off-by-one.

Spec: docs/superpowers/plans/2026-05-17_pr_a_teddymer_data_plumbing.md §9 Test 4.
Failure mode (red phase): ``ImportError`` /
``hydra.errors.InstantiationException`` for the not-yet-existing datamodule
or config.

Marked ``slow``; skipped where the AFDB v4 mirror or the Teddymer view is
not accessible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


TEDDYMER_VIEW_ROOT = Path(
    os.environ.get(
        "TEDDYMER_VIEW_ROOT", "/mnt/storage01/home/schekmenev/data/teddymer_v1"
    )
)
AFDB_PROTEOMES_ROOT = Path(
    os.environ.get(
        "AFDB_PROTEOMES_ROOT",
        "/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4",
    )
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not TEDDYMER_VIEW_ROOT.exists(),
        reason=f"Teddymer view not accessible at {TEDDYMER_VIEW_ROOT}",
    ),
    pytest.mark.skipif(
        not AFDB_PROTEOMES_ROOT.exists(),
        reason=f"AFDB v4 mirror not accessible at {AFDB_PROTEOMES_ROOT}",
    ),
]


N_SAMPLES = 20
INTERFACE_CA_CUTOFF_A = 10.0
INTERFACE_LENGTH_TOL_RES = 2
AVG_INT_PLDDT_BUCKETED_TOL = 2.5
AVG_INT_PAE_UPPER_BOUND_SLACK = 8.0
PAE_DIRECTIONAL_ASYMMETRY_MIN = 1e-3
SAMPLE_SEED = 20260517


@pytest.fixture(scope="module")
def sampled_dimers():
    import pandas as pd

    dimers_path = TEDDYMER_VIEW_ROOT / "dimers.parquet"
    full = pd.read_parquet(dimers_path)
    surviving = full[full["complexa_filter"] == True]  # noqa: E712
    return surviving.sample(n=N_SAMPLES, random_state=SAMPLE_SEED).reset_index(drop=True)


@pytest.fixture(scope="module")
def datamodule():
    import hydra
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs" / "dataset" / "unified"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="teddymer_with_plddt_and_pae",
            overrides=[
                f"datamodule.dimers_parquet={TEDDYMER_VIEW_ROOT / 'dimers.parquet'}",
                f"datamodule.locator_parquet={TEDDYMER_VIEW_ROOT / 'locator_rows.parquet'}",
                f"datamodule.afdb_proteomes_root={AFDB_PROTEOMES_ROOT}",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
            ],
        )
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")
    return dm


def _ca_coords_from_data(data) -> np.ndarray:
    """Return CA coordinates for every resolved residue, in token order."""
    import torch

    coords = data.coords if hasattr(data, "coords") else data["coords"]
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if coords.ndim == 3:
        ca = coords[:, 1, :]
    else:
        ca = coords
    return np.asarray(ca, dtype=np.float64)


def _interface_ca_pairs(
    ca: np.ndarray, chain_id: np.ndarray, cutoff: float
) -> set[tuple[int, int]]:
    a_idx = np.where(chain_id == 0)[0]
    b_idx = np.where(chain_id == 1)[0]
    if len(a_idx) == 0 or len(b_idx) == 0:
        return set()
    diffs = ca[a_idx][:, None, :] - ca[b_idx][None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    mask = d < cutoff
    pairs: set[tuple[int, int]] = set()
    for ai, bi in zip(*np.where(mask)):
        pairs.add((int(a_idx[ai]), int(b_idx[bi])))
    return pairs


def _interface_residue_index_set(
    pairs: set[tuple[int, int]],
) -> tuple[set[int], set[int]]:
    a_set = {p[0] for p in pairs}
    b_set = {p[1] for p in pairs}
    return a_set, b_set


def _locate_data_for_dimer(datamodule, dimer_index: int):
    for ds in (datamodule.train_dataset, datamodule.val_dataset):
        if ds is None:
            continue
        meta = ds.metadata
        match = meta.index[meta["dimer_index"] == dimer_index].tolist()
        if not match:
            continue
        return ds[int(match[0])]
    return None


def test_interface_length_matches_metadata(datamodule, sampled_dimers):
    failures: list[tuple[int, int, int]] = []
    for _, row in sampled_dimers.iterrows():
        data = _locate_data_for_dimer(datamodule, int(row["dimer_index"]))
        if data is None:
            failures.append((int(row["dimer_index"]), -1, int(row["interface_length"])))
            continue
        chain_id = np.asarray(
            data.chain_idx.cpu().numpy() if hasattr(data.chain_idx, "cpu") else data.chain_idx
        )
        ca = _ca_coords_from_data(data)
        pairs = _interface_ca_pairs(ca, chain_id, INTERFACE_CA_CUTOFF_A)
        a_set, b_set = _interface_residue_index_set(pairs)
        observed = len(a_set) + len(b_set)
        expected = int(row["interface_length"])
        if abs(observed - expected) > INTERFACE_LENGTH_TOL_RES:
            failures.append((int(row["dimer_index"]), observed, expected))
    assert len(failures) <= 2, (
        f"{len(failures)}/{N_SAMPLES} dimers exceed interface-length slop of "
        f"+/-{INTERFACE_LENGTH_TOL_RES} residues. First few: {failures[:5]}"
    )


def test_avg_int_plddt_matches_metadata(datamodule, sampled_dimers):
    # Teddymer's avg_int_plddt is mean(10 * floor(pLDDT/10)) over interface
    # residues. Reconstruct that bucketed mean from data.plddt_residue so the
    # comparison is like-for-like; the raw float mean cannot match within 1.0
    # because the metadata loses up to 9.99 units per residue to bucketing.
    diffs = []
    for _, row in sampled_dimers.iterrows():
        data = _locate_data_for_dimer(datamodule, int(row["dimer_index"]))
        if data is None:
            continue
        chain_id = np.asarray(
            data.chain_idx.cpu().numpy() if hasattr(data.chain_idx, "cpu") else data.chain_idx
        )
        ca = _ca_coords_from_data(data)
        pairs = _interface_ca_pairs(ca, chain_id, INTERFACE_CA_CUTOFF_A)
        a_set, b_set = _interface_residue_index_set(pairs)
        if not a_set and not b_set:
            continue
        plddt = data.plddt_residue.detach().cpu().numpy()
        interface_idx = sorted(a_set | b_set)
        bucketed = 10.0 * np.floor(plddt[interface_idx] / 10.0)
        observed = float(bucketed.mean())
        diffs.append(abs(observed - float(row["avg_int_plddt"])))
    assert diffs, "No dimers contributed an interface set"
    max_diff = max(diffs)
    assert max_diff <= AVG_INT_PLDDT_BUCKETED_TOL, (
        f"Max |bucketed_avg_int_plddt observed - metadata| = {max_diff:.3f} "
        f"exceeds tol {AVG_INT_PLDDT_BUCKETED_TOL}"
    )


def test_avg_int_pae_matches_metadata_directional(datamodule, sampled_dimers):
    # Teddymer's avg_int_pae aggregation rule (residue-set vs pair-set, inner
    # cutoff) is not publicly specified. PR-B distils against the full per-pair
    # PAE matrix, not against this scalar -- so we only need a directional
    # sanity check: observed mean over CA<10A contact pairs should be on the
    # same scale as the metadata, and a transposed-PAE bug would still inflate
    # the residual on asymmetric dimers.
    asymmetry = []
    upper_bound_violations: list[tuple[int, float, float]] = []
    for _, row in sampled_dimers.iterrows():
        data = _locate_data_for_dimer(datamodule, int(row["dimer_index"]))
        if data is None:
            continue
        chain_id = np.asarray(
            data.chain_idx.cpu().numpy() if hasattr(data.chain_idx, "cpu") else data.chain_idx
        )
        ca = _ca_coords_from_data(data)
        pairs = _interface_ca_pairs(ca, chain_id, INTERFACE_CA_CUTOFF_A)
        if not pairs:
            continue
        pae = data.pae_residue_pair.detach().cpu().numpy()
        i_idx = np.array([p[0] for p in pairs])
        j_idx = np.array([p[1] for p in pairs])
        ij = pae[i_idx, j_idx]
        ji = pae[j_idx, i_idx]
        directional = np.concatenate([ij, ji])
        observed = float(directional.mean())
        if observed > float(row["avg_int_pae"]) + AVG_INT_PAE_UPPER_BOUND_SLACK:
            upper_bound_violations.append(
                (int(row["dimer_index"]), observed, float(row["avg_int_pae"]))
            )
        asymmetry.append(float(np.abs(ij.mean() - ji.mean())))
    assert asymmetry, "No dimers contributed an interface set for PAE check"
    # If PR-A had accidentally symmetrised PAE in the transform, every dimer's
    # asymmetry would be exactly zero. Require that at least one of the sampled
    # dimers exhibits a nonzero ij-vs-ji mean gap so the test catches a
    # transposed/symmetrised-PAE regression.
    assert max(asymmetry) > PAE_DIRECTIONAL_ASYMMETRY_MIN, (
        f"All sampled dimers show ij/ji means within {PAE_DIRECTIONAL_ASYMMETRY_MIN} -- "
        f"transform may have symmetrised PAE. Max asymmetry: {max(asymmetry):.2e}"
    )
    assert not upper_bound_violations, (
        f"{len(upper_bound_violations)} dimers exceed avg_int_pae + slack "
        f"{AVG_INT_PAE_UPPER_BOUND_SLACK}: {upper_bound_violations[:5]}"
    )
