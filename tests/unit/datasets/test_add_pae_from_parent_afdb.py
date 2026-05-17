"""Red-phase tests for ``AddPAEFromParentAFDB``.

The transform reads the parent monomer's AFDB
``*-predicted_aligned_error_v4.json.gz`` member at a known tar byte offset,
slices the L_full x L_full PAE matrix through the residue-number frame
``[chain_A_residues, chain_B_residues]`` (1-indexed), and emits the
directional ``pae_residue_pair`` plus a binned and masked twin.

Spec: docs/superpowers/plans/2026-05-17_pr_a_teddymer_data_plumbing.md §9 Test 2.
Failure mode (red phase): ``AttributeError`` — the transform does not exist
yet.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.unit.datasets.conftest import build_fake_afdb_tar


def _diagonal_directional_pae(n: int) -> np.ndarray:
    """Construct a matrix where ``pae[i, j] = (i + 2*j) mod 32``.

    Cheap, integer-valued, and asymmetric (`(i, j) != (j, i)` whenever
    `i != j`) — exactly what we need to detect a missing-transpose bug.
    """
    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    return ((rows + 2 * cols) % 32).astype(np.int64)


def _confidence_score_dense(n: int) -> list[float]:
    return [50.0] * n


def _make_data_with_locator(
    fake,
    intervals_a: list[dict],
    intervals_b: list[dict],
    afdb_root,
):
    from proteinfoundation.datasets.transforms import Data

    return Data(
        example_id="test_dimer",
        teddymer_locator={
            "A": {
                "source_tar_relpath": fake.tar_relpath,
                "pae_member_offset": fake.pae.offset,
                "pae_member_size": fake.pae.size,
                "pae_is_gz": fake.pae.is_gz,
                "afdb_id": fake.afdb_id,
            },
            "B": {
                "source_tar_relpath": fake.tar_relpath,
                "pae_member_offset": fake.pae.offset,
                "pae_member_size": fake.pae.size,
                "pae_is_gz": fake.pae.is_gz,
                "afdb_id": fake.afdb_id,
            },
            "intervals_A": intervals_a,
            "intervals_B": intervals_b,
            "afdb_proteomes_root": str(afdb_root),
        },
    )


def _instantiate_transform(afdb_root):
    from proteinfoundation.datasets.transforms import AddPAEFromParentAFDB

    return AddPAEFromParentAFDB(
        bin_width=0.5,
        max_bins=64,
        scale_max=31.75,
    )


def test_pae_shape_dtype_and_values_match_submatrix(tmp_path):
    n = 10
    matrix = _diagonal_directional_pae(n)
    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.pae_residue_pair.shape == (n, n)
    assert out.pae_residue_pair.dtype == torch.float32
    expected = torch.from_numpy(matrix.astype(np.float32))
    assert torch.allclose(out.pae_residue_pair, expected)


def test_pae_is_directional_no_symmetrisation(tmp_path):
    n = 10
    matrix = _diagonal_directional_pae(n)
    assert matrix[2, 5] != matrix[5, 2]

    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    L = out.pae_residue_pair.shape[0]
    off_diag_mask = ~torch.eye(L, dtype=torch.bool)
    delta = (out.pae_residue_pair - out.pae_residue_pair.T).abs()
    assert delta[off_diag_mask].median().item() > 1e-3
    assert out.pae_residue_pair[2, 5].item() == pytest.approx(float(matrix[2, 5]))
    assert out.pae_residue_pair[5, 2].item() == pytest.approx(float(matrix[5, 2]))


def test_pae_bin_matches_pae_to_bin(tmp_path):
    n = 10
    matrix = _diagonal_directional_pae(n)
    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    from proteinfoundation.confidence.losses import pae_to_bin

    expected = pae_to_bin(out.pae_residue_pair, 0.5, 64)
    assert out.pae_bin.shape == (n, n)
    assert out.pae_bin.dtype == torch.int64
    assert torch.equal(out.pae_bin, expected)


def test_pae_bin_clip_ceiling_round_trip(tmp_path):
    n = 4
    matrix = np.full((n, n), 31, dtype=np.int64)
    matrix[0, 0] = 0
    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 2}],
        intervals_b=[{"lo": 3, "hi": 4}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.pae_bin[0, 0].item() == 0
    assert out.pae_bin[1, 2].item() == 62
    assert out.pae_bin[2, 1].item() == 62

    from proteinfoundation.confidence.losses import pae_to_bin

    top = pae_to_bin(torch.tensor([31.75], dtype=torch.float32), 0.5, 64)
    assert top.item() == 63


def test_pae_mask_true_inside_both_endpoints(tmp_path):
    n = 10
    matrix = _diagonal_directional_pae(n)
    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.pae_mask.shape == (n, n)
    assert out.pae_mask.dtype == torch.bool
    assert bool(out.pae_mask.all().item())


def test_pae_transposed_index_regression(tmp_path):
    """Regression: catches a transposed (i, j) -> (j, i) bug.

    Builds a matrix where `pae[i, j] = 2 * i + j` so every off-diagonal
    cell asymmetric. Two adjacent intervals (1-5 and 6-10) flatten to
    contiguous 0-indexed positions [0..9]; asserting on specific cell
    values pins down the axis convention, where the equivalent
    median-based check (item #4) would tolerate a global transpose.
    """
    n = 10
    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    matrix = (2 * rows + cols).astype(np.int64)
    assert matrix[2, 5] == 9
    assert matrix[5, 2] == 12

    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.pae_residue_pair[2, 5].item() == pytest.approx(9.0)
    assert out.pae_residue_pair[5, 2].item() == pytest.approx(12.0)


def test_pae_discontinuous_domain_residue_indexing(tmp_path):
    n_full = 10
    matrix = _diagonal_directional_pae(n_full)
    fake = build_fake_afdb_tar(
        tmp_path, "AF-FAKE0001-F1", _confidence_score_dense(n_full), matrix
    )
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 2}, {"lo": 5, "hi": 7}],
        intervals_b=[{"lo": 8, "hi": 9}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    expanded = [1, 2, 5, 6, 7, 8, 9]
    L = len(expanded)
    assert out.pae_residue_pair.shape == (L, L)
    expected = matrix[np.ix_([r - 1 for r in expanded], [r - 1 for r in expanded])]
    assert torch.allclose(
        out.pae_residue_pair,
        torch.from_numpy(expected.astype(np.float32)),
    )
    assert out.pae_residue_pair[0, 0].item() == pytest.approx(float(matrix[0, 0]))
    assert out.pae_residue_pair[0, 2].item() == pytest.approx(float(matrix[0, 4]))
