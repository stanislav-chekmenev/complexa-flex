"""Red-phase tests for ``AddPLDDTFromParentAFDB``.

The transform reads the parent monomer's per-residue ``confidenceScore``
from an AFDB ``*-confidence_v4.json.gz`` member by tar byte offset, slices
it through the ``residue_intervals_{A,B}`` 1-indexed spans on the Data
object's ``teddymer_locator`` attribute, and writes the standard
``plddt_residue`` / ``plddt_bin`` / ``plddt_mask`` triple that the existing
pLDDT confidence head already consumes.

Spec: docs/superpowers/plans/2026-05-17_pr_a_teddymer_data_plumbing.md §9 Test 1.
Failure mode (red phase): ``AttributeError`` — the transform does not exist
yet.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.unit.datasets.conftest import build_fake_afdb_tar


def _zero_pae(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.int64)


def _make_data_with_locator(
    fake,
    intervals_a: list[dict],
    intervals_b: list[dict],
    afdb_root,
):
    from proteinfoundation.datasets.transforms import Data

    data = Data()
    data.example_id = "test_dimer"
    data.teddymer_locator = {
        "A": {
            "source_tar_relpath": fake.tar_relpath,
            "conf_member_offset": fake.conf.offset,
            "conf_member_size": fake.conf.size,
            "conf_is_gz": fake.conf.is_gz,
            "afdb_id": fake.afdb_id,
        },
        "B": {
            "source_tar_relpath": fake.tar_relpath,
            "conf_member_offset": fake.conf.offset,
            "conf_member_size": fake.conf.size,
            "conf_is_gz": fake.conf.is_gz,
            "afdb_id": fake.afdb_id,
        },
        "intervals_A": intervals_a,
        "intervals_B": intervals_b,
        "afdb_proteomes_root": str(afdb_root),
    }
    return data


def _instantiate_transform(afdb_root):
    from proteinfoundation.datasets.transforms import AddPLDDTFromParentAFDB

    return AddPLDDTFromParentAFDB(
        bin_width=2.0,
        max_bins=50,
        scale_max=100.0,
    )


def test_plddt_residue_values_and_dtype(tmp_path):
    scores = [80.0, 50.0, 0.0, 99.0, 70.0, 12.0, 34.0, 88.0, 100.0, 5.0]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.plddt_residue.shape == (10,)
    assert out.plddt_residue.dtype == torch.float32
    assert torch.allclose(
        out.plddt_residue,
        torch.tensor(scores, dtype=torch.float32),
    )


def test_plddt_bin_matches_plddt_to_bin(tmp_path):
    from proteinfoundation.confidence.losses import plddt_to_bin

    scores = [80.0, 50.0, 0.0, 99.0, 70.0, 12.0, 34.0, 88.0, 100.0, 5.0]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    expected = plddt_to_bin(torch.tensor(scores, dtype=torch.float32), 2.0, 50)
    assert out.plddt_bin.dtype == torch.int64
    assert out.plddt_bin.shape == (10,)
    assert torch.equal(out.plddt_bin, expected)
    assert out.plddt_bin[0].item() == 40
    assert out.plddt_bin[1].item() == 25
    assert out.plddt_bin[2].item() == 0
    assert out.plddt_bin[3].item() == 49
    assert out.plddt_bin[4].item() == 35


def test_plddt_mask_true_inside_parent(tmp_path):
    scores = [80.0, 50.0, 65.0, 99.0, 70.0, 12.0, 34.0, 88.0, 100.0, 5.0]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.plddt_mask.dtype == torch.bool
    assert out.plddt_mask.shape == (10,)
    assert bool(out.plddt_mask.all().item())


def test_plddt_discontinuous_domain(tmp_path):
    scores = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 2}, {"lo": 5, "hi": 7}],
        intervals_b=[],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    expected_residues = [1, 2, 5, 6, 7]
    expected_scores = torch.tensor(
        [scores[r - 1] for r in expected_residues], dtype=torch.float32
    )
    assert out.plddt_residue.shape == (5,)
    assert torch.allclose(out.plddt_residue, expected_scores)


def test_plddt_single_residue_domain(tmp_path):
    scores = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 3, "hi": 3}],
        intervals_b=[],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.plddt_residue.shape == (1,)
    assert out.plddt_bin.shape == (1,)
    assert out.plddt_mask.shape == (1,)
    assert out.plddt_residue.item() == pytest.approx(30.0)


def test_plddt_all_zero_confidence(tmp_path):
    scores = [0.0] * 10
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 5}],
        intervals_b=[{"lo": 6, "hi": 10}],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert torch.all(out.plddt_residue == 0.0)
    assert torch.all(out.plddt_bin == 0)
    assert not bool(out.plddt_mask.any().item())


def test_plddt_out_of_bounds_residue_does_not_raise(tmp_path):
    scores = [10.0 * (i + 1) for i in range(10)]
    fake = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, _zero_pae(10))
    data = _make_data_with_locator(
        fake,
        intervals_a=[{"lo": 1, "hi": 20}],
        intervals_b=[],
        afdb_root=tmp_path,
    )
    transform = _instantiate_transform(tmp_path)

    out = transform(data)

    assert out.plddt_residue.shape == (20,)
    assert bool(out.plddt_mask[:10].all().item())
    assert not bool(out.plddt_mask[10:].any().item())
    assert torch.all(out.plddt_residue[10:] == 0.0)
    assert torch.all(out.plddt_bin[10:] == 0)
