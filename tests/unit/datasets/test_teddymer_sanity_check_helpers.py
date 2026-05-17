from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest


def test_recompute_avg_int_plddt_from_digits_matches_lower_bound():
    """Each digit ``d`` in IntPlddt represents the pLDDT bucket [10*d, 10*d+10);
    Teddymer's ``AvgIntPlddt`` is the mean of ``10*d`` across all interface residues
    (verified empirically against rows 0-7 of nonsingletonrep_metadata.tsv).
    """
    from proteinfoundation.datasets.teddymer.sanity_check import recompute_avg_int_plddt

    # Digits and expected value taken from row 0 (DimerIndex=655) of the real
    # nonsingletonrep_metadata.tsv: IntPlddt =
    #   5556776777777777756677889:99998998888888887888887765663343
    # AvgIntPlddt = 69.8246.
    chain_a = [5, 5, 5, 6, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 6, 6, 7, 7, 8, 8, 9]
    chain_b = [9, 9, 9, 9, 8, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 7, 7, 6, 5, 6, 6, 3, 3, 4, 3]
    avg = recompute_avg_int_plddt(chain_a, chain_b)
    assert avg == pytest.approx(69.8246, abs=0.01)


def test_recompute_avg_int_plddt_handles_empty():
    """Defensive: empty chains should not raise, just return 0.0."""
    from proteinfoundation.datasets.teddymer.sanity_check import recompute_avg_int_plddt

    assert recompute_avg_int_plddt([], []) == 0.0


def _gz_bytes(obj) -> bytes:
    return gzip.compress(json.dumps(obj).encode())


def test_read_confidence_extracts_residue_scores(tmp_path):
    """``read_confidence_from_tar`` should return the per-residue confidence
    scores (which AFDB stores as floats 0-100) given a tar+offset+size."""
    from proteinfoundation.datasets.teddymer.sanity_check import read_confidence_from_tar

    # Build a tar-shaped file with one gzipped JSON member at a known offset.
    payload = {
        "residueNumber": [1, 2, 3, 4, 5],
        "confidenceScore": [54.97, 70.0, 80.5, 92.3, 100.0],
        "confidenceCategory": ["L", "M", "M", "H", "H"],
    }
    blob = _gz_bytes(payload)
    fake_tar = tmp_path / "fake.tar"
    # Pad with 1024 bytes of zero header so offset > 0.
    fake_tar.write_bytes(b"\x00" * 1024 + blob)

    scores = read_confidence_from_tar(fake_tar, offset=1024, size=len(blob))
    assert scores == [54.97, 70.0, 80.5, 92.3, 100.0]
