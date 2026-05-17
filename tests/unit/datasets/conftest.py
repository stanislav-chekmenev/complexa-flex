"""Shared fixtures for Teddymer dataset unit tests.

Builds a tar-shaped file on disk with gzipped CIF / confidence / PAE members
at known byte offsets, mirroring the random-access read pattern that
``read_confidence_from_tar`` (and the upcoming ``read_pae_from_tar`` /
``read_cif_bytes_from_tar``) uses. The byte layout deliberately is *not* a
real tar archive — the readers under test only seek to the recorded offset
and read ``size`` bytes, so the surrounding bytes are irrelevant.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


_HEADER_PAD = b"\x00" * 1024
_BETWEEN_MEMBER_PAD = b"\x00" * 512


@dataclass
class FakeMember:
    name: str
    offset: int
    size: int
    is_gz: bool


@dataclass
class FakeAfdbTar:
    """Bookkeeping for one synthetic AFDB tar.

    The ``conf``, ``pae`` and ``cif`` members are written at known offsets
    inside ``tar_path``. ``confidence_score`` and ``pae_matrix`` carry the
    payload that downstream readers should decode out of the tar.
    """

    afdb_id: str
    tar_path: Path
    tar_relpath: str
    conf: FakeMember
    pae: FakeMember
    cif: FakeMember
    confidence_score: list[float]
    pae_matrix: np.ndarray
    sequence_length: int


def _gz_bytes(obj) -> bytes:
    return gzip.compress(json.dumps(obj).encode())


def _confidence_payload(scores: list[float]) -> bytes:
    return _gz_bytes(
        {
            "residueNumber": list(range(1, len(scores) + 1)),
            "confidenceScore": [float(s) for s in scores],
            "confidenceCategory": ["L"] * len(scores),
        }
    )


def _pae_payload(matrix: np.ndarray, max_pae: float = 31.75) -> bytes:
    return _gz_bytes(
        [
            {
                "predicted_aligned_error": matrix.tolist(),
                "max_predicted_aligned_error": float(max_pae),
            }
        ]
    )


def _minimal_cif_payload(afdb_id: str, n_residues: int) -> bytes:
    """Emit a CIF blob with the minimum atom_site loop the readers need to be
    able to call ``load_structure`` over. The PR-A red-phase tests do not
    parse the CIF — that path is exercised by the green-phase dataset tests
    through the dataset class, not the transform. We still write *something*
    plausible so the bytes are non-empty.
    """
    rows = []
    for r in range(1, n_residues + 1):
        rows.append(
            f"ATOM {r:5d}  CA  ALA A {r:4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{float(r):8.3f}  1.00  50.00           C"
        )
    body = (
        f"data_{afdb_id}\n"
        "loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.type_symbol\n"
        + "\n".join(rows)
        + "\n"
    )
    return gzip.compress(body.encode())


def build_fake_afdb_tar(
    tmp_path: Path,
    afdb_id: str,
    confidence_score: list[float],
    pae_matrix: np.ndarray,
    *,
    tar_subdir: str = "proteomes/v4",
) -> FakeAfdbTar:
    """Write a single fake AFDB tar to ``tmp_path / tar_subdir`` and return
    bookkeeping describing where every member lives.

    ``confidence_score`` and ``pae_matrix`` must agree on sequence length.
    """
    n = len(confidence_score)
    if pae_matrix.shape != (n, n):
        raise ValueError(
            f"pae_matrix.shape={pae_matrix.shape} disagrees with len(confidence_score)={n}"
        )

    tar_dir = tmp_path / tar_subdir
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_relpath = f"{tar_subdir}/{afdb_id}-proteome.tar"
    tar_path = tmp_path / tar_relpath

    conf_blob = _confidence_payload(confidence_score)
    pae_blob = _pae_payload(pae_matrix)
    cif_blob = _minimal_cif_payload(afdb_id, n)

    with open(tar_path, "wb") as f:
        f.write(_HEADER_PAD)
        conf_offset = f.tell()
        f.write(conf_blob)
        f.write(_BETWEEN_MEMBER_PAD)
        pae_offset = f.tell()
        f.write(pae_blob)
        f.write(_BETWEEN_MEMBER_PAD)
        cif_offset = f.tell()
        f.write(cif_blob)

    return FakeAfdbTar(
        afdb_id=afdb_id,
        tar_path=tar_path,
        tar_relpath=tar_relpath,
        conf=FakeMember(
            name=f"{afdb_id}-confidence_v4.json.gz",
            offset=conf_offset,
            size=len(conf_blob),
            is_gz=True,
        ),
        pae=FakeMember(
            name=f"{afdb_id}-predicted_aligned_error_v4.json.gz",
            offset=pae_offset,
            size=len(pae_blob),
            is_gz=True,
        ),
        cif=FakeMember(
            name=f"{afdb_id}-model_v4.cif.gz",
            offset=cif_offset,
            size=len(cif_blob),
            is_gz=True,
        ),
        confidence_score=list(confidence_score),
        pae_matrix=pae_matrix,
        sequence_length=n,
    )


def make_locator_row(
    *,
    dimer_id: str,
    dimer_index: int,
    chain_id: str,
    fake: FakeAfdbTar,
) -> dict:
    """Build one locator-parquet row pointing at the given fake tar."""
    return {
        "dimer_id": dimer_id,
        "dimer_index": dimer_index,
        "chain_id": chain_id,
        "sample_id": fake.afdb_id,
        "afdb_id": fake.afdb_id,
        "uniprot_id": fake.afdb_id.removeprefix("AF-").removesuffix("-F1"),
        "taxonomy_id": "9606",
        "source_tar_relpath": fake.tar_relpath,
        "source_tar_basename": Path(fake.tar_relpath).name,
        "source_version": 4,
        "bucket": 0,
        "has_cif": True,
        "has_pae": True,
        "has_conf": True,
        "cif_member_name": fake.cif.name,
        "cif_member_offset": fake.cif.offset,
        "cif_member_size": fake.cif.size,
        "cif_is_gz": fake.cif.is_gz,
        "pae_member_name": fake.pae.name,
        "pae_member_offset": fake.pae.offset,
        "pae_member_size": fake.pae.size,
        "pae_is_gz": fake.pae.is_gz,
        "conf_member_name": fake.conf.name,
        "conf_member_offset": fake.conf.offset,
        "conf_member_size": fake.conf.size,
        "conf_is_gz": fake.conf.is_gz,
    }


def write_locator_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_dimers_parquet(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return path


@pytest.fixture
def fake_afdb_tar_factory(tmp_path):
    """Factory yielding ``build_fake_afdb_tar`` bound to ``tmp_path``."""

    def _factory(
        afdb_id: str,
        confidence_score: list[float],
        pae_matrix: np.ndarray,
    ) -> FakeAfdbTar:
        return build_fake_afdb_tar(tmp_path, afdb_id, confidence_score, pae_matrix)

    return _factory
