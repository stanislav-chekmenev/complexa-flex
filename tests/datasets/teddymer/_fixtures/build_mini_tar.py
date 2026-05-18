"""Synthetic mini-tar + mini-locator fixture for blob-repack tests.

Mirrors the byte layout the real Teddymer view records in
``locator_rows.parquet``: a flat file with gzipped CIF / PAE / confidence
payloads at known offsets. The repack pipeline must round-trip these
bytes through `data.blob` byte-for-byte.

Why not stdlib ``tarfile``: the dataloader contract uses
``open + seek(offset) + read(size)`` only; the surrounding bytes are
irrelevant. Recording offsets directly is simpler and matches the
existing fixture in ``tests/unit/datasets/conftest.py``.
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


_HEADER_PAD = b"\x00" * 64
_BETWEEN_MEMBER_PAD = b"\x00" * 32


@dataclass
class FakeMember:
    name: str
    offset: int
    size: int
    is_gz: bool


@dataclass
class FakeParent:
    afdb_id: str
    tar_relpath: str
    cif: FakeMember
    pae: FakeMember
    conf: FakeMember
    cif_bytes: bytes
    pae_bytes: bytes
    conf_bytes: bytes


def _gz_json(obj) -> bytes:
    return gzip.compress(json.dumps(obj).encode())


def _confidence_payload(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    scores = [float(50 + rng.integers(0, 50)) for _ in range(n)]
    return _gz_json(
        {
            "residueNumber": list(range(1, n + 1)),
            "confidenceScore": scores,
            "confidenceCategory": ["L"] * n,
        }
    )


def _pae_payload(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, 32, size=(n, n)).tolist()
    return _gz_json(
        [{"predicted_aligned_error": matrix, "max_predicted_aligned_error": 31.75}]
    )


def _cif_payload(afdb_id: str, n: int) -> bytes:
    rows = [
        f"ATOM {r:5d}  CA  ALA A {r:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{float(r):8.3f}  1.00  50.00           C"
        for r in range(1, n + 1)
    ]
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


def write_fake_parent(
    labs_root: Path,
    tar_relpath: str,
    afdb_id: str,
    n_residues: int = 6,
    *,
    seed: int = 0,
    append: bool = False,
) -> FakeParent:
    """Append one (cif, pae, conf) member triple to ``labs_root / tar_relpath``.

    When ``append=True`` and the tar already exists, the new members are
    appended after the existing bytes (the multi-parent-per-tar case).
    """
    tar_path = labs_root / tar_relpath
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    cif_blob = _cif_payload(afdb_id, n_residues)
    pae_blob = _pae_payload(n_residues, seed=seed)
    conf_blob = _confidence_payload(n_residues, seed=seed + 1)

    mode = "ab" if append and tar_path.exists() else "wb"
    with open(tar_path, mode) as f:
        if mode == "wb":
            f.write(_HEADER_PAD)
        cif_offset = f.tell()
        f.write(cif_blob)
        f.write(_BETWEEN_MEMBER_PAD)
        pae_offset = f.tell()
        f.write(pae_blob)
        f.write(_BETWEEN_MEMBER_PAD)
        conf_offset = f.tell()
        f.write(conf_blob)

    return FakeParent(
        afdb_id=afdb_id,
        tar_relpath=tar_relpath,
        cif=FakeMember(
            name=f"{afdb_id}-model_v4.cif.gz",
            offset=cif_offset,
            size=len(cif_blob),
            is_gz=True,
        ),
        pae=FakeMember(
            name=f"{afdb_id}-predicted_aligned_error_v4.json.gz",
            offset=pae_offset,
            size=len(pae_blob),
            is_gz=True,
        ),
        conf=FakeMember(
            name=f"{afdb_id}-confidence_v4.json.gz",
            offset=conf_offset,
            size=len(conf_blob),
            is_gz=True,
        ),
        cif_bytes=cif_blob,
        pae_bytes=pae_blob,
        conf_bytes=conf_blob,
    )


def locator_row_for(
    *,
    dimer_id: str,
    dimer_index: int,
    chain_id: str,
    parent: FakeParent,
) -> dict:
    return {
        "dimer_id": dimer_id,
        "dimer_index": dimer_index,
        "chain_id": chain_id,
        "sample_id": parent.afdb_id,
        "afdb_id": parent.afdb_id,
        "uniprot_id": parent.afdb_id.removeprefix("AF-").removesuffix("-F1"),
        "taxonomy_id": "9606",
        "source_tar_relpath": parent.tar_relpath,
        "source_tar_basename": Path(parent.tar_relpath).name,
        "source_version": 4,
        "bucket": 0,
        "has_cif": True,
        "has_pae": True,
        "has_conf": True,
        "cif_member_name": parent.cif.name,
        "cif_member_offset": parent.cif.offset,
        "cif_member_size": parent.cif.size,
        "cif_is_gz": parent.cif.is_gz,
        "pae_member_name": parent.pae.name,
        "pae_member_offset": parent.pae.offset,
        "pae_member_size": parent.pae.size,
        "pae_is_gz": parent.pae.is_gz,
        "conf_member_name": parent.conf.name,
        "conf_member_offset": parent.conf.offset,
        "conf_member_size": parent.conf.size,
        "conf_is_gz": parent.conf.is_gz,
    }


def write_locator_parquet(path: Path, rows: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_dimers_parquet(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_parquet(path)
    return path


def build_mini_view(
    tmp_path: Path,
    *,
    labs_subdir: str = "labs",
    view_subdir: str = "view",
) -> dict:
    """Build a 4-row synthetic view that exercises the dedup, cross-tar
    and intra-monomer-pair logic of `plan_layout`.

    Layout:
      - tar T1 carries parents P1 and P2 (multi-parent per tar)
      - tar T2 carries parent P3
      - dimer D1 = P1(A) x P1(B)  — intra-monomer dimer (same parent on both chains)
      - dimer D2 = P2(A) x P3(B)  — cross-tar dimer

    => 4 locator rows, 3 unique parents.
    """
    labs_root = tmp_path / labs_subdir
    view_root = tmp_path / view_subdir
    view_root.mkdir(parents=True, exist_ok=True)

    p1 = write_fake_parent(labs_root, "proteomes/v4/T1.tar", "AF-P1-F1", seed=11)
    p2 = write_fake_parent(
        labs_root, "proteomes/v4/T1.tar", "AF-P2-F1", seed=22, append=True
    )
    p3 = write_fake_parent(labs_root, "proteomes/v4/T2.tar", "AF-P3-F1", seed=33)

    rows = [
        locator_row_for(dimer_id="D1", dimer_index=1, chain_id="A", parent=p1),
        locator_row_for(dimer_id="D1", dimer_index=1, chain_id="B", parent=p1),
        locator_row_for(dimer_id="D2", dimer_index=2, chain_id="A", parent=p2),
        locator_row_for(dimer_id="D2", dimer_index=2, chain_id="B", parent=p3),
    ]
    locator_path = write_locator_parquet(view_root / "locator_rows.parquet", rows)

    dimers_rows = [
        {
            "dimer_id": "D1",
            "dimer_index": 1,
            "parent_afdb_id": p1.afdb_id,
            "complexa_filter": True,
        },
        {
            "dimer_id": "D2",
            "dimer_index": 2,
            "parent_afdb_id": p2.afdb_id,
            "complexa_filter": True,
        },
    ]
    dimers_path = write_dimers_parquet(view_root / "dimers.parquet", dimers_rows)

    return {
        "labs_root": labs_root,
        "view_root": view_root,
        "locator_path": locator_path,
        "dimers_path": dimers_path,
        "parents": {p1.afdb_id: p1, p2.afdb_id: p2, p3.afdb_id: p3},
    }
