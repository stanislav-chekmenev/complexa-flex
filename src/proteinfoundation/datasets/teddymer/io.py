"""Tar-byte-offset readers for the Teddymer view.

The Teddymer view records member byte offsets and sizes for each AFDB tar
in ``locator_rows.parquet``. The readers in this module accept those raw
``(tar_path, offset, size, is_gz)`` tuples and decode the three payload
kinds we depend on: per-residue confidence (pLDDT), predicted-aligned-error
matrix, and the model CIF bytes. They are the single source of truth for
tar-offset reads in the dataset path.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def _read_tar_member_bytes(
    tar_path: Path | str,
    offset: int,
    size: int,
    is_gz: bool,
) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(int(offset))
        blob = f.read(int(size))
    if is_gz:
        blob = gzip.decompress(blob)
    return blob


def read_confidence_from_tar(
    tar_path: Path | str,
    offset: int,
    size: int,
    is_gz: bool = True,
) -> list[float]:
    """Return the per-residue ``confidenceScore`` list from an AFDB
    ``*-confidence_v4.json.gz`` member at byte ``offset`` with length
    ``size`` inside ``tar_path``.
    """
    blob = _read_tar_member_bytes(tar_path, offset, size, is_gz)
    obj = json.loads(blob)
    return [float(x) for x in obj["confidenceScore"]]


def read_pae_from_tar(
    tar_path: Path | str,
    offset: int,
    size: int,
    is_gz: bool = True,
) -> np.ndarray:
    """Return the ``L_full x L_full`` predicted-aligned-error matrix from an
    AFDB ``*-predicted_aligned_error_v4.json.gz`` member.

    AFDB stores the payload as ``[{"predicted_aligned_error": <L x L int>,
    "max_predicted_aligned_error": <int>}]``; values are integer Angstrom.
    """
    blob = _read_tar_member_bytes(tar_path, offset, size, is_gz)
    obj = json.loads(blob)
    if isinstance(obj, list):
        obj = obj[0]
    return np.asarray(obj["predicted_aligned_error"], dtype=np.int64)


def read_cif_bytes_from_tar(
    tar_path: Path | str,
    offset: int,
    size: int,
    is_gz: bool = True,
) -> bytes:
    """Return the raw (decompressed if ``is_gz``) CIF text bytes for the
    member at the given offset. The caller decides how to parse them.
    """
    return _read_tar_member_bytes(tar_path, offset, size, is_gz)
