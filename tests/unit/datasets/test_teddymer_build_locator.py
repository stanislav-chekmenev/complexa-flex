from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


_INVENTORY_FIELDS = [
    ("sample_id", pa.string()),
    ("afdb_id", pa.string()),
    ("uniprot_id", pa.string()),
    ("taxonomy_id", pa.string()),
    ("source_tar_relpath", pa.string()),
    ("source_tar_basename", pa.string()),
    ("source_version", pa.int32()),
    ("bucket", pa.int32()),
    ("has_cif", pa.bool_()),
    ("has_pae", pa.bool_()),
    ("has_conf", pa.bool_()),
    ("cif_member_name", pa.string()),
    ("cif_member_offset", pa.int64()),
    ("cif_member_size", pa.int64()),
    ("cif_is_gz", pa.bool_()),
    ("pae_member_name", pa.string()),
    ("pae_member_offset", pa.int64()),
    ("pae_member_size", pa.int64()),
    ("pae_is_gz", pa.bool_()),
    ("conf_member_name", pa.string()),
    ("conf_member_offset", pa.int64()),
    ("conf_member_size", pa.int64()),
    ("conf_is_gz", pa.bool_()),
]
_INVENTORY_SCHEMA = pa.schema(_INVENTORY_FIELDS)


def _write_fake_inventory_batch(inv_dir: Path, afdb_ids: list[str], idx: int) -> Path:
    rows = []
    for i, aid in enumerate(afdb_ids):
        rows.append(
            {
                "sample_id": aid, "afdb_id": aid,
                "uniprot_id": aid.removeprefix("AF-").removesuffix("-F1"),
                "taxonomy_id": "9606",
                "source_tar_relpath": f"proteome-tax_id-9606-{i + 10 * idx}_v4.tar",
                "source_tar_basename": f"proteome-tax_id-9606-{i + 10 * idx}_v4.tar",
                "source_version": 4, "bucket": i + 10 * idx,
                "has_cif": True, "has_pae": True, "has_conf": True,
                "cif_member_name": f"{aid}-model_v4.cif.gz",
                "cif_member_offset": 1000 * (i + 1 + 100 * idx),
                "cif_member_size": 500, "cif_is_gz": True,
                "pae_member_name": f"{aid}-predicted_aligned_error_v4.json.gz",
                "pae_member_offset": 2000 * (i + 1 + 100 * idx),
                "pae_member_size": 800, "pae_is_gz": True,
                "conf_member_name": f"{aid}-confidence_v4.json.gz",
                "conf_member_offset": 3000 * (i + 1 + 100 * idx),
                "conf_member_size": 300, "conf_is_gz": True,
            }
        )
    table = pa.Table.from_pylist(rows, schema=_INVENTORY_SCHEMA)
    path = inv_dir / f"w0_b{idx:06d}.parquet"
    pq.write_table(table, path)
    return path


def test_build_locator_rows_two_chains_per_dimer(tmp_path):
    from proteinfoundation.data.teddymer.build_locator import build_locator_rows

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_batch(inv_dir, ["AF-A0A005-F1", "AF-A0A009E3M2-F1"], idx=0)

    dimers = pd.DataFrame(
        {
            "dimer_id": ["7DI_AF-A0A005-F1-model_v4", "34DI_AF-A0A009E3M2-F1-model_v4"],
            "dimer_index": [7, 34],
            "parent_afdb_id": ["AF-A0A005-F1", "AF-A0A009E3M2-F1"],
        }
    )

    locator = build_locator_rows(dimers, inv_dir)
    assert len(locator) == 4  # two dimers x two chains

    chains = set(locator["chain_id"].unique())
    assert chains == {"A", "B"}

    row = locator[(locator["dimer_index"] == 7) & (locator["chain_id"] == "A")].iloc[0]
    assert row["afdb_id"] == "AF-A0A005-F1"
    assert row["cif_member_offset"] == 1000  # bucket i=0, idx=0 -> 1000


def test_build_locator_rows_raises_on_missing_afdb_id(tmp_path):
    from proteinfoundation.data.teddymer.build_locator import build_locator_rows

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_batch(inv_dir, ["AF-A0A005-F1"], idx=0)

    dimers = pd.DataFrame(
        {
            "dimer_id": ["999DI_AF-MISSING-F1-model_v4"],
            "dimer_index": [999],
            "parent_afdb_id": ["AF-MISSING-F1"],
        }
    )
    with pytest.raises(ValueError, match="missing from AFDB inventory"):
        build_locator_rows(dimers, inv_dir)


def test_build_locator_rows_dedups_duplicate_inventory_rows(tmp_path):
    """If an afdb_id appears in two batches (shouldn't happen but defensive),
    we keep the first occurrence so the row count remains 2 x N_dimers.
    """
    from proteinfoundation.data.teddymer.build_locator import build_locator_rows

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_batch(inv_dir, ["AF-A0A005-F1"], idx=0)
    _write_fake_inventory_batch(inv_dir, ["AF-A0A005-F1"], idx=1)

    dimers = pd.DataFrame(
        {
            "dimer_id": ["7DI_AF-A0A005-F1-model_v4"],
            "dimer_index": [7],
            "parent_afdb_id": ["AF-A0A005-F1"],
        }
    )
    locator = build_locator_rows(dimers, inv_dir)
    assert len(locator) == 2


def test_build_locator_rows_streams_only_needed_rows(tmp_path):
    """The join must work even when the AFDB inventory is much larger than
    the dimers set — most rows are irrelevant and should be filtered out per-batch.
    """
    from proteinfoundation.data.teddymer.build_locator import build_locator_rows

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    # Two batches, each carrying 50 unrelated AFDB IDs plus the two we need.
    _write_fake_inventory_batch(
        inv_dir,
        [f"AF-NOISE{i:03d}-F1" for i in range(50)] + ["AF-A0A005-F1"],
        idx=0,
    )
    _write_fake_inventory_batch(
        inv_dir,
        [f"AF-OTHER{i:03d}-F1" for i in range(50)] + ["AF-A0A009E3M2-F1"],
        idx=1,
    )

    dimers = pd.DataFrame(
        {
            "dimer_id": ["7DI_AF-A0A005-F1-model_v4", "34DI_AF-A0A009E3M2-F1-model_v4"],
            "dimer_index": [7, 34],
            "parent_afdb_id": ["AF-A0A005-F1", "AF-A0A009E3M2-F1"],
        }
    )
    locator = build_locator_rows(dimers, inv_dir)
    assert len(locator) == 4
    assert set(locator["afdb_id"].unique()) == {"AF-A0A005-F1", "AF-A0A009E3M2-F1"}


def test_build_locator_rows_output_schema_matches_la_proteina(tmp_path):
    """The output must carry every column the la_proteina_afdb_512_v1 locator
    parquet carries, plus our 3 extra columns (dimer_id, dimer_index, chain_id).
    """
    from proteinfoundation.data.teddymer.build_locator import (
        build_locator_rows,
        LOCATOR_INVENTORY_COLS,
    )

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_batch(inv_dir, ["AF-A0A005-F1"], idx=0)

    dimers = pd.DataFrame(
        {
            "dimer_id": ["7DI_AF-A0A005-F1-model_v4"],
            "dimer_index": [7],
            "parent_afdb_id": ["AF-A0A005-F1"],
        }
    )
    locator = build_locator_rows(dimers, inv_dir)

    expected = ["dimer_id", "dimer_index", "chain_id"] + LOCATOR_INVENTORY_COLS
    assert list(locator.columns) == expected
