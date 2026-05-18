"""Regression test for the AFDB master-inventory shard glob in build_locator.py.

The AFDB master inventory at
``/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/``
is sharded as ``w{0..N}_b*.parquet`` (multiple workers x 64 batches each).
``_load_needed_inventory_rows`` globs ``w*_b*.parquet`` so every worker's
shards are scanned; this test pins that behaviour so the glob can't regress
back to ``w0_b*.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


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


def _write_fake_inventory_shard(
    inv_dir: Path, worker: int, batch: int, afdb_ids: list[str]
) -> Path:
    rows = []
    for i, aid in enumerate(afdb_ids):
        rows.append(
            {
                "sample_id": aid, "afdb_id": aid,
                "uniprot_id": aid.removeprefix("AF-").removesuffix("-F1"),
                "taxonomy_id": "9606",
                "source_tar_relpath": f"proteome-w{worker}-b{batch}-{i}_v4.tar",
                "source_tar_basename": f"proteome-w{worker}-b{batch}-{i}_v4.tar",
                "source_version": 4, "bucket": worker * 1000 + batch * 10 + i,
                "has_cif": True, "has_pae": True, "has_conf": True,
                "cif_member_name": f"{aid}-model_v4.cif.gz",
                "cif_member_offset": 1000 * (1 + i) + 100_000 * worker + 1_000_000 * batch,
                "cif_member_size": 500, "cif_is_gz": True,
                "pae_member_name": f"{aid}-predicted_aligned_error_v4.json.gz",
                "pae_member_offset": 2000, "pae_member_size": 800, "pae_is_gz": True,
                "conf_member_name": f"{aid}-confidence_v4.json.gz",
                "conf_member_offset": 3000, "conf_member_size": 300, "conf_is_gz": True,
            }
        )
    table = pa.Table.from_pylist(rows, schema=_INVENTORY_SCHEMA)
    path = inv_dir / f"w{worker}_b{batch:06d}.parquet"
    pq.write_table(table, path)
    return path


def test_load_needed_inventory_rows_reads_all_workers(tmp_path):
    """``_load_needed_inventory_rows`` must scan every ``w{N}_b*.parquet`` shard,
    not just worker 0. With the ``w*_b*.parquet`` glob both rows are returned
    and the join sees the full multi-worker inventory.
    """
    from proteinfoundation.datasets.teddymer.build_locator import (
        _load_needed_inventory_rows,
    )

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_shard(inv_dir, worker=0, batch=0, afdb_ids=["AF-W0ONLY-F1"])
    _write_fake_inventory_shard(inv_dir, worker=1, batch=0, afdb_ids=["AF-W1ONLY-F1"])

    needed = {"AF-W0ONLY-F1", "AF-W1ONLY-F1"}
    df = _load_needed_inventory_rows(inv_dir, needed)

    found = set(df["afdb_id"].astype(str))
    assert found == needed, (
        f"expected both worker shards' rows, got {found} "
        f"(missing {needed - found}). Glob in build_locator.py likely still "
        f"restricts to w0_b*.parquet."
    )


def test_load_needed_inventory_rows_w0_only_still_works(tmp_path):
    """Sanity baseline: when every needed afdb_id is in a w0 shard, the loader
    returns them. Guards against an over-eager glob fix that breaks w0-only
    inventories.
    """
    from proteinfoundation.datasets.teddymer.build_locator import (
        _load_needed_inventory_rows,
    )

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _write_fake_inventory_shard(
        inv_dir, worker=0, batch=0, afdb_ids=["AF-A0A005-F1", "AF-A0A009E3M2-F1"]
    )

    needed = {"AF-A0A005-F1", "AF-A0A009E3M2-F1"}
    df = _load_needed_inventory_rows(inv_dir, needed)

    assert set(df["afdb_id"].astype(str)) == needed
