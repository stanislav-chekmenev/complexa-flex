"""Join Teddymer dimer parent AFDB ids against the AFDB master inventory to
emit ``locator_rows.parquet``.

The AFDB master inventory at
``/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/``
is sharded into ~500 ``w0_b*.parquet`` files, each carrying tar-byte-offset
locators for a few hundred thousand entries. This module streams them, keeps
only the rows whose ``afdb_id`` is needed by the Teddymer dimers set, and
expands each dimer into two rows (chain A and chain B) sharing the same
parent monomer locator.

The output schema mirrors
``inventory_first/views/la_proteina_afdb_512_v1/locator_rows.parquet`` with
three extra leading columns: ``dimer_id``, ``dimer_index``, ``chain_id``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


LOCATOR_INVENTORY_COLS: list[str] = [
    "sample_id", "afdb_id", "uniprot_id", "taxonomy_id",
    "source_tar_relpath", "source_tar_basename", "source_version", "bucket",
    "has_cif", "has_pae", "has_conf",
    "cif_member_name", "cif_member_offset", "cif_member_size", "cif_is_gz",
    "pae_member_name", "pae_member_offset", "pae_member_size", "pae_is_gz",
    "conf_member_name", "conf_member_offset", "conf_member_size", "conf_is_gz",
]


def _load_needed_inventory_rows(
    inventory_dir: Path, needed_afdb_ids: set[str]
) -> pd.DataFrame:
    """Scan every ``w0_b*.parquet`` under ``inventory_dir`` and concatenate the
    rows whose ``afdb_id`` is in ``needed_afdb_ids``. Drops duplicate afdb_ids
    keeping the first occurrence (defensive — shouldn't happen, but harmless)."""
    batch_paths = sorted(Path(inventory_dir).glob("w0_b*.parquet"))
    if not batch_paths:
        raise FileNotFoundError(f"No w0_b*.parquet under {inventory_dir}")

    dataset = ds.dataset([str(p) for p in batch_paths], format="parquet")
    filt = ds.field("afdb_id").isin(list(needed_afdb_ids))
    table: pa.Table = dataset.to_table(filter=filt, columns=LOCATOR_INVENTORY_COLS)
    df = table.to_pandas()
    return df.drop_duplicates(subset=["afdb_id"], keep="first").reset_index(drop=True)


def build_locator_rows(dimers: pd.DataFrame, inventory_dir: Path) -> pd.DataFrame:
    """Build the ``locator_rows.parquet`` DataFrame from ``dimers`` and the
    AFDB master inventory at ``inventory_dir``.
    """
    needed = set(dimers["parent_afdb_id"].astype(str).unique())
    inv = _load_needed_inventory_rows(Path(inventory_dir), needed)

    found = set(inv["afdb_id"].astype(str))
    missing = needed - found
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} parent_afdb_id values missing from AFDB inventory. "
            f"Examples: {sample}"
        )

    # Two-chain expansion: cross dimers x {A, B}, then join to inventory.
    chains = pd.DataFrame({"chain_id": ["A", "B"]})
    expanded = dimers[["dimer_id", "dimer_index", "parent_afdb_id"]].merge(
        chains, how="cross"
    )
    locator = expanded.merge(
        inv, left_on="parent_afdb_id", right_on="afdb_id", how="left", validate="m:1"
    ).drop(columns=["parent_afdb_id"])

    locator = locator.sort_values(["dimer_index", "chain_id"]).reset_index(drop=True)

    out_cols = ["dimer_id", "dimer_index", "chain_id"] + LOCATOR_INVENTORY_COLS
    return locator[out_cols]


def write_locator_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
