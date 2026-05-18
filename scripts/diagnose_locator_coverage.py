"""Diagnose Teddymer locator coverage against w0-only vs full w0..w7 AFDB inventory.

One-shot diagnostic. Streams all ``w*_b*.parquet`` shards under the AFDB master
inventory and reports how many Teddymer dimers' parent_afdb_id values are found
in (a) the w0-only 64 shards that ``build_locator.py`` currently globs and
(b) the full w0..w7 512-shard inventory that recently landed. The delta
quantifies how much coverage the glob fix is expected to recover.

Run via:
    uv run python scripts/diagnose_locator_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


DIMERS_PARQUET = Path(
    "/mnt/storage01/home/schekmenev/data/teddymer_v1/dimers.parquet"
)
INVENTORY_DIR = Path(
    "/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/"
)


def _found_afdb_ids(shards: list[Path], needed: list[str]) -> set[str]:
    """Stream ``shards`` and return the subset of ``needed`` afdb_ids present.

    Uses pyarrow.dataset with an ``isin`` filter and projects only ``afdb_id``
    to keep the materialised table tiny regardless of total shard size.
    """
    dataset = ds.dataset([str(p) for p in shards], format="parquet")
    filt = ds.field("afdb_id").isin(needed)
    table = dataset.to_table(filter=filt, columns=["afdb_id"])
    return set(table.column("afdb_id").to_pylist())


def main() -> None:
    print(f"[diag] loading dimers from {DIMERS_PARQUET}")
    dimers = pd.read_parquet(DIMERS_PARQUET, columns=["parent_afdb_id"])
    needed_unique = sorted(set(dimers["parent_afdb_id"].astype(str)))
    n_dimers_total = len(dimers)
    n_unique_parents = len(needed_unique)
    print(
        f"[diag] {n_dimers_total} total dimers, "
        f"{n_unique_parents} unique parent_afdb_ids"
    )

    all_shards = sorted(INVENTORY_DIR.glob("w*_b*.parquet"))
    w0_shards = sorted(INVENTORY_DIR.glob("w0_b*.parquet"))
    print(
        f"[diag] inventory shards: total={len(all_shards)}, w0-only={len(w0_shards)}"
    )
    worker_counts: dict[str, int] = {}
    for p in all_shards:
        worker = p.name.split("_", 1)[0]
        worker_counts[worker] = worker_counts.get(worker, 0) + 1
    print(f"[diag] shards per worker: {dict(sorted(worker_counts.items()))}")

    print("[diag] scanning w0-only inventory (current build_locator.py behaviour)...")
    found_w0 = _found_afdb_ids(w0_shards, needed_unique)
    print(f"[diag] w0-only found {len(found_w0)} / {n_unique_parents} unique parents")

    print("[diag] scanning full w0..w7 inventory...")
    found_full = _found_afdb_ids(all_shards, needed_unique)
    print(
        f"[diag] full w0..w7 found {len(found_full)} / {n_unique_parents} unique parents"
    )

    new_parents = sorted(found_full - found_w0)
    lost_parents = sorted(found_w0 - found_full)
    print(f"[diag] parents recovered by non-w0 shards: {len(new_parents)}")
    print(f"[diag] parents in w0 but NOT in full scan (sanity, expect 0): {len(lost_parents)}")

    is_w0 = dimers["parent_afdb_id"].isin(found_w0)
    is_full = dimers["parent_afdb_id"].isin(found_full)
    dimers_w0 = int(is_w0.sum())
    dimers_full = int(is_full.sum())
    dimers_delta = dimers_full - dimers_w0

    def _pct(x: int, total: int) -> str:
        return f"{x} ({100.0 * x / total:.2f}% of {total})"

    print()
    print("=" * 72)
    print("Teddymer locator coverage diagnostic")
    print("=" * 72)
    print(f"total dimers                              : {n_dimers_total}")
    print(f"unique parent_afdb_ids                    : {n_unique_parents}")
    print(f"w0-only inventory shards                  : {len(w0_shards)}")
    print(f"full w0..w7 inventory shards              : {len(all_shards)}")
    print(f"unique parents covered by w0-only         : {_pct(len(found_w0), n_unique_parents)}")
    print(f"unique parents covered by full w0..w7     : {_pct(len(found_full), n_unique_parents)}")
    print(f"unique parents recovered by glob fix      : {len(new_parents)}")
    print(f"dimers covered by w0-only inventory       : {_pct(dimers_w0, n_dimers_total)}")
    print(f"dimers covered by full w0..w7 inventory   : {_pct(dimers_full, n_dimers_total)}")
    print(f"dimers recovered by glob fix (delta)      : {dimers_delta}")
    print("=" * 72)

    if new_parents:
        print("[diag] example parent_afdb_ids found only in non-w0 shards (first 10):")
        for aid in new_parents[:10]:
            print(f"  - {aid}")

    if lost_parents:
        print("[diag] WARNING: parents in w0 but missing from full scan (first 10):")
        for aid in lost_parents[:10]:
            print(f"  - {aid}")


if __name__ == "__main__":
    main()
