"""Rebuild Teddymer ``locator_rows.parquet`` in place after fixing the
``w0_b*.parquet -> w*_b*.parquet`` glob bug in ``build_locator.py``.

The dimer table (``dimers.parquet``) is unchanged; only the inventory join
needs to re-run against the full multi-worker AFDB master inventory. This
script backs up the old locator parquet (once), rejoins, writes the new
locator parquet, and updates ``view_config.yaml`` in place with rebuilt
counts and an ISO timestamp.

Run via:
    uv run python scripts/rebuild_teddymer_locator.py
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from proteinfoundation.datasets.teddymer.build_locator import (
    build_locator_rows,
    write_locator_parquet,
)


VIEW_DIR = Path("/mnt/storage01/home/schekmenev/data/teddymer_v1")
DIMERS_PATH = VIEW_DIR / "dimers.parquet"
LOCATOR_PATH = VIEW_DIR / "locator_rows.parquet"
LOCATOR_BACKUP_PATH = VIEW_DIR / "locator_rows.parquet.pre_glob_fix.bak"
VIEW_CONFIG_PATH = VIEW_DIR / "view_config.yaml"
INVENTORY_DIR = Path(
    "/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/"
)


def main() -> None:
    print(f"[rebuild] loading dimers from {DIMERS_PATH}")
    dimers = pd.read_parquet(DIMERS_PATH)
    n_dimers_total = len(dimers)
    print(f"[rebuild] {n_dimers_total} dimers loaded")

    if LOCATOR_BACKUP_PATH.exists():
        print(f"[rebuild] backup already exists at {LOCATOR_BACKUP_PATH}; not overwriting")
    elif LOCATOR_PATH.exists():
        print(f"[rebuild] backing up old locator -> {LOCATOR_BACKUP_PATH}")
        shutil.move(str(LOCATOR_PATH), str(LOCATOR_BACKUP_PATH))
    else:
        print(f"[rebuild] no existing locator parquet at {LOCATOR_PATH}; skipping backup")

    if LOCATOR_BACKUP_PATH.exists():
        old_df = pd.read_parquet(LOCATOR_BACKUP_PATH, columns=["dimer_id"])
        old_n_locator_rows = len(old_df)
    else:
        old_n_locator_rows = 0
    print(f"[rebuild] old n_locator_rows = {old_n_locator_rows}")

    print(f"[rebuild] joining against AFDB master inventory at {INVENTORY_DIR}")
    locator = build_locator_rows(
        dimers, inventory_dir=INVENTORY_DIR, on_missing="drop"
    )
    new_n_locator_rows = len(locator)
    n_dimers_with_inventory = new_n_locator_rows // 2
    coverage_pct = 100.0 * n_dimers_with_inventory / n_dimers_total
    print(f"[rebuild] new n_locator_rows = {new_n_locator_rows}")
    print(f"[rebuild] n_dimers_with_inventory = {n_dimers_with_inventory}")
    print(f"[rebuild] coverage = {coverage_pct:.4f}% ({n_dimers_with_inventory}/{n_dimers_total})")

    print(f"[rebuild] writing new locator parquet -> {LOCATOR_PATH}")
    write_locator_parquet(locator, LOCATOR_PATH)

    print(f"[rebuild] updating {VIEW_CONFIG_PATH} in place")
    with VIEW_CONFIG_PATH.open("r") as fh:
        view_config = yaml.safe_load(fh)
    view_config["n_dimers_with_inventory"] = n_dimers_with_inventory
    view_config["n_locator_rows"] = new_n_locator_rows
    view_config["rebuilt_at"] = datetime.now(timezone.utc).isoformat()
    view_config["rebuild_note"] = (
        "locator rebuilt after fixing w0_b* -> w*_b* glob bug in build_locator.py"
    )
    with VIEW_CONFIG_PATH.open("w") as fh:
        yaml.safe_dump(view_config, fh, sort_keys=False)

    delta = new_n_locator_rows - old_n_locator_rows
    print()
    print("=" * 72)
    print("Teddymer locator rebuild summary")
    print("=" * 72)
    print(f"old n_locator_rows                : {old_n_locator_rows}")
    print(f"new n_locator_rows                : {new_n_locator_rows}")
    print(f"delta                             : {delta}")
    print(f"n_dimers_total                    : {n_dimers_total}")
    print(f"n_dimers_with_inventory           : {n_dimers_with_inventory}")
    print(f"coverage %                        : {coverage_pct:.4f}")
    print(f"backup path                       : {LOCATOR_BACKUP_PATH}")
    print("=" * 72)

    readback = pd.read_parquet(LOCATOR_PATH, columns=["dimer_id"])
    assert len(readback) == 2 * n_dimers_with_inventory, (
        f"sanity: locator parquet has {len(readback)} rows, "
        f"expected 2 * {n_dimers_with_inventory} = {2 * n_dimers_with_inventory}"
    )
    print(f"[rebuild] sanity check OK: readback {len(readback)} == 2 * {n_dimers_with_inventory}")


if __name__ == "__main__":
    main()
