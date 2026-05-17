"""End-to-end Teddymer view builder.

Reads the staged Teddymer release (``nonsingletonrep_metadata.tsv`` + the
``teddymer_repdb_h`` / ``teddymer_repdb_h.index`` MMseqs2 DB), joins parent
AFDB ids against the AFDB v4 master inventory, and emits ``dimers.parquet``
(with ``complexa_filter`` column) and ``locator_rows.parquet``.

CLI usage:

    python -m proteinfoundation.datasets.teddymer.build_view \\
        --staging   /path/to/teddymer/_raw \\
        --inventory /path/to/afdb_v4/.../inventory/manifests/batches \\
        --out       /path/to/teddymer_v1 \\
        --on-missing drop \\
        --sanity-n  50 \\
        --afdb-proteomes /path/to/afdb_v4/proteomes/v4
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from proteinfoundation.datasets.teddymer.build_locator import (
    build_locator_rows,
    write_locator_parquet,
)
from proteinfoundation.datasets.teddymer.parse_repdb_h import (
    add_complexa_filter,
    build_dimers_table,
    write_dimers_parquet,
)
from proteinfoundation.datasets.teddymer.sanity_check import sanity_check_dimers


logger = logging.getLogger(__name__)


VIEW_NAME = "teddymer_v1"


def run(
    staging: Path,
    inventory: Path,
    out: Path,
    afdb_proteomes: Path | None,
    sanity_n: int,
    on_missing: str,
) -> dict[str, Any]:
    """Build dimers.parquet + locator_rows.parquet under ``out``.

    Returns a dict with summary counts (``n_dimers``, ``n_complexa_filter``,
    ``n_locator_rows``, ``n_sanity_failures``).
    """
    staging = Path(staging)
    inventory = Path(inventory)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    h_path = staging / "teddymer_repdb" / "teddymer_repdb_h"
    idx_path = staging / "teddymer_repdb" / "teddymer_repdb_h.index"
    meta_path = staging / "nonsingletonrep_metadata.tsv"
    for p in (h_path, idx_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    logger.info("Parsing _h + metadata at %s ...", staging)
    dimers = build_dimers_table(h_path, idx_path, meta_path)
    dimers = add_complexa_filter(dimers)
    n_dimers = len(dimers)
    n_complexa = int(dimers["complexa_filter"].sum())
    logger.info("dimers rows=%d, complexa_filter=%d", n_dimers, n_complexa)

    logger.info("Joining against AFDB inventory at %s (on_missing=%s) ...",
                inventory, on_missing)
    locator = build_locator_rows(dimers, inventory, on_missing=on_missing)
    n_locator = len(locator)
    n_dimers_with_inventory = int(locator["dimer_index"].nunique())
    coverage = n_dimers_with_inventory / max(n_dimers, 1)
    logger.info(
        "locator rows=%d, dimers with inventory=%d/%d (%.2f%%)",
        n_locator, n_dimers_with_inventory, n_dimers, 100.0 * coverage,
    )

    dimers_path = out / "dimers.parquet"
    locator_path = out / "locator_rows.parquet"
    write_dimers_parquet(dimers, dimers_path)
    write_locator_parquet(locator, locator_path)
    logger.info("wrote %s (%d rows) and %s (%d rows)",
                dimers_path, n_dimers, locator_path, n_locator)

    n_sanity_failures: int | None = None
    if sanity_n > 0:
        if afdb_proteomes is None:
            raise ValueError("sanity_n > 0 requires --afdb-proteomes to be set")
        sampled_indices = set(locator["dimer_index"].unique())
        candidates = dimers[dimers["dimer_index"].isin(sampled_indices)]
        n_sample = min(sanity_n, len(candidates))
        sample = candidates.sample(n=n_sample, random_state=20260517).reset_index(drop=True)
        sample_locator = locator[locator["dimer_index"].isin(sample["dimer_index"])]
        failures = sanity_check_dimers(sample, sample_locator, afdb_proteomes=Path(afdb_proteomes))
        n_sanity_failures = len(failures)
        logger.info("sanity check: %d/%d failed", n_sanity_failures, n_sample)
        if failures:
            for dimer_index, reason in failures[:5]:
                logger.error("sanity failure dimer=%d reason=%s", dimer_index, reason)
            raise RuntimeError(
                f"sanity check failed on {n_sanity_failures}/{n_sample} dimers"
            )

    view_config = {
        "name": VIEW_NAME,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inventory_source": str(inventory),
        "raw_input_dir": str(staging),
        "afdb_proteomes": str(afdb_proteomes) if afdb_proteomes else None,
        "dimers_path": str(dimers_path),
        "locator_rows_path": str(locator_path),
        "row_count": n_dimers,
        "n_complexa_filter": n_complexa,
        "n_dimers_with_inventory": n_dimers_with_inventory,
        "n_locator_rows": n_locator,
        "on_missing": on_missing,
        "sanity_n": sanity_n,
        "sanity_failures": n_sanity_failures,
    }
    (out / "view_config.yaml").write_text(yaml.safe_dump(view_config, sort_keys=False))

    return {
        "n_dimers": n_dimers,
        "n_complexa_filter": n_complexa,
        "n_dimers_with_inventory": n_dimers_with_inventory,
        "n_locator_rows": n_locator,
        "n_sanity_failures": n_sanity_failures,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--staging", type=Path, required=True,
                   help="Path containing nonsingletonrep_metadata.tsv and teddymer_repdb/")
    p.add_argument("--inventory", type=Path, required=True,
                   help="Directory of AFDB master inventory w0_b*.parquet batches")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir for dimers.parquet and locator_rows.parquet")
    p.add_argument("--on-missing", choices=["raise", "drop"], default="drop",
                   help="Behaviour when a parent_afdb_id is not in the inventory.")
    p.add_argument("--sanity-n", type=int, default=50,
                   help="Number of dimers to spot-check against AFDB tars. 0 disables.")
    p.add_argument("--afdb-proteomes", type=Path,
                   default=Path("/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4"),
                   help="AFDB v4 proteomes directory (sanity check only).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = run(
        staging=args.staging,
        inventory=args.inventory,
        out=args.out,
        afdb_proteomes=args.afdb_proteomes,
        sanity_n=args.sanity_n,
        on_missing=args.on_missing,
    )
    logger.info("done: %s", summary)


if __name__ == "__main__":
    main()
