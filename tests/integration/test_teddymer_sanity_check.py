"""Integration sanity check (step 4 of the 2026-05-17 handoff).

Builds the full ``dimers.parquet`` from the staged Teddymer release, joins
50 random non-singleton-rep dimers against the AFDB v4 master inventory, pulls
the parent monomer's confidence JSON from the AFDB tar at the recorded byte
offsets, and verifies:

1. ``AvgIntPlddt`` from metadata is reproduced (to within 0.5 pLDDT) by
   recomputing from the per-chain digit-bucket ``IntPlddt`` strings using
   the lower-bound bucket convention.
2. Every TED-domain residue interval lies inside the AFDB sequence length —
   catching residue-numbering / off-by-one bugs that would otherwise corrupt
   training labels downstream.

Skipped unless both the staged Teddymer release and the AFDB v4 mirror are
present (so CI without those mounts simply skips this test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

TEDDYMER_STAGING = Path("/mnt/storage01/home/schekmenev/data/teddymer_v1")
AFDB_INVENTORY = Path(
    "/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches"
)
AFDB_PROTEOMES = Path("/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4")

pytestmark = pytest.mark.skipif(
    not (
        (TEDDYMER_STAGING / "_raw" / "nonsingletonrep_metadata.tsv").exists()
        and AFDB_INVENTORY.exists()
        and AFDB_PROTEOMES.exists()
    ),
    reason="Sanity check requires staged Teddymer + AFDB v4 mirror; skipped in CI.",
)


@pytest.fixture(scope="module")
def dimers_full():
    from proteinfoundation.data.teddymer.parse_repdb_h import build_dimers_table

    return build_dimers_table(
        TEDDYMER_STAGING / "_raw" / "teddymer_repdb" / "teddymer_repdb_h",
        TEDDYMER_STAGING / "_raw" / "teddymer_repdb" / "teddymer_repdb_h.index",
        TEDDYMER_STAGING / "_raw" / "nonsingletonrep_metadata.tsv",
    )


@pytest.fixture(scope="module")
def joined_locator(dimers_full):
    """Build the full locator, dropping dimers whose parent monomer is not in
    the local AFDB master inventory (it only covers ~12% of AFDB v4)."""
    from proteinfoundation.data.teddymer.build_locator import build_locator_rows

    return build_locator_rows(dimers_full, AFDB_INVENTORY, on_missing="drop")


@pytest.fixture(scope="module")
def dimers_sample(dimers_full, joined_locator):
    """Sample 50 dimers from the in-inventory subset so the AFDB round-trip is
    actually exercised."""
    in_inventory = set(joined_locator["dimer_index"].unique())
    survivors = dimers_full[dimers_full["dimer_index"].isin(in_inventory)]
    return survivors.sample(n=50, random_state=20260517).reset_index(drop=True)


def test_50_dimers_round_trip_through_afdb(dimers_sample, joined_locator):
    from proteinfoundation.data.teddymer.sanity_check import sanity_check_dimers

    locator_for_sample = joined_locator[
        joined_locator["dimer_index"].isin(dimers_sample["dimer_index"])
    ]
    failures = sanity_check_dimers(
        dimers_sample, locator_for_sample, afdb_proteomes=AFDB_PROTEOMES
    )
    assert not failures, (
        f"{len(failures)} of 50 dimers failed AFDB-join sanity check. "
        f"First 5: {failures[:5]}"
    )
