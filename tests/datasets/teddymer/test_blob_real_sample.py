"""Labs-only round-trip on a small random sample of the production locator.

Picks 50 random parents from the staged Teddymer view, repacks them into
a tmp blob, and byte-compares CIF / PAE / confidence vs the labs tars.

This is the gate the user runs by hand before / after the real ~30-min
repack. Marked `requires_labs` so CI / synthetic runs skip it.

Maps to plan §8 row `test_repack_real_sample`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

LABS_ROOT = Path("/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4")
VIEW_LOCATOR = Path(
    "/mnt/storage01/home/schekmenev/data/teddymer_v1/locator_rows.parquet"
)


pytestmark = pytest.mark.requires_labs


@pytest.fixture(scope="module")
def prod_locator():
    if not VIEW_LOCATOR.exists() or not LABS_ROOT.exists():
        pytest.skip("Teddymer view or labs root not mounted")
    return pq.read_table(VIEW_LOCATOR).to_pandas()


def test_repack_real_sample(tmp_path, prod_locator):
    from proteinfoundation.datasets.teddymer.build_blob import (
        plan_layout,
        repack,
        rewrite_locator,
    )
    from proteinfoundation.datasets.teddymer.io import (
        read_cif_bytes_from_tar,
        read_confidence_from_tar,
        read_pae_from_tar,
    )

    rng_seed = int(os.environ.get("TEDDYMER_BLOB_SAMPLE_SEED", "0"))
    parents = (
        prod_locator[["afdb_id"]]
        .drop_duplicates()
        .sample(50, random_state=rng_seed)["afdb_id"]
        .tolist()
    )
    sub = prod_locator[prod_locator["afdb_id"].isin(parents)].reset_index(drop=True)

    plan = plan_layout(sub)
    blob_path = tmp_path / "data.blob"
    dst_offsets = repack(plan, labs_root=LABS_ROOT, out_blob=blob_path)
    new_loc = rewrite_locator(sub, dst_offsets)

    for _, row in plan.iterrows():
        afdb_id = row["parent_afdb_id"]
        labs_tar = LABS_ROOT / row["source_tar_relpath"]
        cif_off, pae_off, conf_off = dst_offsets[afdb_id]

        cif_labs = read_cif_bytes_from_tar(
            labs_tar, int(row["src_cif_off"]), int(row["src_cif_size"]),
            bool(row["cif_is_gz"]),
        )
        cif_blob = read_cif_bytes_from_tar(
            blob_path, cif_off, int(row["src_cif_size"]), bool(row["cif_is_gz"]),
        )
        assert cif_labs == cif_blob, f"CIF mismatch on {afdb_id}"

        pae_labs = read_pae_from_tar(
            labs_tar, int(row["src_pae_off"]), int(row["src_pae_size"]),
            bool(row["pae_is_gz"]),
        )
        pae_blob = read_pae_from_tar(
            blob_path, pae_off, int(row["src_pae_size"]), bool(row["pae_is_gz"]),
        )
        assert (pae_labs == pae_blob).all(), f"PAE mismatch on {afdb_id}"

        conf_labs = read_confidence_from_tar(
            labs_tar, int(row["src_conf_off"]), int(row["src_conf_size"]),
            bool(row["conf_is_gz"]),
        )
        conf_blob = read_confidence_from_tar(
            blob_path, conf_off, int(row["src_conf_size"]), bool(row["conf_is_gz"]),
        )
        assert conf_labs == conf_blob, f"conf mismatch on {afdb_id}"

    assert (new_loc["source_tar_relpath"] == "data.blob").all()
