"""Unit tests for `proteinfoundation.datasets.teddymer.build_blob`.

Drives the design of `plan_layout`, `repack`, and `rewrite_locator`. These
tests are the contract the implementation must satisfy. All synthetic;
the labs-mounted round-trip lives in `test_blob_real_sample.py`.

Maps to plan §8 test table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from proteinfoundation.datasets.teddymer import build_blob as build_blob_mod
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

from tests.datasets.teddymer._fixtures.build_mini_tar import (
    build_mini_view,
    locator_row_for,
    write_fake_parent,
    write_locator_parquet,
)


def _locator_df(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def test_plan_layout_dedups_intra_monomer_pairs(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    assert len(loc) == 4

    plan = plan_layout(loc)

    assert len(plan) == 3
    assert set(plan["parent_afdb_id"]) == {"AF-P1-F1", "AF-P2-F1", "AF-P3-F1"}
    assert int(plan.iloc[0]["dst_cif_off"]) == 0


def test_plan_layout_groups_by_source_tar(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])

    plan = plan_layout(loc)

    keys = list(zip(plan["source_tar_relpath"], plan["src_cif_off"]))
    assert keys == sorted(keys)


def test_plan_layout_monotone_dst_offsets(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])

    plan = plan_layout(loc).reset_index(drop=True)

    for _, row in plan.iterrows():
        assert int(row["dst_cif_off"]) < int(row["dst_pae_off"])
        assert int(row["dst_pae_off"]) < int(row["dst_conf_off"])

    for i in range(len(plan) - 1):
        cur = plan.iloc[i]
        nxt = plan.iloc[i + 1]
        cur_end = int(cur["dst_conf_off"]) + int(cur["src_conf_size"])
        assert int(nxt["dst_cif_off"]) >= cur_end


def test_repack_byte_roundtrip(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    blob_path = out_dir / "data.blob"

    plan = plan_layout(loc)
    dst_offsets = repack(plan, labs_root=info["labs_root"], out_blob=blob_path)
    new_loc = rewrite_locator(loc, dst_offsets)

    assert blob_path.exists()
    for parent_id, parent in info["parents"].items():
        cif_off, pae_off, conf_off = dst_offsets[parent_id]

        cif_raw = read_cif_bytes_from_tar(blob_path, cif_off, parent.cif.size, False)
        assert cif_raw == parent.cif_bytes

        cif_dec = read_cif_bytes_from_tar(blob_path, cif_off, parent.cif.size, True)
        assert cif_dec == read_cif_bytes_from_tar(
            info["labs_root"] / parent.tar_relpath,
            parent.cif.offset,
            parent.cif.size,
            True,
        )

        pae_via_blob = read_pae_from_tar(blob_path, pae_off, parent.pae.size, True)
        pae_via_labs = read_pae_from_tar(
            info["labs_root"] / parent.tar_relpath,
            parent.pae.offset,
            parent.pae.size,
            True,
        )
        assert (pae_via_blob == pae_via_labs).all()

        conf_via_blob = read_confidence_from_tar(
            blob_path, conf_off, parent.conf.size, True
        )
        conf_via_labs = read_confidence_from_tar(
            info["labs_root"] / parent.tar_relpath,
            parent.conf.offset,
            parent.conf.size,
            True,
        )
        assert conf_via_blob == conf_via_labs

    assert (new_loc["source_tar_relpath"] == "data.blob").all()


def test_repack_writes_tmp_and_renames_atomically(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    blob_path = out_dir / "data.blob"

    plan = plan_layout(loc)
    repack(plan, labs_root=info["labs_root"], out_blob=blob_path)

    assert blob_path.exists()
    assert not (out_dir / "data.blob.tmp").exists()


def test_rewrite_locator_preserves_row_count_and_schema(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    blob_path = tmp_path / "out" / "data.blob"
    blob_path.parent.mkdir()

    plan = plan_layout(loc)
    dst_offsets = repack(plan, labs_root=info["labs_root"], out_blob=blob_path)
    new_loc = rewrite_locator(loc, dst_offsets)

    assert set(new_loc.columns) == set(loc.columns)
    assert len(new_loc) == len(loc)
    assert (new_loc["source_tar_relpath"] == "data.blob").all()
    assert (new_loc["source_tar_basename"] == "data.blob").all()


def test_rewrite_locator_intra_monomer_dimer_shares_offsets(tmp_path):
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    blob_path = tmp_path / "out" / "data.blob"
    blob_path.parent.mkdir()

    plan = plan_layout(loc)
    dst_offsets = repack(plan, labs_root=info["labs_root"], out_blob=blob_path)
    new_loc = rewrite_locator(loc, dst_offsets)

    d1 = new_loc[new_loc["dimer_id"] == "D1"].sort_values("chain_id").reset_index(drop=True)
    assert len(d1) == 2
    assert int(d1.iloc[0]["cif_member_offset"]) == int(d1.iloc[1]["cif_member_offset"])
    assert int(d1.iloc[0]["pae_member_offset"]) == int(d1.iloc[1]["pae_member_offset"])
    assert int(d1.iloc[0]["conf_member_offset"]) == int(d1.iloc[1]["conf_member_offset"])


def test_main_writes_view_config_to_snapshot_dir_when_outside_out_dir(
    tmp_path, monkeypatch
):
    """``main()`` writes ``view_config.yaml`` BOTH next to the staged
    out_dir AND next to the labs-side snapshot, so the integrity gate's
    version-string fast path has a file to read from both sides
    (plan section 4.3 + R12)."""
    info = build_mini_view(tmp_path)
    out_dir = tmp_path / "out"
    snapshot_dir = tmp_path / "snapshot_dir"
    snapshot_dir.mkdir()
    snapshot_path = snapshot_dir / "data.md5"

    argv = [
        "build_blob",
        "--locator", str(info["locator_path"]),
        "--labs-root", str(info["labs_root"]),
        "--out-dir", str(out_dir),
        "--dimers", str(info["dimers_path"]),
        "--snapshot-path", str(snapshot_path),
        "--progress-every", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_blob_mod.main()

    view_config_out = out_dir / "view_config.yaml"
    view_config_snap = snapshot_dir / "view_config.yaml"
    assert view_config_out.exists()
    assert view_config_snap.exists()

    vc_out = yaml.safe_load(view_config_out.read_text())
    vc_snap = yaml.safe_load(view_config_snap.read_text())

    assert vc_out["version"] == vc_snap["version"]
    expected_md5_keys = {"data.blob", "locator_rows.parquet", "dimers.parquet"}
    assert set(vc_out["md5s"].keys()) == expected_md5_keys
    assert set(vc_snap["md5s"].keys()) == expected_md5_keys
    assert vc_out["md5s"] == vc_snap["md5s"]


def test_repack_unlinks_tmp_on_exception(tmp_path, monkeypatch):
    """If a write fails mid-repack, the ``.tmp`` file must be unlinked so
    a crashed sbatch does not leak 100+ GB of partial blob on /netscratch."""
    info = build_mini_view(tmp_path)
    loc = _locator_df(info["locator_path"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    blob_path = out_dir / "data.blob"
    plan = plan_layout(loc)

    real_read = build_blob_mod._read_member_bytes
    call_counter = {"n": 0}

    def flaky_read(src_fh, off, size):
        call_counter["n"] += 1
        if call_counter["n"] >= 4:
            raise OSError("simulated read failure")
        return real_read(src_fh, off, size)

    monkeypatch.setattr(build_blob_mod, "_read_member_bytes", flaky_read)

    with pytest.raises(OSError, match="simulated read failure"):
        repack(plan, labs_root=info["labs_root"], out_blob=blob_path)

    assert not (out_dir / "data.blob.tmp").exists()
    assert not blob_path.exists()
