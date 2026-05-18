"""Unit tests for `proteinfoundation.datasets.teddymer.{build_blob,integrity}`.

Covers md5 streaming, sidecar format, view_config emission, and the
train-time `verify_teddymer_blob_integrity` gate (synthetic; the labs
variant is `test_blob_real_sample.py`).

Maps to plan §8 test table.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from proteinfoundation.datasets.teddymer.build_blob import (
    compute_md5,
    plan_layout,
    repack,
    rewrite_locator,
    write_md5_sidecar,
    write_view_config,
)
from proteinfoundation.datasets.teddymer.integrity import (
    verify_teddymer_blob_integrity,
)

from tests.datasets.teddymer._fixtures.build_mini_tar import build_mini_view


def _stage_blob_view(tmp_path: Path) -> dict:
    """Run plan + repack + rewrite + sidecar + view_config end-to-end on the
    mini fixture, returning the staged out_dir and snapshot path."""
    info = build_mini_view(tmp_path)
    loc = pq.read_table(info["locator_path"]).to_pandas()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    plan = plan_layout(loc)
    blob_path = out_dir / "data.blob"
    dst_offsets = repack(plan, labs_root=info["labs_root"], out_blob=blob_path)
    new_loc = rewrite_locator(loc, dst_offsets)

    locator_dst = out_dir / "locator_rows.parquet"
    pq.write_table(pa.Table.from_pandas(new_loc, preserve_index=False), locator_dst)

    dimers_dst = out_dir / "dimers.parquet"
    shutil.copyfile(info["dimers_path"], dimers_dst)

    sidecar = out_dir / "data.md5"
    write_md5_sidecar(
        [blob_path, locator_dst, dimers_dst], sidecar_path=sidecar, relative_to=out_dir
    )
    snapshot = tmp_path / "snapshot.md5"
    shutil.copyfile(sidecar, snapshot)

    write_view_config(
        out_dir,
        version="teddymer_v1_blob_20260518_abcd1234",
        blob_path=blob_path,
        locator_path=locator_dst,
        dimers_path=dimers_dst,
        labs_root=info["labs_root"],
        md5s={
            "data.blob": compute_md5(blob_path),
            "locator_rows.parquet": compute_md5(locator_dst),
            "dimers.parquet": compute_md5(dimers_dst),
        },
    )

    return {
        "out_dir": out_dir,
        "blob_path": blob_path,
        "locator_path": locator_dst,
        "dimers_path": dimers_dst,
        "sidecar": sidecar,
        "snapshot": snapshot,
    }


def test_compute_md5_streams_in_chunks(tmp_path):
    rng_path = tmp_path / "buf.bin"
    payload = os.urandom(32 * 1024 * 1024)
    rng_path.write_bytes(payload)

    truth = hashlib.md5(payload).hexdigest()

    assert compute_md5(rng_path, chunk_bytes=4096) == truth
    assert compute_md5(rng_path, chunk_bytes=8 * 1024 * 1024) == truth
    assert compute_md5(rng_path, chunk_bytes=len(payload)) == truth


def test_md5_sidecar_format_matches_md5sum(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    c.write_bytes(b"gamma")

    sidecar = tmp_path / "data.md5"
    write_md5_sidecar([a, b, c], sidecar_path=sidecar, relative_to=tmp_path)

    lines = sidecar.read_text().splitlines()
    assert len(lines) == 3
    parsed = [(ln[:32], ln[34:]) for ln in lines]
    names = [name for _, name in parsed]
    assert names == sorted(names)
    for digest, _ in parsed:
        assert len(digest) == 32
        int(digest, 16)


def test_view_config_records_version_and_md5s(tmp_path):
    staged = _stage_blob_view(tmp_path)
    vc = yaml.safe_load((staged["out_dir"] / "view_config.yaml").read_text())

    assert vc["version"].startswith("teddymer_v1_blob_")
    assert set(vc["md5s"].keys()) == {"data.blob", "locator_rows.parquet", "dimers.parquet"}
    assert "labs_root" in vc
    assert "build_timestamp" in vc


def test_verify_integrity_passes_on_fresh_blob(tmp_path):
    staged = _stage_blob_view(tmp_path)
    result = verify_teddymer_blob_integrity(
        view_root=staged["out_dir"],
        snapshot_path=staged["snapshot"],
    )
    assert result is None


def test_verify_integrity_fails_on_tampered_blob(tmp_path):
    staged = _stage_blob_view(tmp_path)
    with open(staged["blob_path"], "r+b") as f:
        f.seek(0)
        f.write(b"\xff")

    from proteinfoundation.datasets.teddymer.integrity import IntegrityError

    with pytest.raises(IntegrityError) as excinfo:
        verify_teddymer_blob_integrity(
            view_root=staged["out_dir"],
            snapshot_path=staged["snapshot"],
        )
    msg = str(excinfo.value)
    assert "data.blob" in msg
    assert str(staged["snapshot"]) in msg


def test_verify_integrity_fails_on_missing_snapshot(tmp_path):
    staged = _stage_blob_view(tmp_path)
    staged["snapshot"].unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        verify_teddymer_blob_integrity(
            view_root=staged["out_dir"],
            snapshot_path=staged["snapshot"],
        )
    assert str(staged["snapshot"]) in str(excinfo.value)


def test_verify_integrity_fails_on_missing_blob(tmp_path):
    staged = _stage_blob_view(tmp_path)
    staged["blob_path"].unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        verify_teddymer_blob_integrity(
            view_root=staged["out_dir"],
            snapshot_path=staged["snapshot"],
        )
    assert "data.blob" in str(excinfo.value)


def _train_cfg_skeleton(integrity_block):
    """Minimal Hydra config skeleton accepted by `train_confidence.main`."""
    from omegaconf import OmegaConf
    return OmegaConf.create(
        {
            "seed": 0,
            "integrity": integrity_block,
            "confidence": {"head": {}},
            "training": {
                "trunk_ckpt_path": "x",
                "autoencoder_ckpt_path": "x",
                "trunk_eval_t": 0.99,
                "opt": {
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "warmup_steps": 0,
                    "min_lr": 1e-6,
                },
                "loss": {},
            },
            "data": {"datamodule": {}},
            "trainer": {},
        }
    )


def test_train_entry_invokes_integrity_when_enabled():
    """Integrity hook gates on cfg.integrity.enabled; flipping it on must
    cause the train entry to call `verify_teddymer_blob_integrity` exactly
    once before head construction."""
    from proteinfoundation.confidence import train_confidence

    cfg = _train_cfg_skeleton(
        {
            "enabled": True,
            "view_root": "/tmp/fake-view",
            "snapshot_path": "/tmp/fake-snap.md5",
        }
    )

    with mock.patch(
        "proteinfoundation.datasets.teddymer.integrity.verify_teddymer_blob_integrity"
    ) as spy, mock.patch.object(
        train_confidence,
        "build_confidence_head_from_cfg",
        side_effect=RuntimeError("stop after integrity"),
    ):
        with pytest.raises(RuntimeError, match="stop after integrity"):
            train_confidence.main.__wrapped__(cfg)
        assert spy.call_count == 1


def test_train_entry_skips_integrity_when_disabled():
    """`integrity.enabled=false` must skip the verify call entirely."""
    from proteinfoundation.confidence import train_confidence

    cfg = _train_cfg_skeleton({"enabled": False})

    with mock.patch(
        "proteinfoundation.datasets.teddymer.integrity.verify_teddymer_blob_integrity"
    ) as spy, mock.patch.object(
        train_confidence,
        "build_confidence_head_from_cfg",
        side_effect=RuntimeError("stop after integrity"),
    ):
        with pytest.raises(RuntimeError, match="stop after integrity"):
            train_confidence.main.__wrapped__(cfg)
        assert spy.call_count == 0
