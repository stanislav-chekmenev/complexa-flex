"""Red-phase tests for ``configs/dataset/unified/teddymer_with_plddt_and_pae.yaml``
and the new ``TeddymerDimerDataModule``.

The config is the Hydra entry-point that ties together the parquet view at
``/mnt/storage01/home/schekmenev/data/teddymer_v1/`` and the new transforms.
This test composes the config against ``tmp_path`` fixtures, instantiates
the datamodule, and pulls one batch from the train dataloader to confirm
the dataset-level ``complexa_filter == True`` filter binds, that the
expected fields land on the batch, and that variable-length dimers pad
correctly.

Spec: docs/superpowers/plans/2026-05-17_pr_a_teddymer_data_plumbing.md §9 Test 3.
Failure mode (red phase): Hydra ``Could not resolve`` / ``InstantiationException``
for ``proteinfoundation.datasets.teddymer.dataset.TeddymerDimerDataModule``,
or ``FileNotFoundError`` on ``teddymer_with_plddt_and_pae.yaml`` while the
config itself does not yet exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.unit.datasets.conftest import (
    build_fake_afdb_tar,
    make_locator_row,
    write_dimers_parquet,
    write_locator_parquet,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def _build_view(tmp_path: Path):
    """Build a 4-dimer fake view: 2 parents x 2 chains each = 8 locator rows,
    with exactly 3 of the 4 dimers having ``complexa_filter == True``.
    """
    n_full = 12
    matrix = np.zeros((n_full, n_full), dtype=np.int64)
    scores = [50.0 + i for i in range(n_full)]
    fake_p1 = build_fake_afdb_tar(tmp_path, "AF-FAKE0001-F1", scores, matrix)
    fake_p2 = build_fake_afdb_tar(tmp_path, "AF-FAKE0002-F1", scores, matrix)

    dimers = [
        {
            "dimer_id": "D1", "dimer_index": 1,
            "parent_afdb_id": "AF-FAKE0001-F1",
            "residue_intervals_A": [{"lo": 1, "hi": 4}],
            "residue_intervals_B": [{"lo": 5, "hi": 8}],
            "interface_length": 12, "avg_int_plddt": 80.0, "avg_int_pae": 5.0,
            "complexa_filter": True,
        },
        {
            "dimer_id": "D2", "dimer_index": 2,
            "parent_afdb_id": "AF-FAKE0001-F1",
            "residue_intervals_A": [{"lo": 2, "hi": 6}],
            "residue_intervals_B": [{"lo": 7, "hi": 10}],
            "interface_length": 15, "avg_int_plddt": 75.0, "avg_int_pae": 4.0,
            "complexa_filter": True,
        },
        {
            "dimer_id": "D3", "dimer_index": 3,
            "parent_afdb_id": "AF-FAKE0002-F1",
            "residue_intervals_A": [{"lo": 1, "hi": 3}],
            "residue_intervals_B": [{"lo": 4, "hi": 6}],
            "interface_length": 8, "avg_int_plddt": 71.0, "avg_int_pae": 8.0,
            "complexa_filter": True,
        },
        {
            "dimer_id": "D4", "dimer_index": 4,
            "parent_afdb_id": "AF-FAKE0002-F1",
            "residue_intervals_A": [{"lo": 1, "hi": 3}],
            "residue_intervals_B": [{"lo": 4, "hi": 6}],
            "interface_length": 5, "avg_int_plddt": 60.0, "avg_int_pae": 20.0,
            "complexa_filter": False,
        },
    ]
    dimers_path = write_dimers_parquet(tmp_path / "dimers.parquet", dimers)

    locator_rows = []
    fake_by_parent = {
        "AF-FAKE0001-F1": fake_p1,
        "AF-FAKE0002-F1": fake_p2,
    }
    for d in dimers:
        fake = fake_by_parent[d["parent_afdb_id"]]
        locator_rows.append(
            make_locator_row(
                dimer_id=d["dimer_id"],
                dimer_index=d["dimer_index"],
                chain_id="A",
                fake=fake,
            )
        )
        locator_rows.append(
            make_locator_row(
                dimer_id=d["dimer_id"],
                dimer_index=d["dimer_index"],
                chain_id="B",
                fake=fake,
            )
        )
    locator_path = write_locator_parquet(tmp_path / "locator_rows.parquet", locator_rows)

    return dimers_path, locator_path


def _compose_cfg(dimers_path: Path, locator_path: Path, afdb_root: Path):
    config_dir = CONFIG_DIR / "dataset" / "unified"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="teddymer_with_plddt_and_pae",
            overrides=[
                f"datamodule.dimers_parquet={dimers_path}",
                f"datamodule.locator_parquet={locator_path}",
                f"datamodule.afdb_proteomes_root={afdb_root}",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "datamodule.train_split=0.5",
            ],
        )


def test_config_composes_and_instantiates_datamodule(tmp_path):
    dimers_path, locator_path = _build_view(tmp_path)
    cfg = _compose_cfg(dimers_path, locator_path, tmp_path)

    dm = hydra.utils.instantiate(cfg.datamodule)

    assert dm.__class__.__name__ == "TeddymerDimerDataModule"


def test_relaxed_filters_drop_only_small_interfaces(tmp_path):
    """The YAML filter list now expresses the thresholds explicitly:
    ``interface_length > 10``, ``avg_int_plddt > 30.0``, ``avg_int_pae < 25.0``.

    Against the 4-dimer fake fixture (D1 il=12, D2 il=15, D3 il=8, D4 il=5),
    only D1 and D2 pass — D3 and D4 are dropped on the interface-length cut.
    D1/D2's pLDDT (80, 75) and PAE (5, 4) are far inside the relaxed bounds.
    The fixture's old D3 row (il=8, plddt=71, pae=8) previously slipped through
    on the legacy ``complexa_filter == True`` rule (which baked plddt>70 + pae<10
    into the parquet column) but is dropped by the new explicit filter because
    its interface is too small.
    """
    dimers_path, locator_path = _build_view(tmp_path)
    cfg = _compose_cfg(dimers_path, locator_path, tmp_path)

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    n_train = len(dm.train_dataset)
    n_val = len(dm.val_dataset)
    assert n_train + n_val == 2


def test_yaml_filters_pin_relaxed_thresholds():
    """Pin the exact filter strings in the YAML so a future edit can't silently
    drift the cuts. The three thresholds together define the supervised pool;
    changing any of them changes which dimers the confidence head trains on.
    """
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR / "dataset" / "unified"), version_base="1.3"
    ):
        cfg = compose(config_name="teddymer_with_plddt_and_pae")

    filters = list(cfg.datamodule.filters)
    assert filters == [
        "interface_length > 10",
        "avg_int_plddt > 30.0",
        "avg_int_pae < 25.0",
    ], (
        f"Teddymer confidence-distill filter contract drifted. Got {filters}. "
        "The relaxed thresholds (>30 pLDDT, <25 PAE) replace the Complexa-paper "
        "defaults (>70 / <10) so the head sees low-confidence interfaces too."
    )


def test_one_train_batch_has_expected_fields_and_dtypes(tmp_path):
    dimers_path, locator_path = _build_view(tmp_path)
    cfg = _compose_cfg(dimers_path, locator_path, tmp_path)

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    required = {
        "plddt_residue", "plddt_bin", "plddt_mask",
        "pae_residue_pair", "pae_bin", "pae_mask",
        "chain_idx",
    }
    have = set(batch.keys()) if hasattr(batch, "keys") else set(vars(batch).keys())
    missing = required - have
    assert not missing, f"batch missing fields: {missing}"

    plddt_residue = batch["plddt_residue"]
    pae_residue_pair = batch["pae_residue_pair"]
    pae_bin = batch["pae_bin"]
    pae_mask = batch["pae_mask"]
    chain_idx = batch["chain_idx"]

    assert plddt_residue.dtype == torch.float32
    assert pae_residue_pair.dtype == torch.float32
    assert pae_bin.dtype == torch.int64
    assert pae_mask.dtype == torch.bool
    assert chain_idx.dtype == torch.int8

    assert plddt_residue.dim() == 2
    assert pae_residue_pair.dim() == 3


def test_pae_padding_mask_off_outside_real_extent(tmp_path):
    dimers_path, locator_path = _build_view(tmp_path)
    cfg = _compose_cfg(dimers_path, locator_path, tmp_path)

    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    pae_residue_pair = batch["pae_residue_pair"]
    pae_mask = batch["pae_mask"]
    B, L1, L2 = pae_residue_pair.shape
    assert L1 == L2

    plddt_mask = batch["plddt_mask"]
    for b in range(B):
        real_len = int(plddt_mask[b].sum().item())
        if real_len < L1:
            padded_block = pae_mask[b, real_len:, :]
            assert not bool(padded_block.any().item())
            padded_block_t = pae_mask[b, :, real_len:]
            assert not bool(padded_block_t.any().item())
