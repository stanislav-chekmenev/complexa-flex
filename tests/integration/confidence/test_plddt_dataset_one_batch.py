"""End-to-end integration test for the AFDB-with-pLDDT dataset wiring.

Verifies the Hydra config composes, the `atom37_transforms` list resolves to a
real `AddPLDDTFromBFactor`, and that running it over the committed synthetic
fixture yields a `Data` object with the expected pLDDT fields, dtypes, and
ranges. Uses a test-only `FixtureStructureDataset` to avoid touching real AFDB
CIF files on disk.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir

from proteinfoundation.datasets.transforms import AddPLDDTFromBFactor

from tests.integration.confidence._fixture_loader import FixtureStructureDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "afdb_one_protein" / "synthetic_atom37.pt"


def _compose_dataset_cfg():
    import os

    os.environ.setdefault("DATA_PATH", "/tmp")
    with initialize_config_dir(config_dir=str(CONFIG_DIR / "dataset" / "unified"), version_base="1.3"):
        return compose(
            config_name="afdb_monomers_with_plddt",
            return_hydra_config=False,
        )


def test_afdb_with_plddt_yaml_composes_and_drops_plddt_filter() -> None:
    cfg = _compose_dataset_cfg()

    filters = list(cfg.datamodule.get("filters", []) or [])
    assert all("plddt" not in str(f) for f in filters), (
        f"Expected `plddt > 70` filter dropped, but got filters: {filters}"
    )
    assert any("length" in str(f) for f in filters), (
        f"Expected `length < N` filter retained, got: {filters}"
    )


def test_atom37_transforms_includes_add_plddt() -> None:
    cfg = _compose_dataset_cfg()

    targets = [t.get("_target_") for t in cfg.datamodule.atom37_transforms]
    assert "proteinfoundation.datasets.transforms.AddPLDDTFromBFactor" in targets


def test_end_to_end_one_batch_has_plddt_fields() -> None:
    cfg = _compose_dataset_cfg()

    transforms = []
    for t in cfg.datamodule.atom37_transforms:
        instance = hydra.utils.instantiate(t)
        transforms.append(instance)

    AddPLDDTFromBFactor._warned = False
    dataset = FixtureStructureDataset(
        fixture_path=FIXTURE_PATH,
        atom37_transforms=transforms,
        length=2,
    )

    data = dataset[0]

    assert hasattr(data, "plddt_residue"), "plddt_residue missing from Data"
    assert hasattr(data, "plddt_bin"), "plddt_bin missing from Data"
    assert hasattr(data, "plddt_mask"), "plddt_mask missing from Data"

    n = data.coords.shape[0]
    assert data.plddt_residue.shape == (n,), data.plddt_residue.shape
    assert data.plddt_bin.shape == (n,), data.plddt_bin.shape
    assert data.plddt_mask.shape == (n,), data.plddt_mask.shape

    assert data.plddt_residue.dtype == torch.float32
    assert data.plddt_bin.dtype == torch.int64
    assert data.plddt_mask.dtype == torch.bool

    assert torch.isfinite(data.plddt_residue).all()
    assert (data.plddt_residue >= 0).all()
    assert (data.plddt_residue <= 100).all()
    assert (data.plddt_bin >= 0).all()
    assert (data.plddt_bin <= 49).all()


def test_atom_b_factor_threaded_through_fixture() -> None:
    cfg = _compose_dataset_cfg()
    AddPLDDTFromBFactor._warned = False
    dataset = FixtureStructureDataset(
        fixture_path=FIXTURE_PATH,
        atom37_transforms=[hydra.utils.instantiate(cfg.datamodule.atom37_transforms[0])],
        length=1,
    )
    data = dataset[0]
    assert hasattr(data, "atom_b_factor"), "Fixture must include atom_b_factor for pLDDT extraction"
    assert data.atom_b_factor.shape == (data.coords.shape[0], 37)
    assert data.atom_b_factor.dtype == torch.float32
