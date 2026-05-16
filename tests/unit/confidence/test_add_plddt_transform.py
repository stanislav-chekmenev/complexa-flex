"""Tests for the `AddPLDDTFromBFactor` atom37 transform.

Covers happy-path aggregation, bin assignment, validity mask, normalised-scale
warning, and warn-once semantics.
"""

from __future__ import annotations

import logging

import pytest
import torch
from loguru import logger

from proteinfoundation.datasets.transforms import AddPLDDTFromBFactor, Data


def _make_data(b_factors: list[list[float]], coord_mask: torch.Tensor) -> Data:
    atom_b_factor = torch.tensor(b_factors, dtype=torch.float32)
    n = atom_b_factor.shape[0]
    return Data(
        coords=torch.zeros(n, 37, 3, dtype=torch.float32),
        coord_mask=coord_mask,
        atom_b_factor=atom_b_factor,
        residue_type=torch.zeros(n, dtype=torch.long),
    )


def test_add_plddt_transform_writes_expected_fields() -> None:
    n = 5
    b_factors = [
        [80.0] * 37,
        [50.0] * 37,
        [0.0] * 37,
        [99.0] * 37,
        [70.0] * 37,
    ]
    coord_mask = torch.ones(n, 37, dtype=torch.bool)
    coord_mask[2, :] = False
    data = _make_data(b_factors, coord_mask)

    out = AddPLDDTFromBFactor()(data)

    assert out is data
    assert torch.allclose(
        out.plddt_residue,
        torch.tensor([80.0, 50.0, 0.0, 99.0, 70.0], dtype=torch.float32),
    )
    assert out.plddt_residue.dtype == torch.float32

    assert out.plddt_bin.dtype == torch.int64
    assert torch.equal(
        out.plddt_bin,
        torch.tensor([40, 25, 0, 49, 35], dtype=torch.int64),
    )

    assert out.plddt_mask.dtype == torch.bool
    assert torch.equal(
        out.plddt_mask,
        torch.tensor([True, True, False, True, True], dtype=torch.bool),
    )


def test_add_plddt_transform_clamps_above_scale_max() -> None:
    b_factors = [[120.0] * 37]
    coord_mask = torch.ones(1, 37, dtype=torch.bool)
    data = _make_data(b_factors, coord_mask)

    out = AddPLDDTFromBFactor()(data)

    assert out.plddt_residue.item() == pytest.approx(100.0)
    assert out.plddt_bin.item() == 49
    assert out.plddt_mask.item() is True


def test_add_plddt_transform_means_over_valid_atoms_only() -> None:
    row = [10.0, 90.0] + [0.0] * 35
    coord_mask = torch.zeros(1, 37, dtype=torch.bool)
    coord_mask[0, 0] = True
    coord_mask[0, 1] = True
    data = _make_data([row], coord_mask)

    out = AddPLDDTFromBFactor()(data)

    assert out.plddt_residue.item() == pytest.approx(50.0)


def test_add_plddt_transform_warns_when_b_factor_is_normalised(caplog) -> None:
    AddPLDDTFromBFactor._warned = False
    b_factors = [[0.8] * 37, [0.5] * 37]
    coord_mask = torch.ones(2, 37, dtype=torch.bool)
    data = _make_data(b_factors, coord_mask)

    handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
    try:
        with caplog.at_level(logging.WARNING):
            AddPLDDTFromBFactor()(data)
    finally:
        logger.remove(handler_id)

    assert any("normalised" in m.lower() or "b-factor" in m.lower() for m in caplog.messages), (
        f"Expected normalised-scale warning, got messages: {caplog.messages}"
    )


def test_add_plddt_transform_warns_once_across_calls(caplog) -> None:
    AddPLDDTFromBFactor._warned = False
    b_factors = [[0.8] * 37]
    coord_mask = torch.ones(1, 37, dtype=torch.bool)

    transform = AddPLDDTFromBFactor()

    handler_id = logger.add(caplog.handler, format="{message}", level="WARNING")
    try:
        with caplog.at_level(logging.WARNING):
            transform(_make_data(b_factors, coord_mask))
            transform(_make_data(b_factors, coord_mask))
    finally:
        logger.remove(handler_id)

    matching = [m for m in caplog.messages if "b-factor" in m.lower() or "normalised" in m.lower()]
    assert len(matching) == 1, f"Expected exactly one warning across calls, got: {caplog.messages}"


def test_add_plddt_transform_does_not_raise_on_normalised_input() -> None:
    AddPLDDTFromBFactor._warned = False
    b_factors = [[0.8] * 37]
    coord_mask = torch.ones(1, 37, dtype=torch.bool)
    data = _make_data(b_factors, coord_mask)

    out = AddPLDDTFromBFactor()(data)

    assert torch.isfinite(out.plddt_residue).all()
    assert out.plddt_bin.dtype == torch.int64
