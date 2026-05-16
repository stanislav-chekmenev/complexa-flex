"""Tests for `plddt_to_bin` on the AF2 pLDDT [0, 100] scale with 50 bins of width 2."""

import pytest
import torch

from proteinfoundation.confidence.losses import plddt_to_bin


@pytest.mark.parametrize(
    ("plddt", "expected"),
    [
        (0.0, 0),
        (2.0 - 1e-6, 0),
        (2.0, 1),
        (50.0, 25),
        (99.999, 49),
        (100.0, 49),
        (1000.0, 49),
        (-5.0, 0),
    ],
)
def test_plddt_to_bin_scalar(plddt: float, expected: int) -> None:
    out = plddt_to_bin(torch.tensor(plddt, dtype=torch.float32))
    assert out.dtype == torch.int64
    assert out.item() == expected


def test_plddt_to_bin_vector() -> None:
    inp = torch.tensor([0.0, 50.0, 100.0], dtype=torch.float32)
    out = plddt_to_bin(inp)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([0, 25, 49], dtype=torch.int64))


def test_plddt_to_bin_accepts_python_float() -> None:
    assert plddt_to_bin(50.0).item() == 25
    assert plddt_to_bin(0).item() == 0


def test_plddt_to_bin_preserves_shape() -> None:
    inp = torch.linspace(0.0, 100.0, steps=11).reshape(11, 1)
    out = plddt_to_bin(inp)
    assert out.shape == inp.shape
    assert out.dtype == torch.int64


def test_plddt_to_bin_custom_bins() -> None:
    out = plddt_to_bin(torch.tensor([0.0, 0.5, 1.0]), bin_width=0.5, num_bins=2)
    assert torch.equal(out, torch.tensor([0, 1, 1], dtype=torch.int64))
