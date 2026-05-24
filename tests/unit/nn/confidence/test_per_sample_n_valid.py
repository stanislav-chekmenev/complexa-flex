"""Tests for `_per_sample_n_valid` symmetry contract.

The helper assumes the standard residue-pair mask construction
`mask_eff = m[:, None] & m[:, :, None]`, which is symmetric in
(i, j). Callers that feed an asymmetric mask have a bug upstream; the
helper now asserts to surface it.
"""

from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn.confidence._metrics import _per_sample_n_valid


def test_symmetric_mask_passes_and_returns_n_residues() -> None:
    m = torch.tensor([[True, True, True, False], [True, True, False, False]])
    mask_eff = m[:, :, None] & m[:, None, :]
    n = _per_sample_n_valid(mask_eff)
    assert n.tolist() == [3, 2]


def test_asymmetric_mask_triggers_assert() -> None:
    mask_eff = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask_eff[0, 0, 1] = True
    assert not torch.equal(mask_eff, mask_eff.transpose(-1, -2))
    with pytest.raises(AssertionError, match="symmetric"):
        _per_sample_n_valid(mask_eff)
