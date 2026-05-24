"""Tests for `interface_pair_mask`.

The helper returns a `[B, L, L]` boolean tensor that is True where
residues `i` and `j` belong to different chains AND both rows/cols are
valid under the pair mask. It is the row/col filter that the interface
metrics (i_pae, min_ipae, ipTM, ipSAE) consume.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence._metrics import interface_pair_mask


def test_dimer_full_mask_has_eight_off_block_pairs() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    out = interface_pair_mask(chain_idx, mask_eff)
    assert out.shape == (1, 4, 4)
    assert out.dtype == torch.bool
    assert out.sum().item() == 8


def test_dimer_off_block_diagonal_positions_correct() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    out = interface_pair_mask(chain_idx, mask_eff)[0]
    expected = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    assert torch.equal(out, expected)


def test_monomer_all_false() -> None:
    chain_idx = torch.tensor([[0, 0, 0, 0]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    out = interface_pair_mask(chain_idx, mask_eff)
    assert out.sum().item() == 0


def test_masked_residue_zeros_row_and_col() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]])
    mask_eff = torch.ones(1, 4, 4, dtype=torch.float32)
    mask_eff[0, 2, :] = 0.0
    mask_eff[0, :, 2] = 0.0
    out = interface_pair_mask(chain_idx, mask_eff)[0]
    assert not out[2].any().item()
    assert not out[:, 2].any().item()
    assert out.sum().item() == 4


def test_batch_with_padding_independent() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 0]])
    mask_eff = torch.ones(2, 4, 4, dtype=torch.float32)
    mask_eff[1, 2:, :] = 0.0
    mask_eff[1, :, 2:] = 0.0
    out = interface_pair_mask(chain_idx, mask_eff)
    assert out[0].sum().item() == 8
    assert out[1].sum().item() == 2
