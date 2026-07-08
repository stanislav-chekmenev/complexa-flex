"""Unit: scorer `provisional_success` ipAE is the MEAN interface i_pae.

Owner-decided (locked): the interface-PAE reduction feeding
`provisional_success` and `total_reward = -ipae` must be the MEAN over the
cross-chain interface -- the colabdesign/AlphaProteo `i_pAE` definition that
the analyze stage gates on -- NOT the per-row `min_ipae`.

Two parity anchors:
1. fp64 numpy reference computed in-test (community-metric-parity rule).
2. Bit-for-bit equality to `_metrics.i_pae(..., reduce="per_sample")`, the
   exact ported function `PaeHead` logs and `colabdesign_utils.py` maps to
   the `i_pAE` CSV column (verbatim colabdesign port; its fp64 reference
   test is tests/unit/nn/confidence/test_i_pae.py). This proves the scorer
   calls the same reduction analyze relies on, not a re-derivation.
"""

from __future__ import annotations

import numpy as np
import torch

from proteinfoundation.confidence.inference_scorer import ConfidenceHeadScorer
from proteinfoundation.nn.confidence._metrics import i_pae, interface_pair_mask


def _fp64_mean_interface(
    pae_ev: np.ndarray, chain_idx: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    b, l, _ = pae_ev.shape
    out = np.full((b,), np.nan, dtype=np.float64)
    for s in range(b):
        m = mask[s].astype(bool)
        cross = chain_idx[s][:, None] != chain_idx[s][None, :]
        inter = (m[:, None] & m[None, :]) & cross
        denom = inter.sum()
        if denom:
            out[s] = (pae_ev[s] * inter).sum() / denom
    return out


def _hand_built_pae() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Asymmetric, non-uniform interface so mean != min: chain 0 = {0,1},
    # chain 1 = {2,3}. Off-diagonal cross-chain block carries distinct values.
    chain_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.bool)
    pae_ev = torch.tensor(
        [
            [
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 6.0, 8.0],
                [3.0, 5.0, 0.0, 0.0],
                [7.0, 9.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    return pae_ev, chain_idx, mask


def test_scorer_ipae_equals_fp64_mean_reference() -> None:
    pae_ev, chain_idx, mask = _hand_built_pae()
    plddt_ev = torch.full((1, 4), 95.0)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    ref = _fp64_mean_interface(
        pae_ev.double().numpy(), chain_idx.numpy(), mask.numpy()
    )
    # mean over 8 cross-chain pairs {2,4,6,8,3,5,7,9} = 44/8 = 5.5
    assert abs(ref[0] - 5.5) < 1e-12
    np.testing.assert_allclose(out["ipae"].double().numpy(), ref, rtol=0, atol=1e-5)


def test_scorer_ipae_is_bit_identical_to_ported_i_pae() -> None:
    pae_ev, chain_idx, mask = _hand_built_pae()
    plddt_ev = torch.full((1, 4), 95.0)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    pair_valid = mask[:, :, None] & mask[:, None, :]
    inter = interface_pair_mask(chain_idx, pair_valid)
    expected = i_pae(pae_ev.float(), inter, reduce="per_sample")

    assert torch.equal(out["ipae"], expected.float())
