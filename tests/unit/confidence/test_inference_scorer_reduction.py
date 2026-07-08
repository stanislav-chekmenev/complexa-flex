"""Unit: `ConfidenceHeadScorer._reduce` interface / pLDDT / success math.

The reduction is the load-bearing scorer contract wired into best-of-N
CSV columns and plots, so it is unit-tested in isolation from any
checkpoint or trunk:

- `ipae` == `i_pae(pae_ev, interface_mask, reduce="per_sample")` in
  Angstroms -- the MEAN over the cross-chain interface (owner-decided,
  matching the colabdesign/AlphaProteo `i_pAE` definition the analyze
  stage gates on), locked against an fp64 numpy reference per the
  community-metric-parity rule. (Previously `min_ipae`, the MIN over
  interface rows; the reduction changed min -> mean, so this file's
  reference and expected boundary values are the mean.)
- `complex_plddt` == masked-mean(plddt_ev) / 100 on the 0-1 scale,
- `provisional_success` == (ipae < 7) & (complex_plddt > 0.9),
- a single-chain sample yields an empty interface -> ipae NaN ->
  success False,
- pLDDT and ipAE thresholds bite at the exact boundaries with NO x31
  applied to the Angstrom ipAE.
"""

from __future__ import annotations

import numpy as np
import torch

from proteinfoundation.confidence.inference_scorer import ConfidenceHeadScorer


def _np_i_pae_per_sample(
    pae_ev: np.ndarray, chain_idx: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """fp64 reference for `i_pae(..., reduce="per_sample")` (MEAN interface).

    Cross-chain valid pairs; mean EV over all interface pairs; NaN for
    samples without any interface pair. Mirrors `_metrics.i_pae`, the
    verbatim colabdesign `i_pAE` port that the analyze stage and PaeHead
    consume (see tests/unit/nn/confidence/test_i_pae.py).
    """
    b, l, _ = pae_ev.shape
    out = np.full((b,), np.nan, dtype=np.float64)
    for s in range(b):
        m = mask[s].astype(bool)
        pair_valid = m[:, None] & m[None, :]
        cross = chain_idx[s][:, None] != chain_idx[s][None, :]
        inter = pair_valid & cross
        denom = inter.sum()
        if denom == 0:
            continue
        out[s] = (pae_ev[s] * inter).sum() / denom
    return out


def _make_pae_ev(chain_idx: torch.Tensor, seed: int = 0) -> torch.Tensor:
    b, l = chain_idx.shape
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, l, l, generator=g) * 30.0


def test_ipae_matches_i_pae_mean_fp64_reference() -> None:
    """Scorer ipae == MEAN interface i_pae (owner-decided), fp64-parity.

    With a non-uniform PAE map the mean and the old min diverge, so this
    pins the min -> mean change: the fp64 reference is the mean over all
    cross-chain pairs, matching `_metrics.i_pae`.
    """
    chain_idx = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    pae_ev = _make_pae_ev(chain_idx)
    plddt_ev = torch.full(chain_idx.shape, 50.0)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    ref = _np_i_pae_per_sample(
        pae_ev.double().numpy(), chain_idx.numpy(), mask.numpy()
    )
    np.testing.assert_allclose(out["ipae"].double().numpy(), ref, rtol=0, atol=1e-5)


def test_complex_plddt_is_masked_mean_over_100() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])
    plddt_ev = torch.tensor([[90.0, 80.0, 70.0, 999.0]])  # padded col excluded
    pae_ev = _make_pae_ev(chain_idx)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    expected = (90.0 + 80.0 + 70.0) / 3.0 / 100.0
    assert abs(float(out["complex_plddt"][0]) - expected) < 1e-6


def test_single_chain_yields_nan_ipae_and_false_success() -> None:
    chain_idx = torch.zeros(1, 5, dtype=torch.long)  # one chain -> no interface
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    plddt_ev = torch.full(chain_idx.shape, 95.0)  # high plddt so only ipae gates
    pae_ev = _make_pae_ev(chain_idx)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert torch.isnan(out["ipae"]).all()
    assert not bool(out["provisional_success"][0])


def test_two_chain_yields_finite_ipae() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    plddt_ev = torch.full(chain_idx.shape, 95.0)
    pae_ev = _make_pae_ev(chain_idx)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert torch.isfinite(out["ipae"]).all()


def test_plddt_threshold_bites_at_0p9() -> None:
    chain_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    # low, uniform ipAE so ipae passes and pLDDT is the deciding gate.
    pae_ev = torch.full((2, 2, 2), 1.0)
    plddt_ev = torch.tensor([[95.0, 95.0], [85.0, 85.0]])  # 0.95 vs 0.85

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert bool(out["provisional_success"][0]) is True
    assert bool(out["provisional_success"][1]) is False


def test_ipae_threshold_bites_at_7_angstroms_no_x31() -> None:
    chain_idx = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    plddt_ev = torch.full((2, 2), 95.0)  # 0.95 so pLDDT always passes
    # Uniform PAE so the mean interface ipae equals the fill value directly:
    # 6.9 A (< 7) passes; 7.1 A (>= 7) fails. If a x31 scale leaked, 6.9 would
    # blow past 7 and this sample would flip to failure.
    pae_ev = torch.empty(2, 2, 2)
    pae_ev[0] = 6.9
    pae_ev[1] = 7.1

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert abs(float(out["ipae"][0]) - 6.9) < 1e-5
    assert bool(out["provisional_success"][0]) is True
    assert bool(out["provisional_success"][1]) is False


def test_reduce_output_schema() -> None:
    chain_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = torch.ones_like(chain_idx, dtype=torch.bool)
    pae_ev = _make_pae_ev(chain_idx)
    plddt_ev = torch.full(chain_idx.shape, 50.0)

    out = ConfidenceHeadScorer._reduce(pae_ev, plddt_ev, chain_idx, mask)

    assert set(out.keys()) == {"ipae", "complex_plddt", "provisional_success"}
    assert out["ipae"].shape == (1,)
    assert out["complex_plddt"].shape == (1,)
    assert out["provisional_success"].shape == (1,)
    assert out["ipae"].dtype == torch.float32
    assert out["complex_plddt"].dtype == torch.float32
    assert out["provisional_success"].dtype == torch.bool
