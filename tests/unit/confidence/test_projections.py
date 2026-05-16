"""Contract tests for `SeqProjection` and `PairProjection`.

Covers the identity branch (`in_dim == out_dim`, parameter-free) and the
dim-changing branch (`LayerNorm + Linear`), and that both branches apply
the mask consistently.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.projections import PairProjection, SeqProjection


B, N = 2, 5


def _mask(b: int = B, n: int = N) -> torch.Tensor:
    m = torch.ones(b, n, dtype=torch.bool)
    m[0, 3:] = False
    return m


def test_seq_projection_identity_branch() -> None:
    proj = SeqProjection(in_dim=16, out_dim=16)
    assert proj.identity
    assert sum(p.numel() for p in proj.parameters()) == 0

    s = torch.randn(B, N, 16)
    mask = _mask()
    out = proj(s, mask)

    assert out.shape == s.shape
    assert torch.all(out[0, 3:] == 0.0)
    assert torch.allclose(out[mask], s[mask])


def test_seq_projection_dim_changing_branch() -> None:
    proj = SeqProjection(in_dim=16, out_dim=8)
    assert not proj.identity
    assert sum(p.numel() for p in proj.parameters()) > 0

    s = torch.randn(B, N, 16)
    mask = _mask()
    out = proj(s, mask)

    assert out.shape == (B, N, 8)
    assert torch.all(out[0, 3:] == 0.0)
    assert torch.isfinite(out[mask]).all()


def test_pair_projection_identity_branch() -> None:
    proj = PairProjection(in_dim=8, out_dim=8)
    assert proj.identity
    assert sum(p.numel() for p in proj.parameters()) == 0

    z = torch.randn(B, N, N, 8)
    mask = _mask()
    out = proj(z, mask)

    assert out.shape == z.shape
    pair_mask = (mask[:, None, :] & mask[:, :, None])
    assert torch.all(out[~pair_mask] == 0.0)
    assert torch.allclose(out[pair_mask], z[pair_mask])


def test_pair_projection_dim_changing_branch() -> None:
    proj = PairProjection(in_dim=8, out_dim=4)
    assert not proj.identity
    assert sum(p.numel() for p in proj.parameters()) > 0

    z = torch.randn(B, N, N, 8)
    mask = _mask()
    out = proj(z, mask)

    assert out.shape == (B, N, N, 4)
    pair_mask = (mask[:, None, :] & mask[:, :, None])
    assert torch.all(out[~pair_mask] == 0.0)
    assert torch.isfinite(out[pair_mask]).all()
