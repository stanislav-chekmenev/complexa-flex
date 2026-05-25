"""Smoke test for the vendored Boltz-1 PairformerLayer.

Pins that the layer instantiates standalone, that forward shapes and
dtypes are preserved, and that the residue / pair mask is honoured at
padded positions.
"""

from __future__ import annotations

import pytest
import torch

from community_models.boltz.model.modules.pairformer import PairformerLayer


def _make_inputs(
    B: int = 2,
    L: int = 16,
    s_dim: int = 384,
    z_dim: int = 128,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.randn(B, L, s_dim, dtype=dtype)
    z = torch.randn(B, L, L, z_dim, dtype=dtype)
    mask = torch.ones(B, L, dtype=dtype)
    mask[1, -4:] = 0.0
    pair_mask = mask[:, :, None] * mask[:, None, :]
    return s, z, mask, pair_mask


def _build_layer(s_dim: int = 384, z_dim: int = 128) -> PairformerLayer:
    layer = PairformerLayer(
        token_s=s_dim,
        token_z=z_dim,
        num_heads=16,
        dropout=0.0,
    )
    layer.eval()
    return layer


def test_pairformer_layer_forward_shapes_and_dtype() -> None:
    torch.manual_seed(0)
    layer = _build_layer()
    s, z, mask, pair_mask = _make_inputs()
    with torch.no_grad():
        s_out, z_out = layer(s=s, z=z, mask=mask, pair_mask=pair_mask)

    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
    assert s_out.dtype == s.dtype
    assert z_out.dtype == z.dtype


def test_pairformer_layer_respects_mask_at_padded_positions() -> None:
    """Zero-input at padded positions plus pair_mask=0 at any cell touching
    a padded residue must leave the output near-zero there, while valid
    positions retain O(1) magnitude. Triangle-attention softmax over
    large-negative bias still leaves epsilon contributions in fp32, so the
    threshold is set to give a clear order-of-magnitude separation rather
    than bit-identity.
    """
    torch.manual_seed(0)
    layer = _build_layer()
    s, z, mask, pair_mask = _make_inputs()

    s = s * mask[..., None]
    z = z * pair_mask[..., None]

    with torch.no_grad():
        s_out, z_out = layer(s=s, z=z, mask=mask, pair_mask=pair_mask)

    pad_rows = mask[1] == 0
    valid_rows = mask[1] == 1

    pad_s = s_out[1, pad_rows]
    pad_z_rows = z_out[1, pad_rows, :]
    pad_z_cols = z_out[1, :, pad_rows]

    valid_s = s_out[1, valid_rows]
    valid_z = z_out[1][valid_rows][:, valid_rows]

    pad_atol = 1e-4
    assert pad_s.abs().max() < pad_atol, f"padded s leak: {pad_s.abs().max().item()}"
    assert pad_z_rows.abs().max() < pad_atol, (
        f"padded z rows leak: {pad_z_rows.abs().max().item()}"
    )
    assert pad_z_cols.abs().max() < pad_atol, (
        f"padded z cols leak: {pad_z_cols.abs().max().item()}"
    )

    assert valid_s.abs().max() > 1e-2, (
        f"valid s collapsed: {valid_s.abs().max().item()}"
    )
    assert valid_z.abs().max() > 1e-2, (
        f"valid z collapsed: {valid_z.abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pairformer_layer_forward_bf16_mixed_cuda() -> None:
    torch.manual_seed(0)
    layer = _build_layer().cuda()
    s, z, mask, pair_mask = _make_inputs()
    s, z, mask, pair_mask = s.cuda(), z.cuda(), mask.cuda(), pair_mask.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            s_out, z_out = layer(s=s, z=z, mask=mask, pair_mask=pair_mask)

    assert torch.isfinite(s_out).all()
    assert torch.isfinite(z_out).all()
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
