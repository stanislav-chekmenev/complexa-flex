"""Pin the 4-layer Boltz-1 pairformer stack wrapper.

Just a thin nn.Module that owns 4 PairformerLayer's and the mask
re-application after each layer. No semantic novelty; the test is mostly
shape preservation and the mask post-condition.

The Boltz-1 PairformerLayer.forward requires a `pair_mask`; the wrapper
derives it internally from `mask`. The wrapper exposes `s_dim` / `z_dim`
kwargs even though the vendored layer uses `token_s` / `token_z`.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack


def test_stack_default_4_layers():
    stack = QgPairformerStack()
    assert len(stack.layers) == 4


def test_stack_forward_shapes_and_mask():
    stack = QgPairformerStack(n_layers=4, s_dim=384, z_dim=128, num_heads=16).eval()
    B, n = 2, 16
    s = torch.randn(B, n, 384)
    z = torch.randn(B, n, n, 128)
    mask = torch.ones(B, n)
    mask[1, -4:] = 0.0

    s = s * mask[..., None]
    z = z * mask[:, :, None, None] * mask[:, None, :, None]

    with torch.no_grad():
        s_out, z_out = stack(s, z, mask)
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape

    assert torch.allclose(s_out[1, -4:], torch.zeros_like(s_out[1, -4:]), atol=1e-6)
    assert torch.allclose(z_out[1, -4:, :], torch.zeros_like(z_out[1, -4:, :]), atol=1e-6)
    assert torch.allclose(z_out[1, :, -4:], torch.zeros_like(z_out[1, :, -4:]), atol=1e-6)
