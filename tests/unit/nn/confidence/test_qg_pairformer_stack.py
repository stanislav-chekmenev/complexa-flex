"""Pin the Boltz-1 pairformer stack wrapper.

A thin nn.Module that owns N PairformerLayer's and the mask
re-application after each layer. The Boltz-1 PairformerLayer.forward
requires a `pair_mask`; the wrapper derives it internally from `mask`.
The wrapper exposes `s_dim` / `z_dim` kwargs even though the vendored
layer uses `token_s` / `token_z`.

Tests pin:
  - shape preservation and mask post-condition (default args)
  - chunked tri-attn (`chunk_size_tri_attn`) is numerically equivalent
    to the un-chunked path
  - reentrant gradient checkpointing (`gradient_checkpointing=True`) is
    numerically equivalent in both forward and backward to the
    non-checkpointed path
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack


def _make_inputs(B: int = 2, n: int = 12, s_dim: int = 32, z_dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, n, s_dim, generator=g)
    z = torch.randn(B, n, n, z_dim, generator=g)
    mask = torch.ones(B, n)
    mask[1, -3:] = 0.0
    s = s * mask[..., None]
    z = z * mask[:, :, None, None] * mask[:, None, :, None]
    return s, z, mask


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


def _build_stack(*, chunk_size_tri_attn=None, gradient_checkpointing=False, seed: int = 7):
    torch.manual_seed(seed)
    return QgPairformerStack(
        n_layers=2,
        s_dim=32,
        z_dim=16,
        num_heads=4,
        chunk_size_tri_attn=chunk_size_tri_attn,
        gradient_checkpointing=gradient_checkpointing,
    )


def test_chunk_size_tri_attn_matches_unchunked_forward():
    s, z, mask = _make_inputs()
    stack_full = _build_stack(chunk_size_tri_attn=None).eval()
    stack_chunked = _build_stack(chunk_size_tri_attn=4).eval()

    with torch.no_grad():
        s_full, z_full = stack_full(s.clone(), z.clone(), mask)
        s_chunk, z_chunk = stack_chunked(s.clone(), z.clone(), mask)

    assert torch.allclose(s_full, s_chunk, atol=1e-5, rtol=1e-5)
    assert torch.allclose(z_full, z_chunk, atol=1e-5, rtol=1e-5)


def _train_step(stack, s, z, mask):
    stack.train()
    s = s.clone().requires_grad_(True)
    z = z.clone().requires_grad_(True)
    s_out, z_out = stack(s, z, mask)
    loss = s_out.float().pow(2).sum() + z_out.float().pow(2).sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in stack.named_parameters() if p.grad is not None}
    return s_out.detach(), z_out.detach(), loss.detach(), grads


def test_gradient_checkpointing_matches_no_checkpoint_forward_and_backward():
    s, z, mask = _make_inputs()

    stack_no_ckpt = _build_stack(gradient_checkpointing=False)
    stack_ckpt = _build_stack(gradient_checkpointing=True)
    stack_ckpt.load_state_dict(stack_no_ckpt.state_dict())

    torch.manual_seed(0)
    s_a, z_a, loss_a, grads_a = _train_step(stack_no_ckpt, s, z, mask)
    torch.manual_seed(0)
    s_b, z_b, loss_b, grads_b = _train_step(stack_ckpt, s, z, mask)

    assert torch.allclose(s_a, s_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(z_a, z_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(loss_a, loss_b, atol=1e-4, rtol=1e-5)

    assert set(grads_a.keys()) == set(grads_b.keys()) and len(grads_a) > 0
    for name, g_a in grads_a.items():
        g_b = grads_b[name]
        assert torch.allclose(g_a, g_b, atol=1e-4, rtol=1e-4), name
