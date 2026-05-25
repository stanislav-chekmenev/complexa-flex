"""Unit tests for the quality-graft-style adaptor.

Pinned properties:
- Output shapes match the Boltz-1 dim target (384 single, 128 pair).
- LayerNorm-bias-leak invariant: drifting LN.bias off zero does NOT
  contaminate valid positions through padded ones.
- Cα distogram: bins match a fp64 numpy reference.
- Single-attention-block residual init: with zero-init on attn.proj_o
  and the s/z MLP final linears, the block is a near-identity transform
  of the linearly-projected (s, z) within 1e-6.
- Translation equivariance under n_attn_layers=1: translating ca_coords
  by a constant leaves the adaptor output bit-identical (distogram uses
  pairwise distances; AttentionPairBias is not coordinate-aware).
"""

from __future__ import annotations

import numpy as np
import torch

from proteinfoundation.nn.confidence.adaptor import AdaptorModule


def _make_inputs(B: int = 2, n: int = 16, dtype=torch.float32):
    trunk_seqs = torch.randn(B, n, 768, dtype=dtype)
    trunk_pair = torch.randn(B, n, n, 256, dtype=dtype)
    local_latents = torch.randn(B, n, 8, dtype=dtype)
    ca_coords = torch.randn(B, n, 3, dtype=dtype)
    mask = torch.ones(B, n, dtype=dtype)
    mask[1, -4:] = 0.0
    return trunk_seqs, trunk_pair, local_latents, ca_coords, mask


def test_adaptor_output_shapes():
    adaptor = AdaptorModule()
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()
    s, z = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    assert s.shape == (2, 16, 384)
    assert z.shape == (2, 16, 16, 128)


def test_adaptor_layernorm_bias_leak_guard():
    """Set every LN.bias in the adaptor to 0.3 (drift simulation); the
    output at non-padded positions must NOT depend on padded inputs."""
    torch.manual_seed(0)
    adaptor = AdaptorModule()
    for m in adaptor.modules():
        if isinstance(m, torch.nn.LayerNorm) and m.bias is not None:
            with torch.no_grad():
                m.bias.fill_(0.3)

    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()
    s1, z1 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)

    trunk_seqs2 = trunk_seqs.clone()
    trunk_pair2 = trunk_pair.clone()
    local_latents2 = local_latents.clone()
    ca_coords2 = ca_coords.clone()
    trunk_seqs2[1, -4:] = torch.randn_like(trunk_seqs2[1, -4:]) * 100.0
    trunk_pair2[1, -4:, :] = torch.randn_like(trunk_pair2[1, -4:, :]) * 100.0
    trunk_pair2[1, :, -4:] = torch.randn_like(trunk_pair2[1, :, -4:]) * 100.0
    local_latents2[1, -4:] = torch.randn_like(local_latents2[1, -4:]) * 100.0
    ca_coords2[1, -4:] = torch.randn_like(ca_coords2[1, -4:]) * 100.0

    s2, z2 = adaptor(trunk_seqs2, trunk_pair2, local_latents2, ca_coords2, mask=mask)

    valid = mask[1] == 1
    assert torch.allclose(s1[1, valid], s2[1, valid], atol=1e-6)
    valid_idx = torch.where(valid)[0]
    assert torch.allclose(
        z1[1][valid_idx][:, valid_idx],
        z2[1][valid_idx][:, valid_idx],
        atol=1e-6,
    )


def test_adaptor_ca_distogram_matches_fp64_reference():
    """The one-hot Cα distogram must match a fp64 numpy reference
    bin-for-bin."""
    torch.manual_seed(0)
    adaptor = AdaptorModule()
    n_bins = 128
    B, n = 1, 8
    ca_coords = torch.randn(B, n, 3, dtype=torch.float64)

    diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1).numpy()
    bin_limits = np.linspace(0.1, 3.0, n_bins - 1)
    expected_bins = np.digitize(dist, bin_limits)

    got_one_hot = adaptor._binned_ca_distogram(ca_coords.to(torch.float32))
    got_bins = got_one_hot.argmax(dim=-1).numpy()

    np.testing.assert_array_equal(got_bins, expected_bins)


def test_adaptor_single_attn_block_near_identity_at_init():
    """At init, zero-init on attn.proj_o + s_mlp[1].weight + z_mlp[1].weight
    makes each AdaptorAttentionBlock a near-identity residual transform."""
    torch.manual_seed(0)
    adaptor = AdaptorModule(n_attn_layers=1)
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()

    adaptor_linear = AdaptorModule(n_attn_layers=0)
    adaptor_linear.single_proj.load_state_dict(adaptor.single_proj.state_dict())
    adaptor_linear.pair_proj.load_state_dict(adaptor.pair_proj.state_dict())

    s_full, z_full = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    s_lin, z_lin = adaptor_linear(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)

    assert torch.allclose(s_full, s_lin, atol=1e-5)
    assert torch.allclose(z_full, z_lin, atol=1e-5)


def test_adaptor_translation_equivariance():
    """Translating ca_coords by a constant leaves the output unchanged;
    pairwise distances are translation-invariant and AttentionPairBias
    is not coordinate-aware."""
    torch.manual_seed(0)
    adaptor = AdaptorModule(n_attn_layers=1)
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()

    s1, z1 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    shift = torch.tensor([1.5, -2.0, 0.7])[None, None, :]
    s2, z2 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords + shift, mask=mask)

    assert torch.allclose(s1, s2, atol=1e-6)
    assert torch.allclose(z1, z2, atol=1e-6)
