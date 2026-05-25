"""Contract tests for `MultiHeadLoss` weighting (qg-style backbone).

Two distinct weighting layers govern the multi-head sidecar loss:

- **Within-head** `loss.ce_weight / loss.ev_weight` -- already lives on
  each head's constructor and is exercised by `test_pae_head` /
  `test_plddt_head`. This file does not retest it.
- **Across-head** `weight` -- the `MultiHeadLoss` multiplier on each head's
  total. `weight == 0.0` is a hard short-circuit: the head's
  `compute_loss_and_metrics` is not called and zero gradient reaches its
  parameters.

The gradient-flow assertions are load-bearing: a regression that
silently leaks gradients into a `weight: 0.0` head defeats the
short-circuit and produces stale parameter updates under DDP.
"""

from __future__ import annotations

import torch

from proteinfoundation.confidence.losses import MultiHeadLoss
from proteinfoundation.nn.confidence._losses import combined_plddt_loss
from proteinfoundation.nn.confidence.adaptor import AdaptorModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack


B, L = 2, 8
TRUNK_TOKEN_DIM = 768
TRUNK_PAIR_DIM = 256
LATENT_DIM = 8
ADAPTOR_S_DIM = 384
ADAPTOR_Z_DIM = 128
NUM_PLDDT_BINS = 50
NUM_PAE_BINS = 64


def _placeholder_trunk() -> ConfidenceTrunk:
    return ConfidenceTrunk(
        token_dim=TRUNK_TOKEN_DIM,
        pair_repr_dim=TRUNK_PAIR_DIM,
        n_blocks=1,
    )


def _make_wrapper() -> MultiHeadConfidence:
    trunk = _placeholder_trunk()
    plddt = PLDDTHead(
        trunk=trunk,
        token_dim=TRUNK_TOKEN_DIM,
        pair_repr_dim=TRUNK_PAIR_DIM,
        d_in_token=ADAPTOR_S_DIM,
        num_plddt_bins=NUM_PLDDT_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )
    pae = PaeHead(
        trunk=trunk,
        token_dim=TRUNK_TOKEN_DIM,
        pair_repr_dim=TRUNK_PAIR_DIM,
        d_in_pair_token=ADAPTOR_Z_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
    )
    return MultiHeadConfidence(
        adaptor=AdaptorModule(),
        backbone=QgPairformerStack(n_layers=4),
        children={"plddt": plddt, "pae": pae},
    )


def _make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "trunk_seqs": torch.randn(B, L, TRUNK_TOKEN_DIM, generator=g),
        "trunk_pair": torch.randn(B, L, L, TRUNK_PAIR_DIM, generator=g),
        "local_latents": torch.randn(B, L, LATENT_DIM, generator=g),
        "ca_coords": torch.randn(B, L, 3, generator=g),
        "mask": torch.ones(B, L),
    }


def _make_batch(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "mask": torch.ones(B, L, dtype=torch.bool),
        "plddt_residue": torch.rand(B, L, generator=g) * 100.0,
        "plddt_bin": torch.randint(0, NUM_PLDDT_BINS, (B, L), generator=g),
        "plddt_mask": torch.ones(B, L, dtype=torch.bool),
        "pae_residue_pair": torch.rand(B, L, L, generator=g) * 32.0,
        "pae_bin": torch.randint(0, NUM_PAE_BINS, (B, L, L), generator=g),
        "pae_mask": torch.ones(B, L, L, dtype=torch.bool),
    }


def _masks_all_true() -> dict[str, torch.Tensor]:
    return {
        "plddt": torch.ones(B, L, dtype=torch.float32),
        "pae": torch.ones(B, L, L, dtype=torch.float32),
    }


def _grad_is_effectively_zero(p: torch.nn.Parameter) -> bool:
    if p.grad is None:
        return True
    return float(p.grad.abs().max().item()) == 0.0


def _grad_has_signal(p: torch.nn.Parameter) -> bool:
    return p.grad is not None and float(p.grad.abs().max().item()) > 0.0


def test_weight_zero_on_pae_disables_pae_gradients_and_matches_single_head_loss() -> None:
    wrapper = _make_wrapper()
    wrapper.train()
    out = wrapper(**_make_inputs(seed=0))
    batch = _make_batch(seed=1)
    masks = _masks_all_true()

    loss_fn = MultiHeadLoss({"plddt": 1.0, "pae": 0.0})
    total, _ = loss_fn(out, wrapper.children_heads, batch, masks)

    plddt_head = wrapper.children_heads["plddt"]
    expected_plddt_total, _ = combined_plddt_loss(
        student_logits=out["plddt"]["plddt_logits"].float(),
        plddt_bin_labels=batch["plddt_bin"],
        plddt_continuous=batch["plddt_residue"],
        mask=masks["plddt"],
        bin_centers=plddt_head.bin_centers,
        ce_weight=plddt_head.ce_weight,
        smooth_l1_weight=plddt_head.ev_weight,
        label_smoothing=plddt_head.label_smoothing,
    )
    assert torch.allclose(total, expected_plddt_total, atol=1e-5), (
        f"weight=0 PAE branch must short-circuit; total={total.item():.6f} "
        f"vs expected pLDDT-only={expected_plddt_total.item():.6f}"
    )

    total.backward()

    pae_head = wrapper.children_heads["pae"]
    pae_params = list(pae_head.logits_linear.parameters()) + list(
        pae_head.logits_norm.parameters()
    )
    assert pae_params, "PaeHead must expose its own trainable parameters"
    for p in pae_params:
        assert _grad_is_effectively_zero(p), (
            "PaeHead parameter received gradient under weight=0; "
            f"max-abs-grad={p.grad.abs().max().item() if p.grad is not None else 'None'}"
        )

    plddt_params = list(plddt_head.logits_linear.parameters())
    assert any(_grad_has_signal(p) for p in plddt_params), (
        "no pLDDT logits-linear parameter received gradient under weight=1"
    )


def test_both_active_gradients_reach_both_heads_and_shared_backbone() -> None:
    wrapper = _make_wrapper()
    wrapper.train()
    out = wrapper(**_make_inputs(seed=2))
    batch = _make_batch(seed=3)
    masks = _masks_all_true()

    loss_fn = MultiHeadLoss({"plddt": 0.7, "pae": 0.7})
    total, _ = loss_fn(out, wrapper.children_heads, batch, masks)
    total.backward()

    plddt_head = wrapper.children_heads["plddt"]
    pae_head = wrapper.children_heads["pae"]

    assert any(_grad_has_signal(p) for p in plddt_head.logits_linear.parameters()), (
        "no PLDDTHead.logits_linear parameter received gradient"
    )
    assert any(_grad_has_signal(p) for p in pae_head.logits_linear.parameters()), (
        "no PaeHead.logits_linear parameter received gradient"
    )
    assert any(_grad_has_signal(p) for p in wrapper.backbone.parameters()), (
        "no backbone parameter received gradient under joint-active weights"
    )
    assert any(_grad_has_signal(p) for p in wrapper.adaptor.parameters()), (
        "no adaptor parameter received gradient under joint-active weights"
    )


def test_plddt_zero_only_grads_pae_and_shared_backbone() -> None:
    wrapper = _make_wrapper()
    wrapper.train()
    out = wrapper(**_make_inputs(seed=4))
    batch = _make_batch(seed=5)
    masks = _masks_all_true()

    loss_fn = MultiHeadLoss({"plddt": 0.0, "pae": 1.0})
    total, _ = loss_fn(out, wrapper.children_heads, batch, masks)
    total.backward()

    plddt_head = wrapper.children_heads["plddt"]
    pae_head = wrapper.children_heads["pae"]

    for p in plddt_head.logits_linear.parameters():
        assert _grad_is_effectively_zero(p), (
            "PLDDTHead.logits_linear received gradient under plddt weight=0"
        )

    assert any(_grad_has_signal(p) for p in pae_head.logits_linear.parameters()), (
        "no PaeHead.logits_linear parameter received gradient under pae weight=1"
    )
    assert any(_grad_has_signal(p) for p in wrapper.backbone.parameters()), (
        "no backbone parameter received gradient when only PAE is active"
    )


def test_log_dict_structure_and_total_equals_weighted_sum() -> None:
    wrapper = _make_wrapper().eval()
    with torch.no_grad():
        out = wrapper(**_make_inputs(seed=6))
    batch = _make_batch(seed=7)
    masks = _masks_all_true()

    weights = {"plddt": 0.7, "pae": 0.3}
    loss_fn = MultiHeadLoss(weights)
    total, log_dict = loss_fn(out, wrapper.children_heads, batch, masks)

    required = {
        "plddt/total",
        "plddt/loss_ce",
        "plddt/loss_smooth_l1",
        "pae/total",
        "pae/loss_ce",
        "pae/loss_smooth_l1",
    }
    missing = required - set(log_dict.keys())
    assert not missing, (
        f"log_dict missing required keys: {sorted(missing)}; "
        f"got: {sorted(log_dict.keys())}"
    )

    recomputed = (
        weights["plddt"] * log_dict["plddt/total"]
        + weights["pae"] * log_dict["pae/total"]
    )
    assert torch.allclose(total, recomputed, atol=1e-5), (
        f"total ({total.item():.6f}) must equal sum of weight * per-head total "
        f"({recomputed.item():.6f})"
    )
