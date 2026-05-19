"""Contract tests for `PaeHead` (PR-B Slice 2).

Covers `(B, L, L, num_pae_bins)` logits shape, gradient flow into both head
and trunk parameters, and the uniform-logits midpoint property for
`logits_to_expected_value`. The bin convention is the AF2 PAE standard:
`num_bins=64`, `bin_width=0.5`, centers `[0.25, 0.75, ..., 31.75]` so
uniform softmax expectation equals `(0.25 + 31.75) / 2 = 16.0`.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


B, N = 2, 10
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
NUM_PAE_BINS = 64


def _make_trunk() -> ConfidenceTrunk:
    return ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=True,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
    )


def _make_head(trunk: ConfidenceTrunk | None = None) -> PaeHead:
    return PaeHead(
        trunk=trunk if trunk is not None else _make_trunk(),
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
    )


def _make_inputs(b: int = B, n: int = N, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    return s, z, mask, cond


def test_forward_returns_pae_logits_shape() -> None:
    head = _make_head().eval()
    s, z, mask, cond = _make_inputs()
    out = head(s, z, mask, cond)
    assert "pae_logits" in out
    assert out["pae_logits"].shape == (B, N, N, NUM_PAE_BINS)


def test_gradients_flow_to_head_and_trunk() -> None:
    trunk = _make_trunk()
    head = _make_head(trunk=trunk)
    head.train()
    s, z, mask, cond = _make_inputs()
    out = head(s, z, mask, cond)
    loss = (out["pae_logits"] ** 2).mean()
    loss.backward()

    head_only_params = [
        (n, p) for n, p in head.named_parameters() if not n.startswith("trunk.")
    ]
    trunk_params = [(n, p) for n, p in trunk.named_parameters()]
    assert head_only_params, "head must have its own parameters outside the trunk"

    head_grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for _, p in head_only_params
    )
    trunk_grad_seen = any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for _, p in trunk_params
    )
    assert head_grad_seen, "no head-only parameter received gradient"
    assert trunk_grad_seen, "no trunk parameter received gradient"


def test_uniform_logits_expected_value_is_midpoint() -> None:
    head = _make_head().eval()
    logits = torch.zeros(B, N, N, NUM_PAE_BINS)
    ev = head.logits_to_expected_value(logits)
    assert ev.shape == (B, N, N)
    assert torch.allclose(ev, torch.full_like(ev, 16.0), atol=1e-4)


def _make_loss_batch(b: int = B, n: int = N, seed: int = 1) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pae_bin": torch.randint(0, NUM_PAE_BINS, (b, n, n), generator=g),
        "pae_residue_pair": torch.rand(b, n, n, generator=g) * 31.75,
    }


def test_loss_total_emitted_only_outside_train_stage() -> None:
    """Train-time log_dict must NOT carry `loss_total` (would duplicate `loss`)."""
    head = _make_head().eval()
    s, z, mask, _ = _make_inputs(seed=2)
    out = head._predict(s, z, mask)
    mask_eff = (mask[:, None, :] & mask[:, :, None]).to(torch.float32)
    batch = _make_loss_batch()

    _, train_log = head.compute_loss_and_metrics(out, batch, mask_eff, stage="train")
    assert "loss" in train_log
    assert "loss_total" not in train_log, (
        "loss_total at train time duplicates `loss` and pollutes wandb step-keys"
    )

    _, val_log = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    assert "loss" in val_log
    assert "loss_total" in val_log
    assert torch.equal(val_log["loss"], val_log["loss_total"])
