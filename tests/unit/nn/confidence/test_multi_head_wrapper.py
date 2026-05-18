"""Contract tests for `MultiHeadConfidence` (PR-B Slice 3).

The wrapper is a `BaseConfidenceHead` itself: it owns the trunk, holds an
`nn.ModuleDict` of child heads, runs the trunk exactly once per forward
pass, and dispatches `(s, z, mask)` to every child's `_predict`. The
load-bearing properties are:

- Trunk-once efficiency: the shared trunk runs once regardless of how
  many children consume `(s, z)`.
- Per-head missing-label tolerance: an all-False mask on one child does
  not contaminate the total loss with NaN; the wrapper's loss path
  short-circuits the empty-mask head.
- `expected_trunk_eval_t` parity: a child whose pinned `t` differs from
  the wrapper's must fail at construction with a message naming the
  offender (debuggability).
- Non-empty `output_name_root`: children must self-identify so the
  Lightning module can prefix logs deterministically.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from proteinfoundation.nn.confidence._losses import combined_plddt_loss
from proteinfoundation.nn.confidence.base import BaseConfidenceHead, ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.confidence.losses import MultiHeadLoss


B, L = 2, 10
TOKEN_DIM = 64
PAIR_REPR_DIM = 32
DIM_COND = 32
NUM_PLDDT_BINS = 50
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


def _make_plddt_head(trunk: ConfidenceTrunk) -> PLDDTHead:
    return PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_PLDDT_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )


def _make_pae_head(trunk: ConfidenceTrunk) -> PaeHead:
    return PaeHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
    )


def _make_wrapper() -> MultiHeadConfidence:
    trunk = _make_trunk()
    plddt = _make_plddt_head(trunk)
    pae = _make_pae_head(trunk)
    return MultiHeadConfidence(
        trunk=trunk,
        children={"plddt": plddt, "pae": pae},
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
    )


def _make_inputs(b: int = B, n: int = L, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, TOKEN_DIM, generator=g)
    z = torch.randn(b, n, n, PAIR_REPR_DIM, generator=g)
    mask = torch.ones(b, n, dtype=torch.bool)
    cond = torch.randn(b, n, DIM_COND, generator=g)
    return s, z, mask, cond


def _make_batch(seed: int = 0) -> dict:
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


def test_forward_returns_dict_of_per_head_outputs() -> None:
    wrapper = _make_wrapper().eval()
    s, z, mask, cond = _make_inputs()
    out = wrapper(s, z, mask, cond)
    assert set(out.keys()) == {"plddt", "pae"}
    assert "plddt_logits" in out["plddt"]
    assert "pae_logits" in out["pae"]
    assert out["plddt"]["plddt_logits"].shape == (B, L, NUM_PLDDT_BINS)
    assert out["pae"]["pae_logits"].shape == (B, L, L, NUM_PAE_BINS)


def test_trunk_runs_exactly_once_per_forward() -> None:
    wrapper = _make_wrapper().eval()
    real_trunk_forward = wrapper.trunk.forward
    counter = {"n": 0}

    def _counting_forward(*args, **kwargs):
        counter["n"] += 1
        return real_trunk_forward(*args, **kwargs)

    wrapper.trunk.forward = _counting_forward  # type: ignore[method-assign]
    try:
        s, z, mask, cond = _make_inputs()
        _ = wrapper(s, z, mask, cond)
    finally:
        wrapper.trunk.forward = real_trunk_forward  # type: ignore[method-assign]

    assert counter["n"] == 1, (
        f"trunk forward must run exactly once per wrapper forward, ran {counter['n']}"
    )


def test_all_false_pae_mask_does_not_contaminate_total_loss() -> None:
    wrapper = _make_wrapper().eval()
    s, z, mask, cond = _make_inputs(seed=1)
    out = wrapper(s, z, mask, cond)
    batch = _make_batch(seed=2)

    plddt_mask_eff = batch["plddt_mask"].to(torch.float32)
    pae_mask_all_false = torch.zeros(B, L, L, dtype=torch.float32)
    masks_by_head = {"plddt": plddt_mask_eff, "pae": pae_mask_all_false}

    loss_fn = MultiHeadLoss({"plddt": 1.0, "pae": 1.0})
    total, log_dict = loss_fn(out, wrapper.children_heads, batch, masks_by_head)

    assert torch.isfinite(total), "total loss must be finite under all-False PAE mask"

    plddt_total, _ = combined_plddt_loss(
        student_logits=out["plddt"]["plddt_logits"].float(),
        plddt_bin_labels=batch["plddt_bin"],
        plddt_continuous=batch["plddt_residue"],
        mask=plddt_mask_eff,
        bin_centers=wrapper.children_heads["plddt"].bin_centers,
        ce_weight=0.9,
        smooth_l1_weight=0.1,
        label_smoothing=0.0,
    )
    assert torch.allclose(total, plddt_total, atol=1e-5), (
        f"all-False PAE mask must skip the PAE branch entirely; "
        f"total={total.item():.6f}, expected plddt-only={plddt_total.item():.6f}"
    )


def test_mismatched_expected_trunk_eval_t_raises_with_named_offender() -> None:
    trunk = _make_trunk()

    class _FakeMismatchHead(BaseConfidenceHead):
        expected_trunk_eval_t: float = 0.5
        output_name_root: str = "fake"
        output_keys: tuple[str, ...] = ("fake_logits",)

        def _predict(self, s, z, mask):
            return {"fake_logits": torch.zeros_like(s[..., :1])}

        def compute_loss_and_metrics(self, out, batch, mask_eff, *, stage="train"):
            zero = out["fake_logits"].sum() * 0.0
            return zero, {"loss": zero, "loss_ce": zero, "loss_smooth_l1": zero}

    bad = _FakeMismatchHead(
        trunk=trunk, token_dim=TOKEN_DIM, pair_repr_dim=PAIR_REPR_DIM
    )

    with pytest.raises(ValueError) as exc_info:
        MultiHeadConfidence(
            trunk=trunk,
            children={"fake": bad},
            token_dim=TOKEN_DIM,
            pair_repr_dim=PAIR_REPR_DIM,
        )
    msg = str(exc_info.value)
    assert "fake" in msg, f"error must name the offending child head: {msg}"
    assert "0.5" in msg, f"error must name the offending eval_t value: {msg}"


def test_zero_loss_weight_emits_runtime_warning() -> None:
    trunk = _make_trunk()
    plddt = _make_plddt_head(trunk)
    pae = _make_pae_head(trunk)

    with pytest.warns(RuntimeWarning, match=r"loss_weights.*0\.0"):
        MultiHeadConfidence(
            trunk=trunk,
            children={"plddt": plddt, "pae": pae},
            token_dim=TOKEN_DIM,
            pair_repr_dim=PAIR_REPR_DIM,
            loss_weights={"plddt": 1.0, "pae": 0.0},
        )


def test_empty_output_name_root_raises_with_named_child_key() -> None:
    trunk = _make_trunk()

    class _FakeNamelessHead(BaseConfidenceHead):
        expected_trunk_eval_t: float = 0.99
        output_name_root: str = ""
        output_keys: tuple[str, ...] = ("nameless_logits",)

        def _predict(self, s, z, mask):
            return {"nameless_logits": torch.zeros_like(s[..., :1])}

        def compute_loss_and_metrics(self, out, batch, mask_eff, *, stage="train"):
            zero = out["nameless_logits"].sum() * 0.0
            return zero, {"loss": zero, "loss_ce": zero, "loss_smooth_l1": zero}

    nameless = _FakeNamelessHead(
        trunk=trunk, token_dim=TOKEN_DIM, pair_repr_dim=PAIR_REPR_DIM
    )

    with pytest.raises(ValueError) as exc_info:
        MultiHeadConfidence(
            trunk=trunk,
            children={"empty_name_head": nameless},
            token_dim=TOKEN_DIM,
            pair_repr_dim=PAIR_REPR_DIM,
        )
    msg = str(exc_info.value)
    assert "empty_name_head" in msg, (
        f"error must name the offending child dict key: {msg}"
    )
    assert "output_name_root" in msg, (
        f"error must mention 'output_name_root': {msg}"
    )
    assert "non-empty" in msg or "empty" in msg, (
        f"error must mention emptiness explicitly: {msg}"
    )
