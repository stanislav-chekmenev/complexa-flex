"""Pin the quality-graft-style MultiHeadConfidence backbone.

Backbone: AdaptorModule -> QgPairformerStack(n_layers=4). Heads:
PLDDTHead(d_in_token=384), PaeHead(d_in_pair_token=128).

Outputs: {"plddt": {"plddt_logits": [B,L,n_plddt_bins]},
          "pae":   {"pae_logits":   [B,L,L,n_pae_bins]}}.
"""

from __future__ import annotations

import pytest
import torch

from proteinfoundation.nn.confidence.adaptor import AdaptorModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack


def _placeholder_trunk() -> ConfidenceTrunk:
    return ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)


def _build_head() -> MultiHeadConfidence:
    placeholder_trunk = _placeholder_trunk()
    plddt = PLDDTHead(
        trunk=placeholder_trunk,
        token_dim=768,
        d_in_token=384,
        num_plddt_bins=50,
        bin_min=0.0,
        bin_max=100.0,
        ce_weight=1.0,
        ev_weight=0.0,
        label_smoothing=0.0,
    )
    pae = PaeHead(
        trunk=placeholder_trunk,
        pair_repr_dim=256,
        d_in_pair_token=128,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        ce_weight=1.0,
        ev_weight=0.0,
        label_smoothing=0.0,
    )

    return MultiHeadConfidence(
        adaptor=AdaptorModule(),
        backbone=QgPairformerStack(n_layers=4),
        children={"plddt": plddt, "pae": pae},
        loss_weights={"plddt": 1.0, "pae": 1.0},
    )


def _fake_intermediates(B: int = 2, L: int = 16) -> dict:
    return {
        "trunk_seqs": torch.randn(B, L, 768),
        "trunk_pair": torch.randn(B, L, L, 256),
        "local_latents": torch.randn(B, L, 8),
        "ca_coords": torch.randn(B, L, 3),
        "mask": torch.ones(B, L),
    }


def test_multi_head_forward_shapes():
    head = _build_head().eval()
    inputs = _fake_intermediates(B=2, L=16)
    with torch.no_grad():
        out = head(**inputs)
    assert set(out.keys()) == {"plddt", "pae"}
    assert out["plddt"]["plddt_logits"].shape == (2, 16, 50)
    assert out["pae"]["pae_logits"].shape == (2, 16, 16, 64)


def test_multi_head_zero_weight_emits_warning():
    placeholder_trunk = _placeholder_trunk()
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    with pytest.warns(RuntimeWarning, match=r"loss_weights\[\['pae'\]\] == 0\.0"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
            loss_weights={"plddt": 1.0, "pae": 0.0},
        )


def test_multi_head_empty_output_name_root_raises():
    placeholder_trunk = _placeholder_trunk()
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    pae.output_name_root = ""
    with pytest.raises(ValueError, match="empty output_name_root"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
        )


def test_multi_head_mismatched_trunk_eval_t_raises():
    placeholder_trunk = _placeholder_trunk()
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    pae.expected_trunk_eval_t = 0.5
    with pytest.raises(ValueError, match="expected_trunk_eval_t"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
        )


def test_multi_head_runs_one_backward_pass():
    head = _build_head()
    inputs = _fake_intermediates(B=2, L=16)
    out = head(**inputs)
    loss = (
        torch.nn.functional.softmax(out["plddt"]["plddt_logits"], dim=-1).log().mean()
        + torch.nn.functional.softmax(out["pae"]["pae_logits"], dim=-1).log().mean()
    )
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())


def test_placeholder_trunks_dropped_from_children():
    """Pin the load-bearing trunk-pop in `MultiHeadConfidence.__init__`.

    Hydra has to satisfy `BaseConfidenceHead.__init__`'s `trunk:` arg, so
    the yaml ships a placeholder `ConfidenceTrunk` per child. Under the
    qg-style wrapper that trunk is dead weight (~800 MB at full dims)
    and the wrapper's `__init__` pops it from each child's `_modules`.
    A future refactor that re-attaches a trunk under any name would
    silently re-introduce the leak into the optimizer / DDP all-reduce
    set / checkpoint state_dict — only catchable on a GPU smoke node
    with full ckpts staged. Pin the invariant here in the fast suite.
    """
    head = _build_head()

    # 1. Each child's _modules has no "trunk" entry after construction.
    for name, child in head.children_heads.items():
        assert "trunk" not in child._modules, (
            f"child {name!r} still carries `trunk` in _modules; "
            f"placeholder leaked into optimizer / DDP / checkpoint surface."
        )

    # 2. state_dict() has no `children_heads.<name>.trunk.*` keys.
    leaked = [
        k for k in head.state_dict().keys()
        if k.startswith("children_heads.") and ".trunk." in k
    ]
    assert not leaked, f"trunk params leaked into state_dict: {leaked[:5]}"

    # 3. named_parameters() has no trunk params under any child.
    leaked_params = [
        n for n, _ in head.children_heads.named_parameters() if ".trunk." in n
    ]
    assert not leaked_params, (
        f"trunk params leaked into children_heads.named_parameters(): "
        f"{leaked_params[:5]}"
    )
