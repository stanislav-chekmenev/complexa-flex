# PR-3 Plan — Confidence Head Base Class + `PLDDTHead`

Status: draft, ready for review.
Date: 2026-05-16.
Spec: `docs/superpowers/specs/2026-05-16-confidence-head-distillation-design.md` (APPROVED).

---

## 1. Branch and base

- **Branch.** `feat/confidence-head-base-class` (off `merge_quality_graft`).
- **Base commit.** `8250d9a` (PR-2 merged).
- **Worktree.** `.claude/worktrees/pr3-head-base/`.
- **Merge target.** `merge_quality_graft`. No merge to `dev` until the full task lands.
- **Depends on.** PR-1 (trunk `expose_intermediates`) and PR-2 (`plddt_to_bin` helper) — both merged.

## 2. Goal and non-goals

### Goal
Land the trainable, light-weight confidence-head architecture and its Hydra-instantiable registry, with **no** training-loop code and **no** edits to `proteina.py` or `local_latents_transformer*.py`. After this PR:

- `proteinfoundation.nn.confidence` exposes `BaseConfidenceHead`, `ConfidenceTrunk`, `PLDDTHead`, `build_confidence_head_from_cfg`.
- `hydra.utils.instantiate(cfg_plddt_head)` returns a fully-wired `PLDDTHead` ready to consume `(s, z, mask, cond)` from the trunk intermediates.
- Unit + integration tests cover the head architecture, masking, registry, and the wiring between a tiny `LocalLatentsTransformer` (with `expose_intermediates=True`) and a tiny `PLDDTHead`.

### Non-goals
- No CE / SmoothL1 loss — PR-4.
- No `compute_loss` body on `BaseConfidenceHead` (the abstract signature is declared; PR-4 implements concrete losses).
- No sidecar Lightning module, no training entry point, no metrics, no sbatch — PR-4 / PR-6.
- No edits to `proteina.py`, `local_latents_transformer.py`, `local_latents_transformer_v2.py`, `losses.py` (beyond, if strictly needed, a small head-internal helper; spec keeps PR-3 head-only).
- No multi-head registry entries beyond `plddt`.
- No future ipTM/ipAE/ipLDDT implementations; only the *seats* for them (`z` carried + symmetrised, `chain_id` optional kwarg with zero default).

## 3. Files to add (one-line purpose each)

Source (`src/proteinfoundation/nn/confidence/`):
- `__init__.py` — re-exports `BaseConfidenceHead`, `ConfidenceTrunk`, `PLDDTHead`, `build_confidence_head_from_cfg`, `register_confidence_head`, `CONFIDENCE_HEAD_REGISTRY`.
- `base.py` — `ConfidenceTrunk(nn.Module)` + `BaseConfidenceHead(nn.Module, ABC)`.
- `projections.py` — `SeqProjection(in_dim, out_dim)` and `PairProjection(in_dim, out_dim)`; identity (no params) when `in_dim == out_dim`, else `LayerNorm + Linear(bias=False)`.
- `plddt_head.py` — `PLDDTHead(BaseConfidenceHead)`.
- `registry.py` — `CONFIDENCE_HEAD_REGISTRY: dict[str, type]`, `@register_confidence_head(name)`, `build_confidence_head_from_cfg(cfg)`.

Configs (`configs/nn/confidence/`):
- `base.yaml` — shared trunk hyperparams (`token_dim=768`, `pair_repr_dim=256`, `n_blocks=4`, `n_heads=16`, `dim_cond=256`, `use_tri_mult=True`, `use_tri_attn=False`, `use_qkln=True`, `dropout=0.1`, `update_pair_repr_every_n=1`, `expects_external_cond=True`).
- `plddt_head.yaml` — composes `base`; sets `_target_: proteinfoundation.nn.confidence.plddt_head.PLDDTHead`, `name: plddt`, `num_plddt_bins=50`, `bin_min=0.0`, `bin_max=100.0`.

Tests (TDD — written first):
- `tests/unit/confidence/__init__.py`
- `tests/unit/confidence/test_confidence_trunk.py`
- `tests/unit/confidence/test_plddt_head.py`
- `tests/unit/confidence/test_confidence_registry.py`
- `tests/unit/confidence/test_head_invariance.py`
- `tests/integration/confidence/__init__.py`
- `tests/integration/confidence/test_head_with_trunk_intermediates.py`

## 4. Class signatures (sketch — no bodies)

### `ConfidenceTrunk` (in `base.py`)

```python
class ConfidenceTrunk(nn.Module):
    """Shared light-weight pair-biased stack reused by all confidence heads.

    Built from N x (MultiheadAttnAndTransition + optional PairReprUpdate).
    Inputs `(s, z, mask, cond)` are produced upstream by the frozen
    LocalLatentsTransformer trunk (via `expose_intermediates=True`) and
    by the sidecar's FeatureFactory time embedder at t=0.99 (cond).

    Outputs `(s_refined, z_refined)` are LayerNormed; z is symmetrised
    in-place at the end. Caller still owns the optional concat-to-orig
    trimming (Risk R7).
    """

    def __init__(
        self,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 16,
        dim_cond: int = 256,
        use_tri_mult: bool = True,
        use_tri_attn: bool = False,
        use_qkln: bool = True,
        dropout: float = 0.1,
        update_pair_repr_every_n: int = 1,
        expects_external_cond: bool = True,
    ) -> None: ...

    def forward(
        self,
        s: torch.Tensor,           # [b, n, token_dim]
        z: torch.Tensor,           # [b, n, n, pair_repr_dim]
        mask: torch.Tensor,        # [b, n] bool
        cond: torch.Tensor,        # [b, n, dim_cond]  REQUIRED (not Optional)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (s_refined: [b, n, token_dim], z_refined: [b, n, n, pair_repr_dim]).
        Both mask-zeroed; z_refined symmetrised (z + z.transpose(-3, -2)) / 2.
        Both LayerNormed (per-channel zero-mean unit-var on masked positions).
        """
```

Internals: `n_blocks` of `MultiheadAttnAndTransition(dim_token=token_dim, dim_pair=pair_repr_dim, nheads=n_heads, dim_cond=dim_cond, residual_mha=True, residual_transition=True, parallel_mha_transition=False, use_attn_pair_bias=True, use_qkln=use_qkln, dropout=dropout)`. `n_blocks - 1` interleaved `PairReprUpdate(token_dim, pair_repr_dim, use_tri_mult, use_tri_attn, dropout)` (gated by `update_pair_repr_every_n`, matching the trunk's existing pattern). Final `LayerNorm(token_dim)` on `s` and `LayerNorm(pair_repr_dim)` on `z`, then symmetrisation.

`cond` is **mandatory** (no `Optional`); passing `cond=None` raises a `TypeError` at the python signature level. Justification: `MultiheadAttnAndTransition` calls AdaLN, which has no fallback path. Documented in the docstring.

### `BaseConfidenceHead` (in `base.py`)

```python
class BaseConfidenceHead(nn.Module, ABC):
    """Abstract base. Owns trunk; subclasses own the prediction MLPs."""

    output_keys: tuple[str, ...]  # class attribute, override in subclass

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
    ) -> None: ...

    @abstractmethod
    def _predict(
        self,
        s: torch.Tensor,        # [b, n, token_dim] LayerNormed
        z: torch.Tensor,        # [b, n, n, pair_repr_dim] LayerNormed + symm
        mask: torch.Tensor,     # [b, n] bool
    ) -> dict[str, torch.Tensor]: ...

    def forward(
        self,
        s: torch.Tensor,                       # [b, n, token_dim]
        z: torch.Tensor,                       # [b, n, n, pair_repr_dim]
        mask: torch.Tensor,                    # [b, n] bool
        cond: torch.Tensor,                    # [b, n, dim_cond]
        chain_id: torch.Tensor | None = None,  # [b, n] int64; defaults to zeros (monomer)
    ) -> dict[str, torch.Tensor]:
        """Runs trunk then delegates to _predict. Returns the subclass dict.

        chain_id is currently unused by PLDDTHead; PR-3 wires it so future
        ipTM/ipAE/ipLDDT subclasses can consume it without an API break.
        Default zero tensor handled inside forward when None.
        """

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Abstract / NotImplementedError in PR-3. PR-4 implements per-head losses."""
        raise NotImplementedError("compute_loss is implemented in PR-4")
```

### `PLDDTHead` (in `plddt_head.py`)

```python
@register_confidence_head("plddt")
class PLDDTHead(BaseConfidenceHead):
    output_keys = ("plddt_logits",)

    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_plddt_bins: int = 50,
        bin_min: float = 0.0,
        bin_max: float = 100.0,
    ) -> None:
        """Adds final LayerNorm(token_dim) + Linear(token_dim, num_plddt_bins).
        Registers `bin_centers` buffer ([num_plddt_bins], midpoints of equal
        bins on [bin_min, bin_max]) for logits_to_expected_value.
        """

    def _predict(self, s, z, mask) -> dict[str, torch.Tensor]:
        """Returns {"plddt_logits": [b, n, num_plddt_bins]}, mask-zeroed."""

    def logits_to_expected_value(
        self, logits: torch.Tensor  # [b, n, num_plddt_bins]
    ) -> torch.Tensor:                # [b, n] in [bin_min, bin_max]
        """Bin-center-weighted softmax mean. fp32 internally."""
```

### Projections (in `projections.py`)

```python
class SeqProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None: ...
    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...

class PairProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None: ...
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: ...
```

When `in_dim == out_dim` both are `nn.Identity` semantics (no params, no LN); when they differ, they apply `LayerNorm + Linear(bias=False)` to the last dim. Used by callers (PR-4 sidecar) when trunk and head dims diverge; **not** wired inside `ConfidenceTrunk` itself — the trunk assumes inputs already in `(token_dim, pair_repr_dim)`. The projection layer is exported so PR-4 can compose it before the head call.

### Registry (in `registry.py`)

```python
CONFIDENCE_HEAD_REGISTRY: dict[str, type[BaseConfidenceHead]] = {}

def register_confidence_head(name: str) -> Callable[[type], type]: ...

def build_confidence_head_from_cfg(cfg) -> BaseConfidenceHead:
    """Hydra-aware factory. Two supported styles:
       1. cfg has `_target_` -> hydra.utils.instantiate(cfg).
       2. cfg has `name` -> look up CONFIDENCE_HEAD_REGISTRY[name] then
          instantiate with cfg-as-kwargs.
       Unknown name -> ValueError with the list of known names.
    """
```

Registry is populated by import side-effect: `plddt_head.py` imports `register_confidence_head` and decorates `PLDDTHead`. `__init__.py` imports `plddt_head` to make the registration happen at package load.

## 5. Hydra configs

### `configs/nn/confidence/base.yaml`

```yaml
# Shared confidence-trunk hyperparams. Consumed via Hydra composition by
# concrete head configs (plddt_head.yaml).
trunk:
  _target_: proteinfoundation.nn.confidence.base.ConfidenceTrunk
  token_dim: 768
  pair_repr_dim: 256
  n_blocks: 4
  n_heads: 16
  dim_cond: 256
  use_tri_mult: True
  use_tri_attn: False
  use_qkln: True
  dropout: 0.1
  update_pair_repr_every_n: 1
  expects_external_cond: True
```

### `configs/nn/confidence/plddt_head.yaml`

```yaml
defaults:
  - base
  - _self_

_target_: proteinfoundation.nn.confidence.plddt_head.PLDDTHead
name: plddt
token_dim: 768
pair_repr_dim: 256
num_plddt_bins: 50
bin_min: 0.0
bin_max: 100.0
```

`name` is duplicated alongside `_target_` for the dual factory style (§4 registry).

## 6. Tests (TDD, written before implementation)

All tests construct synthetic `cond` of shape `[b, n, dim_cond]` (random or zeros) — PR-3 does not depend on the trunk's `FeatureFactory`; PR-4 will wire the real one.

### `tests/unit/confidence/test_confidence_trunk.py`
Targets §4 `ConfidenceTrunk` contract.
- `test_forward_shapes_match_inputs` — `(b=2, n=11, token_dim=64, pair_repr_dim=32)` toy trunk; assert `s_out.shape == s_in.shape`, `z_out.shape == z_in.shape`.
- `test_mask_zeroes_padded_positions` — set `mask[0, 8:] = False`; assert `s_out[0, 8:, :] == 0` and `z_out[0, 8:, :, :] == 0`.
- `test_z_is_symmetric_after_forward` — assert `torch.allclose(z_out, z_out.transpose(-3, -2), atol=1e-5)`.
- `test_s_layernorm_output_sanity` — on masked positions only, per-channel std bounded (>0.1 < 5.0) and mean near zero — coarse LayerNorm-on-output sanity, not a strict equality.
- `test_cond_required` — calling `trunk(s, z, mask)` (no cond) raises `TypeError`.

### `tests/unit/confidence/test_plddt_head.py`
Targets §4 `PLDDTHead`.
- `test_forward_returns_plddt_logits` — output dict has key `"plddt_logits"` with shape `[b, n, 50]`.
- `test_logits_to_expected_value_in_range` — random logits => expected value tensor lies in `[0.0, 100.0]` everywhere.
- `test_uniform_logits_yields_midpoint` — zero logits (uniform softmax) => expected value `~= 50.0` for the default bins (atol 1e-4 because bin centers are 1, 3, 5, ..., 99 -> mean = 50).
- `test_mask_zero_safe` — padded positions can have arbitrary logits; only test that the call does not NaN and the unpadded logits remain finite.

### `tests/unit/confidence/test_confidence_registry.py`
Targets §4 `registry.py`.
- `test_register_decorator_populates_registry` — `"plddt" in CONFIDENCE_HEAD_REGISTRY` and value is `PLDDTHead`.
- `test_build_from_yaml_via_target` — load `configs/nn/confidence/plddt_head.yaml`, call `build_confidence_head_from_cfg(cfg)`, assert isinstance `PLDDTHead`.
- `test_build_from_yaml_via_name` — strip `_target_` from the loaded cfg, call factory, assert isinstance `PLDDTHead`.
- `test_unknown_name_raises` — `build_confidence_head_from_cfg(OmegaConf.create({"name": "iptm_v2"}))` raises `ValueError` whose message contains the registry's known keys.

### `tests/unit/confidence/test_head_invariance.py`
Targets the structural invariance that matters at this layer: padding permutation.
- `test_padded_position_permutation_invariance` — build batch with `n=8`, `mask = [T]*5 + [F]*3`; permute the padded slice; assert unpadded outputs are bit-identical (`torch.allclose` atol 1e-5).

Note: spec's full rotational equivariance test is **deferred to PR-4 / PR-5** because the head has no coordinates — its inputs `(s, z)` are already rotation-invariant by construction at the trunk boundary. PR-3 only verifies the structural pad-permutation property the head itself owns.

### `tests/integration/confidence/test_head_with_trunk_intermediates.py`
Targets the inter-PR seam (PR-1 hook x PR-3 head).
- `test_tiny_trunk_to_plddt_head_wires` — build a tiny `LocalLatentsTransformer` (smallest legal config; reuse fixtures from existing integration tests if present, else build minimal kwargs in-test) with `expose_intermediates=True`. Build a tiny `PLDDTHead` (matching `token_dim`/`pair_repr_dim`). Run `trunk(input)`, take `nn_out["trunk_intermediates"]`, construct synthetic `cond` of shape `[b, n_extended, dim_cond]`, call `head(s, z, mask, cond)`, assert output shape `(b, n_orig, 50)` (the head sees `n_extended` but the *caller* trims; PR-3 test asserts on the *un-trimmed* `[b, n_extended, 50]` and then on the trimmed `[:, :n_orig, :]`).

## 7. Verification commands

```bash
# Focused
uv run pytest -v \
  tests/unit/confidence/test_confidence_trunk.py \
  tests/unit/confidence/test_plddt_head.py \
  tests/unit/confidence/test_confidence_registry.py \
  tests/unit/confidence/test_head_invariance.py \
  tests/integration/confidence/test_head_with_trunk_intermediates.py

# Full suite — must remain green (32/32 pre-PR-3 tests + ~10-15 new cases)
uv run pytest tests/ -v
```

Exit criterion: focused suite all-pass + full suite all-pass + no new warnings beyond pre-PR-3 baseline. Abort criterion: focused suite cannot be made to pass without modifying `local_latents_transformer*.py`, `proteina.py`, or `confidence/losses.py` (any of these would indicate the PR scope is wrong — escalate).

## 8. Reviewer panel

Per `CLAUDE.md` and spec §7, mandatory + domain-relevant agents:

- **`code-review-debug-complexity-expert` (mandatory)** — overall code hygiene, complexity, dead code, abstraction proliferation. Focus: is `projections.py` overkill for the same-dim case? Are class hierarchies one-level only? Any duplicated logic with `local_latents_transformer.py`?
- **`ml-protein-architect`** — Hydra composition, package layout under `proteinfoundation`, naming consistency with the rest of `nn/`. Focus: does `configs/nn/confidence/plddt_head.yaml` compose cleanly with the rest of the Hydra tree, and is the `_target_` path correct?
- **`ml-software-pytorch-jax-expert`** — torch internals around `MultiheadAttnAndTransition` reuse: DDP unused-params risk if any sub-module of `ConfidenceTrunk` has params not exercised on every forward (e.g. `PairReprUpdate` when `update_pair_repr_every_n` skips a layer), autograd hooks for the frozen-trunk consumer story, bf16 stability of triangle multiplication at `n_blocks=4`.
- **`generative-protein-scientist`** — does the architecture (`n_blocks=4`, `use_tri_attn=False`, single-rep + pair-biased) match the design intent for a pLDDT student head, and is `logits_to_expected_value` the right bin-center mean (vs alternative parameterisations like expected `arg`max-shifted)?

## 9. Risk register (PR-3 specific)

| # | Risk | Sev | Lik | Mitigation |
|---|---|---|---|---|
| R-A | `cond=None` reaches `MultiheadAttnAndTransition` -> AdaLN call fails or silently breaks at runtime under autocast | High | Med | Make `cond` a **mandatory positional kwarg** of `ConfidenceTrunk.forward` (no `Optional`); document in docstring; `test_cond_required` enforces. |
| R-B | `dim_cond` mismatch between head config (256) and what the sidecar's `FeatureFactory` produces (must equal trunk's `dim_cond`, also 256 by default per `local_latents_score_nn_160M.yaml`) | High | Low-Med | Expose `dim_cond` as an explicit kwarg on `ConfidenceTrunk`; default 256; add a `dim_cond` row in `configs/nn/confidence/base.yaml`; document in `__init__` docstring that this must equal the trunk's `dim_cond`. PR-4's sidecar will verify at construction. |
| R-C | DDP unused-params if `update_pair_repr_every_n` skips the final layer's pair update — params of unused `PairReprUpdate` modules will not get gradients | Med | Med (deferred to PR-4) | Default `update_pair_repr_every_n=1` so every block updates pair. Document the constraint in `ConfidenceTrunk.__init__`. PR-4 sets DDP `find_unused_parameters=False` in pure-distillation. Note for reviewer: not a PR-3-test risk; flag for PR-4. |
| R-D | bf16 instability in triangle multiplication at `n_blocks=4` (spec Risk 5) | Med | Low (PR-3 tests run fp32) | `use_tri_attn=False` already reduces exposure. PR-3 tests run in fp32. Flag explicitly for PR-4 to verify bf16 on real data. |
| R-E | Concat-feature cropping (spec Risk 7) — the head emits `[b, n_extended, num_bins]` but the loss must operate on `[:, :n_orig]` | Med | Med | PR-3 head does **not** trim — the caller does. Integration test asserts both the un-trimmed `[b, n_extended, 50]` and the trimmed `[:, :n_orig, :]`. Document in `BaseConfidenceHead.forward` docstring: trimming is caller responsibility, head consumes whatever mask it receives. |

## 10. Out of scope (re-statement)

- No CE loss, no SmoothL1 — PR-4 owns the loss in `proteinfoundation.confidence.losses`.
- No sidecar Lightning module, no training entry point, no metrics, no checkpoint plumbing — PR-4.
- No edits to `proteina.py`, `local_latents_transformer.py`, or `local_latents_transformer_v2.py`.
- No new dataset transforms (PR-2 done).
- No multi-head registry entries beyond `plddt`.
- No equivariance-under-rotation test at this layer (head consumes `(s, z)`; rotational invariance is the trunk's property; PR-5 owns end-to-end equivariance).
- No sbatch / DDP work — PR-6.

## 11. Handoff

Hand to `ml-protein-architect` for tests-first implementation. Open questions: none — all blocking decisions resolved in spec §0.
