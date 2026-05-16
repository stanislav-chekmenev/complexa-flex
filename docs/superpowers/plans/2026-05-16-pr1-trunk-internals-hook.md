# PR-1 — trunk `expose_intermediates` hook

## 1. Branch and base

- **Branch.** `feat/confidence-trunk-internals-hook`
- **Base.** `merge_quality_graft`
- **Merges into.** `merge_quality_graft`

## 2. Goal and non-goals

- **Goal.** Add an `expose_intermediates: bool = False` kwarg to `LocalLatentsTransformer` (v1 and v2). When `True`, `forward(input)` returns one extra dict key, `nn_out["trunk_intermediates"]`, carrying the post-trunk `s`, `z`, masks, and `n_orig`. When `False` (the default), `forward` is **bit-identical** to today.
- **Non-goals.** No head, no sidecar, no dataset changes, no `proteina.py` edits, no Hydra-config edits to existing trunk configs.

## 3. Files to edit

- `src/proteinfoundation/nn/local_latents_transformer.py`
- `src/proteinfoundation/nn/local_latents_transformer_v2.py`

### 3.1 `__init__` edit (both v1 and v2, identical)

Insert immediately after `self.output_param = kwargs["output_parameterization"]`:

- `self.expose_intermediates = bool(kwargs.get("expose_intermediates", False))`
- Docstring sentence: "When `expose_intermediates=True`, `forward(input)` additionally returns `nn_out['trunk_intermediates']` carrying the post-trunk `(s, z, mask, orig_mask, n_orig)`. Default `False`; bit-identical to the legacy forward when off. The sidecar `ConfidenceDistillationModule` (PR-4) flips this on programmatically after instantiation — do not set it in trunk Hydra configs."

### 3.2 `forward` edit (both files)

After the existing trim block and **before** `nn_out = {}`:

- Capture `s_out = seqs * mask[..., None]` (post-trunk sequence repr, mask-zeroed, **not** trimmed to `n_orig`).
- Capture `z_out = pair_rep` (final pair repr; not trimmed).
- Build `intermediates = {"s": s_out, "z": z_out, "mask": mask, "orig_mask": orig_mask, "n_orig": int(n_orig)}`.

Then assemble `nn_out` exactly as today, and append:

```python
if self.expose_intermediates:
    nn_out["trunk_intermediates"] = intermediates
```

**Control-flow invariant.** The bool path through the existing forward must not branch on `expose_intermediates`; the only place the flag is read is the final guarded append. This makes the off-by-default contract automatically defendable.

### 3.3 Shape contract returned

`nn_out["trunk_intermediates"]`:

- `s`: `[b, n_extended, token_dim]`, dtype matches `seqs`.
- `z`: `[b, n_extended, n_extended, pair_repr_dim]`, dtype matches `pair_rep`.
- `mask`: `[b, n_extended]`, `bool`. Concat-extended mask.
- `orig_mask`: `[b, n_orig]`, `bool`. Pre-concat mask.
- `n_orig`: Python `int`. Reader is responsible for `s[:, :n_orig]` / `z[:, :n_orig, :n_orig]` if it wants the original-only slice.

### 3.4 Hydra surface

- `expose_intermediates` is a plain kwarg consumed by `LocalLatentsTransformer(**kwargs)`, default `False`. **No** existing Hydra config under `configs/nn/` is touched in PR-1.
- The flag is documented in the `__init__` docstring as the entry point for the sidecar; PR-4 will do `self.proteina.nn.expose_intermediates = True` after construction.

## 4. Tests (TDD — write **before** implementation)

### 4.1 `tests/integration/confidence/test_trunk_feature_extraction.py`

- Build a minimal `LocalLatentsTransformer` (v1 and v2) with small dims (e.g. `nlayers=2, token_dim=64, pair_repr_dim=32`) for speed.
- Synthetic `input` dict with `b=2, n=16`, mask all-true.
- Default `False` → no `trunk_intermediates` key.
- `mdl.expose_intermediates = True` → present with 5 keys, shapes as specified.
- Repeat for v2.
- Sub-case with `use_concat=True`: assert `s.shape[1] == n_extended > n_orig`, `orig_mask.shape[1] == n_orig`.

### 4.2 `tests/regression/test_flow_matching_loss_unchanged.py`

- Seed torch + numpy.
- Build trunk with `expose_intermediates=False` (default) from a tiny fixed config.
- Run `forward(input)` on fixed-seed input.
- Assert bit-exact equality against snapshot at `tests/fixtures/trunk_baseline_<sha>.pt`.
- Fixture regeneration: if file is absent and `REGEN_FIXTURES=1`, write the snapshot and `pytest.skip("fixture regenerated")`. Otherwise `pytest.skip("baseline fixture missing; run with REGEN_FIXTURES=1 once on merge_quality_graft HEAD then commit")`.
- Sub-case: set `expose_intermediates = True`, run again, assert same `bb_ca` / `local_latents`.

## 5. Verification commands

1. `uv run pytest tests/integration/confidence/test_trunk_feature_extraction.py -v`
2. `uv run pytest tests/regression/test_flow_matching_loss_unchanged.py -v` (first with `REGEN_FIXTURES=1`, commit snapshot, then without).
3. 1-step smoke: `uv run python -m proteinfoundation.train ... +trainer.max_steps=1` on `training_local_latents`.

## 6. Reviewer panel

- **`code-review-debug-complexity-expert`** (mandatory). Focus: contract integrity of the off-path; absence of branching that could perturb FM behaviour.
- **`ml-protein-architect`**. Focus: package-layout conformance; flag consumed via `**kwargs`.
- **`ml-software-pytorch-jax-expert`**. Focus: autograd/`torch.compile` interaction; captured `s, z` do not retain unwanted grad; bf16/fp32 dtype preserved.

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Off-by-default contract violation | `test_flow_matching_loss_unchanged.py` with `torch.equal` |
| R2 | Captured tensors retain grad on off-path | Caller wraps in `torch.no_grad()` (PR-4 responsibility); document |
| R3 | v1/v2 hook drifts | Tests run against both files in same suite |
| R4 | Concat-extended `mask` conflated with `orig_mask` | Explicit two-key contract + sub-case test |
| R5 | bf16/fp32 mismatch downstream | PR-4's responsibility; document |
