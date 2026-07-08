# Quality-graft confidence head port + pAE-derived metric correlations — Design

**Date.** 2026-05-25
**Author.** Stanislav Chekmenev (with Claude as implementer/coordinator)
**Repo.** `complexa-flex` (proteinfoundation)
**Branch base.** `dev`
**Approach.** A (Investigate → PR #1 head rewrite → PR #2 metrics)

> **Status (2026-07-07, GC note):** MIXED — §6 PR #2 (pAE metric correlations) **shipped** as complexa-flex PR #24 and is live. §5 PR #1 (QG head rewrite) shipped as complexa-flex PR #23 then was **fully reverted** by complexa-flex PR #26 (`revert/multi-head-pre-qg`); `adaptor.py` / `qg_pairformer_stack.py` / `community_models/boltz/` no longer exist and `MultiHeadConfidence` is back on `ConfidenceTrunk`. Kept live (not archived) because it is half-reverted; treat §5 as historical. **PR-number collision warning:** the #23/#24 above are complexa-flex PRs; the separate QG/La-Proteina *frozen-feature probe* landed 2026-07-07 in the **quality-graft** repo under its OWN PRs #23 (local_only wiring) + #24 (probe lib) — unrelated numbers in a different repo. Frozen-probe brief archived at `docs/archive/handoff/2026-07-07_qg_frozen_feature_probe.md`.

---

## 1. Motivation

The current multi-head pLDDT+PAE Teddymer distillation run (`distillation_teddymer_multihead` on multi-node 2×2 h100nvl, CE-only, val every 1000 steps) converges to pLDDT Pearson R ≈ 0.88 starting from ≈ 0.78 at step 0. In a sister project, [`quality-graft`](file:///mnt/storage01/home/schekmenev/projects/quality-graft) (which distils pLDDT from a La-Proteina trunk whose intermediates are architecturally identical to complexa's), the same pLDDT-distillation objective takes Pearson R from 0.2 to 0.99 over training. The 0.78 starting Pearson R is suspicious — either a label/feature leak in complexa's confidence pipeline, or the current confidence-head architecture saturates immediately and never recovers the headroom.

Quality-graft's recipe — one thin attention adaptor that projects La-Proteina trunk intermediates `(trunk_seqs, trunk_pair, local_latents, ca_coords)` into Boltz-1 dimensions `(s=384, z=128)`, then four Boltz-1-style pairformer layers with triangular attention, then linear readouts — is the architectural template we want to reproduce.

A second, separable problem: the existing 10 pAE-derived metrics that `PaeHead` logs at validation time (`i_pae`, `min_ipae`, `i_ptm`, `i_ptm_energy`, six ipSAE variants) report the population mean of the *predicted-pAE-derived* scalar. They tell you whether the predicted scalar drifts over training, but they do not tell you whether the predicted scalar tracks the *ground-truth-pAE-derived* scalar per dimer. We want paired MAE and Pearson R between `M(pae_pred)` and `M(pae_gt)` across the val set, for every M.

The work splits cleanly into a Phase 0 investigation report and two PRs.

## 2. Goals

- Reproduce quality-graft's confidence-head architecture inside complexa's `MultiHeadConfidence` wrapper, retaining all current Lightning-side contracts (loss aggregation, `output_name_root`, `expected_trunk_eval_t`, DDP `find_unused_parameters=true` + `static_graph=true`, rank-0 loguru gate, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `RESUME_CKPT_PATH` sbatch contract).
- Expose `ca_coords` (predicted Cα per residue) in `LocalLatentsTransformer.trunk_intermediates`, mirroring quality-graft's trunk hook.
- Diagnose the 0.78 Pearson R floor before the rewrite lands. If a fixable leak is identified, fix it independently. If no leak is found, the rewrite proceeds anyway and the diagnosis becomes a non-blocker.
- Add per-sample-paired MAE and Pearson R (and Spearman) between every pAE-derived metric computed from predicted vs. ground-truth PAE, logged at validation time alongside the existing scalar logs.

## 3. Non-goals

- No change to single-head configs (`PlddtHead`-alone, `PaeHead`-alone) that wrap `ConfidenceTrunk` directly. Those keep working unchanged.
- No port of Boltz-1's MSA, template, distogram, or non-pairformer modules. We vendor the minimum slice required for the pairformer-style layer.
- No introduction of Boltz-1 pretrained weight loading. The 4 pairformer layers are randomly initialised — quality-graft's freezing policy is not reproduced here (per user decision).
- No deletion of the existing `distillation_teddymer_multihead.yaml` config; it stays for A/B comparison.
- No fix-up of the `ConfidenceTrunk` (used by single-head configs) unless the Phase 0 leak audit identifies a bug there. The trunk's LayerNorm-bias-leak guard at `src/proteinfoundation/nn/confidence/base.py:146-147` stays as-is.

## 4. Phase 0 — Investigation report (no PR)

### 4.1 Deliverable

`docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md`. One subagent, read-only, single report.

### 4.2 Subagent

`code-review-debug-complexity-expert`. Has Read/Grep/Glob/Bash/WebSearch. No code writes.

### 4.3 Two-thread mandate

**Thread A — Quality-graft architecture dossier.** Reading `/mnt/storage01/home/schekmenev/projects/quality-graft/src/{quality_graft,boltz}/`, produce file:line citations for:

- `AdaptorModule` (already partially known from `src/quality_graft/models/adaptor.py`): confirms `(trunk_seqs[B,n,768] + local_latents[B,n,8]) → single_proj[B,n,384]`, `trunk_pair[B,n,n,256] → pair_proj[B,n,n,128]`, Cα-Cα distogram with 128 bins over `[0.1, 3.0]` nm added into `z`, `n_attn_layers=1` of `AttentionPairBias`.
- `ConfidenceHead` (`src/quality_graft/models/confidence_head.py`): backbone class, exact pairformer layer count, freeze/train policy, head readouts, loss recipe.
- Boltz-1 module dependency list — the exact set of files under `src/boltz/model/` that the head transitively imports. This becomes the `community_models/boltz/` manifest for PR #1.
- Quality-graft's Lightning module (`src/quality_graft/training/`): LR schedule, warmup, EMA, gradient clipping, mixed-precision policy. Diff against complexa's `ConfidenceModule`.
- Dataset-side label generation: pLDDT/PAE label scale + binning, label transforms. Diff against complexa's `AddPLDDTFromBFactor` (AF2 `[0, 100]` scale, 50 bins of width 2 for pLDDT).

**Thread B — Complexa 0.78-floor leak audit.** Reading `src/proteinfoundation/nn/confidence/base.py`, `src/proteinfoundation/nn/local_latents_transformer.py`, `src/proteinfoundation/nn/confidence/plddt_head.py`, the multihead-Teddymer config + sbatch, identify any path by which the head could receive information correlated with the pLDDT label at init:

1. Does `local_latents` (the 8-dim per-residue latent from the complexa trunk) carry pLDDT-correlated signal? Complexa was trained on AlphaFold-derived structures, so the trunk may have learned to encode confidence-correlated features.
2. Does `CenteringTransform(full, bb_ca)` expose chain-level pLDDT through the Cα-COM recentering?
3. Are GT pLDDT labels accidentally leaking into a feature dict the trunk consumes? Grep every `AddPLDDTFromBFactor` / `pae` / `plddt` reference in the dataset config tree.
4. Is `trunk_eval_t = 0.99` close enough to the data manifold that the trunk features already saturate on pLDDT-correlated geometry?
5. Cross-reference: quality-graft consumes the same trunk-intermediate set `(trunk_seqs, trunk_pair, local_latents, ca_coords)` and observes 0.2 → 0.99. If the leak were in any of these tensors per se (regardless of head architecture), quality-graft would *also* show the 0.78 floor. The agent must compare explicitly and report whether the architecturally-identical trunk reveals the same artefact.

### 4.4 Report structure

```
1. Executive summary (≤ 200 words)
2. Quality-graft architecture inventory
   2.1 AdaptorModule (already partially known; confirm/correct)
   2.2 ConfidenceHead body
   2.3 Boltz-1 dependency manifest (list of files to copy)
   2.4 Loss / metrics recipe
   2.5 Training-loop knobs (LR, warmup, EMA, grad clip, AMP)
   2.6 Data pipeline differences vs complexa
3. Complexa 0.78-floor leak audit
   3.1 Audit checklist with verdict + evidence per item
   3.2 Probability ranking of root cause
   3.3 Recommended diagnostic experiment (if any) to confirm before PR #1
4. Open questions for main thread
```

### 4.5 Stop conditions

Both threads complete, or 90 minutes wall-time, whichever first.

### 4.6 Not-found policy

If Thread B identifies no fixable leak, **PR #1 proceeds anyway**. The leak audit's findings (positive or negative) are recorded in the report and referenced in PR #1's description. The 0.78-floor question may simply resolve when the head changes; the project does not block on it.

If Thread B identifies a fixable leak independent of the head rewrite, a quick fix-bug PR opens *before* PR #1, so PR #1 is evaluated against a clean baseline.

## 5. PR #1 — Quality-graft head port

### 5.1 Branch + scope

Branch `feat/qg-confidence-head` off `dev`. Replaces the trainable backbone of `MultiHeadConfidence` with: adaptor (1 thin attention layer) → 4 Boltz-1 pairformer layers → existing `PlddtHead` / `PaeHead` readouts retargeted to Boltz-1 dims `(s=384, z=128)`. Frozen complexa trunk and Lightning-side contracts stay exactly as today.

### 5.2 New code in-tree

**`community_models/boltz/`** — minimum vendored Boltz-1 slice. Investigation report fixes the exact file list; expected manifest (mirroring quality-graft's `src/boltz/model/`):

- `community_models/boltz/__init__.py` — empty.
- `community_models/boltz/model/layers/attention.py` — `AttentionPairBias` (used by adaptor's attention block AND inside the pairformer single-update path).
- `community_models/boltz/model/layers/triangular_attention/{primitives.py, attention.py}` — `TriangleAttentionStartingNode`, `TriangleAttentionEndingNode`.
- `community_models/boltz/model/layers/triangular_mult.py` — `TriangleMultiplicationOutgoing`, `TriangleMultiplicationIncoming`.
- `community_models/boltz/model/layers/transition.py` — `Transition`.
- `community_models/boltz/model/layers/dropout.py` — `get_dropout_mask` (if used).
- `community_models/boltz/model/modules/pairformer.py` — `PairformerLayer`.
- `community_models/boltz/README.md` — provenance (upstream repo URL, commit hash, license).

Vendored code is copied **verbatim**. Edits permitted: removing dead imports/paths that pull in non-pairformer code, replacing `boltz.model.layers.x` imports with `community_models.boltz.model.layers.x`. No semantic edits.

**`src/proteinfoundation/nn/confidence/adaptor.py`** — port of quality-graft's `AdaptorModule` adapted to complexa naming:

- Dims pinned: `trunk_dim=768`, `pair_dim=256`, `latent_dim=8`, `target_s_dim=384`, `target_z_dim=128`, `n_attn_layers=1`, `num_heads=16`, distogram range `[0.1, 3.0]` nm with 128 bins.
- Drops `source_mode="hybrid"` and decoder-fusion path (complexa has no decoder analogue in confidence-distill).
- Imports `AttentionPairBias` from `community_models.boltz.model.layers.attention`.
- LayerNorm-bias-leak invariant honoured: every projection into `s` is followed by `* mask[..., None]`, every projection into `z` by `* mask[:, :, None, None] * mask[:, None, :, None]`.

**`src/proteinfoundation/nn/confidence/qg_pairformer_stack.py`** — thin wrapper around 4 stacked `community_models.boltz.model.modules.pairformer.PairformerLayer`. Owns the residue-mask plumbing and post-stack mask re-application. Random init (no Boltz-1 checkpoint dependency).

**`src/proteinfoundation/nn/confidence/multi_head.py`** — swap the backbone:

- `ConfidenceTrunk` (the trainable copy of complexa's trunk with pair-update layers) is removed from `MultiHeadConfidence`.
- `MultiHeadConfidence` is constructed with sub-modules: `AdaptorModule`, `QgPairformerStack(n_layers=4)`, the child heads.
- New flow: `(s_trunk, z_trunk, local_latents, ca_coords, mask) → AdaptorModule → (s_384, z_128) → QgPairformerStack(4 layers) → (s_384, z_128) → {plddt_head._predict(s_384), pae_head._predict(z_128)}`.
- `MultiHeadLoss`, `loss_weights`, `expected_trunk_eval_t` assertion, `output_name_root` non-empty assertion, `weight=0.0`-DDP `RuntimeWarning` all unchanged.

**`src/proteinfoundation/nn/local_latents_transformer.py`** — extend `trunk_intermediates` dict by one key:

- Add `"ca_coords": ca_nm_out` (pre-trim `[B, n_extended, 3]`) at the existing `intermediates = {...}` assignment.
- `expose_intermediates=False` default path stays bit-identical (regression contract: `tests/regression/test_flow_matching_loss_unchanged.py`).

**`src/proteinfoundation/nn/confidence/plddt_head.py` and `pae_head.py`** — minimal additions:

- `__init__` accepts `d_in_token` (PlddtHead) / `d_in_pair_token` (PaeHead) kwargs, default 768 / 256 respectively (preserves single-head config behaviour).
- Readout `Linear(d_in, n_bins)` reads the new in-dim from the kwarg.
- Inside `MultiHeadConfidence`, the heads are constructed with `d_in_token=384` / `d_in_pair_token=128`.

**`src/proteinfoundation/nn/confidence/base.py`** — `ConfidenceTrunk` stays available for single-head configs unchanged. `MultiHeadConfidence` no longer instantiates it.

### 5.3 New config

**`configs/confidence/distillation_teddymer_qg_multihead.yaml`** — composes the same defaults as `distillation_teddymer_multihead.yaml` (Teddymer dataset with geometry-only filter, DDP `find_unused_parameters=true` + `static_graph=true`, WandB fragment, rank-0 loguru gate). Swaps `model.head._target_` to the new `MultiHeadConfidence` path; exposes adaptor + pairformer hyperparameters.

The legacy `distillation_teddymer_multihead.yaml` stays in-tree for A/B comparison.

**`scripts/train_confidence_teddymer_qg_multihead.sbatch`** — copy of `scripts/train_confidence_teddymer_pae.sbatch` with `--config-name=confidence/distillation_teddymer_qg_multihead` and per-config `run_name`. Same multi-node h100 wiring (2×2 h100nvl), same venv tarball staging, same `RESUME_CKPT_PATH` contract, same `PYTORCH_ALLOC_CONF=expandable_segments:True` and `NCCL_IB_DISABLE=1` exports.

### 5.4 Tests (written first, TDD)

The implementer writes tests **before** the corresponding implementation. Required tests:

1. **`tests/unit/community_models/boltz/test_pairformer_layer_smoke.py`** — instantiate one `PairformerLayer(s_dim=384, z_dim=128, num_heads=16)`, forward random `(s, z, mask)` of shape `(2, 128, 384)` / `(2, 128, 128, 128)` / `(2, 128)`, assert output shapes, dtypes, and mask preservation (padded rows/cols zero).

2. **`tests/unit/nn/confidence/test_adaptor_module.py`** — instantiate `AdaptorModule`, forward `(trunk_seqs[2,n,768], trunk_pair[2,n,n,256], local_latents[2,n,8], ca_coords[2,n,3], mask[2,n])`, assert output shapes `(2,n,384)` / `(2,n,n,128)`. Properties:
   - **LayerNorm-bias-leak guard:** set every `LayerNorm.bias` in the adaptor to `0.3` (drift simulation), feed input differing only at padded positions, assert outputs are bit-identical at all positions.
   - **Distogram correctness:** synthetic `ca_coords` with pre-computed pairwise distances; assert the one-hot distogram bins exactly match a fp64 numpy reference (`bin_limits = linspace(0.1, 3.0, n_bins - 1)` then `bucketize`).
   - **Single-attention-block residual init:** with the upstream zero-init on `attn.proj_o`, `s_mlp[1].weight`, `z_mlp[1].weight`, the block is a near-identity transform of the linearly-projected `(s, z)` to within `1e-6`.
   - **Translation equivariance:** with `n_attn_layers=1` (the production setting), translate `ca_coords` by a constant; adaptor output bit-identical. Holds because `s` depends only on `trunk_seqs + local_latents` (no coordinate input) and `z` depends on `ca_coords` only through pairwise distances. The `AttentionPairBias` block is not coordinate-aware.

3. **`tests/unit/nn/confidence/test_multi_head_qg_backbone.py`** — instantiate the new `MultiHeadConfidence` with adaptor + 4-layer stack + both `PlddtHead` and `PaeHead` at the new dims; forward at `B=2, L=64`; assert:
   - Output dict has keys `{plddt, pae}` with shapes `[B,L,n_plddt_bins]` and `[B,L,L,n_pae_bins]`.
   - `expected_trunk_eval_t` mismatch fires the existing assertion.
   - `weight=0.0` short-circuit: `loss_weights.pae = 0.0` with `find_unused_parameters=True, static_graph=True` runs one optimisation step without error; construction-time `RuntimeWarning` is emitted.
   - `output_name_root` non-empty assertion fires for empty input.

4. **`tests/integration/confidence/test_trunk_intermediates_ca_coords.py`** — `LocalLatentsTransformer(expose_intermediates=True)` exposes `trunk_intermediates["ca_coords"].shape == (B, n_extended, 3)`; `expose_intermediates=False` does not expose `trunk_intermediates` at all.

5. **`tests/regression/test_flow_matching_loss_unchanged.py`** — existing; must remain green.

6. **`tests/smoke/test_qg_multihead_one_step.py`** — one-step train on a 4-sample Teddymer micro-batch using the new config. Loss finite, grad norm > 0, optimizer step succeeds. Val metrics dict must contain (at minimum) `val/multi/total`, `val/plddt/total`, `val/pae/total`, `val/plddt/pearson`, `val/pae/pearson`, `val/plddt/mae`, `val/pae/mae`. (The full key set is enumerated in `MultiHeadConfidence`'s log contract; this test pins the headline keys only.)

### 5.5 Implementation order (vertical slice → horizontal expansion)

1. Vendor Boltz-1 minimal slice; write `test_pairformer_layer_smoke.py`; make it pass.
2. Port `AdaptorModule`; write `test_adaptor_module.py` (4 properties); make them pass.
3. Extend `LocalLatentsTransformer.trunk_intermediates` with `ca_coords`; write `test_trunk_intermediates_ca_coords.py`; make it pass; verify `test_flow_matching_loss_unchanged.py` still green.
4. Retarget `PlddtHead` / `PaeHead` for new `d_in_token` / `d_in_pair_token` kwargs (default-preserving); verify existing head tests still green.
5. Swap `MultiHeadConfidence` backbone; write `test_multi_head_qg_backbone.py`; make it pass.
6. Add the new config + sbatch; run `test_qg_multihead_one_step.py`; make it pass.
7. Smoke-launch on 1 h100 node, 4 GPU, ~1000 steps; confirm the convergence trajectory.

### 5.6 Reviewer panel

Mandatory:

- `code-review-debug-complexity-expert` (mandatory on every PR per CLAUDE.md).
- `ml-protein-architect` (confidence-distill subsystem layout + Hydra configs).
- `ml-software-pytorch-jax-expert` (vendored triangular-attention modules, DDP + checkpoint interaction, dim swaps + memory budget).
- `generative-protein-scientist` (the architectural recipe is a generative-modeling design call).

All four must approve; loop until convergence; stuck-PR escape hatch (terminate + report + email `schekmenev@aithyra.at`) per CLAUDE.md.

## 6. PR #2 — pAE-derived metric MAE + Pearson R

### 6.1 Branch + sequencing

Branch `feat/pae-derived-metric-correlation` off `dev`. **Sequenced after PR #1 merges to `dev`**: this PR refactors `PaeHead.compute_loss_and_metrics`, which PR #1 also touches (for the dim swap). Opening PR #2 in parallel risks a merge conflict on the same method.

### 6.2 Scope

For each of the 10 pAE-derived metrics in `nn/confidence/_metrics.py` (`i_pae`, `min_ipae`, `i_ptm`, `i_ptm_energy`, six ipSAE variants), additionally log per-sample-paired MAE, Pearson R, and Spearman between GT-pAE-computed metric and predicted-pAE-computed metric across the val set. Existing scalar `mean_pred` logs stay.

### 6.3 Per-sample-paired accumulation under DDP (Option A)

Use `torchmetrics.MetricCollection` with `PearsonCorrCoef` / `MeanAbsoluteError` / `SpearmanCorrCoef`. Torchmetrics handles DDP sync. Hand-rolled `all_gather` is rejected (Option B in design).

`PaeHead.__init__` instantiates `self.val_metric_correlations = nn.ModuleDict({m_name: MetricCollection({"mae": MeanAbsoluteError(), "pearson": PearsonCorrCoef(), "spearman": SpearmanCorrCoef()}) for m_name in METRIC_NAMES})`. `.update(M_pred, M_gt)` is called per val step from `compute_loss_and_metrics`. The Lightning module's `on_validation_epoch_end` consumes `.compute()` and logs under `val/pae/{m_name}/{mae,pearson,spearman}`. Train-time path accumulates nothing.

### 6.4 `M_pred` from the predicted PAE distribution

`PaeHead` predicts a per-pair softmax over PAE bins. The reduction to scalar PAE is pinned by the CLAUDE.md community-metric-parity rule: **expected-value PAE**, `PAE_pred[i,j] = sum_k p[i,j,k] * bin_center[k]`. This matches colabdesign's `get_ipsae_loss` and AlphaFold/Boltz public APIs. Argmax and sampling are rejected.

The expected-value reduction runs in **fp32** at runtime; the parity test runs in fp64 with `1e-5` tolerance.

### 6.5 New code

**`src/proteinfoundation/nn/confidence/pae_head.py`** — minimal additions:

- `__init__` accepts `track_metric_correlations: bool = True` (off-switch for dev iteration). When True, instantiates `self.val_metric_correlations` as a `nn.ModuleDict` of per-metric `MetricCollection`s.
- Private helper `_expected_value_pae(self, pae_logits) -> Tensor` (fp32, output shape `[B, L, L]`).
- `compute_loss_and_metrics(out, batch, mask_eff, *, stage="val")` adds, when `stage == "val"`:
  - `pae_pred = self._expected_value_pae(out["pae"])`.
  - For each metric `M_fn` in the existing 10-metric set, compute `M_pred = M_fn(pae_pred, batch, mask_eff, reduce="per_sample")` and `M_gt = M_fn(batch["pae"], batch, mask_eff, reduce="per_sample")`.
  - `self.val_metric_correlations[m].update(M_pred, M_gt)`.
  - Log `mean_pred = M_pred.mean()` (existing) and `mean_gt = M_gt.mean()` (new) under `val/pae/{m}/mean_{pred,gt}`.

**Refactor of `nn/confidence/_metrics.py`** — add `reduce: Literal["per_sample", "batch_mean"] = "batch_mean"` to each of the 10 metric functions. `per_sample` returns `[B]`; `batch_mean` returns the existing scalar. Default unchanged — no caller breaks. The formula is unchanged; only the final reduction is parameterised, so the colabdesign-parity contract survives.

**`src/proteinfoundation/confidence/train_confidence.py` (Lightning module)** — `on_validation_epoch_end` reads `self.head.val_metric_correlations.compute()` (when the head exposes it) and logs the dict via `self.log_dict(...)`. The multihead case reads from each child head that has the attribute.

### 6.6 Tests

1. **`tests/unit/nn/confidence/test_pae_head_metric_correlations.py`** — three cases:
   - **Identity:** `pae_logits` constructed so expected-value PAE exactly equals `batch["pae"]`. MAE == 0, Pearson R == 1.0 (or `nan` when variance is zero — assert explicitly).
   - **Constant-offset:** predicted PAE = GT + c. For translation-equivariant metrics (`i_pae`): MAE == c, Pearson R == 1.0. For non-equivariant metrics (ipSAE family): MAE > 0, Pearson R well-defined.
   - **Random:** large `(pae_pred, pae_gt)` random; compare against fp64 numpy reference within `1e-5`.

2. **`tests/unit/nn/confidence/test_expected_value_pae_parity.py`** — fp64 numpy reference of `softmax(pae_logits) → sum(p * bin_centers)`; runtime fp32 path matches within `1e-5` on random logits.

3. **`tests/unit/nn/confidence/test_metrics_per_sample_reduction.py`** — for each of the 10 metric functions: `M(pae, batch, mask, reduce="per_sample").mean() ≈ M(pae, batch, mask, reduce="batch_mean")` within `1e-6`. Pins that per-sample is a strict refinement of batch-mean (no semantic drift).

4. **`tests/integration/confidence/test_multi_head_val_epoch_logs_correlations.py`** — one full val epoch on 8 synthetic Teddymer-shaped samples; assert logged keys include `val/pae/{m}/{mae,pearson,spearman,mean_pred,mean_gt}` for every `m`.

5. **`tests/integration/confidence/test_metric_collection_ddp.py`** — parametrised over `world_size in {1, 2}` (skip when CUDA unavailable or `world_size > available GPUs`). Rank-0 epoch-end value equals single-process aggregation of the same val set.

### 6.7 Reviewer panel

Mandatory:

- `code-review-debug-complexity-expert` (mandatory).
- `generative-protein-scientist` (expected-value-PAE choice + scientific interpretation of pair-metric correlation).
- `ml-software-pytorch-jax-expert` (`torchmetrics.MetricCollection` DDP sync, fp32-vs-fp64 reductions, padding handling).

`ml-protein-architect` is not on the panel unless the PR ends up touching the config tree beyond a one-line `track_metric_correlations:` knob.

## 7. Risks and mitigations

- **Risk: Boltz-1 vendored code accidentally pulls a non-pairformer transitive dependency.** Mitigation: investigation report ships the explicit file manifest; implementer copies only those files; CI test `test_pairformer_layer_smoke.py` instantiates the layer in isolation from anything else.
- **Risk: dim swap to 384/128 violates a Hydra config assumption elsewhere in `configs/confidence/`.** Mitigation: only the new `distillation_teddymer_qg_multihead.yaml` opts into the new backbone. Existing configs keep their existing trunk + dims.
- **Risk: torchmetrics `PearsonCorrCoef` DDP sync silently averages instead of correctly concatenating per-sample tensors.** Mitigation: the DDP-correctness test (`test_metric_collection_ddp.py`) compares 1-GPU vs 2-GPU aggregation against the same single-process reference. If torchmetrics is mis-syncing we catch it before launch.
- **Risk: the rewrite does NOT fix the 0.78 floor — the leak (if any) was upstream of the head.** Mitigation: the Phase 0 leak audit reduces this risk before PR #1 lands. If the audit found no leak and the rewrite also starts at 0.78, we have isolated the problem to the trunk-output information content, which is a separate investigation.
- **Risk: PR sequencing — opening PR #2 before PR #1 merges produces a merge conflict on `pae_head.py`.** Mitigation: explicit sequencing rule. PR #2 opens after PR #1 merges to `dev`.
- **Risk: backwards-compat break in `MultiHeadConfidence` for existing runs.** Mitigation: existing config `distillation_teddymer_multihead.yaml` stays in-tree. New config opts into the new backbone. The `model.head._target_` change is opt-in per config.

## 8. Verification and acceptance

### Phase 0
- Report exists at the spec path with all four numbered sections populated.
- Investigation findings reviewed by main thread + user.

### PR #1
- All seven test files in §5.4 pass locally with the pinned `.venv/`.
- `tests/regression/test_flow_matching_loss_unchanged.py` still green.
- Smoke train on 1 h100 node × 4 GPU runs for 1000 steps without OOM, NCCL deadlock, or DDP "marked as ready twice" / "parameters not used" errors.
- All four reviewers approve.

### PR #2
- All five test files in §6.6 pass locally with the pinned `.venv/`.
- The 1000-step smoke run from PR #1's verification, re-run with PR #2 applied, logs every `val/pae/{m}/{mae,pearson,spearman,mean_pred,mean_gt}` key in WandB.
- All three reviewers approve.

## 9. Out-of-scope follow-ups

- Pretrained Boltz-1 pairformer weight loading. If the from-scratch init does not match quality-graft's convergence, this becomes a separate PR with a documented checkpoint source.
- Per-head filter decoupling under `MultiHeadConfidence` (deferred from PR #22; see `confidence_distill_followup_open.md` memory).
- Pinning the head's deployment target (AF2-mimic vs calibrated estimator) — same deferred memory.
- Single-head configs (`PlddtHead`-alone, `PaeHead`-alone) keep using `ConfidenceTrunk`; porting them to the qg-style backbone is not part of this design.
