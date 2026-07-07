# Handoff — Confidence-head best-of-N binder search

**Date:** 2026-07-07
**Branch:** `confhead_best_of_n_search` (off `train_blob_v2`; not pushed; all work uncommitted)

## 1. TL;DR

Best-of-N binder search scored by the **trained confidence head** instead of an AF2 fold, so
the search loop produces provisional successes with no folding-model call. **Shipped
(uncommitted, 21 tests green + real-checkpoint CPU smoke green):** `ConfidenceHeadScorer`, the
`predict_step` hook (AF2 bypassed in search, `total_reward=-ipae`), a per-sample sidecar CSV, a
16 GPU-hour `PredictTimeBudget` callback, the Hydra pipeline config, a Fig-21-style plot +
parity script, and the per-target sbatch. **In flight:** the full GPU pipeline smoke (job
**85858**, PDL1, tiny budget) is queued but **blocked on a free GPU slot**. **Not done:**
`score_concat_reuse` is implemented but not wired into generation (the ablation column is
absent until it is). Nothing is committed.

## 2. Key design decision (do not re-litigate — verified 3 ways)

The confidence head's **interface ipAE is out-of-distribution on the binder-gen concat frame**.
The head was distilled only on co-diffused native Teddymer dimers (both chains as regular
diffused residues in one frame, `n_concat=0`, cross-chain pair block from the normal
`pair_repr_builder`). Binder generation injects the target as **concat features** (extended
`[binder|target]`, cross-block from `ConcatPairFeaturesFactory` — a projection the head never
saw). So the primary score re-presents the generated complex in the **native co-diffused frame**
(Alt B): decode → assemble `[binderA|targetB]` as regular residues (no `x_target`), re-encode,
run the head at `t=0.99`. The cheap concat-latent reuse is kept only as an uncalibrated ablation
for a Spearman-vs-AF2 parity study. Full rationale + file:line: memory
`confhead_interface_extended_frame`; adjudication trace in this session (adversarial review +
generative-protein-scientist). ipAE compared in **Angstroms, no ×31**; pLDDT EV 0–100 → /100 for
the 0.9 threshold.

## 3. What is on disk (uncommitted on `confhead_best_of_n_search`; do not recreate)

New files:
- `src/proteinfoundation/confidence/inference_scorer.py` — `ConfidenceHeadScorer`:
  `score_native(complex_prots)` (primary), `_reduce(pae_ev, plddt_ev, chain_idx, mask)` (pure),
  `score_concat_reuse(...)` (ablation), `from_module(...)`, `_build_native_batch(...)`,
  `_build_head(...)` (rebuilds the multihead from Hydra config for `load_from_checkpoint(head=...)`).
- `src/proteinfoundation/utils/predict_time_budget.py` — `PredictTimeBudget` Lightning callback
  (raises `StopIteration` in `on_predict_batch_start`; Lightning 2.5 predict loop ignores
  `trainer.should_stop`).
- `configs/search_binder_confhead_local_pipeline.yaml` — binder_generate (`search.algorithm:
  best-of-n`, `reward_model: null`, `confidence_scorer:{enabled,ckpt_path:${oc.env:CONF_CKPT_PATH},
  trunk_eval_t}`, `time_budget_hours:16`, `nsamples:1000`) + binder_evaluate (AF2 = final filter)
  + binder_analyze.
- `script_utils/plot/plot_successes_vs_gpu_hours.py` — joins the analyze CSV with the sidecar on
  `metadata_tag`, plots cumulative unique **AF2-confirmed** successes vs `elapsed_gpu_hours`,
  emits a parity JSON (Spearman native-ipae/reuse-ipae vs AF2 ipae).
- `scripts/search_binder_confhead.sbatch` — 1 GPU, 20h wall, venv-tarball staging +
  cross-partition `/netscratch` fallback, stages `complexa.ckpt`/`complexa_ae.ckpt` + the
  confidence ckpt, validates `TARGET_TASK` against the 12 easy targets, runs `complexa design`,
  then the plot. Honours `TIME_BUDGET_HOURS`/`EXTRA_OVERRIDES` env for smoke runs.
- Tests (all green): `tests/unit/confidence/test_inference_scorer_reduction.py` (7),
  `tests/integration/confidence/test_inference_scorer_score_native.py` (2, incl. the batch-key
  regression guard), `test_predict_step_confidence_wiring.py`, `test_predict_time_budget.py`,
  `test_save_predictions_confidence_sidecar.py`, `tests/unit/plot/test_plot_successes_vs_gpu_hours.py` (6).

Modified (tracked, uncommitted): `src/proteinfoundation/proteina.py` (`predict_step` +
`_confidence_scorer_cfg`/`_get_confidence_scorer`/`_score_finals_with_confidence` helpers),
`src/proteinfoundation/generate.py` (sidecar CSV in `save_predictions`; `PredictTimeBudget` +
`limit_predict_batches` on the predict Trainer).

Checkpoints (reachable): trunk `ckpts/complexa.ckpt` + `ckpts/complexa_ae.ckpt`; trained
multihead `/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt`
(the sbatch default `CONF_CKPT_PATH`).

**Real-ckpt CPU smoke caught + fixed 3 bugs** (now pinned by the score_native regression test):
(1) `load_from_checkpoint` needs `head=` (head is an `nn.Module`, not rebuilt from hparams);
(2) `_build_native_batch` must set `coords` (Å), `coords_nm`, `coord_mask` [B,L,37], and `chains`
(=chain_idx, 2-valued chain-id — NOT a 0/1 mask); (3) `mask_dict["coords"]` must be [B,L,37,3]
because `ProductSpaceFlowMatcher.process_batch` derives the residue mask as `[...,0,0]`.

## 4. What is NOT on disk yet (the gap)

1. **Full GPU pipeline smoke result.** Job 85858 (`bon_smoke`, PDL1, `TIME_BUDGET_HOURS=0.05`,
   `nsamples=4`) is queued/pending — no logs yet. Generation needs CUDA (login node has none).
   When a GPU frees, check `logs/bon_85858.{out,err}`, then confirm the plot + parity JSON rendered.
2. **`score_concat_reuse` wiring into generation.** The method exists + is unit-safe, but the
   generation-time concat trunk intermediates are not threaded out of `full_simulation` → search →
   `predict_step`. Until wired, the `ipae_reuse` column is absent and the parity study has only the
   native-frame arm.
3. **No git commit.** All the above is uncommitted.

## 5. Architecture reference (mirror these)

- Scorer frozen-forward mirrors `src/proteinfoundation/confidence/lightning_module.py` `_forward`
  (~line 294 on this branch) minus the training-only label-mask/trim; reuses `_compute_cond` /
  `_pad_cond_to_n_ext`.
- Interface math: `src/proteinfoundation/nn/confidence/_metrics.py` `interface_pair_mask` (342),
  `min_ipae` (391), `i_pae` (358), `reduce="per_sample"`.
- Bin→EV: `pae_head.py::pae_ev_from_logits` (150), `plddt_head.py::logits_to_expected_value` (90).
- Joint complex the scorer consumes: `utils/sample_utils.py::prepend_target_to_samples` (95) —
  `final_prots` = `{coors Å, residue_type, chain_index (2-valued), mask}`; set for binder gen via
  `prepend_target=binder_gen_only` (`datasets/gen_dataset.py:597`).
- Filter gotcha: `filter.py:14` `dropna(subset=["total_reward"])` — hence `total_reward=-ipae`.
- Foldseek per-sample cluster id is in `clusters_binder_successful_self/cluster_assignments_*.csv`
  (cols `cluster_index,sample_index,path_name`, keyed by pdb_path), NOT in the analyze CSV; the
  plot auto-joins on pdb_path basename (`--cluster-col` override).

## 6. PR plan

When landing into `dev` (branch is currently off `train_blob_v2`; rebase/confirm target with the
user). Likely **one T3 PR** (force-T3 triggers hit: touches inference/search *semantics*, adds a
new inference-time scoring subsystem, and a `.ckpt` load path). Tests-first is already satisfied.

- **Scope:** the scorer + `predict_step` hook + CSV/timebudget plumbing + config + plot + sbatch.
- **Panel:** `code-review-debug-complexity-expert` (mandatory) + `generative-protein-scientist`
  (is the native-frame in-distribution argument + the parity study sound; is `total_reward=-ipae`
  the right best-of-N rank key) + `ml-software-pytorch-jax-expert` (the frozen no_grad/inference_mode
  forward, the Lightning predict-loop `StopIteration` cutoff, `load_from_checkpoint(head=...)`).
- **Verify before review:** the GPU smoke (gap #1) must be green end-to-end; run the parity ablation
  on ≥1 real target and report Spearman(native,AF2) vs Spearman(reuse,AF2).
- If `score_concat_reuse` wiring (gap #2) is deferred, ship the native-frame arm alone and note the
  ablation is a follow-up; do not leave a half-wired reuse path.

## 7. Open questions for the user

1. **Target branch / rebase:** land into `dev`? The branch sits off `train_blob_v2`; confirm the
   intended base before opening the PR.
2. **Is `total_reward=-ipae` the right best-of-N ranking key**, and is the native-frame ipAE
   calibration acceptable (the parity study answers this empirically once the GPU run lands)?
3. **Wire `score_concat_reuse` now or defer?** It needs threading generation trunk intermediates
   through the search stack (non-trivial).

## 8. Reference paths

- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Configs: `configs/` (pipeline: `configs/search_binder_confhead_local_pipeline.yaml`)
- Source: `src/proteinfoundation/confidence/inference_scorer.py`, `.../utils/predict_time_budget.py`
- Scripts: `scripts/search_binder_confhead.sbatch`; plot `script_utils/plot/plot_successes_vs_gpu_hours.py`
- Confidence ckpt: `/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt`
- venv tarball: `$PROJECT_ROOT/venv.tar.gz` + `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`
- Plan (this session): `~/.claude/plans/linear-discovering-tulip.md`
- Smoke job logs (when it runs): `logs/bon_85858.{out,err}`

## 9. Project conventions reminder

- Branch off `dev` (main); PRs into `dev`. No Claude / AI attribution in commits or PRs.
- `module load gh` in the same Bash call before any `gh`.
- Never `uv run` / `uv sync` / `uv lock` — use `.venv/bin/python` (or the staged
  `$ENV_LOCAL/bin/python`) directly; sbatch stages the venv from the tarball, cross-partition
  `/netscratch` fallback mandatory.
- TDD via subagents; tests before implementation (already satisfied here).
- Carried-over from `debug_fast_convergence` (untracked, **do NOT commit**): `scripts/debug/`,
  `scratch/`, `src/proteinfoundation/confidence/probes/`, `tests/.../test_forward_refactor_bit_identical.py`,
  `test_frozen_probe_smoke.py`, `tests/unit/confidence/probes/`. Its `lightning_module.py`
  `extract_frozen_features` refactor is stashed (`git stash@{0}`), NOT on this branch.
