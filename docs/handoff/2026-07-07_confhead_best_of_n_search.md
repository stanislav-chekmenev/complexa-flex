# Handoff — Confidence-head best-of-N binder search

**Date:** 2026-07-07 (created); **updated 2026-07-08** (feature + 3 bug fixes committed).
**Branch:** `confhead_best_of_n_search` (off `train_blob_v2`; **not pushed; committed but not PR'd**).

## 1. TL;DR

Best-of-N binder search scored by the **trained confidence head** instead of an AF2 fold, so
the search loop produces provisional successes with no folding-model call. **Committed 2026-07-08
in two commits:** `ee427df` (feature: scorer, predict_step hook, sidecar CSV, time-budget callback,
config, plot, sbatch, tests) and `ddecf3b` (three bug fixes found on the first real GPU run + their
tests). **In flight:** a full clean GPU run (job **85994**, PDL1) is running on gpu01 to verify all
three fixes end-to-end. **Not done:** `score_concat_reuse` is implemented but not wired into
generation; no PR opened into `dev` yet.

## 2. Three bugs fixed this session (were live in the first GPU run, job 85933)

The first real GPU run (85933) exposed that the pipeline was **not** doing what the head was
designed for. All three are fixed in `ddecf3b`:

1. **Evaluate ran on the wrong population.** `filter.py` selected top-1000-by-`total_reward` and
   moved the rest to `filtered_out_samples/`; evaluate scans `job_*` dirs at the run root, so it
   AF2-refolded the top-1000 pool, **not** the confidence-head successes. Confirmed: 2000 gen → 300
   `provisional_success` → 1000 sent to AF2 (738 failures folded, 38 real successes discarded). Fix:
   `filter.select_top_samples` gates to `provisional_success > 0.5` **before** the top-N cap when the
   column is present; legacy path (no column) byte-identical to the historical top-N selection.
2. **ipAE reduction min → mean.** The scorer's `provisional_success`/`total_reward` used `min_ipae`
   (min over the interface), but the AF2 analyze success gate uses `i_pae` (**mean** over the
   interface = AlphaProteo/colabdesign protein-success `i_pAE`). Owner decided **mean**. Scorer now
   calls the verbatim-ported `i_pae`; pinned by an fp64 parity test. Both Angstroms, no ×31.
3. **ESMFold starved the AF2-scRMSD budget.** The default `binder_evaluate.yaml` runs
   `compute_monomer_metrics: true` (ESMFold designability + codesignability) + `compute_esm_metrics:
   true` on **every** sample as a stage **before** the binder AF2 refold. On 85933 this consumed the
   entire 16h budget (849 esmfold batches; **zero** colabdesign in `evaluate.log`), so the AF2
   complex scRMSD — the one metric the head can't supply — never ran. Fix: the confhead config now
   has a `metric:` override disabling monomer/esm/designability/codesignability/ss and keeping only
   `binder_folding_method: colabdesign` → scRMSD. Full contract in memory
   `confhead_evaluate_scrmsd_only`.

**Relaunch chain:** 85933 (buggy, killed) → 85993 (killed before evaluate, still old config) →
**85994** (running, all three fixes). A Monitor is armed to report at the evaluate stage: population
≈ successes (not 1000) **and** AF2 runs / esmfold off.

## 3. Key design decision (do not re-litigate — verified 3 ways)

The confidence head's **interface ipAE is OOD on the binder-gen concat frame**; the primary score
re-presents the generated complex in the **native co-diffused frame** (Alt B): decode → assemble
`[binderA|targetB]` as regular residues (no `x_target`), re-encode, run the head at `t=0.99`.
Full rationale + file:line: memory `confhead_interface_extended_frame`. ipAE in **Angstroms, no
×31**; pLDDT EV 0–100 → /100 for the 0.9 threshold.

## 4. What is on disk (committed on `confhead_best_of_n_search`; do not recreate)

Committed in `ee427df` (feature) and `ddecf3b` (fixes):
- `src/proteinfoundation/confidence/inference_scorer.py` — `ConfidenceHeadScorer`:
  `score_native` (primary), `_reduce` (pure; now MEAN `i_pae`), `score_concat_reuse` (ablation,
  unwired), `from_module`, `_build_native_batch`, `_build_head`.
- `src/proteinfoundation/utils/predict_time_budget.py` — `PredictTimeBudget` Lightning callback
  (raises `StopIteration` in `on_predict_batch_start`; Lightning 2.5 predict loop ignores
  `trainer.should_stop`).
- `configs/search_binder_confhead_local_pipeline.yaml` — binder_generate (`search.algorithm:
  best-of-n`, `reward_model: null`, `confidence_scorer:{...}`, `time_budget_hours:16`,
  `nsamples:1000`); a `metric:` override (monomer/esm/designability OFF, colabdesign binder refold
  ON → scRMSD); an `aggregation.success_thresholds` override (analyze gate = `binder_scRMSD_ca < 1.5`).
- `src/proteinfoundation/filter.py` — `select_top_samples` with the `provisional_success` gate.
- `script_utils/plot/plot_successes_vs_gpu_hours.py`, `scripts/search_binder_confhead.sbatch`.
- Tests: `tests/unit/test_filter_success_gate.py`, `tests/unit/confidence/test_inference_scorer_ipae_is_mean.py`,
  `test_inference_scorer_reduction.py`, `tests/integration/confidence/test_inference_scorer_score_native.py`,
  `test_predict_step_confidence_wiring.py`, `test_predict_time_budget.py`,
  `test_save_predictions_confidence_sidecar.py`, `tests/unit/plot/`, `test_colabdesign_jax09_compat.py`.

Modified + committed: `src/proteinfoundation/proteina.py` (predict_step hook), `generate.py`
(sidecar CSV + `PredictTimeBudget`/`limit_predict_batches` on the predict Trainer).

Checkpoints (reachable): trunk `ckpts/complexa.ckpt` + `ckpts/complexa_ae.ckpt`; trained multihead
`/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt` (sbatch
default `CONF_CKPT_PATH`; the sbatch stages it to `/netscratch/schekmenev/complexa-bon/ckpt/`).

## 5. What is NOT on disk yet (the gap)

1. **A completed clean GPU run.** 85994 is in flight. When it finishes: confirm evaluate's
   population == `provisional_success` count (NOT 1000), that `evaluate.log` shows colabdesign/AF2
   and NOT esmfold, and that the plot + parity JSON rendered. Logs: `logs/bon_85994.{out,err}` and
   `logs/design_pipeline_02_PDL1_bon_02_PDL1_85994_*/`.
2. **`score_concat_reuse` wiring into generation.** Method exists + is unit-safe, but generation-time
   concat trunk intermediates are not threaded out of `full_simulation` → search → `predict_step`.
   Until wired, the `ipae_reuse` ablation column is absent and the parity study has only the
   native-frame arm.
3. **No PR yet.** Committed on the branch; not pushed, not opened into `dev`.

## 6. Architecture reference (mirror these)

- Scorer frozen-forward mirrors `src/proteinfoundation/confidence/lightning_module.py` `_forward`
  minus the training-only label-mask/trim.
- Interface math: `src/proteinfoundation/nn/confidence/_metrics.py` `interface_pair_mask`, `i_pae`
  (mean — the one the scorer now uses), `min_ipae`, `reduce="per_sample"`.
- Filter selection + gate: `src/proteinfoundation/filter.py::select_top_samples`.
- Evaluate stage: `binder_folding_method` / monomer metrics wired in
  `src/proteinfoundation/evaluation/monomer_eval.py` + `configs/pipeline/binder/binder_evaluate.yaml`;
  AF2 binder refold in `src/proteinfoundation/utils/colabdesign_utils.py::predict_binder_complex`;
  scRMSD in `src/proteinfoundation/metrics/binder_metrics.py::calculate_prot_prot_binder_rmsd`.
- Analyze success thresholds: `src/proteinfoundation/result_analysis/binder_analysis_utils.py`
  `DEFAULT_PROTEIN_BINDER_THRESHOLDS`.

## 7. PR plan

When landing into `dev` (branch is off `train_blob_v2`; confirm base with the user). Likely **one
T3 PR** (force-T3: touches inference/search *semantics*, new inference-time scoring subsystem, a
`.ckpt` load path). Tests-first satisfied.
- **Scope:** scorer + predict_step hook + CSV/timebudget plumbing + filter gate + config + plot + sbatch.
- **Panel:** `code-review-debug-complexity-expert` (mandatory) + `generative-protein-scientist`
  (native-frame in-distribution argument, parity study, `total_reward=-ipae` mean rank key) +
  `ml-software-pytorch-jax-expert` (frozen no_grad forward, Lightning predict `StopIteration` cutoff,
  `load_from_checkpoint(head=...)`).
- **Verify before review:** the 85994 GPU run green end-to-end; run the parity ablation on ≥1 target.

## 8. Open questions for the user

1. **Target branch / rebase:** land into `dev`? Branch sits off `train_blob_v2`.
2. **Native-frame ipAE calibration vs AF2** — acceptable? The parity study answers this once a full
   run lands. (Min-vs-mean reduction is settled: **mean**, matching AlphaProteo `i_pAE`.)
3. **Wire `score_concat_reuse` now or defer?** Needs threading gen trunk intermediates through search.

## 9. Reference paths

- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Config: `configs/search_binder_confhead_local_pipeline.yaml`
- Source: `src/proteinfoundation/confidence/inference_scorer.py`, `.../filter.py`, `.../utils/predict_time_budget.py`
- Scripts: `scripts/search_binder_confhead.sbatch`; plot `script_utils/plot/plot_successes_vs_gpu_hours.py`
- Confidence ckpt: `/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt`
- venv tarball: `$PROJECT_ROOT/venv.tar.gz` + `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`
- Current run logs: `logs/bon_85994.{out,err}`

## 10. Project conventions reminder

- Branch off `dev` (main); PRs into `dev`. No Claude / AI attribution in commits or PRs.
- `module load gh` in the same Bash call before any `gh`.
- Never `uv run` / `uv sync` / `uv lock` — use `.venv/bin/python` (or staged `$ENV_LOCAL/bin/python`)
  directly; sbatch stages the venv from the tarball, cross-partition `/netscratch` fallback mandatory.
- TDD via subagents; tests before implementation.
- Carried-over from `debug_fast_convergence` (untracked, **do NOT commit**): `scripts/debug/`,
  `scratch/`, `src/proteinfoundation/confidence/probes/`, `tests/.../test_forward_refactor_bit_identical.py`,
  `test_frozen_probe_smoke.py`, `tests/unit/confidence/probes/`.
