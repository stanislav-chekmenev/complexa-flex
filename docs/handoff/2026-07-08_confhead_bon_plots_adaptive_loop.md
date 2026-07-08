# Handoff — Confhead BoN: plot/timing bug fixes + adaptive-loop feature + two-curve plot

**Date:** 2026-07-08. **Branch:** `confhead_best_of_n_search` (off `train_blob_v2`; **not pushed; this session's work is UNCOMMITTED in the working tree; no PR**).
Supersedes the in-flight parts of `2026-07-07_confhead_best_of_n_search.md` (which is archived this `/ul`). Prior status: [[confhead_bon_search_impl_status]]. Parity caveat: [[confhead_parity_range_restriction]].

## 1. TL;DR
The evaluate-population fixes from the prior session are committed (`ee427df` + `ddecf3b`) and run 85994 (PDL1) reached analyze + rendered the first plot. **This session** found and fixed **three bugs in the successes-vs-GPU-hours plot/timing artifact**, added an **adaptive generate/evaluate loop** (3000 → +1000/round until 100 AF2 successes or 16h), and made the plot show **two success curves**. All reviewed T3 (code-review: no blocking; generative-protein-scientist: 3 methodology blockers, all resolved). **Everything this session is uncommitted.** Not run on GPU (adaptive loop), no PR.

## 2. What shipped this session (all UNCOMMITTED, TDD, 52 tests green)

**Three bug fixes (single-round plot + timing):**
1. **Join blowup / flat curve.** `metadata_tag = bon_orig{s}_r{r}` collided across ~63 search iterations → plot `pd.merge` fanned 254×63 = 15860 rows → curve flat at 35 from 0.4–1.5h. Fixed: iteration-unique tags `bon_it{i}_orig{s}_r{r}` (per-instance counter in `best_of_n_search.py`) → clean 1:1 merge for new runs; legacy colliding-tag runs use a documented best-effort positional join. **Legacy-join limitation (documented in `_positional_join` docstring):** `id_k` is a per-length counter and the sidecar has no length column, so multi-length legacy buckets can mispair (120/254 rows on 85994 are in ambiguous buckets). Affects re-plots of pre-fix data only; new runs are exact.
2. **GPU-hours axis missing AF2 eval time.** `elapsed_gpu_hours` was generation-only (per-batch stamp, `predict_time_budget.py`); AF2 eval is a separate later stage with only aggregate timing. Fixed: `evaluate.py` writes per-sample `eval_finish_s`; plot x = gen_hours + eval_finish_s/3600, degrading to generation-only for legacy runs without the column.
3. **Parity ~0 is an artifact.** Fixed the join so parity n = 254 (not 15860); rescaled AF2 ipAE to Å for display; report both Spearman + Pearson. The number is now caveated (`valid_head_quality_evidence: false` + on-figure footnote) because it is range-restricted — see [[confhead_parity_range_restriction]].

**Gate split (owner decision):** head provisional gate `SUCCESS_PLDDT_01 = 0.92`; AF2 final/headline/target gate = canonical **0.90** AlphaProteo. Two distinct gates — now a CLAUDE.md invariant (confidence-distill section).

**Two success curves (owner spec):** (1) full AF2 AlphaProteo (0.90) headline; (2) `head_plus_scrmsd` = head provisional_success (head ipAE<7 & pLDDT>0.92) AND AF2-multimer scRMSD<1.5. The old interface-blind AF2-scRMSD-only diagnostic was removed. On 85994: curve1=35, curve2=248.

**Adaptive loop feature:** round 1 = 3000 samples, +1000/round until 100 AF2-confirmed uniques (0.90 gate) OR 16h cumulative wall-clock (checked between rounds; a round in flight finishes → can overshoot by one round). Multi-round plot stitches per-round timelines onto one cumulative-GPU-hours axis with both curves.

## 3. What is on disk (working tree, uncommitted; do not recreate)
- **Modified:** `src/proteinfoundation/search/best_of_n_search.py` (unique tags), `src/proteinfoundation/evaluate.py` (`_attach_eval_finish_times` → `eval_finish_s`), `src/proteinfoundation/confidence/inference_scorer.py` (`SUCCESS_PLDDT_01 = 0.92`), `script_utils/plot/plot_successes_vs_gpu_hours.py` (2-mode join, two gates, parity caveat, honest Foldseek annotation, eval-time on x), `tests/unit/plot/test_plot_successes_vs_gpu_hours.py`.
- **New:** `src/proteinfoundation/search/adaptive_bon_loop.py`, `configs/search_binder_confhead_adaptive_pipeline.yaml`, `scripts/search_binder_confhead_adaptive.sbatch`, `script_utils/plot/plot_successes_vs_gpu_hours_multiround.py`, and tests: `tests/unit/search/test_adaptive_bon_loop.py`, `test_adaptive_pipeline_config.py`, `test_best_of_n_tags_unique.py`, `tests/unit/plot/test_plot_join_and_gates.py`, `test_plot_successes_vs_gpu_hours_multiround.py`, `tests/unit/evaluation/` (eval-finish test).
- **Verified artifacts:** re-plot of legacy 85994 at `/tmp/twocurve_verify.png` (curve1=35, curve2=248, both rise, no plateau).
- **Checkpoints (reachable):** trunk `ckpts/complexa.ckpt` + `ckpts/complexa_ae.ckpt`; multihead `/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt` (sbatch default `CONF_CKPT_PATH`).
- **NOT ours — do NOT commit** (carried from `debug_fast_convergence`): `scripts/debug/`, `scratch/`, `src/proteinfoundation/confidence/probes/`, `tests/integration/confidence/test_forward_refactor_bit_identical.py`, `test_frozen_probe_smoke.py`, `tests/unit/confidence/probes/`. Also a pre-existing unrelated dirty `scripts/train_confidence_teddymer_multihead.sbatch`. And 24 whitespace-churn `community_models/colabdesign/*` files (pre-session).

## 4. What is NOT on disk yet (the gap)
1. **Commit this session's work** (explicit paths — see §3; never `-A`).
2. **Adaptive-loop GPU run** to completion on a real target; render the multi-round plot. Never run on GPU yet — only CPU control-logic tests.
3. **Valid full-pool stratified parity experiment** ([[confhead_parity_range_restriction]]): ~200–300 AF2 refolds across confidence strata (head-accepted AND head-rejected) + precision-at-gate confusion. Deferred follow-up.
4. **`score_concat_reuse` wiring** into generation (unchanged from prior handoff; `ipae_reuse` column still absent).
5. **PR into `dev`.**

## 5. Architecture reference (mirror these)
- Iteration counter + tag: `src/proteinfoundation/search/best_of_n_search.py` (`self._iteration`, `bon_it{i}_orig{s}_r{r}`).
- Two-mode join: `script_utils/plot/plot_successes_vs_gpu_hours.py::join_analyze_and_sidecar` / `_positional_join`; gates `_GATES`, `af2_confirmed_mask(df, gate=...)`; parity `compute_parity` + `_PARITY_RANGE_RESTRICTION_CAVEAT`.
- Per-sample eval time: `src/proteinfoundation/evaluate.py::_attach_eval_finish_times` (wraps `run_binder_evaluation`).
- Adaptive loop control (CPU-testable, injected `run_round`): `src/proteinfoundation/search/adaptive_bon_loop.py::run_adaptive_loop`; AF2 gate `af2_confirmed_mask` uses `ADAPTIVE_AF2_SUCCESS_THRESHOLDS = DEFAULT_PROTEIN_BINDER_THRESHOLDS` (0.90).
- Multi-round curves: `script_utils/plot/plot_successes_vs_gpu_hours_multiround.py::build_multiround_curves` (imports the single-round gate logic to stay in lockstep).
- Canonical parity constant (never mutate): `src/proteinfoundation/result_analysis/binder_analysis_utils.py::DEFAULT_PROTEIN_BINDER_THRESHOLDS`.

## 6. PR plan
**One T3 PR into `dev`** (force-T3: touches eval semantics, timing/reproducibility, community-parity-adjacent success gates, and a new search subsystem).
- **Scope:** the §3 modified + new files. Exclude the §3 "NOT ours" debug/churn files — stage by explicit path.
- **Tests first:** already satisfied (52 green).
- **Panel:** `code-review-debug-complexity-expert` (mandatory) + `generative-protein-scientist` (gate-split rationale, parity methodology) + `ml-software-pytorch-jax-expert` (eval timing, subprocess/Hydra isolation in the loop). This session already ran that panel on the working tree: code-review = no blockers; scientist = 3 blockers, all fixed. Re-run only reviewers whose domain code changes further.
- **Note:** confirm base branch with the user (branch is off `train_blob_v2`, not `dev`).

## 7. Open questions for the user
1. **Target branch / rebase:** land into `dev`? Branch is off `train_blob_v2`.
2. **Run the adaptive loop now** on which target(s), and confirm 100 / 16h defaults?
3. **Prioritise the full-pool parity study** (§4.3) before or after the PR?
4. **`score_concat_reuse`** — wire now or defer? (Unchanged open item.)

## 8. Reference paths
- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Configs: `configs/search_binder_confhead_local_pipeline.yaml`, `configs/search_binder_confhead_adaptive_pipeline.yaml`
- Source: `src/proteinfoundation/search/{best_of_n_search,adaptive_bon_loop}.py`, `src/proteinfoundation/{evaluate,confidence/inference_scorer}.py`
- Scripts: `scripts/search_binder_confhead{,_adaptive}.sbatch`; plots `script_utils/plot/plot_successes_vs_gpu_hours{,_multiround}.py`
- Confidence ckpt: `/netscratch/schekmenev/complexa-paedistill/runs/78768/checkpoints/multi-007-2.9509.ckpt`
- venv tarball: `$PROJECT_ROOT/venv.tar.gz` + `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`
- Legacy run outputs: `evaluation_results/search_binder_confhead_local_pipeline_02_PDL1_bon_02_PDL1_85994/`

## 9. Launch command (adaptive loop — needs a GPU node)
```
sbatch --export=ALL,TARGET_TASK=02_PDL1 scripts/search_binder_confhead_adaptive.sbatch
```
Overrides: `TARGET_SUCCESSES` (100), `TIME_BUDGET_HOURS` (16, checked between rounds), `ROUND1_NSAMPLES` (3000), `ROUND_NSAMPLES` (1000), `BASE_SEED`, `CONF_CKPT_PATH`. Size SLURM `--time` against worst-case (round-1 + one extra round + AF2), not 16h.

## 10. Project conventions reminder
- Branch off `dev` (main); PRs into `dev`. No Claude / AI attribution in commits or PRs.
- `module load gh` in the same Bash call before any `gh`.
- Never `uv run` / `uv sync` / `uv lock` — use `.venv/bin/python` (or staged `$ENV_LOCAL/bin/python`) directly; sbatch stages the venv from the tarball, cross-partition `/netscratch` fallback mandatory.
- TDD via subagents; tests before implementation.
- Confidence-head gate rule: head provisional 0.92 vs AF2 reporting 0.90 are distinct — never collapse (see CLAUDE.md).
