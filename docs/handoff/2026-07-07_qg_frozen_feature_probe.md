# Handoff — QG/La-Proteina frozen-feature linear-probe diagnostic

**Date:** 2026-07-07
**Primary repo for the work:** `/mnt/storage01/home/schekmenev/projects/quality-graft` (SEPARATE from complexa-flex), branch `main`, all probe work **uncommitted**.
**This handoff lives in complexa-flex** (the session's home repo) because the diagnostic answers the deferred QG-side thread of the complexa frozen-feature probe.

## 1. TL;DR

Ported the Complexa frozen-feature linear-probe diagnostic to the QG/La-Proteina model to measure whether La-Proteina's frozen trunk features are linearly pLDDT-predictive. **Result: La-Proteina's linear pLDDT floor is ~0 across all variants** (vs Complexa's ~0.74). The user's "~15x slower to train" premise was **refuted by on-disk evidence** (the QG training run reached val Pearson 0.97 via a trained adaptor + Boltz pairformer — the signal is nonlinear, not a high linear floor), and the diagnostic was reframed to measure the floor honestly. **Shipped:** probe library + harness + tests (14 unit + heavy smoke green), full run (job 85849) complete, 3 comparison figures. **In flight:** everything is uncommitted in the quality-graft repo; nothing merged. **Not blocked** — the science is done; only the land/discard decision and the (optional) complexa-side PAE probe remain.

## 2. What is on disk (do not recreate)

**In quality-graft repo (`/mnt/storage01/home/schekmenev/projects/quality-graft`, branch `main`, all UNTRACKED unless noted):**
- `src/quality_graft/probes/__init__.py` + `frozen_feature_probe.py` — ported probe lib: `pool_pair_features`, `LinearProbe`, `fit_probe`, `ridge_probe_ev` (verbatim from complexa), `probe_metrics` (adapted: QG `training.metrics` `num_bins:int` API, not a centers tensor), `build_probe_features` (4 variants, no `n_orig` trim), `split_swissprot_for_probe` (reproduces QG val via `cluster_utils.split_dataframe(seed=42)`, then seeded random fit/eval carve). `ProbePartition/ProbeData` carry `num_bins:int`.
- `scripts/run_frozen_feature_probe.py` — Hydra harness; loads `LaProteinaWrapper.from_checkpoint(..., t_value=0.99, deterministic_encode=True)`, drives the datamodule in `local_only`, pre-filters each partition to readable `.pt` via `zipfile.is_zipfile`, logs per-partition skip counts + a val-approximation warning, runs `fit_probe`+`ridge_probe_ev` per variant, mirrors complexa's loguru result format.
- `scripts/run_frozen_feature_probe.sbatch` — h100 job; re-extracts venv from `venv.tar.gz` into `/netscratch/schekmenev/ffprobe_venv/.venv` (the pre-existing `/netscratch/schekmenev/.venv` was corrupt), reuses staged data + ckpts, hard-gates `import hydra,torch`.
- `scripts/plot_frozen_feature_probe_comparison.py` + `scripts/probe_figs/{headline_floor_comparison,la_proteina_learning_curves,floor_curves_overlay}.png` — the deliverable figures.
- `tests/test_frozen_feature_probe.py` (14 unit tests) + `tests/integration/test_frozen_feature_probe_smoke.py` (heavy).
- **`scripts/train.py` (MODIFIED, tracked):** `_build_swissprot_data_module` now reads `local_only` and passes it through (was a no-op → always non-local, which read a malformed CSV). Mirrors `_build_pdb_data_module`.
- Result log: `logs/ffprobe_85849.out` (COMPLETED, exit 0).

**Staged h100 `/netscratch` artifacts (only visible from the h100 partition store, NOT login01):**
- `/netscratch/schekmenev/ffprobe_venv/.venv` — freshly-extracted working venv (py3.11, torch 2.9, hydra 1.3.2).
- `/netscratch/schekmenev/swissprot_v4/` — 229,405 `.pt` + `processed/plddt_status.csv` + `df_swissprot_f1.0_minl30_maxl256.csv`. **~10% of the `.pt` are empty/truncated tars** (unrecoverable; the probe skips them).
- `/netscratch/schekmenev/ckpt/quality_graft/{LD1_ucond_notri_512.ckpt (2.93G), AE1_ucond_512.ckpt (4.10G), boltz1_conf.ckpt}` — La-Proteina wrapper needs the first two only.

**Measured result (job 85849; SwissProt monomers, caps 300/200/200, 3000 SGD steps, seed 0):**

| variant | La-Proteina val_eval (SGD) | ridge val_eval | Complexa val_eval |
|---|---|---|---|
| s_only | -0.015 | 0.057 | 0.731 |
| s_latents (768+8, adaptor input) | -0.015 | — | (no analogue) |
| z_pooled | 0.009 | — | 0.482 |
| s_z | -0.007 | 0.088 | 0.740 |

All La-Proteina learning curves flat at ~0 across 3000 steps; none approach 0.9. Split pool `val_eval=6882`; skip rate ~9-13% (caps met).

## 3. What is NOT on disk yet (the gap)

- **No commit / no PR.** All quality-graft work is uncommitted on `main`. Decision owed: land it (and how — that repo's PR conventions, not complexa's) or keep as a throwaway diagnostic.
- **Complexa-side PAE probe** (the *other* still-open thread, in complexa-flex on `debug_fast_convergence`) — OOM'd, never produced a number. Independent of this QG work. See [[frozen-feature-probe-diagnostic]].
- **Data-integrity follow-up:** ~10% of the staged `swissprot_v4/*.pt` store is truncated. The probe routes around it; a full re-preprocess would be needed for any run requiring the exact 229k set / bit-identical val membership.

## 4. Architecture reference

- Complexa reference ported from: `complexa-flex/src/proteinfoundation/confidence/probes/frozen_feature_probe.py` + `scripts/debug/run_frozen_feature_probe.py`.
- QG feature source: `quality-graft/src/quality_graft/models/la_proteina_wrapper.py::from_checkpoint` / `forward` → `{trunk_seqs [b,n,768], trunk_pair [b,n,n,256], local_latents [b,n,8], ca_coords}`; mask = `batch["mask"]` AFTER forward; `trunk_pair` NOT mask-zeroed (pool must mask).
- QG metrics reused: `quality-graft/src/quality_graft/training/metrics.py` (`_logits_to_continuous(logits, num_bins)`, `pearson_r`, `spearman_r`). Bins: `data/plddt_utils.py` `NUM_PLDDT_BINS=50`.
- Split: `quality-graft/src/la_proteina/proteinfoundation/utils/cluster_utils.py::split_dataframe(..., seed=42)`; local_only frame built in `src/quality_graft/data/datamodule.py::_setup_local_only`.
- `.pt` filename mapping: `pdb_data.py::__getitem__` uses `{pdb}_{chain}.pt` when a `chain` column exists (SwissProt splits stem on first `_`, so `pdb=AF-...-model`, `chain=v4`).

## 5. PR plan

**Only if the user wants the QG diagnostic landed** (it may be throwaway). If landing, it goes into the **quality-graft** repo per *that* repo's conventions (it has a `uv.lock` + its own CLAUDE.md; complexa's dev/PR-tier rules do NOT automatically apply). Suggested split:
- **QG-PR-A — `scripts/train.py` local_only fix.** Small, behaviour-preserving for existing runs (adds a path that was already assumed by config). Tests-first: a unit test that `_build_swissprot_data_module(local_only=true)` yields a datamodule that reads from `plddt_status.csv` not the CSV. Reviewer: `code-review-debug-complexity-expert` + `ml-software-pytorch-jax-expert` (datamodule wiring).
- **QG-PR-B — the probe (library + harness + tests + sbatch + plots).** Diagnostic tooling; the 14 unit + heavy smoke are the coverage. Reviewer: `code-review-debug-complexity-expert` + a domain reviewer (`generative-protein-scientist`: is the ~0-floor reading + nonlinear-signal interpretation sound). Keep the fp64 ridge cross-check as the reference for the GD probe.

No force-T3 complexa triggers apply (different repo; no complexa training/sampling/eval semantics touched).

## 6. Open questions for the user

1. **Land or discard the QG probe?** If land: into quality-graft `main` via PR, or keep uncommitted?
2. **Re-stage the ~10% truncated `swissprot_v4/*.pt`?** Only needed if a future run wants the exact/full val set; the floor result does not.
3. **Still want the complexa-side PAE probe** finished (the other open thread), or is pLDDT enough?

## 7. Reference paths

- complexa-flex root: `/mnt/storage01/home/schekmenev/projects/complexa-flex` (this session's shell cwd; branch `confhead_best_of_n_search` — unrelated in-flight work, do NOT sweep it).
- quality-graft root: `/mnt/storage01/home/schekmenev/projects/quality-graft` (labs view `/mnt/labs/home/schekmenev/projects/quality-graft` is the SAME inode — compute nodes mount labs).
- QG venv tarball: `quality-graft/venv.tar.gz` (3.7G, intact). Staged working venv: `/netscratch/schekmenev/ffprobe_venv/.venv`.
- QG data: `/netscratch/schekmenev/swissprot_v4` (h100 store). QG ckpts: `/netscratch/schekmenev/ckpt/quality_graft/`.
- QG run recipe: `sbatch quality-graft/scripts/run_frozen_feature_probe.sbatch` (h100 partition, ~15 min).

## 8. Project conventions reminder

- complexa-flex: branch off `dev`, PRs into `dev`, no Claude/AI attribution, `module load gh` before `gh`, never `uv run`/`sync`/`lock` (use `.venv/bin/python`), stage venv from tarball in sbatch, TDD via subagents.
- **quality-graft is a different repo:** `main` is its branch, it HAS a `uv.lock`+`.venv`, but STILL call the staged venv python directly (never `uv run` at runtime — the pre-existing staged `.venv` was found corrupt this session). Its data is only visible from the h100 partition store.
