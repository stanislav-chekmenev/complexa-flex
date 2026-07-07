# Handoff — Frozen-feature linear-probe diagnostic (fast-convergence investigation)

**Date:** 2026-07-06
**Branch:** `debug_fast_convergence` (off `dev`; not pushed)

## 1. TL;DR

A frozen-feature linear probe was built to diagnose why the Teddymer confidence head trains suspiciously fast/well: is it a val-split leak (scary) or are the frozen Complexa trunk features near-linearly pLDDT-predictive (benign)? **Result for pLDDT: leak REFUTED, benign.** No train→val gap under the exact seqclust30 split; the frozen trunk already gives a **~0.74 linear pLDDT floor** and the head's nonlinear tri-mult blocks close only the last 0.74→~0.90. **In flight / not done:** the **PAE-side probe OOM-killed** (never produced a number) and the **QG-side comparison probe** (Complexa vs La-Proteina floor) is deferred. All probe scaffolding is **uncommitted** on this branch.

## 2. What is on disk (do not recreate)

All untracked on `debug_fast_convergence` unless noted:

- **Probe library** — `src/proteinfoundation/confidence/probes/frozen_feature_probe.py`
  - `LinearProbe` (single `nn.Linear`), `split_dimers_for_probe` (reuses the datamodule's `_split_metadata` for the exact train/val boundary, then cluster-disjoint fit/eval on `seqclust30`), `pool_pair_features` (masked mean/max/diag pooling of `z`), `fit_probe` (GD, warmup), `ridge_probe_ev` (fp64 closed-form cross-check), `probe_metrics` (reuses `nn.confidence._metrics.pearson_r`/`spearman_r`/`_logits_to_continuous`/`_labels_to_continuous` — correlation math NOT reimplemented).
- **Harness** — `scripts/debug/run_frozen_feature_probe.py` (Hydra entry over `confidence/distillation_teddymer_multihead`), `scripts/debug/run_frozen_feature_probe.sbatch` (gpu), `scripts/debug/run_frozen_feature_probe_cpu.sbatch` (cpu/login01 — the one that actually ran; gpu nodes were draining).
- **lightning_module.py refactor (modified, tracked)** — `_forward` split so the frozen pre-head pipeline is a reusable `extract_frozen_features(batch) -> dict` (returns `inter`, `s`, `z`, `mask_ext`, `cond`, `local_latents`, `ca_coords`, `n_orig`, `orig_mask`, `batch_trimmed`, and `mask_eff`/`masks_by_head`). `_forward` now calls it and is unchanged in behaviour.
- **Tests (untracked)** — `tests/integration/confidence/test_forward_refactor_bit_identical.py` (asserts the refactored `_forward` is bit-identical to legacy), `tests/integration/confidence/test_frozen_probe_smoke.py`, and 8 unit tests under `tests/unit/confidence/probes/` (determinism, feature/label alignment, padding-mask exclusion, metric reuse, only-Linear-trains, pool_pair_features, forward shapes, seqclust30-split usage).
- **Result log** — `logs/probe_ffp_cpu_79572.out` (numbers below), `logs/probe_ffp_cpu_79572.err` (shows `slurmstepd-login01: Detected 1 oom_kill event ... Out Of Memory` — the PAE OOM).

**Measured pLDDT result (seqclust30 split; train_fit=550,907 · train_eval=5,563 · val_eval=5,619 proteins):**

| variant | train_fit | train_eval | val_eval |
|---|---|---|---|
| s-only | 0.742 | 0.711 | 0.731 |
| z-pooled | 0.452 | 0.428 | 0.482 |
| s+z-pooled | 0.756 | 0.724 | 0.740 |

No train→val gap (val marginally higher = noise) → **3b leak REFUTED**. Linear floor ~0.74, not ~0.90 → **3a only PARTIAL**: high linear floor, nonlinear head closes the rest.

## 3. What is NOT on disk yet (the gap)

1. **PAE-side probe result.** The `pae_z_direct` variant caches `z` as `[b, n, n, d_pair=256]` and OOM'd the 96G cpu allocation. No PAE number exists. Fix the harness memory (cache *pooled* / smaller z for the PAE variant, or hard-cap PAE protein count hard), then rerun **only** the PAE variant.
2. **QG-side comparison probe.** Nothing on disk. The only artefact that would answer whether Complexa's ~0.74 floor is *higher* than La-Proteina's. Deferred by the prior sessions.
3. **Decision on the scaffolding itself** — whether to land the `extract_frozen_features` refactor + probe as a PR into `dev`, or keep it branch-local as a throwaway diagnostic. See PR plan.

## 4. Architecture reference

- Refactor site: `src/proteinfoundation/confidence/lightning_module.py` — `extract_frozen_features` (new) and `_forward` (now delegates); `_build_mask_eff`, `_trim_batch`, `_trim_head_output` unchanged.
- Frozen-forward contract the probe depends on: trunk intermediates arrive via `nn_out["trunk_intermediates"]` = `(s, z, local_latents, mask, orig_mask, n_orig)` when `LocalLatentsTransformer.expose_intermediates=True` (CLAUDE.md confidence-distill section).
- Metric reuse target: `src/proteinfoundation/nn/confidence/_metrics.py`.
- Split reuse target: `src/proteinfoundation/datasets/teddymer/dataset.py::_split_metadata`.
- MultiHead centers the probe reads: `module.head.children_heads["plddt"|"pae"].bin_centers`.

## 5. PR plan

**Only open a PR if the user wants the diagnostic scaffolding landed** — it is currently a branch-local investigation and may be intended as throwaway. Confirm with the user first (see §6).

If landing into `dev`:

- **PR-A — `extract_frozen_features` refactor.** Tier **T3** (force-T3: touches the confidence training/forward path, i.e. training/eval semantics — even though guarded bit-identical). Scope: the `lightning_module.py` split + `test_forward_refactor_bit_identical.py`. Tests-first is already satisfied (the bit-identical test exists). Panel: **code-review-debug-complexity-expert (mandatory)** + **ml-protein-architect** (module/forward-contract) + **ml-software-pytorch-jax-expert** (autograd/no_grad boundary of the extracted frozen forward). Verify the bit-identical test passes on a real batch before review.
- **PR-B — probe diagnostic harness.** Tier **T2** (diagnostic tooling under `probes/` + `scripts/debug/`, does not change training/sampling/eval semantics; the 8 unit tests + smoke test are the coverage). Depends on PR-A (imports `extract_frozen_features`). Panel: **code-review-debug-complexity-expert** + one domain reviewer **generative-protein-scientist** (is the probe a valid leak discriminator, is the 0.74/0.90 reading sound). Note: keep the ridge cross-check as the fp64 reference for the GD probe.

Branches are **T-relaxable**: `debug_fast_convergence` is a debug branch, so per CLAUDE.md's target-branch relaxation, if this stays branch-local it needs no PR at all. The tiers above apply only when merging into `dev`.

## 6. Open questions for the user

1. **Land or discard?** Should the `extract_frozen_features` refactor + probe scaffolding be PR'd into `dev`, or is it a throwaway diagnostic that stays on `debug_fast_convergence`?
2. **Is the PAE answer needed?** The pLDDT leak question is settled (benign). Do you want the PAE-side probe fixed-and-rerun, or is pLDDT sufficient to close the fast-convergence worry?
3. **QG comparison priority.** Is the Complexa-vs-La-Proteina floor comparison (QG-side probe) worth building, or is "leak refuted, benign" enough to move on?

## 7. Reference paths

- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Configs: `configs/` (probe uses `confidence/distillation_teddymer_multihead`)
- Source: `src/proteinfoundation/` (probe: `.../confidence/probes/`, refactor: `.../confidence/lightning_module.py`)
- Scripts: `scripts/debug/`
- venv (runtime, used by the cpu sbatch): `/netscratch/schekmenev/complexa-paedistill/.venv/bin/python`
- venv tarball staging: `$PROJECT_ROOT/venv.tar.gz` (labs NFS) + `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz` (per partition)
- Active Teddymer data: `/netscratch/schekmenev/teddymer_v2_blob` (h100 store; launch via `TEDDYMER_VIEW_ROOT`/`AFDB_PROTEOMES_ROOT` override — but the cpu-partition probe reached it via the gpu store, which cpu/login shares)
- Result logs: `logs/probe_ffp_cpu_79572.{out,err}`

## 8. Project conventions reminder

- Branch off `dev` (main branch); PRs into `dev`. This work sits on `debug_fast_convergence`.
- **No Claude / AI attribution** in commits or PR text.
- `module load gh` in the same Bash invocation before any `gh` use.
- **Never** `uv run` / `uv sync` / `uv lock` — use `.venv/bin/python` (or the staged `/netscratch/.../.venv/bin/python`) directly.
- sbatch stages the venv from the tarball (`tar xzf`), never `uv run`; cross-partition `/netscratch` fallback is mandatory.
- TDD via subagents; tests written before implementation (the bit-identical test already gates the refactor).
