# Session handoff — Teddymer interface confidence distillation (steps 6 & 7)

**Date:** 2026-05-17
**Predecessor handoff:** [2026-05-17_teddymer_afdb_label_join.md](2026-05-17_teddymer_afdb_label_join.md)
**Status of upstream work:** ✅ steps 1-5 shipped in PR #8 (merged to `dev`). ✅ glob bug fix + 100% locator rebuild merged to `prepare_teddy` as `e143e3d` / `48b2804`. ❌ steps 6 & 7 not started.

## TL;DR for the next agent

The Teddymer view on disk is now **complete and ready to be consumed by a dataloader**, but no dataloader / dataset config exists yet. The remaining work is two PRs:

1. **PR-A — data plumbing.** Add `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` + `AddPLDDTFromParentAFDB` + `AddPAEFromParentAFDB` transforms + a structural Cα-Cα spot-check.
2. **PR-B — model plumbing.** Move pair-rep symmetrisation out of the trunk into per-head `_predict` so the asymmetric PAE head can opt out. Add the asymmetric PAE head class + Hydra config + sidecar training entry.

PR-A should land first because PR-B's training run depends on it. PR-A can be done in parallel sub-PRs by domain agents because the transforms and the config are independent.

## What is on disk (verified before this handoff)

### Data view

```
/mnt/storage01/home/schekmenev/data/teddymer_v1/
  dimers.parquet                                 587,687 rows, 17 cols (incl. complexa_filter)
  locator_rows.parquet                         1,175,374 rows = 587,687 × 2 chains, 100% coverage
  locator_rows.parquet.pre_glob_fix.bak          146,002 rows  ← old 12.4%-coverage file, kept for diff
  view_config.yaml                              provenance incl. rebuilt_at / rebuild_note
  _raw/
    nonsingletonrep_metadata.tsv
    teddymer_repdb/teddymer_repdb_h + .index + .lookup + .source + .dbtype
```

`locator_rows.parquet` schema (from `LOCATOR_INVENTORY_COLS` in [src/proteinfoundation/datasets/teddymer/build_locator.py](../../src/proteinfoundation/datasets/teddymer/build_locator.py)):

```
dimer_id, dimer_index, chain_id ∈ {A, B},
sample_id, afdb_id, uniprot_id, taxonomy_id,
source_tar_relpath, source_tar_basename, source_version, bucket,
has_cif, has_pae, has_conf,
cif_member_name, cif_member_offset, cif_member_size, cif_is_gz,
pae_member_name, pae_member_offset, pae_member_size, pae_is_gz,
conf_member_name, conf_member_offset, conf_member_size, conf_is_gz
```

Two chain-rows per dimer share the **same** `afdb_id` and therefore the **same** byte offsets — Teddymer is intra-monomer, so a single AFDB tar read serves both chains.

### Code shipped (and where it lives)

- [src/proteinfoundation/datasets/teddymer/parse_repdb_h.py](../../src/proteinfoundation/datasets/teddymer/parse_repdb_h.py) — parses `_h` + metadata, emits `dimers.parquet`, adds `complexa_filter`.
- [src/proteinfoundation/datasets/teddymer/build_locator.py](../../src/proteinfoundation/datasets/teddymer/build_locator.py) — joins parent AFDB ids against `inventory_first/inventory/manifests/batches/`. Glob is now `w*_b*.parquet` (post-fix).
- [src/proteinfoundation/datasets/teddymer/sanity_check.py](../../src/proteinfoundation/datasets/teddymer/sanity_check.py) — 50-dimer AFDB-tar round-trip on `avg_int_plddt`.
- [src/proteinfoundation/datasets/teddymer/build_view.py](../../src/proteinfoundation/datasets/teddymer/build_view.py) — CLI orchestrator; SLURM wrapper at [scripts/preprocess_teddymer.sbatch](../../scripts/preprocess_teddymer.sbatch).
- [scripts/diagnose_locator_coverage.py](../../scripts/diagnose_locator_coverage.py) — one-shot diagnostic showing w0-only vs full w0..w7 coverage. Useful future reference.
- [scripts/rebuild_teddymer_locator.py](../../scripts/rebuild_teddymer_locator.py) — one-shot rebuild that produced the current `locator_rows.parquet`.

### Branch state

- `dev` — has PR #8 merged (steps 1-5 + sanity check).
- `prepare_teddy` — has the glob fix + rebuild (merge commit `e143e3d`, fix commit `48b2804`). **Branch the new work off `prepare_teddy`** so it inherits the glob fix; later open PRs into `dev`.

## What is NOT on disk yet (the gap)

Grep confirmed (`grep -rli "AddPLDDTFromParentAFDB\|AddPAEFromParentAFDB" src configs` → no hits):

- No `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml`.
- No `AddPLDDTFromParentAFDB` transform.
- No `AddPAEFromParentAFDB` transform.
- No Teddymer-aware Lightning datamodule entry.
- No asymmetric PAE head (`pae_head.py`) in [src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/) — only `plddt_head.py` + `plddt_sequence_only_head.py`.
- No interface Cα-Cα spot-check beyond the average-pLDDT sanity check that already shipped.
- `z = (z + z.transpose(-3, -2)) / 2.0` in [src/proteinfoundation/nn/confidence/base.py:129](../../src/proteinfoundation/nn/confidence/base.py#L129) still lives in the trunk — must move to per-head `_predict` before the first asymmetric-PAE training run (CLAUDE.md flag).

## Architecture reference: how the existing pLDDT path works (to mirror)

Read these files first — your work should be structurally analogous.

- [configs/dataset/unified/afdb_monomers_with_plddt.yaml](../../configs/dataset/unified/afdb_monomers_with_plddt.yaml) — the config template. Extends `afdb_monomers` and adds a single `atom37_transform` entry pointing at `AddPLDDTFromBFactor`.
- `AddPLDDTFromBFactor` (in [src/proteinfoundation/datasets/transforms.py](../../src/proteinfoundation/datasets/transforms.py)) — the transform template. Reads CA-atom B-factors from the in-memory CIF, bins into 50 bins of width 2 on `[0, 100]`, writes `plddt_residue`, `plddt_bin`, `plddt_mask`.
- [src/proteinfoundation/nn/confidence/registry.py](../../src/proteinfoundation/nn/confidence/registry.py) — `@register_confidence_head(name)` decorator. New heads subclass `BaseConfidenceHead` in [src/proteinfoundation/nn/confidence/base.py](../../src/proteinfoundation/nn/confidence/base.py) and override `_predict`.
- [src/proteinfoundation/nn/confidence/plddt_head.py](../../src/proteinfoundation/nn/confidence/plddt_head.py) — the head template. Single-residue head, symmetric (consumes `s`, ignores `z` symmetry).
- [src/proteinfoundation/confidence/lightning_module.py](../../src/proteinfoundation/confidence/lightning_module.py) + [losses.py](../../src/proteinfoundation/confidence/losses.py) + [metrics.py](../../src/proteinfoundation/confidence/metrics.py) — sidecar Lightning module. Loss is `0.7 * masked_CE + 0.1 * SmoothL1(EV)`. Read these to see the metric pattern (acc, MAE, Pearson, Spearman, stratified MAE, ECE, reliability diagram) you'll re-use / extend.
- [src/proteinfoundation/confidence/train_confidence.py](../../src/proteinfoundation/confidence/train_confidence.py) — Hydra entry point. Mirror this for the PAE head training entry.
- [scripts/train_confidence_swissprot.sbatch](../../scripts/train_confidence_swissprot.sbatch) — SLURM wrapper template.

## PR-A — data plumbing (do this first)

### Scope

Make `teddymer_with_plddt_and_pae` a loadable, transform-stacked dataset that the existing confidence-head sidecar can already consume *for a pLDDT-on-dimers smoke run*, even before the asymmetric PAE head exists. This decouples data from model.

### Test plan (write tests first, per CLAUDE.md TDD discipline)

All under `tests/unit/datasets/` and `tests/integration/`. Use the same `tmp_path` + tiny-fake-AFDB-tar pattern as [tests/unit/datasets/test_teddymer_build_locator.py](../../tests/unit/datasets/test_teddymer_build_locator.py).

1. `tests/unit/datasets/test_add_plddt_from_parent_afdb.py`
   - Builds a minimal `locator_rows.parquet`-like DataFrame pointing into a tmp AFDB tar with a hand-crafted CIF (10 residues, known B-factors).
   - Asserts the transform fills `plddt_residue` for both chains with the right values, respects `residue_intervals_{A,B}` from `dimers.parquet`, binning matches `AddPLDDTFromBFactor` defaults.
   - Edge cases: discontinuous domain (`RES15-30_37-122`), single-residue domain, all-zero B-factor, missing CA atom (must produce `plddt_mask=False` not crash).

2. `tests/unit/datasets/test_add_pae_from_parent_afdb.py`
   - Same tmp-tar pattern but with hand-crafted `*-predicted_aligned_error_v4.json.gz` payload — known L×L matrix.
   - Asserts the transform yields **directional** PAE (no symmetrisation) on the residue submatrix selected by `residue_intervals_A ∪ residue_intervals_B`.
   - Asserts the **inter-chain block** `PAE[i ∈ A, j ∈ B]` is exposed as a first-class field for the asymmetric head (cheaper than the head re-slicing per batch).
   - Edge case: AFDB's clip ceiling (≈31.75 Å) round-trips losslessly.

3. `tests/unit/datasets/test_teddymer_dataset_config.py`
   - Loads `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` via Hydra compose and `instantiate`s the datamodule on a 4-dimer fake parquet. Asserts one batch comes through with `plddt_residue`, `plddt_bin`, `plddt_mask`, `pae_residue_pair`, `pae_bin`, `pae_mask` populated.

4. `tests/integration/test_teddymer_interface_cca_spot_check.py`
   - Sample 20 real dimers from `/mnt/storage01/home/schekmenev/data/teddymer_v1/dimers.parquet`, pull both chains' atoms from the parent CIF using the locator offsets, compute the Cα-Cα interface set (`d < 8 Å` across the two TED domain index sets), assert `len(interface_set) ≈ interface_length` (column from metadata) and `mean(pLDDT over interface) ≈ avg_int_plddt`. Tolerance: `interface_length` within ±2 residues (residue-numbering frame edge cases); `avg_int_plddt` within 1.0 pLDDT unit. Mark slow.
   - This catches the residue-numbering frame mismatch flagged by structural-biology review in PR #8 that the existing sanity check cannot detect.

### Implementation deliverables

- `src/proteinfoundation/datasets/transforms.py` — add `AddPLDDTFromParentAFDB`, `AddPAEFromParentAFDB`.
  - Constructor takes the locator parquet path + AFDB proteomes root. Each `__call__(sample)` does one random-access tar read at `cif_member_offset` / `pae_member_offset` (use the existing tar-byte-offset reader pattern that the la_proteina_afdb_512_v1 view already uses — look it up via `grep -rli "cif_member_offset" src/`).
  - Lazy loading at dataloader time (per handoff §6 in the predecessor doc — don't materialise PAE to disk; up to 1 MB densified per dimer × 587k dimers = ~600 GB if cached).
  - Output fields:
    - pLDDT: `plddt_residue` (float, [0, 100]), `plddt_bin` (int, 50 bins width 2), `plddt_mask` (bool). Match `AddPLDDTFromBFactor` schema exactly so the existing pLDDT head trains unchanged on Teddymer monomer pLDDT first.
    - PAE: `pae_residue_pair` (L_A+L_B × L_A+L_B float, directional, no symmetrisation), `pae_bin` (int, AF2 standard 64 bins on `[0, 31.75]`), `pae_mask` (bool, off-diagonal + within-chain entries can be optionally masked at loss time, not here). Also emit `pae_inter_chain_block` view (L_A × L_B float) as a convenience field.

- `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` — extend `ted_dimers` (or the closest existing dimer config; check what `ted_dimers.yaml` does first). Add the two transforms. **Drop the `complexa_filter`-only restriction** to expose the full 587k label distribution; let the user filter at experiment time.

- `src/proteinfoundation/datasets/structure_data.py` — if Teddymer needs its own `__getitem__` (two chains, intra-monomer, shared parent), add a `TeddymerDimerDataset` here. Otherwise reuse the existing dimer dataset path and document the choice.

### Reviewers for PR-A

Mandatory:
- [code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md) — correctness, edge cases, memory budget (the 1 MB/dimer × N_workers PAE-densification cost is real; flag if seen).

Domain-add:
- [ml-protein-architect](../../.claude/agents/ml-protein-architect.md) — module layout, config tree conformance, mirroring of `afdb_monomers_with_plddt.yaml`.
- [structural-biology-binder-expert](../../.claude/agents/structural-biology-binder-expert.md) — interface definition (Cα-Cα < 8 Å) is the right one for Teddymer's synthetic dimers; residue-numbering frame mismatch detection plan is sufficient.
- [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md) — dataloader determinism (Teddymer's PAE field is large; check worker-fork behavior and pinned-memory cost).

## PR-B — model plumbing (depends on PR-A landing)

### Scope

Add an asymmetric PAE head and refactor the trunk so symmetric heads (pLDDT, future PDE / ipLDDT / ipTM) can opt in to symmetrisation and asymmetric heads (PAE) can opt out.

### Test plan

1. `tests/regression/test_flow_matching_loss_unchanged.py` already exists — it must continue to pass after the trunk refactor. **This is the load-bearing regression test for the trunk change.**

2. `tests/unit/nn/confidence/test_symmetrisation_per_head.py`
   - Instantiate `ConfidenceTrunk` and confirm `z` is now passed **un-symmetrised** to `_predict` (compare `z` to its transpose; assert at least some i,j entries differ on a random input).
   - Instantiate the existing `PlddtHead` (or a new `PdeHead` once added); assert its `_predict` produces a symmetric output even when fed asymmetric `z`.
   - Instantiate the new `PaeHead`; assert it produces an asymmetric output `PAE_pred[i,j] ≠ PAE_pred[j,i]` on a random input.

3. `tests/unit/nn/confidence/test_pae_head.py`
   - Forward shape: `(B, L, L, n_bins)` with `n_bins=64`.
   - On a trivial fake batch (e.g. zero trunk output), training-loss is finite and gradients flow to head parameters and the trunk projection.

4. `tests/unit/confidence/test_pae_loss_and_metrics.py`
   - `masked_CE` on PAE bins matches a hand-computed value on a tiny 3×3 case.
   - PAE-EV (expected value over bin midpoints) MAE on the same case.
   - Calibration metrics (ECE, per-distance-stratified MAE) compute without NaN on a small case where some pairs are masked.

5. `tests/integration/test_pae_distillation_smoke.py`
   - One Lightning training step on ≤4 Teddymer dimers (use the 50-dimer subset PR-A's spot-check uses to keep IO cheap).
   - Asserts loss decreases over 10 steps on the same micro-batch (overfit-one-batch smoke).

### Implementation deliverables

- [src/proteinfoundation/nn/confidence/base.py](../../src/proteinfoundation/nn/confidence/base.py): move `z = (z + z.transpose(-3, -2)) / 2.0` out of `ConfidenceTrunk.forward` (currently line 129).
  - **Recommended:** option (b) from CLAUDE.md — remove from trunk entirely; require each `_predict` to symmetrise. Cleaner long-term; no `if`-branch in trunk.
  - **Fallback:** option (a) — add `symmetrise_z: bool = True` flag on `ConfidenceTrunk.__init__`; default `True` preserves current behavior. PaeHead config sets it to `False`. Smaller diff but state-leaks the head's needs into the trunk constructor.
  - Whichever you pick, update existing `PlddtHead._predict` to call `z = 0.5 * (z + z.transpose(-3, -2))` itself if it actually uses `z` (it currently consumes `s` only, so the change may be a no-op for it).

- [src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/) `pae_head.py` — new file. `@register_confidence_head("pae")`. Subclass `BaseConfidenceHead`. Consumes asymmetric `z`. Output: 64-bin classification over PAE bins on `[0, 31.75]` Å. Match AF2's bin edges exactly.

- [src/proteinfoundation/confidence/losses.py](../../src/proteinfoundation/confidence/losses.py): add `pae_loss` (`masked_CE + small SmoothL1 on EV`, same recipe family as pLDDT).

- [src/proteinfoundation/confidence/metrics.py](../../src/proteinfoundation/confidence/metrics.py): add pair-level analogues — accuracy, MAE in Å, Pearson, Spearman, stratified MAE by predicted-PAE bucket, equal-width and equal-mass adaptive ECE. Add per-distance-stratified MAE (`d_ij < 8`, `8 ≤ d_ij < 16`, `d_ij ≥ 16` Å) — interfaces are the high-value regime.

- `configs/confidence/distillation_teddymer_pae.yaml` — analogue of [configs/confidence/distillation_swissprot.yaml](../../configs/confidence/distillation_swissprot.yaml). Points dataset at `teddymer_with_plddt_and_pae`, head at `pae`, loss/metric blocks at PAE variants.

- `scripts/train_confidence_teddymer_pae.sbatch` — analogue of [scripts/train_confidence_swissprot.sbatch](../../scripts/train_confidence_swissprot.sbatch).

### Reviewers for PR-B

Mandatory:
- [code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md).

Domain-add:
- [ml-protein-architect](../../.claude/agents/ml-protein-architect.md) — head registry conformance, sidecar pattern integrity.
- [generative-protein-scientist](../../.claude/agents/generative-protein-scientist.md) — PAE bin edges, loss weighting, distance-stratified metric choice.
- [generative-flow-stochastic-math-expert](../../.claude/agents/generative-flow-stochastic-math-expert.md) — bin-classification + EV-regression hybrid is the right discretisation.
- [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md) — symmetrisation refactor's interaction with `torch.compile` and FSDP; trunk regression test passing on multi-GPU.

## TDD orchestration recipe for the next session

Per CLAUDE.md, the main thread coordinates and dispatches subagents:

1. **Plan with [software-planning-architect](../../.claude/agents/software-planning-architect.md).** Translate this handoff into a structured plan for PR-A (data plumbing). Output: written plan with exit/abort criteria, milestones, explicit verification commands. Save under `docs/superpowers/plans/`.

2. **Red phase.** Dispatch [code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md) (acting as test-author) to write all PR-A tests from the test plan above. Confirm they fail with the right error (transform/config doesn't exist), not a setup error.

3. **Green phase.** Dispatch [ml-protein-architect](../../.claude/agents/ml-protein-architect.md) to implement the transforms + config until tests pass. If random-access tar reading is the hard part, escalate to [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md) for the IO loop.

4. **Refactor / review.** Dispatch the four-reviewer panel for PR-A in parallel. All must approve before merging into `prepare_teddy` (or directly into `dev` if `prepare_teddy` is no longer needed).

5. **Repeat for PR-B.** PR-B's red-phase tests can be written in parallel with PR-A's green phase, but the trunk refactor must wait for the trunk regression test (`tests/regression/test_flow_matching_loss_unchanged.py`) to be green on PR-A's branch first.

## Open questions for the user before starting PR-A

- **Filter at dataset-load or at experiment-config?** Memory says Complexa trained on the 510k `complexa_filter`-passing subset. Recommend exposing all 587k via the dataset and filtering at experiment-config time — keeps the dataset reusable for ablations that want the noisier 587k.
- **PAE bin edges.** AF2 uses 64 bins on `[0, 31.75]` Å. Confirm we mirror that exactly (load-bearing for any later distillation-into-multimer-PAE comparison).
- **Trunk symmetrisation refactor: option (a) or (b)?** Recommend (b) per CLAUDE.md. Confirm before refactor lands; option (a) is the quick rollback if (b) causes unexpected breakage.

## Reference paths (most-used during this work)

- View root: `/mnt/storage01/home/schekmenev/data/teddymer_v1/`
- AFDB v4 raw: `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/`
- AFDB master inventory: `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/` (512 shards: `w{0..7}_b{000000..000063}.parquet`, ~216M rows)
- Existing AFDB view template: `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/la_proteina_afdb_512_v1/`
- Confidence subsystem: [src/proteinfoundation/confidence/](../../src/proteinfoundation/confidence/), [src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/)
- Trunk symmetrisation line to refactor: [src/proteinfoundation/nn/confidence/base.py:129](../../src/proteinfoundation/nn/confidence/base.py#L129)
- Eventual cluster `mv` (requires user write access to `/mnt/labs/shared/`): `mv /mnt/storage01/home/schekmenev/data/teddymer_v1/* /mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/teddymer_v1/`. `view_config.yaml` makes the move mechanical — no per-file fixup.

## Project conventions (reminder)

- Branch off `prepare_teddy` (or `dev` if `prepare_teddy` has been merged); PR into `dev`.
- TDD via subagents. `code-review-debug-complexity-expert` mandatory on every PR.
- No Claude attribution in commits or PRs.
- `gh` requires `module load gh` in the same Bash invocation.
- Pinned env: uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5 <2.6.
- Runtime artefacts (`ckpts/`, `wandb/`, sample/eval outputs) stay gitignored.
