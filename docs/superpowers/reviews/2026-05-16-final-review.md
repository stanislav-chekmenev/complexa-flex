# Final Review — Confidence-Head Distillation in Proteina-Complexa

**Date.** 2026-05-16
**Branch.** `merge_quality_graft` at `13e4cbc`
**Spec.** `docs/superpowers/specs/2026-05-16-confidence-head-distillation-design.md`
**Scope.** Port the trainable student confidence head from `quality-graft` into `complexa-flex`, train it to distill AF2 per-residue pLDDT on ~240K SwissProt monomers, deliver an `sbatch`-able SLURM script.

---

## Outcome

**6 PRs merged into `merge_quality_graft`. 110/110 tests pass. The user can launch the distillation run with `sbatch scripts/train_confidence_swissprot.sbatch`.**

| PR | Title | Merge commit | New tests | Reviewers (final round) |
|---|---|---|---:|---|
| PR-1 | trunk `expose_intermediates` hook | `f22f38f` | 8 contract + 2 regression | code-review APPROVE, ml-protein-architect APPROVE, ml-software-pytorch-jax-expert APPROVE |
| PR-2 | AF2 pLDDT data transform | `8250d9a` | 8 binning + 6 transform + 4 integration + 4 Hydra | code-review APPROVE, ml-protein-architect APPROVE, structural-biology-binder-expert APPROVE |
| PR-3 | `BaseConfidenceHead` + `ConfidenceTrunk` + `PLDDTHead` | `b76218f` | 16 head + 4 projection + 1 calibration | code-review APPROVE, ml-protein-architect APPROVE, ml-software-pytorch-jax-expert APPROVE, generative-protein-scientist APPROVE |
| PR-4 | `ConfidenceDistillationModule` sidecar + loss + metrics + entry | `fa4c5ee` | 5 losses + 4 metrics + 2 cond + 3 distill + 1 ckpt | code-review APPROVE (after rd-2 trunk-batch plumbing fix), ml-protein-architect APPROVE, ml-software-pytorch-jax-expert APPROVE, generative-protein-scientist APPROVE |
| PR-5 | held-out eval + sequence-only control + smoke | `d4a89e2` | 22 (sequence-only + ECE-adaptive + reliability + cluster + DDP + concat) | unanimous round-1 APPROVE (no round 2 needed) |
| PR-6 | sbatch + multi-GPU + 11 PR-5 follow-ups + PR-4 cond-shape fix | `13e4cbc` | 6 (concat features + DDP smoke + skewed cluster + empty val + reliability aggregation) | code-review APPROVE, pytorch-jax APPROVE, scientist APPROVE, physics APPROVE; ml-protein-architect rd-1 REQUEST_CHANGES → rd-2 APPROVE |

**Total new tests: 110** (8 PR-1 + 22 PR-2 + 21 PR-3 + 15 PR-4 + 22 PR-5 + 6 PR-6 + 16 from round-2 fixes scattered across PRs). All green.

---

## What was delivered

### New runtime artefacts

- **`src/proteinfoundation/nn/confidence/`** — pair-biased confidence trunk + `BaseConfidenceHead` abstract base + `PLDDTHead` (50-bin AF2 pLDDT classifier with bin-EV reduction) + `SequenceOnlyPLDDTHead` (diagnostic control with `n_blocks=1`) + `SeqProjection`/`PairProjection` + registry/factory.
- **`src/proteinfoundation/confidence/`** — `ConfidenceDistillationModule(L.LightningModule)` sidecar that owns a frozen `Proteina` + trainable head; `_forward` does `add_clean_samples → fm.corrupt_batch → pin t=0.99 → fm.interpolate → no_grad trunk forward → cond-pad → head → CE+SmoothL1 loss`. AdamW + LambdaLR warmup → linear decay. `combined_plddt_loss = 0.7*CE + 0.1*SmoothL1(EV)`, reduction `sum/mask.sum().clamp_min(1)`. Validation logs `val/loss` (CE alias), `val/loss_ce`, `val/loss_smooth_l1`, `val/loss_total`, `val/plddt_accuracy`, `val/plddt_mae`, `val/pearson_r`, `val/spearman_r`, `val/mae_lt50`/`_50_70`/`_70_90`/`_ge90`, `val/ece`, `val/ece_adaptive`. Reliability diagram aggregated across full val epoch, emitted via `logger.log_table` or `.npy` fallback on rank 0 only.
- **`scripts/train_confidence_swissprot.sbatch`** — SLURM script for 2× h100nvl, 3-day wall, bf16-mixed, DDP `find_unused_parameters_false`. Stages trunk + AE ckpt + AFDB parquet to `/netscratch/$USER/complexa-confdistill`. NCCL knobs identical to `quality-graft`. Rsync run dir back.
- **`scripts/run_smoke_train.sh`** — single-GPU `max_steps=100` smoke command.
- **Hydra configs** — `configs/nn/confidence/{base,plddt_head,plddt_sequence_only_head}.yaml`, `configs/training/confidence_distill.yaml`, `configs/confidence/{distillation_swissprot,distillation_swissprot_control}.yaml`, `configs/dataset/unified/afdb_monomers_with_plddt.yaml`.

### Surface area into existing complexa code

- `src/proteinfoundation/nn/local_latents_transformer.py` + `_v2.py` — added one opt-in `expose_intermediates` kwarg (off by default; bit-identical to legacy when off, enforced by `tests/regression/test_flow_matching_loss_unchanged.py` with a `torch.equal` baseline fixture).
- `src/proteinfoundation/datasets/transforms.py` — appended `AddPLDDTFromBFactor` (additive; AF2 `[0, 100]` scale; once-per-rank warning on `b_factor.max() ≤ 1.5`).
- `src/proteinfoundation/datasets/structure_data.py` — threaded `atom_b_factor` through `atomarray_to_atom37`; added optional `cluster_column` / `cluster_seed` kwargs to `StructureDataModule` (`None` default preserves PR-2 baseline; `loguru.warning` fallback if column missing or val empty).
- `src/proteinfoundation/proteina.py` — **never edited.** Sidecar pattern (Option A from spec §0 Q1).
- `src/proteinfoundation/train.py` — **never edited.**

### User-approved key decisions (spec §0)

- `trunk_eval_t = 0.99` (avoids `t=1.0` singularity; near-clean point inside the trunk's training distribution).
- `ckpts/complexa.ckpt` + `ckpts/complexa_ae.ckpt` — production trunk + autoencoder.
- AF2 pLDDT `[0, 100]` scale, 50 bins of width 2, midpoint bin centers `[1, 3, …, 99]`.
- `n_blocks=4, n_heads=16, dim_cond=256, use_tri_mult=True, use_tri_attn=False, dropout=0.1` for the main head.
- SwissProt cluster-30% held-out only in PR-5 (RFdiffusion / Dayhoff transfer deferred).
- AdaLN cond reuses the trunk's existing `FeatureFactory` time embedder at `t=0.99` (not a learned constant).

---

## Review hygiene

### Approval discipline

Every PR went through TDD-via-subagents (plan → tests-first → implement → review panel → merge). Reviewers ran in parallel; CLAUDE.md "all-or-nothing" approval rule enforced. Five PRs (PR-1, PR-2, PR-3, PR-4, PR-6) required at least one round-2 fix; PR-5 was unanimous round-1 APPROVE. No deadlocks; no escape-hatch email needed.

### Reviewer panels per PR

- PR-1, PR-2: 3 reviewers (code-review mandatory + 2 domain).
- PR-3: 4 reviewers (added generative-protein-scientist).
- PR-4: 4 reviewers (same panel).
- PR-5: 4 reviewers.
- PR-6: **5 reviewers** (added `physics-statmech-md-dft-expert` for sbatch resource sanity).

### Blockers found by reviewers

| PR | Blocker | Severity | Resolution |
|---|---|---|---|
| PR-1 (rd 1) | Docstring contract gaps (autograd / aliasing / dtype) | major (REQUEST_CHANGES) | Docstring expanded; rd-2 APPROVE. |
| PR-3 (rd 1) | `name` key in YAML breaks direct `hydra.utils.instantiate`; `chain_id` docstring vs behaviour mismatch | 2× major (REQUEST_CHANGES) | Dropped `name` from YAML; rewrote docstring; rd-2 APPROVE. |
| PR-4 (rd 1) | `t` not stamped before trunk forward; `x_t` never built (no `fm.corrupt_batch` call) | 2× blocker (BLOCK) | Rewrote `_forward` to use `add_clean_samples + corrupt_batch + interpolate + pinned t`; strengthened test stub; rd-2 APPROVE. |
| PR-4 (rd 1) | SmoothL1 weight 0.3 too aggressive (calibration collapse risk) | major (REQUEST_CHANGES) | Reverted to spec §3.8's 0.1; rd-2 APPROVE. |
| PR-6 (rd 1) | sbatch Hydra-override key mismatch (`confidence.trunk_ckpt_path` vs `training.trunk_ckpt_path`) | blocker (REQUEST_CHANGES) | Dropped the overrides; rely on `CKPT_DIR` env-var path. |

### Latent issue caught + fixed

The PR-5 review surfaced a latent bug in PR-4: `_compute_cond` returned `(b, n_orig, dim_cond)` but the head consumed `(b, n_ext, dim_cond)` after the trunk runs on the extended batch. Dormant for monomer training (`n_ext == n_orig`); would silently shape-error or broadcast incorrectly when concat features (chain-id / target seq / interface flags) become active for future ipTM/ipAE work. PR-6 fixed this pre-emptively with `_pad_cond_to_n_ext` (sidecar-side zero-pad). The scientist confirmed in PR-6 round 2 that zero-padding is **architecturally correct** — bit-consistent with how the frozen trunk itself extends conditioning via `torch.cat([c, zero_cond], dim=1)` in `LocalLatentsTransformer.forward`. (The scientist explicitly retracted their earlier PR-5 position recommending `cond_factory` extension.)

---

## Quality signals

- **TDD discipline** held throughout: every commit graph shows test commit landing before implementation. Tests were never weakened to pass.
- **Bit-exact regression guard** for the only non-additive change (PR-1's trunk hook). Implementer independently `git stash`'ed the implementation and reran the regression test against a fixture committed against `merge_quality_graft` HEAD pre-PR-1.
- **No forbidden-path edits** confirmed by every PR review. `proteina.py` was read-only across the entire 6-PR sequence.
- **Style hygiene**: zero Claude attribution in commits or PR bodies (verified per PR by reviewers); zero emojis; type hints on public APIs.

---

## Carried-forward follow-ups (non-blocking; deferred to PR-7+ or separate research threads)

### Engineering hygiene
- DDP `find_unused_parameters` re-evaluation once the multi-GPU run is stable.
- Vectorize the 50-bin Python loop in `expected_calibration_error_adaptive` and `reliability_diagram` via `torch.bucketize + scatter_add_` (~5000 host syncs/epoch otherwise — measurable but not fatal at val cadence).
- `NCCL_P2P_DISABLE=1` re-evaluation (mirrors `quality-graft` for stability; small NVLink underutilization cost).
- Size-check / `md5sum` guard on sbatch staging against partial-failure leftovers.
- Trailing rsync `trap` so outputs surface to project tree even if `srun` aborts mid-rank.
- Move sbatch trainer block under a `configs/trainer/` group for reuse.

### Scientific
- Regression test pinning "control head has zero `pair_update_layers`" + "logits invariant to z input" (PR-7 candidate; 10-line test).
- Capacity-matched control (4-block trunk with `PairReprUpdate.forward` as no-op) **only if** the 1-block control underperforms.
- Adaptive-ECE noise at small val batches — fine once full SwissProt val is in flight.
- `_compute_cond` skips trunk's `transition_c_1`/`_2` post-processing on cond (pre-existing in PR-4, independent of the shape-pad fix).
- bf16 + frozen trunk + triangle attention: consistent per-sequence-deterministic features → fine for distillation, but flipping to fp32 trunk later shifts calibration; document.

### Followups gated by scientific results
- **RFdiffusion / Dayhoff transfer eval** (deferred per spec §0 Q6). The headline question for binder-deployment: does the head's pLDDT prediction transfer from natural SwissProt to designed binders? Scientist's bet: SwissProt-val MAE 3-5 pLDDT, designed-binder MAE 2-3× worse.
- **Sequence-only control comparison**: if val Spearman gap < 0.05 vs the structure-aware head, the head is learning a sequence shortcut. Production smoke must report this.
- **Joint training** with trunk fine-tuned at small weight — only if pure distillation caps below useful Spearman on the designed-binder set.
- **ipTM/ipAE/ipLDDT heads** — separate research thread; `BaseConfidenceHead` is designed for them (`z` always carried + symmetrised + LayerNormed; `chain_id` accepted in `forward`; `_pad_cond_to_n_ext` makes concat-features safe).

---

## Files of record

- Spec: `docs/superpowers/specs/2026-05-16-confidence-head-distillation-design.md`
- Plans: `docs/superpowers/plans/2026-05-16-pr{1,2,3,4,5,6}-*.md` + PR-4 round-2 + PR-3 round-2 implicit in commits.
- PRs: https://github.com/stanislav-chekmenev/complexa-flex/pull/{1,2,3,4,5,6}
- Merge commits: `f22f38f` (PR-1), `8250d9a` (PR-2), `b76218f` (PR-3), `fa4c5ee` (PR-4), `d4a89e2` (PR-5), `13e4cbc` (PR-6).
- Final HEAD: `13e4cbc` on `merge_quality_graft`.

## Production launch command

```
sbatch scripts/train_confidence_swissprot.sbatch
```

Optional override of staging root via `DATA_ROOT` env, of GPU count via `NUM_DEVICES`. The script reads `${oc.env:CKPT_DIR,ckpts}/complexa{,_ae}.ckpt` and `${oc.env:DATA_PATH}/afdb_cifs/metadata.parquet` so the user can repoint without editing the script.
