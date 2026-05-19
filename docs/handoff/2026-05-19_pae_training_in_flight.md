# Handoff — PAE confidence-distillation on H100, training in flight

**Date:** 2026-05-19
**Branch:** `train_pae` (off `dev`, has merge commit `e12c298` pulling latest `dev` into the branch)
**Supersedes:** `2026-05-19_h100_blob_build_then_train.md` (the h100 blob is now staged and verified — that handoff is no longer load-bearing).

## TL;DR

- **Shipped this session (operationally):** the 127 GB Teddymer blob is now staged on **both** `gpu`-partition and `h100`-partition `/netscratch/schekmenev/teddymer_v1_blob/`, md5-verified against the labs-side snapshot (version `teddymer_v1_blob_20260518_068d9c67`). The h100 copy was made via labs-NFS rsync (build_blob.py does not support resume — verified, see new memory `build_blob_no_resume`). The sbatch [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch) has been edited extensively but the edits are **uncommitted**.
- **In flight:** the PAE distillation training has progressed past the integrity gate (~234 s on h100), past wandb auth (after a `WANDB_API_KEY` placeholder fix), past trainer init, and printed `train/pae/loss_step=4.830` at step 1 of job 53632. Backward then failed twice on DDP issues. The DDP fix (find_unused_parameters=true + static_graph=true) was applied as Hydra CLI overrides; submitted as job 53636 which the **user cancelled** at config-dump stage before training started.
- **NOT shipped:** the training run has NOT been verified end-to-end (no first val, no first checkpoint, no wandb run with loss curve). The PR to `dev` is NOT opened. The summary email to `schekmenev@aithyra.at` is NOT sent.

## What is on disk

### Verified
- **GPU-partition blob** at `/netscratch/schekmenev/teddymer_v1_blob/` (head node view) — 127 GB `data.blob`, md5 `068d9c6736f04f6b4d131ab6160301cb`, version `teddymer_v1_blob_20260518_068d9c67`.
- **H100-partition blob** at `/netscratch/schekmenev/teddymer_v1_blob/` (h100 node view, separate physical NVMe) — same 5 files (`data.blob`, `data.md5`, `view_config.yaml`, `dimers.parquet`, `locator_rows.parquet`), `md5sum -c data.md5` passed inside job 53605.
- **Labs-side staging copy** at `/mnt/storage01/home/schekmenev/data/teddymer_v1_blob_stage/` — full 127 GB blob + sidecars, used to bridge gpu→h100. Safe to delete once you trust the h100 copy (it's recreatable from the gpu copy in ~15 min).
- **Labs-side integrity snapshot** at `/mnt/storage01/home/schekmenev/data/teddymer_v1/` — `data.md5` + `view_config.yaml` only (the training-time gate consumes these; configured via `integrity.snapshot_path` in [configs/confidence/distillation_teddymer_pae.yaml](configs/confidence/distillation_teddymer_pae.yaml)).
- **Venv tarball** at `$PROJECT_ROOT/venv.tar.gz` (labs NFS) and `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz` (per-partition; gpu copy verified, h100 copy auto-staged inside the training sbatch's fallback path on first h100 launch).
- **Pre-staged training assets on h100-/netscratch** at `/netscratch/schekmenev/complexa-paedistill/` — `.venv`, `ckpt/complexa.ckpt`, `ckpt/complexa_ae.ckpt`, prior failed run dirs `runs/{53624,53629,53631,53632,53635,53636}/`.

### Uncommitted edits in the working tree (left for the next agent to review)
```
 M CLAUDE.md                                  # adds DDP static_graph rule (~25 lines)
 M scripts/train_confidence_teddymer_pae.sbatch  # WANDB_API_KEY unset + DDP strategy overrides
?? docs/handoff/2026-05-19_h100_blob_build_then_train.md  # prior handoff, now stale
?? docs/handoff/2026-05-19_pae_training_in_flight.md      # this file
```
The `scripts/train_confidence_teddymer_pae.sbatch` edits are the load-bearing fix and need to ship in the PR. The stale handoff `2026-05-19_h100_blob_build_then_train.md` should be **deleted** (or kept, but it points at a build job that's irrelevant now).

### Failed wandb runs to clean up
Multiple `pae-distill-teddymer` runs were started under WandB project `confidence-distillation`. The `run_name` is shared across launches so they should be coalesced by Lightning's `id == run_name` pattern, but verify before launching the real run — if there's clutter, the user may want them deleted.

## What is NOT on disk yet (the gap)

1. **A successful PAE training run** that demonstrates: (a) loss decreases over ≥25 logged steps, (b) first validation epoch completes without crash, (c) first checkpoint saves. None of the 6 attempted jobs (53624, 53629, 53631, 53632, 53635, 53636) reached the val/checkpoint stage.
2. **The PR `train_pae` → `dev`.** Scope, test plan, reviewer panel below.
3. **The summary email** to `schekmenev@aithyra.at`.
4. **A bookkeeping commit** for CLAUDE.md + this handoff on `train_pae` (Step 4 of `/ul` will handle this for the current session; if it fails, do it next session).

## Architecture reference

- Training entry point: [src/proteinfoundation/confidence/train_confidence.py](src/proteinfoundation/confidence/train_confidence.py).
- Lightning sidecar: [src/proteinfoundation/confidence/lightning_module.py](src/proteinfoundation/confidence/lightning_module.py).
- PaeHead: registered in [src/proteinfoundation/nn/confidence/](src/proteinfoundation/nn/confidence/) (see PR #11 in git log).
- PAE config: [configs/confidence/distillation_teddymer_pae.yaml](configs/confidence/distillation_teddymer_pae.yaml) — currently has `strategy: ddp_find_unused_parameters_false` (wrong for multi-GPU; the sbatch overrides via CLI).
- Sbatch: [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch) — the WANDB-unset + DDP-strategy override is at the bottom (`srun --kill-on-bad-exit /usr/bin/env -u WANDB_API_KEY ...`).
- Build_blob with no resume: [src/proteinfoundation/datasets/teddymer/build_blob.py:168-241](src/proteinfoundation/datasets/teddymer/build_blob.py#L168-L241) (atomic tmp+rename, BaseException → unlink).
- Integrity gate: [src/proteinfoundation/datasets/teddymer/integrity.py:88-140](src/proteinfoundation/datasets/teddymer/integrity.py#L88-L140) — two-stage (version compare → full md5).
- DDP multi-head footgun documented in CLAUDE.md `MultiHeadConfidence wrapper` bullet; the new generalisation bullet (added this session, line ~28) makes `static_graph=true` mandatory for ANY multi-GPU confidence-distill run, not just multi-head.

## PR plan — `train_pae` → `dev`

### Scope
**Code/config changes load-bearing for the PR:**
1. `scripts/train_confidence_teddymer_pae.sbatch` — the four edits from this session:
   - `--partition=h100` + `--gres=gpu:h100nvl:2` + `DEVICE_TYPE=h100nvl`.
   - Venv cross-partition fallback (`cp` from `$ENV_TARBALL_SRC` if `$ENV_TARBALL` missing on local `/netscratch`).
   - `srun --kill-on-bad-exit /usr/bin/env -u WANDB_API_KEY "$ENV_LOCAL/bin/python" ...` (the WANDB env strip).
   - DDP strategy overrides (`~trainer.strategy +trainer.strategy._target_=...DDPStrategy +trainer.strategy.find_unused_parameters=true +trainer.strategy.static_graph=true`).
2. (Optional but recommended) Flip the DDP default in all four [configs/confidence/distillation_*.yaml](configs/confidence/) from `strategy: ddp_find_unused_parameters_false` to a structured `DDPStrategy` block:
   ```yaml
   strategy:
     _target_: lightning.pytorch.strategies.DDPStrategy
     find_unused_parameters: true
     static_graph: true
   ```
   Then drop the long CLI overrides from the sbatch. This is the clean-architecture version of the fix.
3. `CLAUDE.md` — the new DDP-strategy bullet under the confidence-distillation section (already edited this session, uncommitted).

**Already-on-disk that does NOT need to be in this PR:**
- The h100-partition blob — it's a runtime artefact on `/netscratch`, not in the repo.
- The labs-staging blob copy — same.
- The venv tarball — same.

### Test plan (TDD per CLAUDE.md — tests first)
- **Smoke test on 2 × h100** of the (committed) sbatch end-to-end: integrity gate passes, ≥25 logged training steps with monotone-ish loss trend, first validation epoch completes, first checkpoint saves under `runs/<jobid>/`. Treat this as the abort gate before merging.
- **Unit-level**: the DDP-strategy yaml override should be loadable via `OmegaConf.load` + `hydra.utils.instantiate` and produce a `DDPStrategy` with both flags `True`. One pytest is enough.
- Existing regression `tests/regression/test_flow_matching_loss_unchanged.py` must continue to pass (unrelated to this PR but must not be broken).
- No code logic changes other than configs + sbatch, so no new module-level unit tests are required.

### Implementation deliverables
- Sbatch edits (5 hunks).
- Config edits (4 yamls × 1 hunk each, OR keep CLI override in sbatch and skip yaml edits — discuss with reviewer panel; the cleaner version is the yaml flip).
- CLAUDE.md edit (1 hunk, already done).
- Optional: delete stale handoff `docs/handoff/2026-05-19_h100_blob_build_then_train.md`.

### Reviewer panel
- **Mandatory:** [code-review-debug-complexity-expert](.claude/agents/code-review-debug-complexity-expert.md) — covers correctness, DDP/static_graph semantics, sbatch portability.
- **Domain add:** [ml-software-pytorch-jax-expert](.claude/agents/ml-software-pytorch-jax-expert.md) — for the DDP + gradient-checkpoint interaction and the `static_graph=true` invariant ("graph IS static" assertion).
- **Domain add:** [ml-protein-architect](.claude/agents/ml-protein-architect.md) — for the Hydra config-tree change (DDPStrategy structured override) and whether to flip in all 4 confidence yamls vs only the PAE one.

## Open questions for the user

1. **Should the DDP-strategy fix land as a yaml-default flip (preferred, affects all 4 confidence configs incl. SwissProt) or as a CLI-only override in the sbatch (minimal blast radius)?** I lean yaml flip — the SwissProt run that shipped in PR #6 must have been single-GPU or the symptom was masked; ANY future multi-GPU launch of any confidence config will hit the same crash. But this changes config defaults that the user may have validated separately. **Ask before opening the PR.**
2. **The 14-char `WANDB_API_KEY=YOUR_WANDB_KEY` placeholder in `.env` was commented out this session.** Should that change ship in the PR? `.env` is gitignored so the change does NOT propagate via git. The commit-out was a belt-and-suspenders move; the load-bearing fix is `/usr/bin/env -u WANDB_API_KEY` in the sbatch. If the user wants the `.env` to stay as a template-with-placeholders, **revert** the comment-out before continuing.
3. **The user cancelled job 53636 at config-dump stage** before training started — was this because of a remote concern (h100 capacity, the long config dump cluttering wandb, something noticed in the edits) that the next agent should address before resubmitting? **Ask.**

## Reference paths

- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Training sbatch: `/mnt/storage01/home/schekmenev/projects/complexa-flex/scripts/train_confidence_teddymer_pae.sbatch`
- PAE config: `/mnt/storage01/home/schekmenev/projects/complexa-flex/configs/confidence/distillation_teddymer_pae.yaml`
- H100 blob: `/netscratch/schekmenev/teddymer_v1_blob/` (h100 node view — invisible from head)
- GPU blob: `/netscratch/schekmenev/teddymer_v1_blob/` (head node view; same path, separate NVMe)
- Labs-staging blob copy: `/mnt/storage01/home/schekmenev/data/teddymer_v1_blob_stage/`
- Labs integrity snapshot: `/mnt/storage01/home/schekmenev/data/teddymer_v1/`
- Logs: `/mnt/storage01/home/schekmenev/projects/complexa-flex/logs/paedistill_*.{out,err}`
- WandB project: `confidence-distillation` (entity `stanislav-chekmenev`), run name `pae-distill-teddymer`.
- User's authorized email: `schekmenev@aithyra.at` (PR summary destination per CLAUDE.md).

## Project conventions reminder

- **Branch base:** new PR-fix work branches off `train_pae` (which already has `dev` merged in via commit `e12c298`). Do not rebase `train_pae` onto `dev` without checking with the user.
- **No Claude attribution** in commits or PR descriptions.
- **`gh` requires `module load gh`** in the same Bash call (environment module; not on PATH).
- **TDD** via subagents — `software-planning-architect` for the PR plan if it grows, `ml-protein-architect` or `ml-software-pytorch-jax-expert` for implementation, `code-review-debug-complexity-expert` mandatory on review.
- **Never `uv run`** — see memory `feedback-uv-run-breaks-venv`. Use `.venv/bin/python` directly.
- **Auto mode** was on for this session and the user kept it on. Continue under that contract unless the user signals otherwise.
