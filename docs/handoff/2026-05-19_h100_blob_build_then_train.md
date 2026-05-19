# Handoff — 2026-05-19 — h100 blob build, then PAE distillation training

## TL;DR for the next agent

User wants the **PAE confidence-head distillation** training run to start on the **h100 partition** and to confirm that **training loss decreases**, that the **first validation epoch completes**, and that **the first checkpoint saves**. Then prepare a PR to `dev` from `train_pae` (do NOT merge), send the user a summary email, and run `/ul`.

This session resolved the venv, integrity-snapshot, and partition-storage blockers and committed those fixes (commit `946fbc1` on `train_pae`, then merged `dev` at `e12c298`). The remaining blocker is that the **122 GB Teddymer data blob** lives only on the **gpu-partition `/netscratch`**, and h100 has its own (separate) `/netscratch` that is empty. To unblock training on h100, a CPU-only blob-repack was submitted on the h100 partition itself: **SLURM job 53585**. At the time of writing (08:24 CEST 2026-05-19) it has processed ~40 k / 555 791 parents at ~2.7 MB/s; ETA roughly 13 h. The user's instruction was: **wait for 53585 to finish, abort the training task, run `/ul`**.

Net result: when the next session resumes via `/recall_and_follow_latest`, blob build 53585 should be **complete**. The training resubmission is the next agent's job — see "PR plan" below.

## What is on disk

### Code / configs (committed)

- Branch `train_pae`, ahead of `dev` by one feature commit + one merge commit:
  - `946fbc1 fixing train bugs` — user-authored commit containing my edits to [scripts/train_confidence_teddymer_pae.sbatch](../../scripts/train_confidence_teddymer_pae.sbatch) (partition swap rtx6000 → h100nvl, venv staging via tarball with cross-partition fallback, `srun ... "$ENV_LOCAL/bin/python" -m ...` instead of `uv run`), [.gitignore](../../.gitignore) (added `venv.tar.gz`), [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml) (`batch_size: 6` → `4`), [configs/training/confidence_distill.yaml](../../configs/training/confidence_distill.yaml) (`max_epochs: 50` → `25`).
  - `e12c298 Merge branch 'dev' into train_pae` — user-authored merge.
- Working tree is clean.

### Filesystem state

- **venv tarball — labs-NFS (cross-partition source-of-truth):** `/mnt/storage01/home/schekmenev/projects/complexa-flex/venv.tar.gz` (3 826 858 794 bytes, 3.6 GB). Gitignored.
- **venv tarball — gpu-partition `/netscratch`:** `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz` (same bytes; staged this session).
- **venv tarball — h100-partition `/netscratch`:** `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz` (same bytes; the sbatch's `cp` fallback already populated this when job 53525 ran on h100 earlier in this session — verified by job 53525's stdout: "venv tarball missing at ...; copying from /mnt/storage01/.../venv.tar.gz" → "env unpacked to /netscratch/.../complexa-paedistill/.venv").
- **Project `.venv`:** `/mnt/storage01/home/schekmenev/projects/complexa-flex/.venv/` (5.1 GB). Rebuilt this session via `env/build_uv_env.sh --clean`. `python -c "import torch, atomworks, graphein, torch_geometric, lightning, hydra; from proteinfoundation.confidence.train_confidence import main; print('OK')"` passes. torch 2.10.0+cu130, lightning 2.5.6, hydra 1.3.1, tqdm 4.66.4.
- **Integrity-gate snapshot — labs NFS:** `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5` (148 B) and `/mnt/storage01/home/schekmenev/data/teddymer_v1/view_config.yaml` (566 B). Staged this session by copying from the gpu-partition blob's sidecar; matches view version `teddymer_v1_blob_20260518_068d9c67`.
- **Teddymer data blob — gpu-partition `/netscratch` (verified):** `/netscratch/schekmenev/teddymer_v1_blob/{data.blob, dimers.parquet, locator_rows.parquet, data.md5, view_config.yaml}`. Total 127 GB. md5s verified clean against `data.md5` sidecar.
- **Teddymer data blob — h100-partition `/netscratch` (in flight):** **partial / not yet complete**. Being written by SLURM job 53585 on node H100Azure02, running [scripts/build_teddymer_blob.sbatch](../../scripts/build_teddymer_blob.sbatch) (mirror with `--partition=h100` instead of `--partition=cpu`, sbatch source at `/tmp/build_teddymer_blob_h100.sbatch` on this session's working host — NOT committed). Live log at `/mnt/storage01/home/schekmenev/projects/complexa-flex/logs/teddymer_blob_repack_h100_53585.err`. Output dir `/netscratch/schekmenev/teddymer_v1_blob/` on the h100 partition. The build is from-scratch (h100's `/netscratch` was empty), not resumable mid-flight, but the build script atomically renames at the end so a clean completion is binary "all there" vs "still working".
- **Stale training run dir on h100:** `/netscratch/schekmenev/complexa-paedistill/runs/53525/` may exist (job 53525 failed at the integrity gate before producing real outputs). Safe to ignore or remove.

### Earlier failed training jobs (for context)

- **53521** (gpu partition, rtx6000nvl) — failed `uv run` resolution on tqdm. Root cause: `uv run` re-resolves pyproject which has `tqdm==4.66.4`, cu130 mirror has only 4.66.5, "first index wins" prevents PyPI fallback. Resolved by switching sbatch to `"$ENV_LOCAL/bin/python"` (no `uv run`).
- **53524** (h100, queued) — cancelled at user request to switch partition.
- **53525** (h100) — failed at integrity gate: snapshot path `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5` didn't exist. Resolved by staging snapshot + view_config.yaml to that path.
- **53585** (h100, blob build) — **still running** at handoff time; see "Open questions" below for the eventual exit-state.

## What is NOT on disk yet (the gap)

1. **Completed h100 blob.** When 53585 finishes, the `/netscratch/schekmenev/teddymer_v1_blob/` on h100 should contain `data.blob` (~127 GB), `dimers.parquet` (~35 MB), `locator_rows.parquet` (~69 MB), `data.md5`, `view_config.yaml`. The sbatch will also write to `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5` (overwriting the existing labs snapshot — should produce identical bytes since the source data and `dimers.parquet`/`locator_rows.parquet` are deterministic from the same input; verify md5 equality before proceeding).
2. **Successful training job** — the actual training run. Needs to be resubmitted via `sbatch scripts/train_confidence_teddymer_pae.sbatch` (h100 partition). User's success criterion: **(a) job runs, (b) training loss decreasing, (c) first validation epoch completes without bug, (d) first checkpoint saves without bug**.
3. **PR from `train_pae` → `dev`** — not yet opened. Scope: the venv-staging + partition-storage fixes (`946fbc1`) plus the integrity-snapshot path staging plus the CLAUDE.md amendments from this `/ul`. Do NOT merge.
4. **Summary email to `schekmenev@aithyra.at`** — once training is healthy, send a brief summary.
5. **Final `/ul`** — by the next agent, once training is confirmed healthy.

## Architecture reference

- **sbatch staging pattern (canonical):** [scripts/train_confidence_teddymer_pae.sbatch:38-73](../../scripts/train_confidence_teddymer_pae.sbatch#L38-L73) — the venv staging + cross-partition fallback. The `ENV_TARBALL_SRC` defaults to `$PROJECT_ROOT/venv.tar.gz` (labs NFS). The training launch at line 138 uses `"$ENV_LOCAL/bin/python"`, never `uv run`. Mirror this pattern in any future sbatch that needs the project venv.
- **Integrity gate:** [src/proteinfoundation/datasets/teddymer/integrity.py:88-140](../../src/proteinfoundation/datasets/teddymer/integrity.py#L88-L140). The gate compares `view_root` (live blob) against `snapshot_path` (canonical md5 list) and ALSO compares `view_config.yaml` version strings between them. Both `data.md5` and `view_config.yaml` must exist beside each other at `snapshot_path.parent`.
- **Training entry point:** [src/proteinfoundation/confidence/train_confidence.py:42-50](../../src/proteinfoundation/confidence/train_confidence.py#L42-L50) — integrity gate fires immediately after `L.seed_everything` and BEFORE head construction.
- **Distillation config:** [configs/confidence/distillation_teddymer_pae.yaml](../../configs/confidence/distillation_teddymer_pae.yaml). `integrity.snapshot_path: /mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`.
- **Blob build (cpu partition default):** [scripts/build_teddymer_blob.sbatch](../../scripts/build_teddymer_blob.sbatch). Job 53585's sbatch is the same content with `--partition=h100`; the file at `/tmp/build_teddymer_blob_h100.sbatch` on the head node is the literal copy. If the next agent needs to re-run on h100 (e.g. 53585 failed), recreate that file or just submit the original sbatch overridden in-flight.

## PR plan

**Single PR — `train_pae` → `dev`.** Do NOT merge; the user wants to review the diff and authorise the merge.

- **Scope:** the venv-staging cross-partition pattern + the h100 partition switch + the integrity-snapshot path documentation + CLAUDE.md amendments from this `/ul`.
- **Test plan:** the test of record is the training run itself — there are no unit tests for sbatch behaviour. Verify in this order before opening the PR:
  1. Blob build 53585 completed cleanly (`sacct -j 53585 -X` shows `COMPLETED`; `tail -n 4` of its stderr shows the "repack done" line; live `data.md5` matches what's already in `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`).
  2. Resubmit `sbatch scripts/train_confidence_teddymer_pae.sbatch`. Job state transitions PENDING → RUNNING.
  3. Integrity gate passes (look for the `Integrity` / `md5` log lines without `IntegrityError`).
  4. Training loss decreases over the first ~25 logged steps (default `log_every_n_steps: 25`).
  5. First validation epoch completes without crash (look for `val/pae/...` log lines).
  6. First checkpoint saves to `$DATA_ROOT/runs/<jid>/...` (look for `ModelCheckpoint` saving line).
- **Implementation deliverables:** none new. The committed `946fbc1` plus this `/ul`'s `CLAUDE.md` + handoff edits are the full delta.
- **Reviewer panel:** mandatory [code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md) + [ml-protein-architect](../../.claude/agents/ml-protein-architect.md) (data pipeline / configs / module layout touch). The PR doesn't touch losses or guidance, so the generative / structural-biology reviewers are not required.

## Open questions for the user

- **Q1.** Job 53585's last observed throughput was ~2.7 MB/s (NVMe write rate, dominated by source-read tar IO). At that rate the full 127 GB blob takes ~13 h, vs the gpu-partition build that finished in roughly the same wall-clock per memory. Is this the expected rate for the h100 partition's storage, or is something pathological (e.g. labs DB share contention) that should be investigated before next time?
- **Q2.** Should the next agent commit a new sbatch — say `scripts/build_teddymer_blob_h100.sbatch` — that explicitly targets the h100 partition (sister to the existing cpu-partition one)? This is currently a `/tmp/` artefact. The committed pattern is "edit `--partition=` in the cpu sbatch", but the agent should not destructively edit a tracked sbatch — having a partition-specific variant is cleaner.
- **Q3.** The labs DB share at `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/teddymer_v1/data.md5` was the ORIGINAL canonical snapshot (per [scripts/build_teddymer_blob.sbatch](../../scripts/build_teddymer_blob.sbatch:28)). The distillation config now points at `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`. This session restaged the latter from the netscratch blob's sidecar. If the labs DB snapshot disagrees with the user's home-NFS snapshot, the training gate compares against the home-NFS one — confirming this is the intended target, not the labs DB one.

## Reference paths

```
PROJECT_ROOT       = /mnt/storage01/home/schekmenev/projects/complexa-flex
BRANCH             = train_pae  (off dev; last commit 946fbc1 + merge e12c298)
PYTHON             = $PROJECT_ROOT/.venv/bin/python
VENV_TARBALL_LABS  = $PROJECT_ROOT/venv.tar.gz                         # 3.6 GB
VENV_TARBALL_GPU   = /netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz   # gpu partition
VENV_TARBALL_H100  = /netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz   # h100 partition (same path, separate physical store)
BLOB_GPU           = /netscratch/schekmenev/teddymer_v1_blob/                 # 127 GB, verified
BLOB_H100          = /netscratch/schekmenev/teddymer_v1_blob/                 # under construction by job 53585
SNAPSHOT_LABSNFS   = /mnt/storage01/home/schekmenev/data/teddymer_v1/{data.md5, view_config.yaml}
LOGS               = $PROJECT_ROOT/logs/paedistill_{53521,53524,53525}.{out,err}
                     $PROJECT_ROOT/logs/teddymer_blob_repack_h100_53585.{out,err}
SBATCH_TRAIN       = scripts/train_confidence_teddymer_pae.sbatch
SBATCH_BLOB_CPU    = scripts/build_teddymer_blob.sbatch
SBATCH_BLOB_H100   = /tmp/build_teddymer_blob_h100.sbatch  # NOT committed; see Q2
TRAIN_CONFIG       = configs/confidence/distillation_teddymer_pae.yaml
DATASET_CONFIG     = configs/dataset/unified/teddymer_with_plddt_and_pae.yaml
```

## Project conventions reminder

- Branch off `dev` for new feature work; `train_pae` is the in-flight branch and should NOT be the base for new features once the PR lands.
- No Claude attribution in commits or PRs (per [CLAUDE.md](../../CLAUDE.md)).
- `gh` needs `module load gh` in the same Bash call.
- TDD via subagents — but the next agent's task is verification of an in-flight training run, not new code, so the discipline applies only to whatever code-change might surface during debugging.
- `/ul` standing rule binds the next session even without typing it explicitly — when training is healthy and the PR is open, run `/ul` and write a follow-up handoff iff there's still in-flight work.
