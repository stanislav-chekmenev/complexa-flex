# PAE first-training milestone — handoff

## TL;DR

PAE confidence-head distillation on Teddymer is now **training successfully end-to-end**: validation at step 2000 (run 53677, h100nvl ×2 DDP, bf16-mixed) reported `val/pae/loss_ce = 1.6373`, `pae_mae = 1.51 Å`, `pearson_r = 0.890`, `spearman_r = 0.905`, `ece = 0.039`, and the first checkpoint `pae-ce-000-1.6373.ckpt` (2.0 GB) is on disk. Three uncommitted fixes on `train_pae` need to land in a PR into `dev`, and the user wants an email summary after merge. Training continues at ~1.37 it/s (~12 h / epoch) and is *not* blocked by any of this paperwork.

## What is on disk

**On `train_pae` (4 uncommitted, of which 1 is the `/ul` bookkeeping commit produced by this handoff):**

- [configs/confidence/distillation_teddymer_pae.yaml](../../configs/confidence/distillation_teddymer_pae.yaml) — adds `val_check_interval: 2000` under the `trainer:` block, so val + checkpoint fires every ~22 min instead of every ~12 h.
- [scripts/train_confidence_teddymer_pae.sbatch](../../scripts/train_confidence_teddymer_pae.sbatch) — adds `export PYTORCH_ALLOC_CONF=expandable_segments:True` between the NCCL block and `cd "$PROJECT_ROOT"`. Carries an inline comment explaining the fragmentation root cause.
- [src/proteinfoundation/nn/confidence/pae_head.py:156-162](../../src/proteinfoundation/nn/confidence/pae_head.py#L156-L162) — gates `"loss_total"` emission to `stage != "train"` (mirrors the convention in `plddt_head.py` and `plddt_sequence_only_head.py`). Fixes the duplicate `train/pae/loss_step` vs `train/pae/loss_total_step` keys in wandb.
- [CLAUDE.md](../../CLAUDE.md) — adds a project-wide rule mandating `PYTORCH_ALLOC_CONF=expandable_segments:True` for any confidence-distill sbatch that uses reentrant ckpt under variable-L inputs (bookkeeping commit from `/ul` step 4).

**Live artifacts (NOT in repo, do not recreate):**

- `/mnt/storage01/home/schekmenev/projects/complexa-flex/confidence-distillation/pae-distill-teddymer/checkpoints/pae-ce-000-1.6373.ckpt` (2.0 GB, written 2026-05-19 15:23, monitor = `val/pae/loss_ce`, `save_top_k=3`).
- WandB run: project `confidence-distillation`, run name + id `pae-distill-teddymer`, entity `stanislav-chekmenev`. Local dir at `wandb/run-20260519_145553-pae-distill-teddymer/`.
- SLURM job 53677 — RUNNING on H100Azure09 (h100 partition). At step ~2700+/63168 of epoch 0 as of this handoff. The persistent Monitor watcher (task `btrvxxf8a`) is streaming progress lines into the session.
- Stale outputs: `/mnt/storage01/home/schekmenev/projects/complexa-flex/outputs/2026-05-19/14-50-42/` has only `.hydra/config.yaml` + `train_confidence.log` (no checkpoints, no `lightning_logs/`). Per [[lightning-ckpt-path-with-wandb]] memory, WandbLogger overrides the dirpath derivation.

## What is NOT on disk yet (the gap)

1. **PR `train_pae` → `dev`** — not opened.
2. **Reviewer-panel approval cycle** — pending. Mandatory `code-review-debug-complexity-expert` + `ml-software-pytorch-jax-expert` (for the allocator + duplicate-loss fix) + `ml-protein-architect` (for the yaml + sbatch + head-loss code).
3. **Merge into `dev`** — pending.
4. **Email to `schekmenev@aithyra.at`** — pending. User wants: training-report summary, PR description, next steps for distillation of other rewards.

## Architecture reference

- Loss-emission convention (head-level): mirror [src/proteinfoundation/nn/confidence/plddt_head.py](../../src/proteinfoundation/nn/confidence/plddt_head.py) — `"loss"` always, `"loss_total"` only when `stage != "train"`. After this PR's fix, `pae_head.py` matches.
- Trainer instantiation entry point: [src/proteinfoundation/confidence/train_confidence.py:53-54](../../src/proteinfoundation/confidence/train_confidence.py#L53-L54). `hydra.utils.instantiate(cfg.trainer, logger=logger, _convert_="partial")` consumes the `trainer:` block in [configs/confidence/distillation_teddymer_pae.yaml](../../configs/confidence/distillation_teddymer_pae.yaml). `val_check_interval` lands on this trainer (not on the partial_autoencoder trainer at `src/proteinfoundation/partial_autoencoder/train.py:271`, which was the user's initial concern but turned out not to apply).
- Reentrant-ckpt fragmentation site: [src/proteinfoundation/nn/modules/pair_update.py](../../src/proteinfoundation/nn/modules/pair_update.py) lines ~82, 88, 105, 107, 110.
- WandbLogger build site (sets `save_dir=str(project_root)`, which is why checkpoints land directly under `$PROJECT_ROOT/confidence-distillation/...`): [src/proteinfoundation/confidence/train_confidence.py](../../src/proteinfoundation/confidence/train_confidence.py).

## PR plan

### PR scope: `train_pae` → `dev` — "PAE training infra fixes + log-key dedup"

**Files (3 — `CLAUDE.md` is committed separately by `/ul` step 4 and will be picked up by the PR diff automatically):**
1. `configs/confidence/distillation_teddymer_pae.yaml` — `val_check_interval: 2000` under `trainer:`.
2. `scripts/train_confidence_teddymer_pae.sbatch` — `PYTORCH_ALLOC_CONF=expandable_segments:True` export with inline justification.
3. `src/proteinfoundation/nn/confidence/pae_head.py` — gate `loss_total` to non-train stage.
4. `CLAUDE.md` — project-wide allocator rule (bookkeeping commit; carried into the PR).

**Test plan (write-tests-first per CLAUDE.md TDD discipline):**

- `pae_head.py` loss-key change is verifiable by extending an existing integration test (or adding a small unit test) that asserts `"loss_total" not in log_dict` when `stage == "train"` and `"loss_total" in log_dict` when `stage == "val"`. A symmetric assertion already exists for `plddt_head.py` at [tests/integration/confidence/test_validation_logging_extension.py](../../tests/integration/confidence/test_validation_logging_extension.py); mirror it for PAE.
- `val_check_interval` is a Hydra config-only change; smoke-instantiate the trainer and assert `trainer.val_check_interval == 2000`.
- The sbatch change is an environment export; verify via shellcheck + dry-run that the env is set before the `srun python` line (no Python-level test possible). The empirical proof is in the job-53677 vs job-53667 outcome.
- CLAUDE.md is documentation; no code test.

Reference: tests already pass after the `loss_total` fix per my mid-session check (126 passed, 62 warnings in 19.73 s).

**Reviewer panel (run in parallel — mandatory per CLAUDE.md):**

- `code-review-debug-complexity-expert` — always required, will scrutinize correctness/edge-cases of the loss-key gating and the yaml/sbatch deltas.
- `ml-software-pytorch-jax-expert` — required because the change touches PyTorch's CUDA allocator config and the loss/log-emission contract of a `nn.Module`.
- `ml-protein-architect` — required because the change touches a Hydra config, an sbatch infrastructure file, and a confidence-head module (project-level architecture).

Run the three reviewers concurrently. If any flags an issue, fix → re-run the full panel → loop until unanimous approval. If the loop fails to converge (per CLAUDE.md "stuck-PR escape hatch"): terminate, report in chat, and email `schekmenev@aithyra.at` summarising the deadlock.

**Merge:** standard `gh pr merge --squash --delete-branch` (or `--merge` if the user prefers preserving the per-commit history — confirm with user style; CLAUDE.md doesn't mandate squash). Note `module load gh` requirement.

### Email after merge

To `schekmenev@aithyra.at` (the only authorized destination per CLAUDE.md). Content:

1. **Training report** — first-val metrics + checkpoint path + throughput + ETA. Include the failure-then-fix arc (53667 OOM → expandable_segments → 53677 clean).
2. **PR description** — PR number, link, what shipped, what tests added.
3. **Next steps for distillation of other rewards** — propose the order: (a) pLDDT on Teddymer (the only other single-head config that exists), (b) joint pLDDT+PAE multi-head (`distillation_teddymer_multihead.yaml` per [[teddymer-pae-distillation-shipped]]), (c) additional heads (PDE, ipLDDT, ipTM) following the pattern in `src/proteinfoundation/nn/confidence/`. Flag that single-head PAE is still mid-epoch — full convergence won't be in until ~12 h × 25 epochs = ~12.5 days, so the next reward distillation can either wait for convergence or start in parallel on a different GPU slice.

No Claude attribution. No secrets. No URLs other than the PR URL.

## Open questions for the user

- **Merge style.** Squash vs merge-commit for PR `train_pae` → `dev`. CLAUDE.md doesn't mandate one. Prior PRs on this repo (e.g. #16) used merge-commits via the GitHub UI. Default to merge-commit unless the user prefers squash.
- **Order of next reward distillation.** pLDDT on Teddymer single-head vs jumping straight to joint multi-head. Should be answered in the email.
- **Whether to keep 53677 running through full convergence**, or kill once a few more checkpoints have landed to free the H100 slot for the next reward distillation. ~12 h × 25 epochs is a long commitment for a *first* training number.

## Reference paths

- Project root: `/mnt/storage01/home/schekmenev/projects/complexa-flex`
- Live PAE checkpoint dir: `/mnt/storage01/home/schekmenev/projects/complexa-flex/confidence-distillation/pae-distill-teddymer/checkpoints/`
- WandB local dir: `/mnt/storage01/home/schekmenev/projects/complexa-flex/wandb/run-20260519_145553-pae-distill-teddymer/`
- SLURM logs: `/mnt/storage01/home/schekmenev/projects/complexa-flex/logs/paedistill_53677.{out,err}`
- Teddymer blob (compute-node side, h100 partition): `/netscratch/schekmenev/teddymer_v1_blob/` — verified by integrity gate at train-start (version `teddymer_v1_blob_20260518_068d9c67`).
- Venv tarball: `$PROJECT_ROOT/venv.tar.gz` (labs NFS, gitignored) → per-partition `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`.

## Project conventions reminder

- Branch base for any new feature work: **`dev`** (CLAUDE.md). This handoff lives on `train_pae`; the next agent's *follow-up* work (after the PR merges) branches off `dev`, not `train_pae`.
- No Claude attribution in commits, PRs, or emails.
- `gh` is an environment module: `module load gh && gh ...` in the same Bash invocation.
- TDD is the default for medium/major changes — write tests first via [.claude/agents/software-planning-architect.md](../../.claude/agents/software-planning-architect.md) and implement via [.claude/agents/ml-protein-architect.md](../../.claude/agents/ml-protein-architect.md) or [.claude/agents/ml-software-pytorch-jax-expert.md](../../.claude/agents/ml-software-pytorch-jax-expert.md).
- Never `uv run`, `uv sync`, or `uv lock`. Use `.venv/bin/python` directly.
- WANDB_API_KEY lives in `~/.netrc` (mode 600), never in `.env`. The sbatch unsets it defensively before `srun`.
- The h100 partition's `/netscratch/schekmenev/` is **invisible from the head node** — inspect via `srun --partition=h100 ... ls`.
