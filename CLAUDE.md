# CLAUDE.md — Proteina-Complexa

Project-specific instructions for Claude working in this repository. These rules apply to every Claude session (main thread and subagents) operating inside `complexa-flex/`.

## Load-on-demand convention

This file keeps every load-bearing invariant inline as a one-line guardrail plus a pointer to a detail file under [docs/claude/](docs/claude/). The detail files carry the rationale, reproduction steps, file:line forensics, and test contracts. **Read the matching detail file BEFORE acting in its area:**

- Before opening a PR → read [docs/claude/pr_protocol.md](docs/claude/pr_protocol.md).
- Before editing the confidence subsystem (`nn/confidence/`, `confidence/`, its configs or sbatches) → read [docs/claude/confidence_subsystem.md](docs/claude/confidence_subsystem.md).
- Before delegating to a subagent → read [docs/claude/subagents.md](docs/claude/subagents.md).
- Before starting a non-trivial change → read [docs/claude/dev_workflow.md](docs/claude/dev_workflow.md).
- At session start / end → read [docs/claude/session_lifecycle.md](docs/claude/session_lifecycle.md).

## Project at a glance

- **Project name.** Proteina-Complexa — atomistic flow-matching generative model for protein–protein and protein–ligand binder design, with test-time search over reward models (AF2 / RF3 / force fields), motif scaffolding (AME), and fold-class-conditioned generation.
- **Distribution / package name.** `proteinfoundation` (see [pyproject.toml](pyproject.toml)). Source lives under [src/proteinfoundation/](src/proteinfoundation/). Do not invent parallel package names.
- **Pinned environment.** uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5,<2.6 (2.6.x breaks checkpoint loading).
- **Never run `uv run` / `uv sync` / `uv lock`** — they re-resolve `pyproject.toml` (no `uv.lock` shipped), silently upgrade torch and remove atomworks / graphein / PyG / jax, trashing the venv. Use `source .venv/bin/activate` or `.venv/bin/python` directly; rebuild via `bash env/build_uv_env.sh --clean`; never substitute plain PyPI (CPU-only) torch wheels. Details: [env/build_uv_env.sh](env/build_uv_env.sh).
- **Compute-node venv staging: extract the pre-built tarball, never `uv run`.** Canonical tarball `$PROJECT_ROOT/venv.tar.gz` (labs NFS) with per-partition copies at `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`; `cp` fallback labs NFS → partition-local `/netscratch` (each partition has its own physical `/netscratch` — fallback mandatory), then `tar xzf`, then `srun ... "$ENV_LOCAL/bin/python" -m ...`. Pattern: [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch).
- **Configs.** Hydra tree under [configs/](configs/) — `generation/`, `nn/`, `nn_ae/`, `pipeline/`, `dataset/`, `design_tasks/`, `confidence/`, etc. Compose, don't duplicate. Mirror naming of existing entries (`search_binder_*`, `evaluate_*`, `analyze_*`).
- **Runtime artefacts** (`ckpts/`, `wandb/`, sample/eval output trees) belong in `.gitignore`. Never commit weights, large PDB dumps, or run directories.

## Confidence-head distillation subsystem

Sidecar Lightning module distils AF2 per-residue pLDDT and per-pair PAE from the frozen complexa trunk into a trainable student head. **Does not modify** `proteinfoundation.proteina` or the main `train.py`. Full rationale + file:line + test contracts: [docs/claude/confidence_subsystem.md](docs/claude/confidence_subsystem.md). Load-bearing invariants:

- **Layering rule:** `nn/confidence/` must NOT import from `confidence/`.
- **Trunk hook + AdaLN:** opt-in `expose_intermediates` (default `False` is bit-identical to legacy); sidecar reuses the trunk `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99`, zero-padded along the residue axis.
- **Pair-repr symmetrisation is the head's job:** `ConfidenceTrunk.forward` does NOT symmetrise `z`; symmetric heads prepend `z = 0.5*(z + z.transpose(-3,-2))`, `PaeHead` consumes `z` directly, `PlddtHead` is `s`-only.
- **WandB fragment reuse:** compose `- /logging/wandb@logging` + per-config `wandb_tags:`; never embed wandb keys in training YAMLs.
- **Rank-0 loguru gate:** every Hydra-launched DDP entry point calls `_gate_loguru_to_rank0()` as the FIRST line of `main()`.
- **DDP:** pair `find_unused_parameters=true` with `static_graph=true` (both mandatory; frozen-trunk params + reentrant ckpt in the head's trunk copy compose two failures).
- **Allocator:** confidence-distill sbatch must export `PYTORCH_ALLOC_CONF=expandable_segments:True` (reentrant ckpt + heterogeneous L fragments the allocator and OOMs after ~900 steps).
- **Resume:** every config exposes `resume_ckpt_path: null` → `trainer.fit(ckpt_path=...)`; sbatch contract via `RESUME_CKPT_PATH` env → Hydra override (orthogonal to `resume_id`).
- **MultiHeadConfidence weighting:** within-head `loss.ce_weight` / `loss.ev_weight`; across-head `loss_weights: {head: scalar}` (`0.0` short-circuits the total but still runs `_predict` — DDP footgun relying on `static_graph=true`).
- **Teddymer filter is geometry-only** (`interface_length > 10`); never pre-filter on `avg_int_plddt` / `avg_int_pae` — filtering on the label distribution by the label is a selection-bias antipattern.
- **LayerNorm-bias-leak invariant:** any `Linear(bias=False)→LayerNorm` residual into a masked sequence rep must multiply post-LN by `* mask` before the add (trained `LN(0)=beta` leaks bias into padded rows, contaminating valid cells); pin with a non-init test.

## Community-standard metric / reward parity

Port any community-standard PAE/pLDDT-derived metric or reward (colabdesign `get_ipsae_loss`, AF ipTM, RFdiff/BindCraft filters, AF3 kernels, Boltz/Chai scores) **verbatim** in the first PR, with a fp64 numpy reference **inside the test file** locking parity within `1e-5`. Keep upstream quirks (`1e-8` additives vs `clamp_min`, magic clips like ipSAE `L >= 27` / ipTM `n_eff >= 19`, two-direction reductions, per-row vs per-sample `d_0`) — they are the contract. Generalisations ship as **separately-named** follow-ups (`ipsae_contact_gated`, `iptm_soft`); never reuse a canonical name. Canonical refs: [community_models/colabdesign/af/loss.py](community_models/colabdesign/af/loss.py), [src/proteinfoundation/rewards/alphafold2_reward_utils.py](src/proteinfoundation/rewards/alphafold2_reward_utils.py). Binds every PR touching `nn/confidence/` `_metrics.py` / `_losses.py` and any future reward head (ipTM, iPLDDT, pAE_interaction, ipSAE).

## Development workflow (TDD default)

TDD is the main strategy for any non-trivial change, executed via the subagents: plan (software-planning-architect) → **write tests first** → implement (ml-protein-architect / ml-software-pytorch-jax-expert), vertical slice first (compiles, trains one step, samples one structure, evals one batch) → review. Minor changes (one-line fix, typo, comment, semantics-neutral config tweak) need no new test. Smoke tests (one-step train, one-sample gen, one-batch eval) are required for any training-loop / sampling / eval-harness change. Full test-quality checklist: [docs/claude/dev_workflow.md](docs/claude/dev_workflow.md).

## Subagent delegation

Route work to the subagent whose `description:` matches the task; dispatch in parallel when angles are independent. Full roster + scopes: [docs/claude/subagents.md](docs/claude/subagents.md).

## Pull-request review protocol

Classify every PR and state the tier on the first line of the PR body. Full panel, domain mapping, target-branch relaxation, blocking-vs-nonblocking disposition, re-review budget, and stuck-PR escape hatch: [docs/claude/pr_protocol.md](docs/claude/pr_protocol.md).

- **T1 — Trivial.** Docs/comment-only, private rename, semantics-neutral config tweak (ckpt/log cadence, batch size, num_workers, `--exclude`, wallclock, partition), CLAUDE.md / memory bookkeeping. No review.
- **T2 — Standard.** One subsystem, bug fix, one new test, behaviour-neutral refactor, config from existing fragments, bounded metric/loss tweak, < ~300 LoC non-test.
- **T3 — Substantial.** New head/architecture/training-stage/dataset-pipeline, cross-module refactor, community-metric port, any training/sampling/eval semantics or reproducibility change, or ≳ 300 LoC non-test.
- **Force-T3 triggers (override size):** training/sampling/eval semantics; a community-standard metric; `.ckpt` round-trip; DDP / strategy / precision wiring; the data-integrity gate.
- **Approval:** T1 self-approve after tests pass. T2/T3 — every appointed reviewer approves or raises only non-blocking findings; mergeable once all blocking findings are addressed. **PRs into `main` get full T3 regardless of size.**

## Git / GitHub conventions

- **`gh` needs `module load gh`** in the same Bash invocation (`module load gh && gh ...` every time; shell state does not persist between Bash calls).
- **No Claude attribution** in commits or PRs — no "Generated with", "Co-Authored-By: Claude", or similar. Author as the user. (Overrides the default Claude Code commit template.)
- **Branching.** Work on a feature branch off `dev` (the main branch). Open PRs into `dev`.
- **Commits.** Small, focused, `verb: short subject`. Group by intent, not by file.
- **Never amend a published commit; never `--no-verify`, `--no-gpg-sign`, or force-push to `dev` / `main`.**
- **`.gitignore`.** Runtime artefacts (`ckpts/`, `wandb/`, sample/eval outputs, `.venv/`, `__pycache__/`, large PDB / mmCIF dumps) stay ignored. `.claude/agents/` is tracked; everything else under `.claude/` stays ignored.

## Style and code hygiene

- Match repo style; type-hint public APIs.
- No speculative abstractions, no half-finished scaffolds, no parallel implementations of an existing module.
- Default to **no comments**; add one only when the *why* is non-obvious (hidden constraint, bug workaround, subtle invariant). Don't narrate the code.
- No emojis in code, tests, configs, commits, or PRs unless the user explicitly asks.
- Prefer editing existing files over creating new ones. Never create `.md` / README files unless the user asks.

## Reproducibility hygiene

- Seeded splits and seeded dataloaders. State the seed where it's set; do not silently re-seed.
- Deterministic ops where they matter (geometry math, eval); document the throughput cost when determinism is on.
- Checkpoints must round-trip on the pinned Lightning version. Do not bump Lightning without verifying weight loading on an existing checkpoint.
- Smoke-train and smoke-sample on a tiny subset is part of "done", not optional.

## Session lifecycle

Start every session with `/recall` (clean prior ship) or `/recall_and_follow_latest` (work in flight — also reads the latest `docs/handoff/`). End every session (on any "done / wrap" signal, even without typing `/ul`) with the `/ul` workflow: refresh `MEMORY.md`, amend `CLAUDE.md` if a new project-wide rule was established, and write a handoff `docs/handoff/YYYY-MM-DD_<topic-slug>.md` **iff** work is left in flight (empty handoffs forbidden). Bookkeeping commit: iff `/ul` produced a `CLAUDE.md` edit and/or new handoff, commit exactly those file(s) on the current branch (`git add` by explicit path, never `-A`); never push. Full mechanics: [docs/claude/session_lifecycle.md](docs/claude/session_lifecycle.md).

**Read memories lazily.** `MEMORY.md` (loaded at session start) and the memory-file pointers it lists are an *index*, not content to preload. Do NOT open the individual memory files under `memory/` — or the `docs/claude/` detail files — up front; reading them all fills the 200K context window. Open a memory (or detail) file only when its one-line hook is relevant to the task at hand, and read only that one. The same rule applies to the `docs/claude/` pointers in the Load-on-demand convention above: fetch on demand, one at a time.
