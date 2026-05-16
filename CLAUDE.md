# CLAUDE.md — Proteina-Complexa

Project-specific instructions for Claude working in this repository. These rules apply to every Claude session (main thread and subagents) operating inside `complexa-flex/`.

## Project at a glance

- **Project name.** Proteina-Complexa — atomistic flow-matching generative model for protein–protein and protein–ligand binder design, with test-time search over reward models (AF2 / RF3 / force fields), motif scaffolding (AME), and fold-class-conditioned generation.
- **Distribution / package name.** `proteinfoundation` (see [pyproject.toml](pyproject.toml)). Source lives under [src/proteinfoundation/](src/proteinfoundation/). Do not invent parallel package names.
- **Pinned environment.** uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5,<2.6 (2.6.x breaks checkpoint loading). Use `uv run …` or activate `.venv/`. Never substitute plain PyPI torch wheels (CPU-only).
- **Configs.** Hydra tree under [configs/](configs/) — `generation/`, `nn/`, `nn_ae/`, `pipeline/`, `dataset/`, `design_tasks/`, `confidence/`, etc. Compose, don't duplicate. Mirror naming of existing entries (`search_binder_*`, `evaluate_*`, `analyze_*`).
- **Runtime artefacts** (`ckpts/`, `wandb/`, sample/eval output trees) belong in `.gitignore`. Never commit weights, large PDB dumps, or run directories.

## Confidence-head distillation subsystem

A sidecar Lightning module distils AF2 per-residue pLDDT from the frozen complexa trunk into a trainable light-weight student head. **Does not modify** `proteinfoundation.proteina` or the main `train.py` entry point.

- **Package layout.** [src/proteinfoundation/nn/confidence/](src/proteinfoundation/nn/confidence/) (heads + trunk + registry) and [src/proteinfoundation/confidence/](src/proteinfoundation/confidence/) (sidecar Lightning module + loss + metrics + Hydra entry point). Heads register via `@register_confidence_head(name)` so new heads (ipTM / ipAE / ipLDDT) drop in by subclassing `BaseConfidenceHead`.
- **Entry point.** `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_swissprot`. SLURM-launch via [scripts/train_confidence_swissprot.sbatch](scripts/train_confidence_swissprot.sbatch).
- **Dataset.** [configs/dataset/unified/afdb_monomers_with_plddt.yaml](configs/dataset/unified/afdb_monomers_with_plddt.yaml) extends the AFDB monomer dataset with `AddPLDDTFromBFactor` (AF2 `[0, 100]` scale, 50 bins of width 2) and optional cluster-30% held-out split via `cluster_column: unicluster`.
- **Loss / metrics.** Combined `0.7 * masked_CE + 0.1 * SmoothL1(EV)`, reduction `sum(loss * mask) / mask.sum().clamp_min(1)`. Validation logs accuracy, MAE (in pLDDT units), Pearson, Spearman, stratified MAE by pLDDT bucket, equal-width ECE, equal-mass adaptive ECE, and a per-bucket reliability diagram (rank-0 only).
- **Trunk hook.** `LocalLatentsTransformer` has an opt-in `expose_intermediates: bool = False` kwarg. When `True`, `forward(input)` adds `nn_out["trunk_intermediates"]` carrying `(s, z, mask, orig_mask, n_orig)`. Default `False` is bit-identical to the legacy forward (enforced by `tests/regression/test_flow_matching_loss_unchanged.py` against a baseline fixture).
- **AdaLN cond.** The sidecar reuses the trunk's existing `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99` to produce the same `cond` the trunk consumes; zero-padded along the residue axis to `n_ext` if concat features extend the sequence.

## Subagent roster

Subagent definitions are tracked under [.claude/agents/](.claude/agents/). Each agent has a defined scope; route work to the agent whose description matches the task:

| Agent | Scope |
| --- | --- |
| [software-planning-architect](.claude/agents/software-planning-architect.md) | Translate ambiguous asks into structured implementation plans. Upstream of implementation. |
| [ml-protein-architect](.claude/agents/ml-protein-architect.md) | Project-level architecture and *writes/edits code* once a plan is agreed. |
| [ml-software-pytorch-jax-expert](.claude/agents/ml-software-pytorch-jax-expert.md) | PyTorch and JAX framework internals (autograd, compile, FSDP/SPMD, profiling, kernels). |
| [code-review-debug-complexity-expert](.claude/agents/code-review-debug-complexity-expert.md) | Code review, systematic debugging, algorithmic / complexity optimisation. |
| [generative-protein-scientist](.claude/agents/generative-protein-scientist.md) | Applied generative modelling for protein structure/sequence (FM, diffusion, guidance, search). |
| [generative-flow-stochastic-math-expert](.claude/agents/generative-flow-stochastic-math-expert.md) | Math of diffusion / flow matching / OT / guidance / SMC. |
| [structural-biology-binder-expert](.claude/agents/structural-biology-binder-expert.md) | Folded protein–protein binder biophysics and field practice. |
| [structural-biology-idp-smallmol-expert](.claude/agents/structural-biology-idp-smallmol-expert.md) | IDP/IDR, motif-driven, and small-molecule binder design. |
| [physics-statmech-md-dft-expert](.claude/agents/physics-statmech-md-dft-expert.md) | Stat mech, MD, free-energy methods, DFT, MLIP literature. |
| [xray-crystallography-binder-ml](.claude/agents/xray-crystallography-binder-ml.md) | X-ray crystallography + ML, oriented to binder validation. |

When delegating, pick the agent whose `description:` line matches the task. If multiple apply, dispatch them in parallel where they are working on independent angles.

## Development workflow

### TDD is the default for major and medium-size tasks

Test-driven development is the **main strategy** for any non-trivial change. The TDD discipline is executed *via the subagents in [.claude/agents/](.claude/agents/)* — the main thread coordinates; agents (planner, implementer, framework expert, reviewer) do the actual planning, writing, and reviewing.

For each non-trivial change:

1. **Plan with [software-planning-architect](.claude/agents/software-planning-architect.md)** — translate the ask into a concrete plan with data contracts, milestones, exit and abort criteria, and an explicit verification strategy.
2. **Write tests first.** Tests are written before the implementation they cover. Implementation is then driven by making the failing tests pass.
3. **Implement with [ml-protein-architect](.claude/agents/ml-protein-architect.md)** (or [ml-software-pytorch-jax-expert](.claude/agents/ml-software-pytorch-jax-expert.md) when the work is framework-internal). Vertical slice first — smallest end-to-end version that compiles, trains one step, samples one structure, evaluates one batch — then horizontal expansion.
4. **Review** with the agents appointed for the PR (see below).

**Minor changes** (one-line fix, typo, comment, config tweak that does not change semantics, isolated documentation edit) do **not** require a new test. Use judgement; err toward writing a test when the change crosses a module boundary, touches a training-loop or eval-harness contract, or modifies a Hydra default that other configs compose.

### What an effective test looks like in this repo

Tests must:

- **Cover the primary logic** of the function / method / class / pipeline they target — not just happy-path smoke.
- **Be effective and complexity-optimised** — exercise the behaviour that matters, not exhaustively enumerate trivial cases. Prefer a small number of targeted tests over a large grid of near-duplicates.
- **Catch real edge cases relevant to this domain.** Empty input, single residue, single atom, masked / NaN entries, very long sequences, multimer crops, ligand-present vs absent, fp16/bf16 vs fp32 numerical drift, equivariance under random rotation+translation, dataloader determinism under the same seed, mismatched device / dtype.
- **Be readable.** Each test makes the property under test obvious from its name and assertions; a reader can tell *what would break* from a one-line read.
- **Pass on every PR.** A PR is not mergeable while any required test fails. CI green is a necessary, not sufficient, gate — review still applies.

Smoke tests (one-step train, one-sample generation, one-batch eval) are required for any change that touches the training loop, sampling pipeline, or eval harness, on top of the unit/integration tests for the changed module.

## Pull-request review protocol

### Choosing reviewers

For every PR, you (the main thread) appoint a panel of reviewer subagents from [.claude/agents/](.claude/agents/) based on what the PR actually touches.

- **[code-review-debug-complexity-expert](.claude/agents/code-review-debug-complexity-expert.md) is mandatory on every PR**, regardless of subject matter.
- Then add the subagents whose domain the PR touches. Examples:
  - PR adds a new loss / schedule / guidance scheme → include [generative-protein-scientist](.claude/agents/generative-protein-scientist.md) and [generative-flow-stochastic-math-expert](.claude/agents/generative-flow-stochastic-math-expert.md).
  - PR adds protein–protein binder logic → include [structural-biology-binder-expert](.claude/agents/structural-biology-binder-expert.md).
  - PR adds IDR / motif / small-molecule logic → include [structural-biology-idp-smallmol-expert](.claude/agents/structural-biology-idp-smallmol-expert.md).
  - PR touches MD reward / DFT / MLIP scoring → include [physics-statmech-md-dft-expert](.claude/agents/physics-statmech-md-dft-expert.md).
  - PR touches PyTorch / JAX internals, FSDP, `torch.compile`, custom kernels → include [ml-software-pytorch-jax-expert](.claude/agents/ml-software-pytorch-jax-expert.md).
  - PR touches data pipeline, configs, or module layout → include [ml-protein-architect](.claude/agents/ml-protein-architect.md).
  - PR touches crystallographic evidence or PDB-derived data handling → include [xray-crystallography-binder-ml](.claude/agents/xray-crystallography-binder-ml.md).

Reviewers run in parallel when their reviews are independent.

### Approval rule (all-or-nothing)

The PR can merge only when **every appointed reviewer approves**. The protocol:

1. Dispatch each appointed reviewer with the PR diff, context, and the test plan.
2. Each reviewer either **approves** the PR or **identifies what's broken** with file:line citations and severity.
3. If any reviewer flags issues:
   - Form a plan to address every issue (use [software-planning-architect](.claude/agents/software-planning-architect.md) if the fix touches design, not just code).
   - Execute the fix (TDD — write or update tests first, then implement).
   - Re-dispatch the **same** panel for another review round.
4. Loop until all reviewers approve.

### Stuck-PR escape hatch

If the review loop does not converge — reviewers keep finding new issues, two reviewers disagree on direction, or the same issue resurfaces after a fix — **stop**. Do not push the PR through. Instead:

1. **Terminate the execution.**
2. **Report the situation in the main chat** with: which PR, which reviewers were on the panel, what they last said, and what step is blocked.
3. **Send an email** to `schekmenev@aithyra.at` (the project owner — Stanislav) summarising the PR, the panel, and the deadlock, stating that the PR is hung and needs human review. Permission to send to that specific address has been granted by the user; **never send to any other address** without explicit user permission, and never include secrets, credentials, or proprietary training data in the email.

Use whatever email transport is available on the host (e.g. `mail`, `mailx`, `sendmail`, `msmtp`, or an authorised API helper). If none is available, fall back to reporting in chat *and* opening a GitHub issue with the same content, and tell the user no transport was available.

## Git / GitHub conventions

- **`gh` needs `module load gh`** in the same Bash invocation before use (it's an environment module on this host, not on PATH by default). Run `module load gh && gh ...` in a single command, every time (shell state does not persist between Bash tool calls).
- **No Claude attribution in commits or PRs.** Commit messages and pull-request descriptions must **not** mention Claude, "Generated with", "Co-Authored-By: Claude", or similar. Author commits and PRs as the user. *(This overrides the default Claude Code commit template.)*
- **Branching.** Work on a feature branch off `dev` (the main branch). Open PRs into `dev`.
- **Commits.** Small, focused, descriptive — `verb: short subject` (e.g. "add interface H-bond reward", "fix FSDP unsharded embedding bug"). Group commits by intent, not by file.
- **Never amend a published commit.** Always create a new commit. Never `--no-verify`, `--no-gpg-sign`, or force-push to `dev` / `main`.
- **`.gitignore`.** Runtime artefacts (`ckpts/`, `wandb/`, sample/eval outputs, `.venv/`, `__pycache__/`, large PDB / mmCIF dumps) stay ignored. `.claude/agents/` is tracked; everything else under `.claude/` stays ignored.

## Style and code hygiene

- Match repo style; type-hint public APIs.
- No speculative abstractions, no half-finished scaffolds, no parallel implementations of an existing module.
- Default to **no comments**. Add a comment only when the *why* is non-obvious (a hidden constraint, a workaround for a specific bug, a subtle invariant). Do not narrate the code; the code says what.
- No emojis in code, tests, configs, commits, or PRs unless the user explicitly asks for them.
- Prefer editing existing files over creating new ones. Never create documentation (`.md`) or README files unless the user asks.

## Reproducibility hygiene

- Seeded splits and seeded dataloaders. State the seed where it's set; do not silently re-seed.
- Deterministic ops where they matter (geometry math, eval). Document the throughput cost when determinism is enabled.
- Checkpoints must round-trip on the pinned Lightning version. Do not bump Lightning without verifying weight loading on an existing checkpoint.
- Smoke-train and smoke-sample on a tiny subset is part of "done", not optional.
