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

- **Package layout.** [src/proteinfoundation/nn/confidence/](src/proteinfoundation/nn/confidence/) (heads + trunk + registry + pure-function `_losses.py` / `_metrics.py`) and [src/proteinfoundation/confidence/](src/proteinfoundation/confidence/) (sidecar Lightning module + `losses.py` / `metrics.py` re-export shims + Hydra entry point). Heads register via `@register_confidence_head(name)` so new heads (ipTM / ipAE / ipLDDT) drop in by subclassing `BaseConfidenceHead`. **Layering rule:** `nn/confidence/` must NOT import from `confidence/` (heads are framework-pure; the sidecar consumes them). The `confidence/{losses,metrics}.py` shims exist only for back-compat and will be removed in a future cleanup PR — new code should import from `proteinfoundation.nn.confidence._{losses,metrics}` directly.
- **Entry point.** `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_swissprot`. SLURM-launch via [scripts/train_confidence_swissprot.sbatch](scripts/train_confidence_swissprot.sbatch).
- **Dataset.** [configs/dataset/unified/afdb_monomers_with_plddt.yaml](configs/dataset/unified/afdb_monomers_with_plddt.yaml) extends the AFDB monomer dataset with `AddPLDDTFromBFactor` (AF2 `[0, 100]` scale, 50 bins of width 2) and optional cluster-30% held-out split via `cluster_column: unicluster`.
- **Loss / metrics.** Combined `0.9 * masked_CE + 0.1 * SmoothL1(EV)`, reduction `sum(loss * mask) / mask.sum().clamp_min(1)`. The recipe lives on the **head** (`ce_weight` / `ev_weight` / `label_smoothing` are constructor kwargs, set via the head's Hydra `loss:` block); each head implements `compute_loss_and_metrics(out, batch, mask_eff, *, stage) -> (loss, log_dict)` and the Lightning module is head-agnostic, prefixing log keys with `{train,val}/{head.output_name_root}/`. The Lightning module's own same-named kwargs are deprecated no-ops since PR #10 and emit `DeprecationWarning`. Validation logs accuracy, MAE (in pLDDT units), Pearson, Spearman, stratified MAE by pLDDT bucket, equal-width ECE, equal-mass adaptive ECE, and a per-bucket reliability diagram (rank-0 only).
- **Trunk hook.** `LocalLatentsTransformer` has an opt-in `expose_intermediates: bool = False` kwarg. When `True`, `forward(input)` adds `nn_out["trunk_intermediates"]` carrying `(s, z, mask, orig_mask, n_orig)`. Default `False` is bit-identical to the legacy forward (enforced by `tests/regression/test_flow_matching_loss_unchanged.py` against a baseline fixture).
- **AdaLN cond.** The sidecar reuses the trunk's existing `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99` to produce the same `cond` the trunk consumes; zero-padded along the residue axis to `n_ext` if concat features extend the sequence.
- **Pair-repr symmetrisation — removed from the trunk in PR #10.** `ConfidenceTrunk.forward` no longer applies `z = (z + z.transpose(-3, -2)) / 2.0`. Symmetric pair-heads (PDE, ipLDDT, ipTM, future) must add `z = 0.5 * (z + z.transpose(-3, -2))` as the first line of their own `_predict`. Asymmetric pair-heads (`PaeHead`, future directional ones) consume `z` directly. `PlddtHead` consumes `s` only, so the change is a no-op for it. Load-bearing regression: `tests/regression/test_flow_matching_loss_unchanged.py` continues to pass.
- **Joint training via `MultiHeadConfidence` wrapper — shipped in PR #10.** Lives at [src/proteinfoundation/nn/confidence/multi_head.py](src/proteinfoundation/nn/confidence/multi_head.py). Registered as `multi_head`. Holds an `nn.ModuleDict` of child heads, runs the trunk once per batch, dispatches the same `(s, z, mask, cond)` to every child's `_predict`, and returns `{head_name: prediction}`. `MultiHeadLoss` at [src/proteinfoundation/nn/confidence/_losses.py](src/proteinfoundation/nn/confidence/_losses.py) aggregates `total = sum(w_i * L_i)` with per-task scalars logged separately. Two weighting layers, kept separate in Hydra: (i) **within-head** `loss.ce_weight` / `loss.ev_weight` (the existing pLDDT recipe family); (ii) **across-head** `loss_weights: {head: scalar}` (multiplies the head's total loss before summing with siblings; `weight: 0.0` is a hard short-circuit — skip the head's loss call entirely). A head whose entire batch mask is False contributes exactly zero, never NaN-from-empty-mean. The wrapper asserts `expected_trunk_eval_t` matches across children at construction AND that every child has a non-empty `output_name_root`; a head needing a different `t` cannot live under this wrapper. Single-head runs (`PlddtHead`-only on SwissProt, `PaeHead`-only on Teddymer for the clean reference number) keep working unchanged — joint training is one Hydra config flip away, not the default. **DDP footgun:** `weight: 0.0` disables the loss but the child's `_predict` still runs in the wrapper's `forward`, so the child's parameters enter the autograd graph with no gradient — `find_unused_parameters=False` raises at backward. Working escape hatches: (a) remove the child from `children:`, (b) `DDP(static_graph=True)`. The wrapper emits a `RuntimeWarning` at construction listing offending heads.

## Session lifecycle (recall and log)

Three project-scoped slash commands govern the start and end of a Claude session in this repo. Their definitions live under [.claude/commands/](.claude/commands/) and are tracked in git so every session sees the same contract.

- **[/recall](.claude/commands/recall.md)** — at session start, restores context by reading `CLAUDE.md` and the auto-memory index `MEMORY.md` in parallel. Use this when picking up routine work where the prior session shipped cleanly (no handoff needed).
- **[/recall_and_follow_latest](.claude/commands/recall_and_follow_latest.md)** — at session start, reads `CLAUDE.md`, `MEMORY.md`, **and** the most recent handoff under `docs/handoff/` (sorted lexicographically; the naming convention `YYYY-MM-DD_<topic-slug>.md` makes this chronological). Use this when resuming work that the prior session left in flight.
- **[/ul](.claude/commands/ul.md)** — "Update and Log" — at session end, refreshes `MEMORY.md` (and any linked memory files) with new durable facts, amends `CLAUDE.md` if the session established a new project-wide rule, and writes a handoff at `docs/handoff/YYYY-MM-DD_<topic-slug>.md` **iff** the session leaves undone tasks the next agent must resume. Empty handoffs are forbidden; a clean-shipped session updates memory + (optionally) `CLAUDE.md` and stops.

**Standing rule (binds every session, whether `/ul` was typed or not):** before the user closes the session — typically when they signal "done for the day", "let's wrap", or anything equivalent — perform the `/ul` workflow. If the session has new durable facts, update memory; if it established a new convention, update `CLAUDE.md`; if it leaves work in flight, write a handoff for the next agent. The next agent will start with `/recall` or `/recall_and_follow_latest`, so the handoff is the only channel by which in-flight context survives. Do not commit any of these changes without explicit user approval — memory lives outside the repo, but `CLAUDE.md` and `docs/handoff/` are tracked, and the user authors the commit.

**Handoff naming convention.** `docs/handoff/YYYY-MM-DD_<topic-slug>.md`. Slug is lowercase, underscore-separated, ASCII. The date is today's date (always use the absolute date, not "today" — the file must be interpretable months later). If two handoffs land on the same date for related but distinct topics, the topic slug disambiguates; do not append `_v2`.

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
