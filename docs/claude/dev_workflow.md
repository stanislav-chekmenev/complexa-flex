# Development workflow — full detail

Read this file before starting a non-trivial change. The core [CLAUDE.md](../../CLAUDE.md) keeps a short TDD summary; the full discipline and the "what an effective test looks like" checklist live here.

## TDD is the default for major and medium-size tasks

Test-driven development is the **main strategy** for any non-trivial change. The TDD discipline is executed *via the subagents in [.claude/agents/](../../.claude/agents/)* — the main thread coordinates; agents (planner, implementer, framework expert, reviewer) do the actual planning, writing, and reviewing.

For each non-trivial change:

1. **Plan with [software-planning-architect](../../.claude/agents/software-planning-architect.md)** — translate the ask into a concrete plan with data contracts, milestones, exit and abort criteria, and an explicit verification strategy.
2. **Write tests first.** Tests are written before the implementation they cover. Implementation is then driven by making the failing tests pass.
3. **Implement with [ml-protein-architect](../../.claude/agents/ml-protein-architect.md)** (or [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md) when the work is framework-internal). Vertical slice first — smallest end-to-end version that compiles, trains one step, samples one structure, evaluates one batch — then horizontal expansion.
4. **Review** with the agents appointed for the PR (see [pr_protocol.md](pr_protocol.md)).

**Minor changes** (one-line fix, typo, comment, config tweak that does not change semantics, isolated documentation edit) do **not** require a new test. Use judgement; err toward writing a test when the change crosses a module boundary, touches a training-loop or eval-harness contract, or modifies a Hydra default that other configs compose.

## What an effective test looks like in this repo

Tests must:

- **Cover the primary logic** of the function / method / class / pipeline they target — not just happy-path smoke.
- **Be effective and complexity-optimised** — exercise the behaviour that matters, not exhaustively enumerate trivial cases. Prefer a small number of targeted tests over a large grid of near-duplicates.
- **Catch real edge cases relevant to this domain.** Empty input, single residue, single atom, masked / NaN entries, very long sequences, multimer crops, ligand-present vs absent, fp16/bf16 vs fp32 numerical drift, equivariance under random rotation+translation, dataloader determinism under the same seed, mismatched device / dtype.
- **Be readable.** Each test makes the property under test obvious from its name and assertions; a reader can tell *what would break* from a one-line read.
- **Pass on every PR.** A PR is not mergeable while any required test fails. CI green is a necessary, not sufficient, gate — review still applies.

Smoke tests (one-step train, one-sample generation, one-batch eval) are required for any change that touches the training loop, sampling pipeline, or eval harness, on top of the unit/integration tests for the changed module.
