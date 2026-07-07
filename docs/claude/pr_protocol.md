# Pull-request review protocol — full detail

Read this file before opening any PR. The core [CLAUDE.md](../../CLAUDE.md) keeps only the tier definitions, force-T3 triggers, and the approval one-liner; the full panel, domain mapping, target-branch relaxation, blocking rules, re-review budget, and stuck-PR escape hatch live here.

The aim of this protocol is to match review cost to PR risk. Most PRs do not need a panel. Reviewers exist to catch real bugs and unsafe designs, not to enforce cosmetics.

## Tier classification (mandatory; state in the PR description)

Before opening a PR, classify it. State the tier on the first line of the PR body so the choice is auditable.

- **T1 — Trivial.** Docs-only edit, comment-only change, rename of a private symbol, config tweak that does not change training/eval semantics (ckpt cadence, log cadence, batch size, num_workers, `--exclude` flag, sbatch wallclock, partition), CLAUDE.md / memory bookkeeping.
- **T2 — Standard.** One subsystem touched, bug fix, one new test, refactor with no behaviour change, new Hydra config composed of existing fragments, metric/loss tweak with bounded blast radius, < ~300 LoC of non-test code.
- **T3 — Substantial.** New head/architecture, new training stage, new dataset pipeline, cross-module refactor, community-metric port, anything that changes training/sampling/eval *semantics*, anything touching reproducibility (seed, ckpt format, schedule), or any PR that adds ≳ 300 LoC of non-test code.

**Force-T3 triggers (override the size heuristic):** touches training/sampling/eval semantics; touches a community-standard metric (see [community metric parity](../../CLAUDE.md#community-standard-metric--reward-parity)); touches `.ckpt` round-trip; touches DDP / strategy / precision wiring; touches the data-integrity gate.

## Panel by tier

| Tier | Reviewers |
| --- | --- |
| **T1** | None. |
| **T2** | **[code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md)** + at most **one** domain reviewer if a domain is meaningfully touched (see roster below). |
| **T3** | **[code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md) (mandatory)** + **2–4 domain reviewers** picked by what the PR touches. |

Domain mapping (used at T2/T3 to pick the domain reviewer(s)):

- New loss / schedule / guidance / sampling → [generative-protein-scientist](../../.claude/agents/generative-protein-scientist.md) and/or [generative-flow-stochastic-math-expert](../../.claude/agents/generative-flow-stochastic-math-expert.md).
- Protein–protein binder logic → [structural-biology-binder-expert](../../.claude/agents/structural-biology-binder-expert.md).
- IDR / motif / small-molecule logic → [structural-biology-idp-smallmol-expert](../../.claude/agents/structural-biology-idp-smallmol-expert.md).
- MD / DFT / MLIP scoring → [physics-statmech-md-dft-expert](../../.claude/agents/physics-statmech-md-dft-expert.md).
- PyTorch / JAX internals, FSDP, `torch.compile`, custom kernels → [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md).
- Data pipeline, configs, module layout → [ml-protein-architect](../../.claude/agents/ml-protein-architect.md).
- Crystallographic evidence / PDB-derived data → [xray-crystallography-binder-ml](../../.claude/agents/xray-crystallography-binder-ml.md).

Reviewers run in parallel when their angles are independent.

## Target-branch relaxation

The table above describes **PRs into `dev`**. For other targets:

- **`main`** — full T3 protocol regardless of diff size. `main` is the release line.
- **`dev`** — apply the tier table verbatim.
- **A `feat/*`, `exp/*`, `test/*`, or other testing/integration branch** that exists to see whether a feature works — **drop one tier** (T3 → T2, T2 → T1, T1 → no PR needed; push directly). Testing branches are throwaway experiments; the cost of a missed bug is reverting the branch, not a wedged `dev`.

## Blocking vs non-blocking findings (the core rule)

Reviewer feedback is not all equal. The protocol distinguishes:

- **Blocking.** A real bug, a correctness/safety hazard, a contract violation, a missing test for behaviour the PR changes, a reproducibility regression, a security or data-integrity issue, or a substantiated claim of broken behaviour with file:line evidence. Blocking findings **must** be addressed before merge.
- **Non-blocking.** Cosmetic preferences, naming bikesheds, "could be cleaner", unverified suggestions, vague guesses, speculative refactors, hypothetical future-proofing. These do **not** gate the PR.

Reviewers must mark each finding `BLOCKING` or `NON-BLOCKING` with a one-line justification. A finding without that mark is treated as non-blocking by default.

**Disposition of non-blocking findings:**

- If the finding is a genuine improvement worth remembering, the agent appends a one-line note to CLAUDE.md (project-wide convention) or to the appropriate `MEMORY.md` entry (anything else durable). No new memory file unless the note doesn't fit anywhere existing.
- Otherwise, the finding is acknowledged in the PR thread ("noted, non-blocking") and dropped.
- **Do not open follow-up PRs to address non-blocking findings unless the user explicitly asks.**

## Approval rule (revised)

A PR can merge when:

- **T1.** No review needed; the author (you) self-approves after tests pass (if any apply).
- **T2 / T3.** Every appointed reviewer either approves, OR raises only non-blocking findings. As soon as all blocking findings are addressed and re-checked, the PR is mergeable — even if reviewers still have non-blocking notes outstanding.

**Re-review budget.** After a fix round for blocking findings, re-dispatch **only the reviewers that raised blockers** — not the full panel. A reviewer that approved or raised only non-blocking findings in round N does not need to be re-asked in round N+1 unless the fix changed code in their domain.

## Stuck-PR escape hatch

If the **blocking** loop does not converge — same blocker resurfaces after a fix, or two reviewers disagree on what the blocker is — **stop**. Do not push the PR through.

1. **Terminate the execution.**
2. **Report in the main chat** with: which PR, which reviewers, what each last said, what's blocked.
3. **Send an email** to `schekmenev@aithyra.at` (the project owner — Stanislav) summarising the PR, the panel, and the deadlock. Permission to send to that specific address has been granted; **never send to any other address** without explicit user permission; never include secrets, credentials, or proprietary training data in the email.

Use whatever email transport is available (`mail`, `mailx`, `sendmail`, `msmtp`, an authorised API helper). If none is available, report in chat *and* open a GitHub issue with the same content, and tell the user no transport was available.
