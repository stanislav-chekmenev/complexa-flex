# CLAUDE.md — Proteina-Complexa

Project-specific instructions for Claude working in this repository. These rules apply to every Claude session (main thread and subagents) operating inside `complexa-flex/`.

## Project at a glance

- **Project name.** Proteina-Complexa — atomistic flow-matching generative model for protein–protein and protein–ligand binder design, with test-time search over reward models (AF2 / RF3 / force fields), motif scaffolding (AME), and fold-class-conditioned generation.
- **Distribution / package name.** `proteinfoundation` (see [pyproject.toml](pyproject.toml)). Source lives under [src/proteinfoundation/](src/proteinfoundation/). Do not invent parallel package names.
- **Pinned environment.** uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5,<2.6 (2.6.x breaks checkpoint loading). **Do not run `uv run`, `uv sync`, or `uv lock` against this project — they re-resolve `pyproject.toml`, which drifts from the actual `.venv` (no `uv.lock` is shipped; multiple packages are installed by [env/build_uv_env.sh](env/build_uv_env.sh) directly). Re-resolution silently upgrades torch and removes atomworks / graphein / PyG / jax — trashing the venv.** Instead: activate `.venv/` (`source .venv/bin/activate`) or invoke `.venv/bin/python` directly. To rebuild the venv from scratch, use `bash env/build_uv_env.sh --clean`. Never substitute plain PyPI torch wheels (CPU-only).
- **Compute-node venv staging.** sbatch scripts must stage the venv by extracting a pre-built tarball, **not** by calling `uv run`. Canonical tarball at `$PROJECT_ROOT/venv.tar.gz` (labs NFS, gitignored) with per-partition copies at `/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz`. See [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch) for the canonical pattern: `cp` fallback from labs NFS → partition-local `/netscratch` if missing, then `tar xzf` into `$DATA_ROOT/.venv`, then `srun ... "$ENV_LOCAL/bin/python" -m ...`. Each compute partition (`gpu`, `h100`) has its own physical `/netscratch` mount despite the shared path — the cross-partition fallback is mandatory.
- **Configs.** Hydra tree under [configs/](configs/) — `generation/`, `nn/`, `nn_ae/`, `pipeline/`, `dataset/`, `design_tasks/`, `confidence/`, etc. Compose, don't duplicate. Mirror naming of existing entries (`search_binder_*`, `evaluate_*`, `analyze_*`).
- **Runtime artefacts** (`ckpts/`, `wandb/`, sample/eval output trees) belong in `.gitignore`. Never commit weights, large PDB dumps, or run directories.

## Confidence-head distillation subsystem

A sidecar Lightning module distils AF2 per-residue pLDDT and per-pair PAE from the frozen complexa trunk into a trainable light-weight student head. **Does not modify** `proteinfoundation.proteina` or the main `train.py` entry point.

- **Package layout.** [src/proteinfoundation/nn/confidence/](src/proteinfoundation/nn/confidence/) holds heads + trunk + registry + pure-function `_losses.py` / `_metrics.py`. [src/proteinfoundation/confidence/](src/proteinfoundation/confidence/) holds the Lightning module + Hydra entry point + back-compat shims; new code imports from `nn.confidence._{losses,metrics}` directly. Heads register via `@register_confidence_head(name)`. **Layering rule:** `nn/confidence/` must NOT import from `confidence/`.
- **Entry point.** `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/<config>`. SLURM-launch via [scripts/train_confidence_swissprot.sbatch](scripts/train_confidence_swissprot.sbatch), [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch), or sibling sbatches.
- **Datasets.** SwissProt monomers via [configs/dataset/unified/afdb_monomers_with_plddt.yaml](configs/dataset/unified/afdb_monomers_with_plddt.yaml) (AF2 `[0, 100]`, 50 bins of width 2, optional 30%-cluster split via `cluster_column: unicluster`). Teddymer dimers via [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](configs/dataset/unified/teddymer_with_plddt_and_pae.yaml).
- **Loss / metrics.** Per-head recipe lives on the head (e.g. pLDDT default `0.9*masked_CE + 0.1*SmoothL1(EV)` with `ce_weight` / `ev_weight` / `label_smoothing` as Hydra `loss:` kwargs). Reduction `sum(loss * mask) / mask.sum().clamp_min(1)`. Each head implements `compute_loss_and_metrics(out, batch, mask_eff, *, stage) -> (loss, log_dict)`; the Lightning module prefixes log keys with `{train,val}/{head.output_name_root}/`. Validation logs accuracy, MAE, Pearson, Spearman, stratified MAE, equal-width / equal-mass ECE, and a per-bucket reliability diagram (rank-0 only). PaeHead also logs 10 interface metrics (i_pae, min_ipae, i_ptm, i_ptm_energy, six ipSAE variants — colabdesign-parity ports).
- **Trunk hook + AdaLN.** `LocalLatentsTransformer` has opt-in `expose_intermediates: bool = False`; when `True`, `nn_out["trunk_intermediates"]` carries `(s, z, local_latents, mask, orig_mask, n_orig)`. Default `False` is bit-identical to legacy (`tests/regression/test_flow_matching_loss_unchanged.py`). Sidecar reuses the trunk's `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99`, zero-padded along the residue axis if concat features extend the sequence.
- **Pair-repr symmetrisation is the head's job.** `ConfidenceTrunk.forward` does NOT symmetrise `z`. Symmetric pair-heads (PDE, ipLDDT, ipTM) prepend `z = 0.5 * (z + z.transpose(-3, -2))` in their `_predict`; asymmetric heads (`PaeHead`) consume `z` directly. `PlddtHead` is `s`-only.
- **WandB logging.** Wired via [configs/logging/wandb.yaml](configs/logging/wandb.yaml) (`@package _global_`, project `confidence-distillation`). Compose with `- /logging/wandb@logging` and set a per-config `wandb_tags:`. Entry point builds `WandbLogger` via `_build_wandb_logger(cfg)`; honours `WANDB_MODE=disabled` and `log_wandb: false`. Top-level `run_name` is the WandB `id` and `name`, so re-launching the same `run_name` *resumes* the run. Future entry points must reuse this fragment — embedding wandb keys in training YAMLs is the anti-pattern.
- **Rank-0-only loguru gate.** Any Hydra-launched DDP entry point must call `_gate_loguru_to_rank0()` as the FIRST line of `main()`. Reads `LOCAL_RANK`, `NODE_RANK`, `RANK` (the last covers `torchrun`); calls `logger.remove()` if any is non-zero. Reference: [src/proteinfoundation/confidence/train_confidence.py:29-34](src/proteinfoundation/confidence/train_confidence.py#L29-L34).
- **Multi-GPU DDP — pair `find_unused_parameters=true` with `static_graph=true`.** All confidence configs ship `DDPStrategy(find_unused_parameters=true, static_graph=true)` as a structured yaml. Two failures compose without both flags: (i) frozen-trunk params don't feed the head loss (`PaeHead` uses `z`, `PlddtHead` uses `s`) → `find_unused_parameters=false` raises "parameters not used"; (ii) the head's embedded `ConfidenceTrunk` ([src/proteinfoundation/nn/confidence/base.py](src/proteinfoundation/nn/confidence/base.py)) uses **reentrant** `torch.utils.checkpoint` in pair-update layers, so `find_unused_parameters=true` alone makes the reducer mark the same param ready twice. The frozen complexa trunk runs under `torch.no_grad()` so its ckpts never enter backward — the reentrant-ckpt site that trips DDP is the head's trunk copy. Same contract applies to the second reentrant-ckpt site, [QgPairformerStack](src/proteinfoundation/nn/confidence/qg_pairformer_stack.py) (wraps `PairformerLayer` in `use_reentrant=True` when `gradient_checkpointing: true`; required at L≥500, n_layers≥6 on h100nvl). `static_graph=true` is what lets the reducer tolerate the double-ready signal; switching to non-reentrant ckpt changes the autograd graph shape and is NOT a drop-in. Numerical equivalence of ckpt + chunked-tri-attn vs the unckpt'd / unchunked baseline is pinned by [tests/unit/nn/confidence/test_qg_pairformer_stack.py](tests/unit/nn/confidence/test_qg_pairformer_stack.py).
- **CUDA allocator — confidence-distill sbatch must export `PYTORCH_ALLOC_CONF=expandable_segments:True`.** Reentrant ckpt in [src/proteinfoundation/nn/modules/pair_update.py](src/proteinfoundation/nn/modules/pair_update.py) plus Teddymer's heterogeneous L (250–550) fragments the caching allocator after ~900 steps and OOMs with multi-GiB reserved-but-unallocated. Goes alongside the NCCL block, *before* `srun python -m ...`; see [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch). `static_graph=true` is reducer-side; `expandable_segments` is allocator-side; both required.
- **Resume-from-checkpoint.** Every confidence-distill config exposes `resume_ckpt_path: null`, forwarded to `trainer.fit(..., ckpt_path=...)`. Orthogonal to `resume_id` (WandB id): `resume_ckpt_path` restores Lightning state (weights + optimizer + LR + step + epoch); `resume_id` re-attaches the WandB run. Sbatch contract: `RESUME_CKPT_PATH` env var → Hydra override; usage `RESUME_CKPT_PATH=/abs/path.ckpt sbatch --export=ALL,RESUME_CKPT_PATH scripts/train_confidence_*.sbatch`. SwissProt sbatch lacks this wiring — add when next resuming SwissProt.
- **Joint training via `MultiHeadConfidence` wrapper.** [src/proteinfoundation/nn/confidence/multi_head.py](src/proteinfoundation/nn/confidence/multi_head.py), registered as `multi_head`. `nn.ModuleDict` of child heads; runs trunk once per batch and dispatches `(s, z, mask, cond, local_latents)` to every child's `_predict`. The shared `ConfidenceTrunk` projects + LN's `local_latents` (8-dim) and adds as a mask-zeroed residual to `s` at trunk entry. `MultiHeadLoss` aggregates `total = sum(w_i * L_i)`. Two Hydra weighting layers: (i) **within-head** `loss.ce_weight` / `loss.ev_weight`; (ii) **across-head** `loss_weights: {head: scalar}` (multiplies the head's total; `0.0` is a hard short-circuit). Wrapper asserts `expected_trunk_eval_t` matches and every child has a non-empty `output_name_root`. **DDP footgun:** `weight: 0.0` still runs `_predict`, so its params enter the autograd graph with no gradient → `find_unused_parameters=False` raises at backward. Escapes: remove from `children:`, or rely on `static_graph=True` (already mandatory above).
- **Teddymer filter is geometry-only.** Supervised pool is `interface_length > 10` at [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](configs/dataset/unified/teddymer_with_plddt_and_pae.yaml). **Confidence-based pre-filters are a selection-bias antipattern**: the head's targets are per-residue pLDDT and per-pair PAE, so filtering on aggregates of those same quantities (`avg_int_plddt`, `avg_int_pae`) truncates the label distribution by the label itself. AF2-multimer, Boltz-1/-2, and Chai-1 filter on data-source quality (resolution, identity clustering, homology) but never on the model's own confidence; Boltz-2 names the antipattern. Only filter by geometric properties independent of the AF2 confidence map. `complexa_filter` is still on `dimers.parquet` but unused. Contract pinned by [tests/unit/datasets/test_teddymer_dataset_config.py::test_yaml_filter_pins_geometry_only_threshold](tests/unit/datasets/test_teddymer_dataset_config.py).
- **LayerNorm-bias-leak invariant.** Any `Sequential(Linear(d_in, d_token, bias=False), LayerNorm(d_token))` projecting an auxiliary signal into a masked sequence representation and added as a residual **must** multiply post-LN by `mask_f` before the add — even if the upstream input is zero at padded positions. At init `beta = 0` makes `LN(0) = 0` look fine, but trained drift gives `LN(0) = beta` and lands the bias in padded `s` rows, contaminating valid `(i, j)` cells via `PairReprUpdate` outer products + triangle multiplication summing over padded `k`. Reference: [src/proteinfoundation/nn/confidence/base.py:146-147](src/proteinfoundation/nn/confidence/base.py#L146-L147). Regression contract: [tests/integration/confidence/test_confidence_trunk_local_latents_proj.py::test_mask_multiply_guards_against_trained_layernorm_bias_leak](tests/integration/confidence/test_confidence_trunk_local_latents_proj.py) — writes `LN.bias = 0.3` and asserts forwards differing only at padded positions are bit-identical everywhere. Any future projection into `s` or `z` must follow `* mask` after the LN, and pin the invariant with a non-init test.

## Session lifecycle (recall and log)

Three project-scoped slash commands govern the start and end of a Claude session in this repo. Their definitions live under [.claude/commands/](.claude/commands/) and are tracked in git so every session sees the same contract.

- **[/recall](.claude/commands/recall.md)** — at session start, restores context by reading `CLAUDE.md` and the auto-memory index `MEMORY.md` in parallel. Use this when picking up routine work where the prior session shipped cleanly (no handoff needed).
- **[/recall_and_follow_latest](.claude/commands/recall_and_follow_latest.md)** — at session start, reads `CLAUDE.md`, `MEMORY.md`, **and** the most recent handoff under `docs/handoff/` (sorted lexicographically; the naming convention `YYYY-MM-DD_<topic-slug>.md` makes this chronological). Use this when resuming work that the prior session left in flight.
- **[/ul](.claude/commands/ul.md)** — "Update and Log" — at session end, refreshes `MEMORY.md` (and any linked memory files) with new durable facts, amends `CLAUDE.md` if the session established a new project-wide rule, and writes a handoff at `docs/handoff/YYYY-MM-DD_<topic-slug>.md` **iff** the session leaves undone tasks the next agent must resume. Empty handoffs are forbidden; a clean-shipped session updates memory + (optionally) `CLAUDE.md` and stops.

**Standing rule (binds every session, whether `/ul` was typed or not):** before the user closes the session — typically when they signal "done for the day", "let's wrap", or anything equivalent — perform the `/ul` workflow. If the session has new durable facts, update memory; if it established a new convention, update `CLAUDE.md`; if it leaves work in flight, write a handoff for the next agent. The next agent will start with `/recall` or `/recall_and_follow_latest`, so the handoff is the only channel by which in-flight context survives.

**Bookkeeping commit (Step 4 of `/ul`):** if and only if the `/ul` pass produced a `CLAUDE.md` edit and/or a new `docs/handoff/` file, the agent **must** create a single commit on the **current branch** (no branch switching, no push) containing exactly those file(s) — `git add` by explicit path, never `-A` / `.`. Other dirty files in the working tree are left untouched. Commit message: `docs: end-of-session bookkeeping — <slug>` (handoff present) or `docs: update CLAUDE.md — <one-line summary>` (CLAUDE.md only). If neither file was produced this `/ul`, skip the commit step silently. Memory updates (Step 1) are never committed by this step — memory lives outside the repo. Push policy: never push from `/ul`; the user pushes when ready.

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

The aim of this protocol is to match review cost to PR risk. Most PRs do not need a panel. Reviewers exist to catch real bugs and unsafe designs, not to enforce cosmetics.

### Tier classification (mandatory; state in the PR description)

Before opening a PR, classify it. State the tier on the first line of the PR body so the choice is auditable.

- **T1 — Trivial.** Docs-only edit, comment-only change, rename of a private symbol, config tweak that does not change training/eval semantics (ckpt cadence, log cadence, batch size, num_workers, `--exclude` flag, sbatch wallclock, partition), CLAUDE.md / memory bookkeeping.
- **T2 — Standard.** One subsystem touched, bug fix, one new test, refactor with no behaviour change, new Hydra config composed of existing fragments, metric/loss tweak with bounded blast radius, < ~300 LoC of non-test code.
- **T3 — Substantial.** New head/architecture, new training stage, new dataset pipeline, cross-module refactor, community-metric port, anything that changes training/sampling/eval *semantics*, anything touching reproducibility (seed, ckpt format, schedule), or any PR that adds ≳ 300 LoC of non-test code.

**Force-T3 triggers (override the size heuristic):** touches training/sampling/eval semantics; touches a community-standard metric ([Community-standard metric / reward parity](#community-standard-metric--reward-parity)); touches `.ckpt` round-trip; touches DDP / strategy / precision wiring; touches the data-integrity gate.

### Panel by tier

| Tier | Reviewers |
| --- | --- |
| **T1** | None. |
| **T2** | **[code-review-debug-complexity-expert](.claude/agents/code-review-debug-complexity-expert.md)** + at most **one** domain reviewer if a domain is meaningfully touched (see roster below). |
| **T3** | **[code-review-debug-complexity-expert](.claude/agents/code-review-debug-complexity-expert.md) (mandatory)** + **2–4 domain reviewers** picked by what the PR touches. |

Domain mapping (used at T2/T3 to pick the domain reviewer(s)):

- New loss / schedule / guidance / sampling → [generative-protein-scientist](.claude/agents/generative-protein-scientist.md) and/or [generative-flow-stochastic-math-expert](.claude/agents/generative-flow-stochastic-math-expert.md).
- Protein–protein binder logic → [structural-biology-binder-expert](.claude/agents/structural-biology-binder-expert.md).
- IDR / motif / small-molecule logic → [structural-biology-idp-smallmol-expert](.claude/agents/structural-biology-idp-smallmol-expert.md).
- MD / DFT / MLIP scoring → [physics-statmech-md-dft-expert](.claude/agents/physics-statmech-md-dft-expert.md).
- PyTorch / JAX internals, FSDP, `torch.compile`, custom kernels → [ml-software-pytorch-jax-expert](.claude/agents/ml-software-pytorch-jax-expert.md).
- Data pipeline, configs, module layout → [ml-protein-architect](.claude/agents/ml-protein-architect.md).
- Crystallographic evidence / PDB-derived data → [xray-crystallography-binder-ml](.claude/agents/xray-crystallography-binder-ml.md).

Reviewers run in parallel when their angles are independent.

### Target-branch relaxation

The table above describes **PRs into `dev`**. For other targets:

- **`main`** — full T3 protocol regardless of diff size. `main` is the release line.
- **`dev`** — apply the tier table verbatim.
- **A `feat/*`, `exp/*`, `test/*`, or other testing/integration branch** that exists to see whether a feature works — **drop one tier** (T3 → T2, T2 → T1, T1 → no PR needed; push directly). Testing branches are throwaway experiments; the cost of a missed bug is reverting the branch, not a wedged `dev`.

### Blocking vs non-blocking findings (the core rule)

Reviewer feedback is not all equal. The protocol distinguishes:

- **Blocking.** A real bug, a correctness/safety hazard, a contract violation, a missing test for behaviour the PR changes, a reproducibility regression, a security or data-integrity issue, or a substantiated claim of broken behaviour with file:line evidence. Blocking findings **must** be addressed before merge.
- **Non-blocking.** Cosmetic preferences, naming bikesheds, "could be cleaner", unverified suggestions, vague guesses, speculative refactors, hypothetical future-proofing. These do **not** gate the PR.

Reviewers must mark each finding `BLOCKING` or `NON-BLOCKING` with a one-line justification. A finding without that mark is treated as non-blocking by default.

**Disposition of non-blocking findings:**

- If the finding is a genuine improvement worth remembering, the agent appends a one-line note to CLAUDE.md (project-wide convention) or to the appropriate `MEMORY.md` entry (anything else durable). No new memory file unless the note doesn't fit anywhere existing.
- Otherwise, the finding is acknowledged in the PR thread ("noted, non-blocking") and dropped.
- **Do not open follow-up PRs to address non-blocking findings unless the user explicitly asks.**

### Approval rule (revised)

A PR can merge when:

- **T1.** No review needed; the author (you) self-approves after tests pass (if any apply).
- **T2 / T3.** Every appointed reviewer either approves, OR raises only non-blocking findings. As soon as all blocking findings are addressed and re-checked, the PR is mergeable — even if reviewers still have non-blocking notes outstanding.

**Re-review budget.** After a fix round for blocking findings, re-dispatch **only the reviewers that raised blockers** — not the full panel. A reviewer that approved or raised only non-blocking findings in round N does not need to be re-asked in round N+1 unless the fix changed code in their domain.

### Stuck-PR escape hatch

If the **blocking** loop does not converge — same blocker resurfaces after a fix, or two reviewers disagree on what the blocker is — **stop**. Do not push the PR through.

1. **Terminate the execution.**
2. **Report in the main chat** with: which PR, which reviewers, what each last said, what's blocked.
3. **Send an email** to `schekmenev@aithyra.at` (the project owner — Stanislav) summarising the PR, the panel, and the deadlock. Permission to send to that specific address has been granted; **never send to any other address** without explicit user permission; never include secrets, credentials, or proprietary training data in the email.

Use whatever email transport is available (`mail`, `mailx`, `sendmail`, `msmtp`, an authorised API helper). If none is available, report in chat *and* open a GitHub issue with the same content, and tell the user no transport was available.

## Community-standard metric / reward parity

When porting any community-standard PAE/pLDDT-derived metric or reward — colabdesign's `get_ipsae_loss`, AlphaFold ipTM, RFdiff/BindCraft binder filters, AF3 confidence kernels, Boltz/Chai scores, or anything else the field publishes as a recognised number — port it **verbatim** in the first PR, with a fp64 reference test locking parity within `1e-5`.

- **Verbatim, not "improved".** Keep the upstream quirks: `1e-8` additives vs `clamp_min`, magic clip thresholds (e.g. ipSAE's `L >= 27`, ipTM's `n_eff >= 19`), two-direction reductions, per-row vs per-sample `d_0`. These quirks are part of the contract; deviating from them silently makes the logged values incomparable to the published binder-design literature (Watson et al., Bennett et al., Cao et al., Pacesa et al. / BindCraft, Yin et al.).
- **Reference lives in the test, not the implementation.** Write a fp64 numpy reproduction of the canonical upstream code inside the test file (e.g. [tests/unit/nn/confidence/test_ipsae_family.py](tests/unit/nn/confidence/test_ipsae_family.py)'s `_colabdesign_ipsae_reference`). The reference is itself reviewable — generative-protein-scientist must confirm it mirrors the upstream source line-by-line before the parity assertion is meaningful.
- **Generalisations come as separately-named follow-ups.** Want a Cα-distance-gated ipSAE variant or a soft-row ipTM aggregation? Ship it in a follow-up PR under a *new* name (`ipsae_contact_gated`, `iptm_soft`). Never reuse the canonical name for a different formulation, even if the new formulation is more principled.
- **Canonical references in-tree.** [community_models/colabdesign/af/loss.py](community_models/colabdesign/af/loss.py) is the colabdesign source of truth; [src/proteinfoundation/rewards/alphafold2_reward_utils.py](src/proteinfoundation/rewards/alphafold2_reward_utils.py) wraps the AF2-side primitives. Cite file:line when defending a port.

This rule binds every PR that touches the `_metrics.py` / `_losses.py` modules under `nn/confidence/` and any future reward heads (ipTM, iPLDDT, pAE_interaction, ipSAE).

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
