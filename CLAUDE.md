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

A sidecar Lightning module distils AF2 per-residue pLDDT from the frozen complexa trunk into a trainable light-weight student head. **Does not modify** `proteinfoundation.proteina` or the main `train.py` entry point.

- **Package layout.** [src/proteinfoundation/nn/confidence/](src/proteinfoundation/nn/confidence/) (heads + trunk + registry + pure-function `_losses.py` / `_metrics.py`) and [src/proteinfoundation/confidence/](src/proteinfoundation/confidence/) (sidecar Lightning module + `losses.py` / `metrics.py` re-export shims + Hydra entry point). Heads register via `@register_confidence_head(name)` so new heads (ipTM / ipAE / ipLDDT) drop in by subclassing `BaseConfidenceHead`. **Layering rule:** `nn/confidence/` must NOT import from `confidence/` (heads are framework-pure; the sidecar consumes them). The `confidence/{losses,metrics}.py` shims exist only for back-compat and will be removed in a future cleanup PR — new code should import from `proteinfoundation.nn.confidence._{losses,metrics}` directly.
- **Entry point.** `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_swissprot`. SLURM-launch via [scripts/train_confidence_swissprot.sbatch](scripts/train_confidence_swissprot.sbatch).
- **Dataset.** [configs/dataset/unified/afdb_monomers_with_plddt.yaml](configs/dataset/unified/afdb_monomers_with_plddt.yaml) extends the AFDB monomer dataset with `AddPLDDTFromBFactor` (AF2 `[0, 100]` scale, 50 bins of width 2) and optional cluster-30% held-out split via `cluster_column: unicluster`.
- **Loss / metrics.** Combined `0.9 * masked_CE + 0.1 * SmoothL1(EV)`, reduction `sum(loss * mask) / mask.sum().clamp_min(1)`. The recipe lives on the **head** (`ce_weight` / `ev_weight` / `label_smoothing` are constructor kwargs, set via the head's Hydra `loss:` block); each head implements `compute_loss_and_metrics(out, batch, mask_eff, *, stage) -> (loss, log_dict)` and the Lightning module is head-agnostic, prefixing log keys with `{train,val}/{head.output_name_root}/`. The Lightning module's own same-named kwargs are deprecated no-ops since PR #10 and emit `DeprecationWarning`. Validation logs accuracy, MAE (in pLDDT units), Pearson, Spearman, stratified MAE by pLDDT bucket, equal-width ECE, equal-mass adaptive ECE, and a per-bucket reliability diagram (rank-0 only).
- **Trunk hook.** `LocalLatentsTransformer` has an opt-in `expose_intermediates: bool = False` kwarg. When `True`, `forward(input)` adds `nn_out["trunk_intermediates"]` carrying `(s, z, mask, orig_mask, n_orig)`. Default `False` is bit-identical to the legacy forward (enforced by `tests/regression/test_flow_matching_loss_unchanged.py` against a baseline fixture).
- **AdaLN cond.** The sidecar reuses the trunk's existing `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99` to produce the same `cond` the trunk consumes; zero-padded along the residue axis to `n_ext` if concat features extend the sequence.
- **Pair-repr symmetrisation — removed from the trunk in PR #10.** `ConfidenceTrunk.forward` no longer applies `z = (z + z.transpose(-3, -2)) / 2.0`. Symmetric pair-heads (PDE, ipLDDT, ipTM, future) must add `z = 0.5 * (z + z.transpose(-3, -2))` as the first line of their own `_predict`. Asymmetric pair-heads (`PaeHead`, future directional ones) consume `z` directly. `PlddtHead` consumes `s` only, so the change is a no-op for it. Load-bearing regression: `tests/regression/test_flow_matching_loss_unchanged.py` continues to pass.
- **WandB logging — shipped in PR #15 (2026-05-19).** Logging is wired via a single composable Hydra fragment at [configs/logging/wandb.yaml](configs/logging/wandb.yaml) (`@package _global_`, project `confidence-distillation`, fields `log_wandb` / `wandb_project` / `wandb_entity` / `wandb_group` / `wandb_tags`). Each of the four confidence configs (`distillation_swissprot`, `distillation_swissprot_control`, `distillation_teddymer_pae`, `distillation_teddymer_multihead`) composes it via `- /logging/wandb@logging` and sets a per-config `wandb_tags:` override (e.g. `["pae", "teddymer"]`). The entry point [src/proteinfoundation/confidence/train_confidence.py](src/proteinfoundation/confidence/train_confidence.py) builds the `WandbLogger` via `_build_wandb_logger(cfg)`, which honours `WANDB_MODE=disabled` (returns `None`) and `log_wandb: false`; the logger is then passed to `hydra.utils.instantiate(cfg.trainer, logger=..., _convert_="partial")`. `wandb_logger.log_hyperparams({"config": OmegaConf.to_container(cfg, resolve=True)})` runs once on rank 0. The run-name is the config's existing top-level `run_name` field, used as both `id` and `name` so re-launching the same `run_name` *resumes* the WandB run rather than starting a new one. **Future training / sampling / eval entry points must reuse this fragment** rather than embed wandb keys directly in their training YAMLs (the legacy `configs/training_local_latents*.yaml` pattern is the anti-pattern — do not propagate it).
- **Rank-0-only loguru gate.** Any Hydra-launched entry point that runs under DDP must call `_gate_loguru_to_rank0()` (or an equivalent gate) as the FIRST line of `main()`, before any `logger.info(...)`. The gate reads `LOCAL_RANK`, `NODE_RANK`, and `RANK` (the last covers `torchrun` / torchelastic launches that don't set `NODE_RANK`); if any is non-zero it calls `logger.remove()` to drop loguru's default sink. `WandbLogger` is already rank-0-only by Lightning contract, so this only affects loguru. Without this gate, the "Composed config:" dump and any future info lines duplicate across ranks. The pattern lives at [src/proteinfoundation/confidence/train_confidence.py:29-34](src/proteinfoundation/confidence/train_confidence.py#L29-L34) and is the reference implementation.
- **CUDA allocator — confidence-distill sbatch must export `PYTORCH_ALLOC_CONF=expandable_segments:True`.** Pair-update layers at [src/proteinfoundation/nn/modules/pair_update.py](src/proteinfoundation/nn/modules/pair_update.py) (lines ~82, 88, 105, 107, 110) wrap their inner blocks in reentrant `torch.utils.checkpoint`. Combined with the Teddymer dimer crop's heterogeneous L distribution (250–550 residues within an epoch), the caching allocator fragments after ~900 steps and OOMs with multi-GiB reserved-but-unallocated — observed at step 927/63168 in job 53667 (5.15 GiB reserved-but-unallocated, 878 MiB free of 93 GB H100). With `expandable_segments:True`, PyTorch grows existing segments rather than demanding fresh contiguous blocks; the same workload reached first val + checkpoint at step 2000 in job 53677 (2026-05-19) and continues to run. The export goes alongside the NCCL block, *before* the `srun python -m ...` line — see [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch) for the canonical placement. The rule generalises: any future confidence-distill (or main-flow) sbatch that touches the head's embedded `ConfidenceTrunk` (or any reentrant-ckpt site) under variable-L inputs must export this allocator config. The DDP fix `static_graph=true` is the reducer-side mitigation; `expandable_segments` is the allocator-side one — both are required, neither subsumes the other.
- **Multi-GPU DDP strategy — must pair `find_unused_parameters=true` with `static_graph=true`.** All four confidence configs (`distillation_swissprot`, `distillation_swissprot_control`, `distillation_teddymer_pae`, `distillation_teddymer_multihead`) ship a structured `DDPStrategy(find_unused_parameters=true, static_graph=true)` block as the yaml default (landed in PR #16 on `train_pae`, 2026-05-19). Two failures compose without both flags: (i) the frozen complexa trunk has parameters that don't feed the head's loss (`PaeHead` consumes `z` only, `PlddtHead` consumes `s` only — neither uses every trunk param), so `find_unused_parameters=false` raises "parameters that were not used in producing the loss"; (ii) the **head's embedded `ConfidenceTrunk`** at [src/proteinfoundation/nn/confidence/base.py](src/proteinfoundation/nn/confidence/base.py) uses reentrant `torch.utils.checkpoint` in its pair-update layers, so with `find_unused_parameters=true` alone DDP's reducer marks the same checkpointed param ready twice and raises "Parameter at index N with name `head.trunk.pair_update_layers.*.linear_2.weight` has been marked as ready twice". The frozen complexa trunk also has gradient checkpointing but runs under `torch.no_grad()` in the sidecar, so its checkpoint calls never enter the backward graph — the reentrant-ckpt site that trips DDP is the head's trunk copy, not the frozen one. Both flags together fix both failures. New confidence configs must include the same structured block (or compose a shared Hydra group once that follow-up lands). Generalisation of the multi-head wrapper memo: `static_graph=true` is required for **any** confidence-distill multi-GPU run, not only when `loss_weights` zeroes a head.
- **Resume-from-checkpoint — shipped in PR #18 (2026-05-22).** Every confidence-distill config exposes a top-level `resume_ckpt_path: null` field. [src/proteinfoundation/confidence/train_confidence.py](src/proteinfoundation/confidence/train_confidence.py) reads it and forwards to `trainer.fit(..., ckpt_path=resume_ckpt_path)`; a rank-0 loguru line announces the resume before `fit`. `resume_ckpt_path` is **orthogonal to `resume_id`**: the former restores Lightning trainer state (weights + optimizer + LR scheduler + global step + epoch); the latter re-attaches to a prior WandB run id. A typical mid-training resume sets both. The sbatch contract is a `RESUME_CKPT_PATH` env var that the script translates into a `resume_ckpt_path=…` Hydra override before `srun python -m ...`; see [scripts/train_confidence_teddymer_pae.sbatch](scripts/train_confidence_teddymer_pae.sbatch) lines ~172-185 for the canonical pattern. Usage: `RESUME_CKPT_PATH=/abs/path/to.ckpt sbatch --export=ALL,RESUME_CKPT_PATH scripts/train_confidence_*.sbatch`. **Any future confidence-distill (or main-flow) sbatch must follow the same env-var → Hydra-override pattern** — embedding the ckpt path as a hardcoded Hydra arg in the sbatch is the anti-pattern. The SwissProt sbatch at [scripts/train_confidence_swissprot.sbatch](scripts/train_confidence_swissprot.sbatch) does not yet have the equivalent wiring; add it when next launching a SwissProt resume.
- **Joint training via `MultiHeadConfidence` wrapper — shipped in PR #10.** Lives at [src/proteinfoundation/nn/confidence/multi_head.py](src/proteinfoundation/nn/confidence/multi_head.py). Registered as `multi_head`. Holds an `nn.ModuleDict` of child heads, runs the trunk once per batch, dispatches the same `(s, z, mask, cond)` to every child's `_predict`, and returns `{head_name: prediction}`. `MultiHeadLoss` at [src/proteinfoundation/nn/confidence/_losses.py](src/proteinfoundation/nn/confidence/_losses.py) aggregates `total = sum(w_i * L_i)` with per-task scalars logged separately. Two weighting layers, kept separate in Hydra: (i) **within-head** `loss.ce_weight` / `loss.ev_weight` (the existing pLDDT recipe family); (ii) **across-head** `loss_weights: {head: scalar}` (multiplies the head's total loss before summing with siblings; `weight: 0.0` is a hard short-circuit — skip the head's loss call entirely). A head whose entire batch mask is False contributes exactly zero, never NaN-from-empty-mean. The wrapper asserts `expected_trunk_eval_t` matches across children at construction AND that every child has a non-empty `output_name_root`; a head needing a different `t` cannot live under this wrapper. Single-head runs (`PlddtHead`-only on SwissProt, `PaeHead`-only on Teddymer for the clean reference number) keep working unchanged — joint training is one Hydra config flip away, not the default. **DDP footgun:** `weight: 0.0` disables the loss but the child's `_predict` still runs in the wrapper's `forward`, so the child's parameters enter the autograd graph with no gradient — `find_unused_parameters=False` raises at backward. Working escape hatches: (a) remove the child from `children:`, (b) `DDP(static_graph=True)`. The wrapper emits a `RuntimeWarning` at construction listing offending heads.

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
