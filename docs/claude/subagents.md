# Subagent roster — full detail

Read this file before delegating to a subagent. The core [CLAUDE.md](../../CLAUDE.md) keeps only the one-line routing rule; the full roster and delegation guidance live here.

Subagent definitions are tracked under [.claude/agents/](../../.claude/agents/). Each agent has a defined scope; route work to the agent whose description matches the task:

| Agent | Scope |
| --- | --- |
| [software-planning-architect](../../.claude/agents/software-planning-architect.md) | Translate ambiguous asks into structured implementation plans. Upstream of implementation. |
| [ml-protein-architect](../../.claude/agents/ml-protein-architect.md) | Project-level architecture and *writes/edits code* once a plan is agreed. |
| [ml-software-pytorch-jax-expert](../../.claude/agents/ml-software-pytorch-jax-expert.md) | PyTorch and JAX framework internals (autograd, compile, FSDP/SPMD, profiling, kernels). |
| [code-review-debug-complexity-expert](../../.claude/agents/code-review-debug-complexity-expert.md) | Code review, systematic debugging, algorithmic / complexity optimisation. |
| [generative-protein-scientist](../../.claude/agents/generative-protein-scientist.md) | Applied generative modelling for protein structure/sequence (FM, diffusion, guidance, search). |
| [generative-flow-stochastic-math-expert](../../.claude/agents/generative-flow-stochastic-math-expert.md) | Math of diffusion / flow matching / OT / guidance / SMC. |
| [structural-biology-binder-expert](../../.claude/agents/structural-biology-binder-expert.md) | Folded protein–protein binder biophysics and field practice. |
| [structural-biology-idp-smallmol-expert](../../.claude/agents/structural-biology-idp-smallmol-expert.md) | IDP/IDR, motif-driven, and small-molecule binder design. |
| [physics-statmech-md-dft-expert](../../.claude/agents/physics-statmech-md-dft-expert.md) | Stat mech, MD, free-energy methods, DFT, MLIP literature. |
| [xray-crystallography-binder-ml](../../.claude/agents/xray-crystallography-binder-ml.md) | X-ray crystallography + ML, oriented to binder validation. |

When delegating, pick the agent whose `description:` line matches the task. If multiple apply, dispatch them in parallel where they are working on independent angles.
