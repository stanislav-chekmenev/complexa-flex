# Agent roster and model routing

Mechanism behind the roster in `CLAUDE.md`'s Subagent roster. Read before adding
an agent, changing a `model:` pin, or debugging an agent that appears to run on
the wrong model. The guard suite is `tests/unit/test_agent_routing.py`.

## Lanes

The model IDs below are LiteLLM gateway deployment names, not Anthropic model
names. Routing works only when the session has gateway access: credentials stay
untracked, and `ANTHROPIC_BASE_URL` must name the gateway without a `/v1`
suffix, because the client appends `/v1/messages` itself.

| Lane | Model | Who |
| --- | --- | --- |
| Main session | `claude-opus-5-5` | `orchestrator` (`model` in `.claude/settings.json`; its agent definition inherits) |
| Review, advice, planning, search | `z-ai/glm-5.3-flash` | `code-review-debug-complexity-expert`, `software-planning-architect`, `generative-protein-scientist`, `generative-flow-stochastic-math-expert`, `structural-biology-binder-expert`, `structural-biology-idp-smallmol-expert`, `physics-statmech-md-dft-expert`, `xray-crystallography-binder-ml`, `Explore`, `Plan`, `general-purpose` |
| Implementation | `gpt-6-luna` | `ml-protein-architect`, `ml-software-pytorch-jax-expert` |

The implementation lane is for the two named implementers; the other eight
specialists and three shadowed built-in names are on the judgement lane. The
`code-review-debug-complexity-expert` may edit code, but its primary role is
review and it remains on the GLM lane to provide a cross-family check on Luna
implementation work. A Luna implementer must not be the sole reviewer of a diff
it wrote.

The main session starts as `orchestrator` from `.claude/settings.json`; its
frontmatter uses `model: inherit` and keeps the full tool set. Delegation is
prompt policy, not a `tools:` restriction. The project files for `Explore`,
`Plan`, and `general-purpose` shadow the built-in definitions, so each file
restates the applicable agent contract as well as pinning its model.

At T2, `CLAUDE.md` requires `code-review-debug-complexity-expert` plus at most
one relevant domain reviewer; at T3 that reviewer is mandatory alongside 2–4
domain reviewers. Select the domain panel from the risk mapping in
`CLAUDE.md`'s Pull-request review protocol, and apply its target-branch
relaxation. The reviewer's model lane does not replace the protocol's blocking
and non-blocking finding rules.

## Sharp edges

- **An agent without `model:` silently inherits.** A missing pin can therefore
  put a reviewer or implementer on the main-session model without an error.
  Keep each roster pin explicit and covered by `tests/unit/test_agent_routing.py`.
- **`CLAUDE_CODE_SUBAGENT_MODEL` outranks every frontmatter pin.** This project
  tolerates it only when it is exactly `claude-opus-5`; any other value collapses
  the documented lanes. It pins subagents, not the main session.
- **Shell exports outrank the settings `env` block.** The routing test checks
  exported family aliases in the shell that runs it, but cannot establish the
  environment of another session.
- **A running session keeps the routing it started with.** Agent definitions and
  settings are loaded at session start; changes do not re-route the session that
  made them. Verify changes from a fresh session.
- **The shadowed names are project agent files.** `Explore`, `Plan`, and
  `general-purpose` replace the built-in prompts, so changes to these files must
  preserve their respective read-only-search, planning, and general-purpose
  contracts.
- **The guard checks repository configuration, not gateway availability or
  runtime model selection.** A fresh-session probe is still required to verify
  that Claude Code honours a pin.

## Adding or re-laning an agent

1. Add or edit `.claude/agents/<name>.md`; set a `name:` equal to the filename
   stem and an explicit full deployment name in `model:`. Add `effort: max` for
   judgement agents; implementers do not carry an `effort` key.
2. Update `LANES` and `GATEWAY_MODELS` in `tests/unit/test_agent_routing.py`.
3. Update the Model column in the roster table in `CLAUDE.md` and the lane table
   above.
4. Update `.claude/agents/orchestrator.md` when the agent's role changes how
   tasks should be routed.
5. Run the routing guard and verify the model from a fresh session. File-level
   assertions cannot prove runtime routing.

## Re-verifying a lane

Start a fresh session after changing a pin, run the agent as the main thread, and
inspect the returned model usage:

```text
claude --agent <name> -p 'Reply with exactly: OK' --output-format json
```

The same probe can check the `agent` setting by launching a plain session and
checking its model usage. The local command confirms the reported model for that
fresh invocation; it does not prove gateway reachability for a different
machine or shell.
