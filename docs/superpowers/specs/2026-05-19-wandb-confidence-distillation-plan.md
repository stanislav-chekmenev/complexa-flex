# WandB logging for confidence-head distillation — implementation plan

**Date:** 2026-05-19
**Branch:** `wandb-confidence-distillation` (off `dev`)
**Spec:** [2026-05-19-wandb-confidence-distillation-design.md](2026-05-19-wandb-confidence-distillation-design.md)
**Status:** ready for implementation

## Divergences from the spec (resolved before milestones)

1. **Test location.** The spec writes tests under `tests/confidence/`. That
   directory does not exist in this repo. The repo's test tree is split
   `tests/unit/confidence/` vs `tests/integration/confidence/`, both with
   `__init__.py` already in place (verified). The four new test files in
   this plan land under those two existing directories — three unit, one
   integration-shaped (the trainer-wiring test). No `__init__.py` or
   `conftest.py` setup is needed.
2. **`WandbLogger` attribute names confirmed.** Verified against the
   pinned `lightning==2.5.6` on this host: a `WandbLogger(project=...,
   id=..., name=..., tags=[...])` exposes `_project`, `_id`, `_name`,
   and stores tags inside `_wandb_init["tags"]` (the public `name` is a
   read-only property derived from `_name`). The spec's asserted
   attribute names are correct; tags must be asserted via
   `logger._wandb_init["tags"]`, not a top-level `_tags`.
3. **Helper signature stays exactly as the spec.** No extra knobs added.

## Hard constraints (verbatim, do not relax)

- **TDD.** Every test in Milestone 0 is written and committed *before*
  the corresponding implementation. Each test must fail (red) when run
  against the unmodified `train_confidence.py`, then pass (green) after
  Milestone 1.
- **Scope.** This PR touches exactly:
  - `src/proteinfoundation/confidence/train_confidence.py` (edit)
  - `configs/logging/wandb.yaml` (new)
  - `configs/confidence/distillation_swissprot.yaml` (edit)
  - `configs/confidence/distillation_swissprot_control.yaml` (edit)
  - `configs/confidence/distillation_teddymer_pae.yaml` (edit)
  - `configs/confidence/distillation_teddymer_multihead.yaml` (edit)
  - `tests/unit/confidence/test_wandb_logger_construction.py` (new)
  - `tests/unit/confidence/test_wandb_logger_disabled.py` (new)
  - `tests/unit/confidence/test_rank0_logging.py` (new)
  - `tests/integration/confidence/test_train_confidence_passes_logger_to_trainer.py` (new)
  Do **not** touch `src/proteinfoundation/nn/confidence/` or
  `src/proteinfoundation/train.py`.
- **No new dependency.** `wandb` and
  `lightning.pytorch.loggers.WandbLogger` already import in this env
  (verified).
- **No Claude attribution** in any commit message or PR description.
- **No emojis** in code, configs, tests, commits, or PRs.

## Reviewer panel (for the eventual PR)

- `code-review-debug-complexity-expert` (mandatory, per `CLAUDE.md`).
- `ml-protein-architect` (Hydra + Lightning glue is exactly this PR's
  scope).

No biology / generative / physics reviewer needed — pure plumbing.

---

## Milestone 0 — failing tests committed first (TDD red)

Goal: land four failing tests on the branch before any implementation.

### Files touched

- `tests/unit/confidence/test_wandb_logger_construction.py` (new)
- `tests/unit/confidence/test_wandb_logger_disabled.py` (new)
- `tests/unit/confidence/test_rank0_logging.py` (new)
- `tests/integration/confidence/test_train_confidence_passes_logger_to_trainer.py` (new)

### Test 1 — `test_wandb_logger_construction.py` (unit)

Imports `_build_wandb_logger` from
`proteinfoundation.confidence.train_confidence`. Patches `wandb.init`
(to prevent any network or `~/.netrc` lookup). Builds a minimal
`DictConfig` via `OmegaConf.create({...})` with:

```python
{
    "run_name": "pae-distill-teddymer",
    "logging": {
        "log_wandb": True,
        "wandb_project": "confidence-distillation",
        "wandb_entity": None,
        "wandb_group": None,
        "wandb_tags": ["pae", "teddymer"],
    },
}
```

Calls `_build_wandb_logger(cfg)` and asserts:

- the return value is an instance of
  `lightning.pytorch.loggers.WandbLogger`,
- `logger._project == "confidence-distillation"`,
- `logger._id == "pae-distill-teddymer"`,
- `logger._name == "pae-distill-teddymer"`,
- `logger._wandb_init["tags"] == ["pae", "teddymer"]`.

Mocks: `unittest.mock.patch("wandb.init")`. Optionally
`monkeypatch.delenv("WANDB_MODE", raising=False)` to neutralise
environment leak from a previous test.

### Test 2 — `test_wandb_logger_disabled.py` (unit)

Two cases, parameterised or two top-level `test_*` functions:

- `test_returns_none_when_log_wandb_false`: cfg has
  `logging.log_wandb=False` (other fields irrelevant); asserts
  `_build_wandb_logger(cfg) is None` **without** patching `wandb.init`
  (the function must short-circuit before any wandb call).
- `test_returns_none_when_wandb_mode_disabled`: cfg has
  `logging.log_wandb=True`; `monkeypatch.setenv("WANDB_MODE",
  "disabled")`; asserts the return is `None`.

Third case (defensive, recommended): `test_returns_none_when_logging_block_absent`
with `cfg = OmegaConf.create({"run_name": "x"})`; asserts the helper
returns `None` and does **not** raise.

### Test 3 — `test_rank0_logging.py` (unit)

Imports `_gate_loguru_to_rank0` from
`proteinfoundation.confidence.train_confidence`. Two cases:

- `test_non_rank0_silences_loguru`: `monkeypatch.setenv("LOCAL_RANK",
  "1")`. After calling `_gate_loguru_to_rank0()`, assert that
  `loguru.logger` has no active handlers (inspect
  `loguru._logger.core.handlers` — it should be an empty dict).
- `test_rank0_keeps_loguru`: `monkeypatch.setenv("LOCAL_RANK", "0")`
  and `monkeypatch.setenv("NODE_RANK", "0")`. After calling the gate,
  assert `loguru._logger.core.handlers` is non-empty.

**Test-isolation note for the implementer:** loguru's handler state is
process-global. Use a fixture (`autouse=True`, function-scoped) that
snapshots handlers before each test and restores them after, so
ordering between these two cases (or with other tests in the same
session) cannot poison the global state. Add this fixture inline in
this test file — do **not** modify the shared
`tests/unit/confidence/conftest.py` (it does not exist; do not create
one for this PR).

### Test 4 — `test_train_confidence_passes_logger_to_trainer.py` (integration)

Lightweight integration-shaped test. Patches, in this order:

- `proteinfoundation.confidence.train_confidence.L.Trainer` →
  `MagicMock` returning a `MagicMock(is_global_zero=True,
  fit=MagicMock())`. This is the assertion target.
- `proteinfoundation.confidence.train_confidence.build_confidence_head_from_cfg`
  → `MagicMock(return_value=MagicMock())`.
- `proteinfoundation.confidence.train_confidence.ConfidenceDistillationModule`
  → `MagicMock(return_value=MagicMock())`.
- `proteinfoundation.confidence.train_confidence.hydra.utils.instantiate`
  → returns `MagicMock()` (covers the datamodule call).
- `proteinfoundation.confidence.train_confidence.L.seed_everything` →
  no-op.
- `wandb.init` → no-op.

Composes a minimal `DictConfig` with the keys `main()` reads:
`seed`, `run_name`, `logging.{log_wandb, wandb_project, wandb_tags,
wandb_entity, wandb_group}`, `integrity.enabled=False`, `confidence`,
`training` (only the fields touched by the call to the
`ConfidenceDistillationModule` constructor — the constructor itself is
mocked, so any subset that does not trip Hydra/OmegaConf access is
fine), `data`, `trainer`.

Calls `train_confidence.main.__wrapped__(cfg)` (bypassing the
`@hydra.main` decorator) and asserts:

- `L.Trainer` was called once,
- the call's `logger=` kwarg is a `WandbLogger` instance,
- `wandb_logger.log_hyperparams` was called exactly once with a single
  positional or keyword payload whose top-level key is `"config"`.

Then re-run with `cfg.logging.log_wandb=False` and assert
`L.Trainer.call_args.kwargs["logger"] is None` and that
`log_hyperparams` was not called.

### Known risk: Milestone 0, Test 4 may be too sticky to mock

The trunk path inside `main()` calls
`ConfidenceDistillationModule(...)` with keyword args drawn from
`cfg.training.{trunk_ckpt_path, autoencoder_ckpt_path, trunk_eval_t,
opt.*, loss.*}`. If the implementer finds that providing all those
keys in the minimal cfg becomes brittle or that
`MagicMock(return_value=...)` for the module class fails because of
OmegaConf attribute access mid-construction, **fall back** to a
thinner pair of unit tests:

- assert that `main()` passes `logger=<built logger>` to `_build_trainer`
  by patching `_build_trainer` itself and inspecting its kwargs;
- separately, unit-test `_build_wandb_logger` already covers the
  construction path.

The plan accepts this fallback explicitly: if Test 4 takes more than
~30 minutes of mock plumbing to stabilise, replace it with the thinner
unit test. Document the choice in the PR description.

### Exit criteria

- `pytest tests/unit/confidence/test_wandb_logger_construction.py
  tests/unit/confidence/test_wandb_logger_disabled.py
  tests/unit/confidence/test_rank0_logging.py
  tests/integration/confidence/test_train_confidence_passes_logger_to_trainer.py`
  collects all four files and fails with `ImportError` or `AttributeError`
  for `_build_wandb_logger` / `_gate_loguru_to_rank0` (proving the tests
  are wired to real symbols, and red).
- All four files added to git in a single TDD commit:
  `tests: failing tests for wandb logger + rank0 loguru gate`.

### Abort criteria

- The `_wandb_init["tags"]` accessor returns something different from
  `["pae", "teddymer"]` on this host (i.e. Lightning silently rewrites
  tags). In that case, swap the assertion to inspect
  `logger.experiment._init_kwargs["tags"]` after patching
  `wandb.init` to a `MagicMock(return_value=MagicMock(...))`, and
  document the divergence at the top of this plan.
- The pinned Lightning is bumped (currently 2.5.6) before this PR
  lands and `_project` / `_id` / `_name` disappear. Treat as a stop
  signal: re-verify on the new version and update the test
  assertions; do not weaken them to "any truthy attribute".

### Risk register

- R0.1 (low): loguru handler-state leakage between tests. Mitigation:
  per-test snapshot/restore fixture inside `test_rank0_logging.py`.
- R0.2 (medium): Test 4 mock surface bigger than expected.
  Mitigation: documented fallback above.

---

## Milestone 1 — implementation, smallest end-to-end (TDD green)

Goal: make all Milestone 0 tests pass.

### Files touched

- `src/proteinfoundation/confidence/train_confidence.py` (edit, single file)
- `configs/logging/wandb.yaml` (new file, new top-level Hydra group)

### Edits to `train_confidence.py`

Add imports at the top of the module (next to the existing imports):

```python
from typing import Optional
from lightning.pytorch.loggers import WandbLogger
```

Add two helpers, between `_build_trainer` and `main`:

```python
def _gate_loguru_to_rank0() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    if local_rank != 0 or node_rank != 0:
        logger.remove()


def _build_wandb_logger(cfg: DictConfig) -> Optional[WandbLogger]:
    log_cfg = cfg.get("logging", None)
    if log_cfg is None or not bool(log_cfg.get("log_wandb", False)):
        return None
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    return WandbLogger(
        project=log_cfg["wandb_project"],
        id=cfg.run_name,
        name=cfg.run_name,
        entity=log_cfg.get("wandb_entity", None),
        group=log_cfg.get("wandb_group", None),
        tags=list(log_cfg.get("wandb_tags", []) or []),
    )
```

(The module's existing `from loguru import logger` import already binds
`logger`; `_gate_loguru_to_rank0` reuses it. Do not shadow with a local
import.)

Rewrite `main()` body to (in order):

1. `_gate_loguru_to_rank0()` — first thing, **before** the existing
   `logger.info("Composed config: ...")` call.
2. `L.seed_everything(...)` (unchanged).
3. `logger.info("Composed config:\n{}", OmegaConf.to_yaml(cfg, resolve=True))`
   (unchanged in content, now rank-0-only by virtue of step 1).
4. Integrity / head / module construction (unchanged).
5. `wandb_logger = _build_wandb_logger(cfg)` — new line, **before**
   `_build_trainer`.
6. Replace `trainer = _build_trainer(cfg)` with
   `trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger,
   _convert_="partial")`. Remove the `_build_trainer` helper if it
   becomes a one-line shim, or keep it and pass `logger` through —
   either is fine; the test asserts on `L.Trainer(...)` kwargs, not on
   `_build_trainer`'s presence. Prefer **keeping** `_build_trainer` and
   threading `logger` through its signature
   (`def _build_trainer(cfg, *, logger)`) so the change to `main()`
   stays minimal and Test 4 can patch either symbol.
7. After trainer construction, if `wandb_logger is not None and
   trainer.is_global_zero:` call `wandb_logger.log_hyperparams({"config":
   OmegaConf.to_container(cfg, resolve=True)})`.
8. `trainer.fit(module, datamodule=datamodule)` (unchanged).

### New file `configs/logging/wandb.yaml`

```yaml
# @package _global_
logging:
  log_wandb: true
  wandb_project: confidence-distillation
  wandb_entity: null
  wandb_group: null
  wandb_tags: []
```

(`wandb_id` is implicitly `${run_name}`, which every confidence config
sets at top level; no new top-level field needed.)

### Exit criteria

- `pytest tests/unit/confidence/test_wandb_logger_construction.py
  tests/unit/confidence/test_wandb_logger_disabled.py
  tests/unit/confidence/test_rank0_logging.py
  tests/integration/confidence/test_train_confidence_passes_logger_to_trainer.py`
  is all green (or all green minus Test 4 if the documented fallback is
  taken, in which case the fallback unit test is added in the same
  commit and is green).
- `pytest tests/unit/confidence/ tests/integration/confidence/` runs
  with no new red — pre-existing tests unaffected.
- `pytest tests/regression/test_flow_matching_loss_unchanged.py` is
  still green (per `CLAUDE.md` standing guard on the trunk).

### Abort criteria

- Lightning's `WandbLogger.__init__` raises at import time on this env
  with the chosen kwarg set (e.g. `tags` removed in a patch bump).
  Re-verify pin (`python -c "import lightning; print(lightning.__version__)"`
  must report `2.5.6`); if drifted, stop and report.
- `hydra.utils.instantiate(cfg.trainer, logger=wandb_logger,
  _convert_="partial")` does not propagate the `logger=` kwarg through
  to `L.Trainer.__init__` because of an existing `logger:` key in any
  `cfg.trainer` block (it is not present in any of the four confidence
  YAMLs — verified — but check before proceeding).

### Risk register

- R1.1 (low): `cfg.trainer.logger` collision. Mitigation: grep the
  four YAMLs (verified absent at plan time).
- R1.2 (medium): `OmegaConf.to_container(cfg, resolve=True)` payload
  trips wandb's hyperparam serialiser (custom OmegaConf nodes or
  non-JSON-serialisable values). Mitigation: the payload is wrapped
  in `{"config": ...}` so wandb sees a single dict; if a serialisation
  warning appears, fall back to `OmegaConf.to_yaml(cfg, resolve=True)`
  as a string under the same key.
- R1.3 (low): `trainer.is_global_zero` raises if `trainer` is a
  `MagicMock` in Test 4. Mitigation: the test sets
  `is_global_zero=True` on the mock explicitly.

---

## Milestone 2 — wire the fragment into all four confidence configs

Goal: every confidence run picks up the WandB logger automatically.

### Files touched

- `configs/confidence/distillation_swissprot.yaml`
- `configs/confidence/distillation_swissprot_control.yaml`
- `configs/confidence/distillation_teddymer_pae.yaml`
- `configs/confidence/distillation_teddymer_multihead.yaml`

### Edit pattern (apply to each config)

Add `- /logging/wandb@logging` to the `defaults:` block (anywhere
before `_self_`), then append a top-level `logging:` override that
sets only `wandb_tags`. Per-config tag set:

- `distillation_swissprot.yaml` → `wandb_tags: ["plddt", "swissprot"]`
- `distillation_swissprot_control.yaml` → `wandb_tags: ["plddt", "swissprot", "control"]`
- `distillation_teddymer_pae.yaml` → `wandb_tags: ["pae", "teddymer"]`
- `distillation_teddymer_multihead.yaml` → `wandb_tags: ["plddt", "pae", "teddymer", "multi"]`

The other `logging.*` fields (project, entity, group) inherit from
`configs/logging/wandb.yaml`. Per-config `wandb_group` overrides are a
nice-to-have but out of scope (the spec keeps `wandb_group: null` and
relies on tags for filtering).

### Exit criteria

- For every config:
  `python -m proteinfoundation.confidence.train_confidence
  --config-name=confidence/<name> --cfg job --resolve` resolves cleanly
  (exit code 0, no Hydra missing-key errors), and the printed config
  contains `logging.log_wandb: true`, `logging.wandb_project:
  confidence-distillation`, and the config-specific `wandb_tags`.
- Add a compose-only unit test next to the existing
  `tests/unit/confidence/test_config_compose.py` (extend, do not
  create a new file): one new test function
  `test_logging_block_composed_for_all_confidence_runs` that loops over
  the four config names, `compose()`s each, and asserts
  `cfg.logging.log_wandb is True` and `cfg.logging.wandb_project ==
  "confidence-distillation"`. This test goes in with the YAML edits in
  the same commit.

### Abort criteria

- Hydra rejects `- /logging/wandb@logging` because of a package /
  path collision with an existing `logging:` key elsewhere in the
  compose graph (verified: no other config defines top-level
  `logging:`). If the rejection appears anyway, rename the Hydra group
  to `wandb` (`configs/wandb/default.yaml`, defaults entry
  `- /wandb/default@logging`) and update tests accordingly.

### Risk register

- R2.1 (low): a future config (out of scope) that already sets
  `logging:` would silently merge. Mitigation: only the four named
  configs are in scope today; the compose-only test will catch a
  regression if a future PR diverges.

---

## Milestone 3 — manual DDP smoke-test (instructions for the implementer)

Goal: confirm the rank-0 gate and the logger-pass-through behave on a
real Lightning launch. Not a CI test.

### Files touched

None (documentation-only in the PR description / handoff).

### Smoke-test recipe

On the dev host, from `complexa-flex/` root, in an env with
`uv` available:

```bash
WANDB_MODE=disabled \
  uv run python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_swissprot \
    trainer.devices=2 \
    trainer.max_steps=1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    +trainer.fast_dev_run=true
```

Confirm by reading the captured stdout/stderr:

1. The "Composed config:" loguru line appears **exactly once** (not
   twice). Grep:
   `... 2>&1 | grep -c "Composed config:"` should print `1`.
2. The Lightning summary line ("`Trainer logger: ...`") shows that
   `trainer.logger` is `None` (because `WANDB_MODE=disabled` returns
   `None` from the helper).
3. The full Hydra resolved config dump appears once and the run exits
   with code 0 after the one-step `fast_dev_run`.

Then re-run **without** `WANDB_MODE=disabled` but with
`WANDB_MODE=offline` so a local wandb dir is created without network
traffic:

```bash
WANDB_MODE=offline \
  uv run python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_teddymer_pae \
    trainer.devices=1 \
    +trainer.fast_dev_run=true
```

Confirm:

- a `wandb/offline-run-*` directory is created,
- inside, `config.yaml` contains `project: confidence-distillation`
  and `name: pae-distill-teddymer`,
- tags include `pae` and `teddymer`.

### Exit criteria

- Both smoke commands return exit code 0.
- The three grep checks for the first command pass.
- The two field checks for the second command pass.

### Abort criteria

- The "Composed config:" line appears twice on the 2-GPU run →
  `_gate_loguru_to_rank0` is not running early enough OR the second
  process is not seeing `LOCAL_RANK` set when the gate runs. Inspect:
  is `@hydra.main` re-entering on the spawned rank? If yes, the gate
  must run inside `main()` (it does — confirm placement). If still
  duplicated, fall back to checking `os.environ.get("RANK")` as well
  as `LOCAL_RANK` (Lightning sets both before re-invoking the entry
  point under DDP-spawn). Document the fix and update Test 3.

### Risk register

- R3.1 (medium): Lightning's DDP launcher (`subprocess`-spawn vs
  `python -m torch.distributed.run`) may set the rank envs at slightly
  different lifecycle points. Mitigation: smoke-test catches this; the
  fix (also gate on `RANK`) is one extra `os.environ.get` and an
  updated test case.
- R3.2 (low): `+trainer.fast_dev_run=true` interacts badly with
  `ModelCheckpoint` callbacks defined in each YAML. Mitigation: known
  Lightning behaviour — `fast_dev_run` disables checkpointing; harmless.

---

## Out-of-scope follow-ups (not this PR)

- Artefact upload of the resolved Hydra config (would mirror trunk
  `train.py`'s `store_n_log_configs`).
- Reusable callback that mirrors per-head reliability diagrams as
  WandB `Image` panels (currently saved to disk only).
- Renaming the WandB project to `reward-distillation` once the heads
  feed a reward stack; trivial config-only change.

## Handoff

Hand to `ml-protein-architect` for implementation, TDD discipline as
sequenced above (Milestone 0 commit lands before Milestone 1 commit
lands before Milestone 2 commit). Review panel on the final PR:
`code-review-debug-complexity-expert` (mandatory) + `ml-protein-architect`.
