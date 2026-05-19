# WandB logging for confidence-head distillation — design

**Date:** 2026-05-19
**Branch:** `wandb-confidence-distillation` (off `dev`)
**Status:** draft (pending implementation)

## Problem

The confidence-head distillation training pipeline (Hydra entry point
`proteinfoundation.confidence.train_confidence`) currently runs without a
Lightning logger configured. The four production configs —
`distillation_swissprot.yaml`, `distillation_swissprot_control.yaml`,
`distillation_teddymer_pae.yaml`, `distillation_teddymer_multihead.yaml` —
all instantiate `cfg.trainer` directly via Hydra, and `cfg.trainer` carries
no `logger:` key, so the runs log to Lightning's default CSV sink only.

The trunk's main `src/proteinfoundation/train.py` already has the WandB
pattern at lines 380–390; this work ports the same shape into the
confidence sidecar without touching the trunk.

A second issue, surfaced after the design pass: loguru `logger.info(...)`
calls inside `train_confidence.py` run on every rank under DDP, so the
"Composed config" dump and any future info lines duplicate across ranks.
`WandbLogger` itself is already rank-0-only (Lightning gates `wandb.init`
internally), so the duplication is purely in the loguru sink, not in WandB.

## Goals

1. WandB logging for every confidence-distillation run, behind a single
   composable Hydra fragment that all four configs include.
2. WandB project name: **`confidence-distillation`** (user-confirmed —
   matches code nomenclature; will not rename to `reward-distillation`).
3. Run name describes what is being trained — use the existing top-level
   `run_name` field per config (`plddt-distill-swissprot`,
   `pae-distill-teddymer`, `plddt-distill-swissprot-control`,
   `multi-plddt-pae-distill-teddymer`). No rename needed.
4. Loguru output is **rank-0-only** under DDP — no duplicated config
   dumps from non-zero ranks.
5. Existing CI / regression tests continue to pass. In particular,
   `tests/regression/test_flow_matching_loss_unchanged.py` must remain
   green — this work does not touch the trunk or `nn/confidence/`.

## Non-goals

- No new metrics. The existing per-head `compute_loss_and_metrics`
  contract (CE / EV / MAE / Pearson / Spearman / ECE / reliability
  diagram) is unchanged; WandB just *receives* what the Lightning module
  already logs.
- No artefact upload (config snapshots, checkpoint mirroring). The trunk
  `train.py` does this via `store_n_log_configs`; we skip it for now
  because Hydra's `.hydra/` directory already preserves the resolved
  config alongside every run. If needed later, it is a one-helper add.
- No change to the trunk training entry point (`src/proteinfoundation/train.py`).
- No change to the sbatch scripts (env vars like `WANDB_API_KEY` are
  already inherited from the user's shell on this host).
- No change to `nn/confidence/` — heads remain framework-pure.

## Architecture

### New Hydra fragment

`configs/logging/wandb.yaml` (single composable block, package
`@logging`):

```yaml
# @package _global_
logging:
  log_wandb: true
  wandb_project: confidence-distillation
  wandb_entity: null            # use $WANDB_ENTITY or default account
  wandb_group: null             # optional, e.g. "pae" / "plddt" / "multi"
  wandb_tags: []                # set per-config, e.g. ["pae", "teddymer"]
  # `wandb_id` defaults to ${run_name}, which is already required by every
  # confidence config — no new top-level field needed.
```

Each of the four confidence configs adds `- /logging/wandb@logging` to
its `defaults:` list and sets sensible `wandb_tags:` overrides
(`["plddt", "swissprot"]`, `["pae", "teddymer"]`,
`["plddt", "swissprot", "control"]`, `["plddt", "pae", "teddymer", "multi"]`).

### Wire-up in `train_confidence.py`

Two small helpers added to `src/proteinfoundation/confidence/train_confidence.py`:

```python
def _gate_loguru_to_rank0() -> None:
    """Silence loguru on non-rank-0 processes under DDP."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    if local_rank != 0 or node_rank != 0:
        from loguru import logger as _l
        _l.remove()


def _build_wandb_logger(cfg: DictConfig) -> WandbLogger | None:
    """Build a WandbLogger from cfg.logging.* or return None if disabled."""
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

`main()` is changed to:

1. Call `_gate_loguru_to_rank0()` **before** any `logger.info(...)`.
2. Build the WandB logger before `_build_trainer`.
3. Inject the logger via Hydra instantiation override:
   `trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, _convert_="partial")`.
   (`cfg.trainer.logger` is not set in any YAML, so this is a pure addition.)
4. On rank 0 (and only if `wandb_logger is not None`), call
   `wandb_logger.log_hyperparams({"config": OmegaConf.to_container(cfg, resolve=True)})`
   so the resolved Hydra tree lands in the run's config panel.

### Why a Hydra fragment, not direct `_target_` in `cfg.trainer.logger`

A `WandbLogger` defined as `_target_: lightning.pytorch.loggers.WandbLogger`
inside `cfg.trainer.logger` would *also* work, but it has two drawbacks:

- It loses the disabled-via-env-var escape hatch (`WANDB_MODE=disabled`)
  unless we add a Hydra resolver, which is more code than the helper.
- It scatters WandB knobs across every trainer block in every config.

Keeping a single `configs/logging/wandb.yaml` is the lighter-weight,
DRY-er pattern and matches how the trunk `train.py` already structures
its `cfg.log.*` block.

## Data flow

```
Hydra compose
  configs/confidence/<run>.yaml
    defaults:
      - /nn/confidence/<head>@confidence.head
      - /dataset/unified/<dataset>@data
      - /training/confidence_distill@training
      - /logging/wandb@logging          ← NEW
      - _self_

train_confidence.main(cfg)
  _gate_loguru_to_rank0()               ← NEW, runs first
  logger.info(OmegaConf.to_yaml(cfg))   ← now rank-0-only
  head = build_confidence_head_from_cfg(cfg.confidence.head)
  module = ConfidenceDistillationModule(...)
  wandb_logger = _build_wandb_logger(cfg)   ← NEW
  datamodule = hydra.utils.instantiate(cfg.data.datamodule)
  trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, ...)
  if wandb_logger is not None and trainer.is_global_zero:
      wandb_logger.log_hyperparams({"config": OmegaConf.to_container(cfg)})
  trainer.fit(module, datamodule=datamodule)
```

## Testing

All new tests live under `tests/confidence/`. Each is small and focused;
no real W&B network traffic and no real training step.

### `tests/confidence/test_wandb_logger_construction.py`

Builds a minimal `DictConfig` with `logging.log_wandb=True`,
`logging.wandb_project="confidence-distillation"`,
`logging.wandb_tags=["pae", "teddymer"]`, `run_name="pae-distill-teddymer"`.
Patches `wandb.init` to a no-op. Calls `_build_wandb_logger(cfg)` and
asserts:
- returns a `WandbLogger` instance
- `logger._project == "confidence-distillation"`
- `logger._id == "pae-distill-teddymer"`
- `logger._name == "pae-distill-teddymer"`
- the tags list propagated

### `tests/confidence/test_wandb_logger_disabled.py`

Two paths must produce `None`:

1. `logging.log_wandb=False` in the config.
2. `logging.log_wandb=True` but environment has `WANDB_MODE=disabled`
   (verifies the env-var escape hatch, useful in CI and on offline
   compute nodes).

### `tests/confidence/test_train_confidence_passes_logger_to_trainer.py`

Integration-shaped but lightweight. Patches `L.Trainer` to a `MagicMock`
returning a `MagicMock(is_global_zero=True)`. Patches
`build_confidence_head_from_cfg`, the datamodule instantiation, and
`ConfidenceDistillationModule`. Composes a minimal cfg and calls
`main(cfg)`. Asserts the trainer was constructed with `logger=<the
WandbLogger>` (or with `logger=None` when `log_wandb=False`).

### `tests/confidence/test_rank0_logging.py`

Two cases:
1. `LOCAL_RANK="1"` in env ⇒ after `_gate_loguru_to_rank0()`, loguru
   has no active handlers (calling `logger.info("x")` produces no
   output).
2. `LOCAL_RANK="0"` and `NODE_RANK="0"` ⇒ handlers untouched.

Uses `monkeypatch` (pytest) for env, captures stderr or inspects
`loguru._logger.core.handlers` to verify state.

### Smoke test (manual, not CI)

After implementation, on the dev host, dry-run with
`WANDB_MODE=disabled` and confirm:
- only rank 0 prints the composed config (when DDP launches with
  `--devices=2`)
- `trainer.logger is None` when disabled
- run name in WandB UI is `pae-distill-teddymer` when enabled

## Review panel

- `code-review-debug-complexity-expert` (mandatory, per CLAUDE.md)
- `ml-protein-architect` (Hydra config tree + Lightning training-loop
  glue — this is exactly the scope)

No biology / generative / physics reviewer needed; this PR is pure
plumbing.

## Rollout / risk

- **Backwards compatibility.** Configs without
  `- /logging/wandb@logging` in their defaults will get `cfg.logging =
  None` in `_build_wandb_logger`, which returns `None`, which trainer
  accepts. So old configs keep working.
- **Reproducibility hygiene (CLAUDE.md §Reproducibility).** Seed and
  determinism are unchanged. The only new env-coupled state is
  `WANDB_API_KEY`, `WANDB_MODE`, `WANDB_ENTITY`, all standard.
- **Checkpoint compatibility.** No change to `state_dict` shape, no
  Lightning version bump.

## Out-of-scope follow-ups

- Artefact upload of the resolved Hydra config (would mirror trunk
  `train.py`'s `store_n_log_configs`).
- A reusable Lightning callback that mirrors per-head reliability
  diagrams into WandB images (currently they are saved to disk by the
  module; uploading them would be a one-line addition once we know the
  callback hook is wanted).
- Renaming the WandB project to `reward-distillation` if/when the heads
  feed a reward stack in earnest. Out of scope today; project name is
  cheap to change later.
