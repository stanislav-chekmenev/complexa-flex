# PR-5 — Held-out evaluation, sequence-only control, smoke run

Status: PLAN — not yet implemented
Date: 2026-05-16
Spec: `docs/superpowers/specs/2026-05-16-confidence-head-distillation-design.md` §3.10
Predecessors: PR-1 (`f22f38f`), PR-2 (`8250d9a`), PR-3 (`b76218f`), PR-4 (`fa4c5ee`)

## 1. Branch and base

- Branch: `feat/confidence-heldout-eval` cut from `merge_quality_graft` at HEAD `fa4c5ee`.
- Worktree: `.claude/worktrees/pr5-heldout-eval/`.
- Target: PR into `merge_quality_graft`.

## 2. Goal and non-goals

**Goal.** Add the held-out-evaluation scaffolding the scientist flagged in PR-4 round-2: cluster-30 % SwissProt split, adaptive-bucket ECE + reliability diagram, a sequence-only diagnostic head, an early-stop / top-k checkpoint policy on `val/loss_ce`, and a single-GPU smoke command. All additive, no edits to PR-4-merged contracts.

**Non-goals.** No real multi-GPU training. No SLURM sbatch (PR-6). No real metadata regeneration (mmseqs2 is documented as an offline prerequisite — we *consume* the column, not produce it). No RFdiffusion / Dayhoff transfer eval. No new ipTM/ipAE/ipLDDT heads. No changes to `proteina.py`, `local_latents_transformer*.py`, `product_space_flow_matcher.py`, `transforms.py`, `losses.py`, `lightning_module.py` (except a strictly additive validation block — see §3), `train_confidence.py`, `nn/confidence/{base,plddt_head,projections,registry}.py`.

## 3. Files to add / edit (one-line purpose each)

- `src/proteinfoundation/nn/confidence/plddt_sequence_only_head.py` — NEW `SequenceOnlyPLDDTHead(BaseConfidenceHead)`; registered as `plddt_sequence_only`. Ignores `z` and the trunk's pair-update path by construction.
- `src/proteinfoundation/confidence/metrics.py` — EDIT (additive only) `expected_calibration_error_adaptive` and `reliability_diagram`; no signature changes to existing functions.
- `src/proteinfoundation/confidence/lightning_module.py` — EDIT (additive only) extend `validation_step` to log `val/ece_adaptive` and, every `reliability_diagram_every_n_epochs` epochs, emit a `[K, 3]` tensor via the trainer's logger (W&B image / TB histogram), with `.npy` fallback under the trainer's `log_dir` when no logger is configured. New `__init__` kwargs `num_bins_ece_adaptive: int = 15`, `reliability_diagram_every_n_epochs: int = 1`, `reliability_diagram_num_bins: int = 10` — all defaulted so PR-4 callers do not break.
- `src/proteinfoundation/datasets/structure_data.py` — EDIT (minimal, additive) add `cluster_column: str | None = None` and `cluster_seed: int = 42` kwargs and a `_cluster_aware_split` private helper. When `cluster_column` is `None`, behaviour is byte-identical to today. **Justification:** the existing `setup` slices rows 0..n_train; chain-level random would already break MMseqs-30 leakage guarantees, so we genuinely need a cluster-aware path. Read the column from `full_metadata`, group by it, shuffle the *cluster IDs* with `numpy.random.default_rng(cluster_seed)`, and assign whole clusters to train / val until `len(train) / len(full) ≥ train_split`.
- `configs/nn/confidence/plddt_sequence_only_head.yaml` — NEW Hydra cfg for the control head; smaller trunk (`n_blocks=2`, `use_tri_mult=False`, `update_pair_repr_every_n=10**6` ≡ "never update pair"); head registered name `plddt_sequence_only`.
- `configs/confidence/distillation_swissprot_control.yaml` — NEW top-level config composing the sequence-only head + the same data + the same training cfg + the new checkpoint/early-stop callbacks.
- `configs/confidence/distillation_swissprot.yaml` — EDIT add a `callbacks:` block: `ModelCheckpoint(monitor=val/loss_ce, mode=min, save_top_k=3, filename="ce-{epoch}-{val/loss_ce:.4f}")` and `EarlyStopping(monitor=val/loss_ce, mode=min, patience=10)`. Add `data.datamodule.cluster_column: unicluster` (with comment: override to null on environments without the column) and `data.datamodule.cluster_seed: 42`. Add a `confidence.module.{num_bins_ece_adaptive, reliability_diagram_every_n_epochs, reliability_diagram_num_bins}` block consumed by `train_confidence.py` via `module = ConfidenceDistillationModule(..., **cfg.confidence.module)`. (This is purely a kwarg-forwarding change in `train_confidence.py` — but the spec forbids edits there; we therefore default these kwargs in the module and *omit* the cfg-side knob from PR-5 if the policy is strictly read. Resolution in §10 Q1.)
- `scripts/run_smoke_train.sh` — NEW documented one-liner.
- Tests (see §6).

## 4. Class signatures (sketch)

```python
# src/proteinfoundation/nn/confidence/plddt_sequence_only_head.py
@register_confidence_head("plddt_sequence_only")
class SequenceOnlyPLDDTHead(BaseConfidenceHead):
    output_keys = ("plddt_logits",)
    def __init__(self, trunk: ConfidenceTrunk, token_dim=768, pair_repr_dim=256,
                 num_plddt_bins=50, bin_min=0.0, bin_max=100.0): ...
    # bin_centers buffer + logits_to_expected_value identical to PLDDTHead.
    def _predict(self, s, z, mask) -> {"plddt_logits": Tensor[b, n, num_plddt_bins]}:
        del z  # explicitly ignored — diagnostic intent
        return {"plddt_logits": self.linear(self.norm(s)) * mask[..., None]}
```

`forward(s, z, mask, cond)` is inherited; the head still calls `self.trunk(s, z, mask, cond)` (the trunk is configured with `update_pair_repr_every_n` set so high it never fires, but `MultiheadAttnAndTransition` *does* still consume `z` for pair-bias). To truly isolate sequence signal we also pass `z = torch.zeros_like(z)` inside `forward` before delegating to the trunk. Override `forward` to zero `z`; document the override.

```python
# additions to metrics.py
def expected_calibration_error_adaptive(
    logits: Tensor, labels_bin: Tensor, mask: Tensor, num_bins_ece: int = 15
) -> Tensor:  # 0-dim fp32 in [0, 1]
    """Equal-mass top-1-prob buckets via quantiles; ties resolved by stable sort."""

def reliability_diagram(
    logits: Tensor, labels_bin: Tensor, mask: Tensor, num_bins_ece: int = 10
) -> Tensor:  # shape (num_bins_ece, 3); columns (conf_b, acc_b, n_b)
    """Equal-width buckets; rows correspond to buckets, empty buckets = (0, 0, 0)."""
```

## 5. Hydra configs

`configs/nn/confidence/plddt_sequence_only_head.yaml`:

```yaml
defaults:
  - base
  - _self_
name: plddt_sequence_only
_target_: proteinfoundation.nn.confidence.plddt_sequence_only_head.SequenceOnlyPLDDTHead
token_dim: 768
pair_repr_dim: 256
num_plddt_bins: 50
bin_min: 0.0
bin_max: 100.0
trunk:
  n_blocks: 2
  use_tri_mult: False
  use_tri_attn: False
  update_pair_repr_every_n: 1000000
```

(`base.yaml` provides `trunk._target_`, `token_dim`, `pair_repr_dim`, `n_heads`, `dim_cond`, `use_qkln`, `dropout`; overridden fields above replace those keys.)

`configs/confidence/distillation_swissprot_control.yaml`:

```yaml
defaults:
  - distillation_swissprot
  - override /nn/confidence/plddt_sequence_only_head@confidence.head
  - _self_
run_name: "plddt-distill-swissprot-control"
```

Edits to `distillation_swissprot.yaml`:

```yaml
trainer:
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: "val/loss_ce"
      mode: "min"
      save_top_k: 3
      filename: "ce-{epoch:03d}-{val/loss_ce:.4f}"
      auto_insert_metric_name: false
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: "val/loss_ce"
      mode: "min"
      patience: 10
data:
  datamodule:
    cluster_column: "unicluster"
    cluster_seed: 42
```

`train_confidence.py` currently constructs `Trainer(**OmegaConf.to_container(cfg.trainer))`, so `callbacks` are honored without any code change — `hydra.utils.instantiate` runs lazily on the `_target_` list because `to_container` keeps the dicts intact… **except it does not**. Verify in the smoke test: if `Trainer` chokes on `_target_` dicts, the cheapest fix is to call `hydra.utils.instantiate(cfg.trainer)` instead of `OmegaConf.to_container` in `train_confidence.py` — but that file is on the no-edit list. Resolution: instantiate callbacks **inside the cfg** by composing them via Hydra's `+trainer.callbacks=...` mechanism is the same issue. **Open question 1 (§10).**

`scripts/run_smoke_train.sh`:

```bash
#!/usr/bin/env bash
# Single-GPU smoke run for PR-5 verification.
set -euo pipefail
CKPT_DIR="${CKPT_DIR:-ckpts}" uv run python -m proteinfoundation.confidence.train_confidence \
  --config-name=confidence/distillation_swissprot \
  trainer.max_steps=100 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=2 \
  trainer.devices=1 \
  trainer.strategy=auto
```

## 6. Tests (TDD, written first)

- `tests/unit/confidence/test_sequence_only_head.py`
  - Registered name `plddt_sequence_only` returns `SequenceOnlyPLDDTHead`.
  - Output shape `(b, n, num_plddt_bins)` for `(b=2, n=37)`.
  - `forward(s, z1, mask, cond) == forward(s, z2, mask, cond)` for any two random `z1, z2` with same shape (within fp32 tolerance) — proves z-independence.
  - Padded positions zeroed.
- `tests/unit/confidence/test_ece_adaptive.py`
  - Perfect calibration (logits one-hot on correct bin): ECE ≈ 0.
  - Worst case (max-confidence on always-wrong bin): ECE > 0.5.
  - Output in `[0, 1]` and 0-dim fp32.
  - Tie-handling: 1000 residues with identical top-1 prob still split across 15 buckets without crash.
  - All-masked input: returns 0 (no NaN).
- `tests/unit/confidence/test_reliability_diagram.py`
  - Shape `(num_bins_ece, 3)`.
  - `sum(col[:, 2]) == mask.sum()`.
  - Uniform logits: `conf_b ≈ 1/num_bins` and `acc_b ≈ 1/num_bins` per non-empty bucket within sampling tolerance.
  - Empty bucket rows are `(0, 0, 0)`.
- `tests/unit/datasets/test_cluster_split.py`
  - Synthetic 100-row dataframe with `unicluster` column of 10 clusters of 10 rows.
  - With `train_split=0.9`, validation contains *whole clusters*, no shared cluster ID across train/val.
  - With `cluster_seed=0` vs `cluster_seed=1`, splits differ.
  - With `cluster_column=None`, behaviour matches PR-2 baseline (assert exact row order preserved).
- `tests/integration/confidence/test_sequence_only_distillation.py`
  - Build `SequenceOnlyPLDDTHead(...)` via `from_components` with a `FakeProteina` (reuse the PR-4 test fixture).
  - One `training_step` runs, loss finite.
  - Only head params receive `grad`; `proteina.parameters()` all `grad is None`.
  - `_predict` does not crash for `n_orig != n_extended`.
- `tests/integration/confidence/test_validation_logging_extension.py`
  - Run one `validation_step` and assert `val/ece_adaptive` is in the logged keys.
  - Assert no PR-4 keys are missing (`val/loss`, `val/loss_ce`, `val/loss_smooth_l1`, `val/loss_total`, `val/ece`, `val/plddt_accuracy`, `val/plddt_mae`, `val/pearson_r`, `val/spearman_r`, `val/mae_*`).
  - When the trainer has no logger, reliability-diagram fallback writes `<log_dir>/reliability_epoch_<E>.npy`.

## 7. Verification commands

Automated:

```bash
uv run pytest tests/ -v
# Pre-PR-5 count remains green + new tests pass.
```

Manual (single-GPU smoke; not a CI step):

```bash
CKPT_DIR=ckpts bash scripts/run_smoke_train.sh
```

Exit criterion: 100 steps complete, `val/loss_ce` is finite, `val/ece_adaptive` is logged, a reliability-diagram artifact appears under the trainer log dir.

Abort criterion: if the cluster split leaves `len(val) == 0` (insufficient clusters in the parquet) — fall back to chain-level random with the seed and open a follow-up ticket.

## 8. Reviewer panel

- `code-review-debug-complexity-expert` (mandatory).
- `ml-protein-architect` (config tree + datamodule edits).
- `ml-software-pytorch-jax-expert` (logger-fallback + callback wiring).
- `generative-protein-scientist` (sequence-only diagnostic semantics; ECE-adaptive bin policy).

## 9. Risk register

| # | Risk | Sev | Lik | Mitigation |
|---|---|---|---|---|
| 1 | `unicluster` column missing from the production parquet. | Med | Med | `cluster_column=None` fallback preserves PR-2 behaviour; smoke test asserts split sanity at setup time and logs a `loguru.warning` with a follow-up TODO. |
| 2 | Additive `validation_step` extension drifts a PR-4 contract (a metric key disappears). | High | Low | `test_validation_logging_extension.py` enumerates every PR-4 key and fails if any are missing. |
| 3 | Reliability-diagram logging API differs (W&B vs TB vs no-logger). | Low | Med | Branch on `isinstance(self.logger, (WandbLogger, TensorBoardLogger))`; everything else writes `.npy`. |
| 4 | `SequenceOnlyPLDDTHead` name collision in the registry. | Low | Low | Unique name `plddt_sequence_only`; existing `plddt` not touched. |
| 5 | Single-GPU smoke can't load `ckpts/complexa.ckpt` (large, not in CI). | Med | High | Documented command only — not run in pytest. CI uses the integration test with `FakeProteina`. |

## 10. Out of scope (PR-6 and follow-ups)

- SLURM sbatch + multi-GPU bring-up — PR-6.
- Actual cluster-30 % training run with metrics on real data — manual smoke is the verification artifact.
- RFdiffusion / Dayhoff-BackboneRef refold + AF2 + held-out eval — deferred (spec Q6).
- ipTM / ipAE / ipLDDT heads — separate work.

## Open questions (must resolve before implementation starts)

1. **`train_confidence.py` is on the no-edit list, but the new `callbacks:` block under `trainer:` will reach `L.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))` as raw dicts containing `_target_`, which `L.Trainer` does not auto-instantiate.** Three options: (a) carve out a one-line exemption — replace `OmegaConf.to_container` with `hydra.utils.instantiate(cfg.trainer, _convert_="partial")` (smallest, cleanest); (b) introduce a tiny helper `proteinfoundation/confidence/_trainer_utils.py` and have `train_confidence.py` not change (impossible without editing the entry point); (c) instantiate callbacks via a Python-side helper hooked into a new module-level entry point. The spec's no-edit clause for `train_confidence.py` is in tension with the spec's callback requirement; the cheapest faithful resolution is **option (a)** and treat it as a one-line additive edit. Hand to `ml-protein-architect` to confirm before implementation.
2. **`reliability_diagram_every_n_epochs` policy** — emit on validation epoch end or only on `trainer.is_last_batch` of validation? Hand to `generative-protein-scientist` (cheap question, default to "last batch only" to avoid mid-epoch noise).
3. **`unicluster` column name** — confirm whether it is `unicluster`, `mmseqs30`, `uniref30`, or something else in the SwissProt parquet at `${DATA_PATH}/afdb_cifs/metadata.parquet`. Cheapest experiment: `python -c "import pandas as pd; print(pd.read_parquet(...).columns.tolist())"`. Hand to user.

Hand to `ml-protein-architect` for implementation once Q1 is resolved.
