# PR-4 — Confidence-head distillation Lightning sidecar

Status: planning
Date: 2026-05-16
Spec: `docs/superpowers/specs/2026-05-16-confidence-head-distillation-design.md`
Predecessors: PR-1 (`f22f38f`), PR-2 (`8250d9a`), PR-3 (`b76218f`) — all on `merge_quality_graft`.

## 1. Branch and base

- Branch: `feat/confidence-distillation-lightning`
- Base: `merge_quality_graft` at HEAD `b76218f`
- Worktree: `.claude/worktrees/pr4-distill-lightning/` (created via `git worktree add`)
- Merge target: `merge_quality_graft`

## 2. Goal and non-goals

### Goal
Add a self-contained Lightning sidecar that distils AF2 pLDDT (per-residue, AF2 `[0,100]` scale) from the frozen `Proteina` trunk into the trainable `PLDDTHead` shipped by PR-3. Deliver: loss module, validation metrics, Lightning module, Hydra entry point, two new configs, unit + integration tests written first.

### Non-goals
- No edits to `src/proteinfoundation/proteina.py` (Option A sidecar pattern, spec §0 Q1).
- No edits to `src/proteinfoundation/train.py`.
- No held-out eval set, no smoke run on real ckpts, no SLURM sbatch — deferred to PR-5 / PR-6.
- No KL distillation (no teacher logits available; spec §3.8).
- No `logits_to_logprob_above(threshold)` (deferred to PR-5).
- No state_dict filter on the frozen trunk; we accept the disk cost (Risk 9 below).
- No joint training / no trunk unfreezing.

## 3. Files to add (one-line purpose)

- `src/proteinfoundation/confidence/lightning_module.py` — `ConfidenceDistillationModule` sidecar: owns frozen `Proteina` + trainable `BaseConfidenceHead`, full train/val step, optimizer + LR schedule, defensive `eval()` re-assertion hooks.
- `src/proteinfoundation/confidence/losses.py` (**extend**) — append `masked_plddt_cross_entropy`, `masked_smooth_l1_on_expected_value`, `combined_plddt_loss`. Keep `plddt_to_bin` untouched.
- `src/proteinfoundation/confidence/metrics.py` — `plddt_accuracy`, `plddt_mae`, `pearson_r`, `spearman_r`, `_logits_to_continuous`, `_labels_to_continuous`, `plddt_mae_stratified`. AF2-scale-aware port.
- `src/proteinfoundation/confidence/train_confidence.py` — Hydra entry point `@hydra.main(config_path=".../configs", config_name="confidence/distillation_swissprot")`. Composes head + dataset + training cfg, instantiates `ConfidenceDistillationModule` and `StructureDataModule`, runs `L.Trainer.fit(...)`.
- `src/proteinfoundation/confidence/__init__.py` (**extend**) — re-export `ConfidenceDistillationModule`, `combined_plddt_loss`. Keep prior exports.
- `configs/training/confidence_distill.yaml` — optimizer/scheduler/training cfg (per spec §6).
- `configs/confidence/distillation_swissprot.yaml` — top-level training cfg composing head + dataset + training.

Tests:
- `tests/unit/confidence/test_losses.py`
- `tests/unit/confidence/test_metrics.py`
- `tests/unit/confidence/test_cond_construction.py`
- `tests/integration/confidence/test_distillation_one_step.py`
- `tests/integration/confidence/test_distillation_checkpoint_roundtrip.py`

## 4. Class and function signatures (sketch only)

### `confidence/lightning_module.py`

```python
class ConfidenceDistillationModule(L.LightningModule):
    """Sidecar trainer for the pLDDT confidence head.

    Owns a frozen `Proteina` (loaded once at __init__ from trunk_ckpt_path +
    autoencoder_ckpt_path) and a trainable BaseConfidenceHead. Forward:
        1. Build cond from t = trunk_eval_t via the trunk's existing
           cond_factory (FeatureFactory).
        2. Under torch.no_grad(), run the trunk to obtain trunk_intermediates
           = {s, z, mask, orig_mask, n_orig}.
        3. Run head.forward(s, z, mask, cond) -> {"plddt_logits": [b, n_ext, 50]}.
        4. Slice logits / labels to orig_mask AND-combined with batch["plddt_mask"].
        5. combined_plddt_loss with ce_weight / smooth_l1_weight.

    Note: the head's ConfidenceTrunk mandates a non-None cond (see
    BaseConfidenceHead.forward); the dropped expects_external_cond flag does
    not relax that requirement.
    """

    def __init__(
        self,
        head: BaseConfidenceHead,
        trunk_ckpt_path: str,
        autoencoder_ckpt_path: str,
        trunk_eval_t: float = 0.99,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        warmup_steps: int = 500,
        min_lr: float = 5e-6,
        ce_weight: float = 0.7,
        smooth_l1_weight: float = 0.3,
        label_smoothing: float = 0.0,
    ) -> None: ...

    def _load_frozen_trunk(
        self, trunk_ckpt_path: str, autoencoder_ckpt_path: str
    ) -> "Proteina":
        """Proteina.load_from_checkpoint(..., autoencoder_ckpt_path=...);
        self.proteina.nn.expose_intermediates = True;
        self.proteina.requires_grad_(False); self.proteina.eval()."""

    def _compute_cond(
        self, batch: dict, n_residues: int
    ) -> torch.Tensor:
        """Stamp batch['t']['<modality>'] = trunk_eval_t for every modality
        the trunk's cond_factory consumes; call self.proteina.nn.cond_factory(
        batch) -> [b, n, dim_cond]. Cached if batch shape repeats."""

    def _forward(self, batch: dict) -> dict:
        """Returns {'logits': [b, n_orig, 50], 'mask_eff': [b, n_orig],
        'labels_bin': [b, n_orig], 'labels_cont': [b, n_orig]}."""

    def training_step(self, batch, batch_idx): ...   # logs train/loss, train/loss_ce, train/loss_smooth_l1
    def validation_step(self, batch, batch_idx): ... # logs val/loss (CE-only), val/plddt_accuracy, val/plddt_mae, val/pearson_r, val/spearman_r, val/mae_lt50, val/mae_50_70, val/mae_70_90, val/mae_ge90
    def configure_optimizers(self): ...              # AdamW on [p for p in self.head.parameters() if p.requires_grad] + LambdaLR warmup + linear decay to min_lr/lr

    # defensive trunk eval() — port from quality_graft/training/lightning_module.py:164-197
    def on_train_epoch_start(self) -> None: ...
    def on_train_batch_start(self, batch, batch_idx) -> None: ...
    def on_validation_epoch_start(self) -> None: ...
```

### `confidence/losses.py` (additions only — `plddt_to_bin` stays)

```python
def masked_plddt_cross_entropy(
    student_logits: Tensor,        # [b, n, num_bins]
    plddt_bin_labels: Tensor,      # [b, n] int64
    mask: Tensor,                  # [b, n] bool or float
    label_smoothing: float = 0.0,
) -> Tensor:
    """Per-residue CE, reduced sum(loss*mask) / mask.sum().clamp_min(1).
    Uses F.cross_entropy(reduction='none', label_smoothing=label_smoothing)
    flattened over the residue axis."""

def masked_smooth_l1_on_expected_value(
    student_logits: Tensor,        # [b, n, num_bins], any dtype
    plddt_continuous: Tensor,      # [b, n] float in [0, bin_max]
    mask: Tensor,                  # [b, n]
    bin_centers: Tensor,           # [num_bins] in fp32, on logits' device
) -> Tensor:
    """ev = softmax(logits.float()) @ bin_centers; smooth_l1(ev, target,
    reduction='none') reduced as sum(.*mask)/mask.sum().clamp_min(1)."""

def combined_plddt_loss(
    student_logits: Tensor,
    plddt_bin_labels: Tensor,
    plddt_continuous: Tensor,
    mask: Tensor,
    bin_centers: Tensor,
    ce_weight: float = 0.7,
    smooth_l1_weight: float = 0.3,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Returns (total, {'loss_ce': l_ce, 'loss_smooth_l1': l_sl1})."""
```

NaN policy: empty-mask returns a zero scalar with grad (caller doesn't divide by zero, no NaN propagates).

### `confidence/metrics.py` (port from `quality_graft/training/metrics.py`, AF2-scale-aware)

```python
def _logits_to_continuous(logits: Tensor, bin_centers: Tensor) -> Tensor: ...   # accepts explicit centers; signature differs from QG (which hard-codes [0,1])
def _labels_to_continuous(labels: Tensor, bin_centers: Tensor) -> Tensor: ...
def plddt_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor: ...
def plddt_mae(
    logits: Tensor, labels: Tensor, mask: Tensor,
    bin_centers: Tensor,  # carries the [0, 100] scale
) -> Tensor: ...
def pearson_r(pred_cont: Tensor, target_cont: Tensor, mask: Tensor) -> Tensor: ...
def spearman_r(pred_cont: Tensor, target_cont: Tensor, mask: Tensor) -> Tensor: ...
def plddt_mae_stratified(
    logits: Tensor, labels: Tensor, mask: Tensor, bin_centers: Tensor,
    buckets: tuple[tuple[float, float], ...] = ((0,50),(50,70),(70,90),(90,100)),
) -> dict[str, Tensor]:
    """Returns {'mae_lt50': ..., 'mae_50_70': ..., 'mae_70_90': ..., 'mae_ge90': ...}.
    Each MAE is over residues whose *target continuous* pLDDT falls in [lo, hi)."""
```

The sidecar reads `bin_centers` off `self.head.bin_centers` (registered buffer on `PLDDTHead`). This keeps metrics scale-agnostic; AF2 `[0, 100]` is encoded in the head buffer only.

## 5. Hydra configs

### `configs/training/confidence_distill.yaml`

```yaml
training_mode: "pure_distillation"
loss_weight: 1.0
trunk_eval_t: 0.99
freeze_trunk: True
trunk_ckpt_path: ${oc.env:CKPT_DIR,ckpts}/complexa.ckpt
autoencoder_ckpt_path: ${oc.env:CKPT_DIR,ckpts}/complexa_ae.ckpt
opt:
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  warmup_steps: 500
  min_lr: 5.0e-6
loss:
  ce_weight: 0.7
  smooth_l1_weight: 0.3
  label_smoothing: 0.0
max_epochs: 50
precision: bf16-mixed
gradient_clip_val: 1.0
```

### `configs/confidence/distillation_swissprot.yaml`

```yaml
defaults:
  - /nn/confidence/plddt_head@confidence.head
  - /dataset/unified/afdb_monomers_with_plddt@data
  - /training/confidence_distill@training
  - _self_

confidence:
  enabled: True

trainer:
  accelerator: gpu
  devices: 1
  precision: ${training.precision}
  max_epochs: ${training.max_epochs}
  gradient_clip_val: ${training.gradient_clip_val}
  log_every_n_steps: 25
  check_val_every_n_epoch: 1

seed: 42
run_name: "plddt-distill-swissprot"
```

The Hydra composition exactly mirrors how `configs/search_binder_*.yaml` compose sibling groups — no new patterns introduced.

## 6. Tests (TDD — written first)

### `tests/unit/confidence/test_losses.py`
- `test_masked_ce_matches_manual_reduction` — hand-build `b=2, n=4, num_bins=5`; CE reduces by `sum(loss*mask) / mask.sum()`, NOT `mean`. Compare to manual.
- `test_masked_ce_empty_mask_finite` — all-zero mask returns scalar `0.0`, not NaN.
- `test_masked_ce_label_smoothing_knob` — `label_smoothing=0.1` differs from `0.0` on the same inputs.
- `test_smooth_l1_on_expected_value_finite` — `b=2, n=8` random logits + `[0,100]` targets; finite, fp32 internally even under bf16 inputs.
- `test_combined_loss_decomposition` — `combined_plddt_loss` equals `0.7 * masked_ce + 0.3 * smooth_l1_ev` on the same inputs.

### `tests/unit/confidence/test_metrics.py`
- `test_plddt_accuracy_hand_computed` — `b=1, n=4` predicted argmax vs labels, mask zeroes one position; assert exact fraction.
- `test_plddt_mae_in_plddt_units` — `bin_centers = [1, 3, 5, ..., 99]`; perfect prediction → MAE 0; one-bin shift → MAE 2.
- `test_pearson_spearman_perfect` — monotone preds vs targets → Pearson and Spearman both 1.
- `test_stratified_mae_bucket_keys` — keys are exactly `mae_lt50, mae_50_70, mae_70_90, mae_ge90`; empty bucket returns 0 with finite value, not NaN.

### `tests/unit/confidence/test_cond_construction.py`
- `test_compute_cond_shape_and_finite` — with a fake `cond_factory` (a small `nn.Linear`-based stub keyed on `t`), `_compute_cond(b=2, n=11, ...)` returns `[2, 11, dim_cond]`, finite.
- `test_compute_cond_deterministic` — two calls with the same batch yield identical tensors (no hidden randomness).

### `tests/integration/confidence/test_distillation_one_step.py`
Builds a *fake* `Proteina` stub (a `nn.Module` exposing `.nn` with `.cond_factory`, `expose_intermediates`, and a forward returning `nn_out["trunk_intermediates"] = {s, z, mask, orig_mask, n_orig}` with random tensors). The real `proteina.ckpt` is GPU-only and is exercised in PR-5 smoke.
- `test_one_step_loss_finite` — synthetic batch with `plddt_bin`, `plddt_residue`, `plddt_mask`, `mask`; one `training_step` returns a finite tensor.
- `test_only_head_params_get_grad` — after `loss.backward()`, every `p` in the fake trunk has `p.grad is None`; every `p` in `self.head` with `requires_grad=True` has finite `p.grad`.
- `test_trunk_frozen_grad_state` — every fake-trunk param has `requires_grad=False`.

### `tests/integration/confidence/test_distillation_checkpoint_roundtrip.py`
- `test_checkpoint_round_trip` — `L.Trainer(max_steps=1, ...)` fit on the fake stub, save ckpt, reload via `ConfidenceDistillationModule.load_from_checkpoint`, assert head state dict matches bit-exactly on Lightning 2.5.x.

Skipped: real `Proteina.load_from_checkpoint` calls (GPU + on-disk ckpts; PR-5 smoke).

## 7. Verification commands

```
uv run pytest tests/unit/confidence/test_losses.py \
              tests/unit/confidence/test_metrics.py \
              tests/unit/confidence/test_cond_construction.py \
              tests/integration/confidence/test_distillation_one_step.py \
              tests/integration/confidence/test_distillation_checkpoint_roundtrip.py -v
uv run pytest tests/ -v   # 53 pre-PR-4 tests stay green; ~15-20 new cases added
```

Exit criterion: all targeted new tests green, all pre-existing 53 tests green.

## 8. Reviewer panel

- **`code-review-debug-complexity-expert`** (mandatory) — overall code quality, dead-code, readability, no speculative abstractions.
- **`ml-protein-architect`** — Hydra composition matches existing `search_binder_*` patterns; `proteina.py` is untouched; config keys align with PR-3 head config; `cond` dim matches trunk `dim_cond=256`.
- **`ml-software-pytorch-jax-expert`** — DDP `find_unused_parameters=False` correct with frozen trunk; `torch.no_grad()` placement; bf16 vs fp32 boundaries for the expected-value computation; Lightning 2.5.x checkpoint round-trip.
- **`generative-protein-scientist`** — loss weighting (`0.7/0.3`), `trunk_eval_t=0.99` justification, validation-metric choice, stratified MAE buckets.

## 9. Risk register

| # | Risk | Sev | Lik | Mitigation |
|---|---|---|---|---|
| 1 | DDP `find_unused_parameters` mismatch with frozen trunk | Med | Med | Optimizer only sees `head.parameters()` with `requires_grad=True`; sidecar sets `Trainer(strategy=DDPStrategy(find_unused_parameters=False))` explicitly. ml-software-pytorch-jax-expert reviews. |
| 2 | Lightning 2.5.x checkpoint round-trip with frozen trunk state inside the module | Med | Low | `test_distillation_checkpoint_roundtrip.py` covers it. Do not bump Lightning. |
| 3 | bf16 instability in triangle ops on long sequences | Med | Med | Confined to head (4 blocks); EV reduction forced fp32 inside `masked_smooth_l1_on_expected_value`; full-length numerical sanity deferred to PR-5 smoke. |
| 4 | `cond` dim mismatch — trunk's FeatureFactory output dim must equal head trunk `dim_cond=256` | High | Low | `_compute_cond` asserts `cond.shape[-1] == self.head.trunk.dim_cond` on first batch; raises with explicit message. |
| 5 | Frozen trunk re-enters `train()` mode at epoch / batch boundary | Med | High | Port defensive `eval()` re-assertion in `on_train_epoch_start` + `on_train_batch_start` from `quality_graft/training/lightning_module.py:164-197`. |
| 6 | Loss reduction inadvertently uses `mean` over the bin dim (silent mis-scaling) | Med | Low | `test_masked_ce_matches_manual_reduction` pins the reduction to `sum / mask.sum().clamp_min(1)`. |
| 7 | `plddt_mask` AND-combine with `orig_mask` omitted — head learns on invalid residues | High | Low | `test_one_step_loss_finite` constructs a batch with `plddt_mask` zeroing half the residues and inspects the effective mask. |
| 8 | The `expects_external_cond` flag (dropped in PR-3 round 2) tempts a future caller to omit `cond` — `BaseConfidenceHead.forward` now silently requires it | Low | Med | Sidecar docstring explicitly states the contract and `_compute_cond`'s role. |
| 9 | Ckpt disk cost — Lightning saves the entire frozen trunk too | Low | High | Accept for PR-4; document in the sidecar docstring; PR-5 (or follow-up) may add a `state_dict` filter. |
| 10 | Hydra `defaults`-list composition typo silently picks up wrong dataset | Low | Med | Top-level config explicitly composes `dataset/unified/afdb_monomers_with_plddt` (the PR-2 dataset); reviewers confirm. |

## 10. Out of scope (PR-5 / PR-6 / follow-up)

- Held-out SwissProt eval set + smoke run + held-out metrics → **PR-5**.
- SLURM sbatch + multi-GPU bring-up + 1-epoch full data → **PR-6**.
- `logits_to_logprob_above(threshold)` exposure (SMC reward consumer) → PR-5 or follow-up.
- Sequence-only control diagnostic head → PR-5.
- `bin_min` / `bin_max` `state_dict` round-trip guard → follow-up.
- ECE / reliability diagram → PR-5.
- ipTM / ipAE / ipLDDT subclasses → separate research thread (spec §2).

## 11. Handoff

Hand to **`ml-protein-architect`** for implementation, with tests written first per the CLAUDE.md TDD discipline. Implementation may begin once this plan is approved by the user.
