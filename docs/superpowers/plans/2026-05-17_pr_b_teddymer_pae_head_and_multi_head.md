# PR-B — Teddymer PaeHead + MultiHeadConfidence — implementation plan

**Date:** 2026-05-17
**Owner of execution:** ml-protein-architect (green phase); code-review-debug-complexity-expert (red phase)
**Branch base:** `teddymer_pr_b_pae_head_and_multi_head` (off PR-A's tip `teddymer_pr_a_data_plumbing`, itself off `prepare_teddy`)
**Target merge branch:** `prepare_teddy` (downstream PR into `dev` after PR-A merges)
**Predecessor plan:** [docs/superpowers/plans/2026-05-17_pr_a_teddymer_data_plumbing.md](2026-05-17_pr_a_teddymer_data_plumbing.md)
**Scope discipline:** PR-B only. Re-running pLDDT distillation on Teddymer is OUT of scope (PR-A already provides that via the existing pLDDT head); PR-B's reference run is **PAE-alone single-head** on Teddymer.

## 1. Goal & non-goals

### Goal
Three coupled deliveries that unblock the first asymmetric-PAE training run on Teddymer:
1. Refactor `ConfidenceTrunk` so `(s, z)` reaches heads un-symmetrised (CLAUDE.md option (b)); every concrete symmetric head opts in by symmetrising inside its own `_predict`.
2. Add an asymmetric `PaeHead` (64-bin classification over `[0, 31.75]` Å, `(B, L, L, 64)` logits) plus its loss + metrics.
3. Add a `MultiHeadConfidence` wrapper that runs the trunk once and dispatches `(s, z, mask, cond)` to each registered child head; pair with a `MultiHeadLoss` aggregator that has two distinct weighting layers (within-head CE/EV and across-head `weight`).

PR-B ships Hydra configs for both the single-head PAE reference run and the multi-head joint pLDDT+PAE run, plus a SLURM wrapper. First training run (decided in PR-A locked-decisions) is single-head PAE.

### Non-goals
- No data-plumbing changes: PR-A delivered `pae_residue_pair`, `pae_bin`, `pae_mask`, `chain_idx`; PR-B consumes them as-is.
- No re-training of pLDDT on Teddymer in PR-B (that is a downstream experiment using existing artefacts).
- No new metric classes beyond pair-level analogues of the pLDDT ones plus the distance-stratified MAE.
- No changes to `proteinfoundation.proteina` or to `train.py`.
- No view move to `/mnt/labs/shared/...`.
- No `MultiHeadConfidence` support for heads whose `expected_trunk_eval_t` differs (fail at construction; future work).

## 2. Locked decisions (carried; do not re-open)

| # | Decision | Effect on plan |
| --- | --- | --- |
| 1 | Trunk symmetrisation refactor = option (b) (remove from trunk; per-head symmetrises) | One-line deletion at [src/proteinfoundation/nn/confidence/base.py:129](../../../src/proteinfoundation/nn/confidence/base.py#L129). `PLDDTHead._predict` consumes `s` only → no-op. `SequenceOnlyPLDDTHead` ditto. |
| 2 | PAE bin edges = AF2 standard: 64 bins of width 0.5 Å on `[0, 31.75]` | `PaeHead.num_bins = 64`, `bin_min = 0.0`, `bin_max = 32.0`. Reuses `pae_to_bin` already in `losses.py`. |
| 3 | Loss recipe per head = `0.9 * masked_CE + 0.1 * SmoothL1(EV)` (matches pLDDT recipe family) | `combined_pae_loss` mirrors `combined_plddt_loss`; same defaults. |
| 4 | First PAE run = single-head config | Both configs ship in PR-B; sbatch defaults to single-head. |
| 5 | Multi-head weighting = two separate Hydra layers (within-head `loss.ce_weight/ev_weight`, across-head `weight`) | Encoded in `distillation_teddymer_multihead.yaml`; `MultiHeadLoss` reads `weight` from a separate parallel dict, not from inside each head's loss block. |
| 6 | `weight: 0.0` is a hard short-circuit | `MultiHeadLoss` skips the head's loss call entirely when `weight == 0.0` (saves compute and guarantees zero gradient). |
| 7 | `expected_trunk_eval_t` mismatch → fail at `__init__` | New attribute on `BaseConfidenceHead`; the wrapper asserts equality across children at construction with a message naming the offending head. |
| 8 | Inter-chain block is NOT a stored field anymore (PR-A dropped it) | `PaeHead` does not need it; the head consumes the full directional `z`. Distance-stratified MAE on validation reconstructs the inter-chain mask from `chain_idx` if needed. |

## 3. Context & constraints

- **Trunk surface (current, after PR-A).** [src/proteinfoundation/nn/confidence/base.py](../../../src/proteinfoundation/nn/confidence/base.py): `ConfidenceTrunk.forward(s, z, mask, cond) -> (s, z)` ends with `z = (z + z.transpose(-3, -2)) / 2.0` at line 129. `BaseConfidenceHead.forward` calls `self.trunk(...)` then `self._predict(s, z, mask)` (line 191-192).
- **Only callers of `ConfidenceTrunk.forward`.** `grep -rn "ConfidenceTrunk\|self.trunk("` confirms two callers: `BaseConfidenceHead.forward` (line 191) and `SequenceOnlyPLDDTHead.forward` (line 63). No code outside the head package consumes the trunk directly. Refactor is local.
- **Concrete head consumers of `z`.** None today. `PLDDTHead._predict` deletes `z` (line 58); `SequenceOnlyPLDDTHead._predict` deletes `z` (line 72). Symmetrisation removal is invisible to both.
- **`expected_trunk_eval_t` is a new attribute.** `grep -rn "expected_trunk_eval_t"` returns empty — PR-B introduces it. Default = `0.99` (CLAUDE.md pin), declared on `BaseConfidenceHead`. Existing heads inherit the default unchanged.
- **Registry surface.** `@register_confidence_head(name)` at [src/proteinfoundation/nn/confidence/registry.py:26](../../../src/proteinfoundation/nn/confidence/registry.py#L26). New head names `"pae"` and `"multi_head"`. `build_confidence_head_from_cfg` (line 39) already handles `_target_` dispatch; multi-head children resolve via the `_target_` path, NOT via `name` (cleaner — child cfgs already have `_target_` from `plddt_head.yaml`).
- **Loss/metrics surface.** [src/proteinfoundation/confidence/losses.py](../../../src/proteinfoundation/confidence/losses.py) already has `pae_to_bin` (PR-A), `_mask_reduce`, `masked_plddt_cross_entropy`, `masked_smooth_l1_on_expected_value`, `combined_plddt_loss`. `pae_to_bin` is generic enough to reuse; PR-B adds masked pair-level analogues.
- **Sidecar lightning module.** [src/proteinfoundation/confidence/lightning_module.py](../../../src/proteinfoundation/confidence/lightning_module.py): `_forward` hard-codes `plddt_*` field access. PR-B must abstract the head's loss + metric path so the same module serves both single-head and multi-head training without forking the module. Approach: introduce a small head-side `compute_loss_and_metrics(out, batch, mask_eff)` protocol so the Lightning module stays head-agnostic. The pLDDT head implements it via the existing `combined_plddt_loss`; the PAE head implements it via `combined_pae_loss`; the multi-head wrapper implements it via `MultiHeadLoss`.
- **Pinned env unchanged.** uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5,<2.6.
- **CLAUDE.md TDD discipline.** Tests first; tests must fail because the artefact is missing, not because of a setup error.

## 4. Stakeholders / handoffs

- **Sidecar Lightning module** consumes the head's `compute_loss_and_metrics` API. The protocol must be flat enough that the existing pLDDT path is a one-method-call refactor, not a rewrite.
- **PR-A consumer contract holds.** Dataset emits `plddt_residue [L]`, `plddt_bin [L]`, `plddt_mask [L]`, `pae_residue_pair [L, L]`, `pae_bin [L, L]`, `pae_mask [L, L]`, `chain_idx [L]`. PR-B reads these.
- **Downstream evaluation/paper figures.** No PR-B handoff — once PR-B trains, the resulting `*.ckpt` is a sidecar checkpoint inspectable via the existing analysis flow.

## 5. Data and module contracts

### `BaseConfidenceHead` (touched at [src/proteinfoundation/nn/confidence/base.py:135](../../../src/proteinfoundation/nn/confidence/base.py#L135))

```
class BaseConfidenceHead(nn.Module, ABC):
    expected_trunk_eval_t: float = 0.99   # NEW class attribute, default = CLAUDE.md pin
    output_keys: tuple[str, ...] = ()
    def __init__(self, trunk, token_dim=768, pair_repr_dim=256) -> None: ...
    @abstractmethod
    def _predict(self, s, z, mask) -> dict[str, Tensor]: ...
    def forward(self, s, z, mask, cond, chain_id=None) -> dict[str, Tensor]: ...   # unchanged surface
    # NEW abstract method (default impl raises so subclasses are forced to think):
    def compute_loss_and_metrics(self, out, batch, mask_eff) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError
```

The `compute_loss_and_metrics` contract:
- `out: dict[str, Tensor]` — the head's forward return (e.g. `{"plddt_logits": ...}` or `{"pae_logits": ...}`).
- `batch: dict[str, Tensor]` — the trimmed-to-`n_orig` batch with `plddt_*` / `pae_*` fields the head needs.
- `mask_eff: Tensor` — `orig_mask & {plddt,pae}_mask` already AND-ed by the caller.
- Returns `(total_loss, {scalar_name: Tensor})`. The dict is everything the Lightning module's `self.log(...)` should emit for that head; the multi-head wrapper concatenates child dicts under a `{head_name}/` prefix.

### `PaeHead` (new file [src/proteinfoundation/nn/confidence/pae_head.py](../../../src/proteinfoundation/nn/confidence/pae_head.py))

```
@register_confidence_head("pae")
class PaeHead(BaseConfidenceHead):
    output_keys: tuple[str, ...] = ("pae_logits",)
    expected_trunk_eval_t: float = 0.99
    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_pae_bins: int = 64,
        bin_min: float = 0.0,
        bin_max: float = 32.0,   # 64 * 0.5 = 32.0; bin centers land at 0.25, 0.75, ..., 31.75
    ) -> None: ...
    def _predict(self, s, z, mask) -> dict[str, Tensor]:
        # NO symmetrisation. z passed through unchanged.
        logits = self.logits_linear(self.logits_norm(z))    # (B, L, L, num_pae_bins)
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None]
        logits = logits * pair_mask
        return {"pae_logits": logits}
    def logits_to_expected_value(self, logits) -> Tensor:
        # softmax over last dim, dot with bin_centers, fp32
    def compute_loss_and_metrics(self, out, batch, mask_eff) -> tuple[Tensor, dict[str, Tensor]]: ...
```

Output: `pae_logits` shape `(B, L, L, 64)`, dtype follows trunk (bf16 in mixed-precision); the `logits_to_expected_value` helper softmaxes in fp32 and dots with `bin_centers` registered as a buffer (`bin_width=0.5`, centers `[0.25, 0.75, …, 31.75]`).

### `MultiHeadConfidence` (new file [src/proteinfoundation/nn/confidence/multi_head.py](../../../src/proteinfoundation/nn/confidence/multi_head.py))

```
@register_confidence_head("multi_head")
class MultiHeadConfidence(BaseConfidenceHead):
    """Runs the shared trunk once; dispatches (s, z, mask, cond) to every child head."""
    output_keys: tuple[str, ...] = ()   # filled in __init__ as flat union of children's keys
    def __init__(
        self,
        trunk: ConfidenceTrunk,
        children: dict[str, BaseConfidenceHead],   # Hydra-instantiated; child trunk arg ignored
        token_dim: int = 768,
        pair_repr_dim: int = 256,
    ) -> None:
        super().__init__(trunk, token_dim, pair_repr_dim)
        # Reparent every child onto the wrapper's trunk so we own the single instance.
        for name, child in children.items():
            child.trunk = self.trunk
        self.children_heads = nn.ModuleDict(children)
        # Assert eval-t parity.
        for name, child in children.items():
            if child.expected_trunk_eval_t != self.expected_trunk_eval_t:
                raise ValueError(
                    f"MultiHeadConfidence: child head {name!r} has "
                    f"expected_trunk_eval_t={child.expected_trunk_eval_t} which "
                    f"!= wrapper's {self.expected_trunk_eval_t}. Use a separate "
                    f"sidecar for heads needing a different t."
                )
    def _predict(self, s, z, mask) -> dict[str, dict[str, Tensor]]:
        # Wrapper does not own a prediction MLP. Required because BaseConfidenceHead
        # declares _predict abstract; this dispatches to children, NOT the trunk.
        return {name: head._predict(s, z, mask) for name, head in self.children_heads.items()}
    def forward(self, s, z, mask, cond, chain_id=None) -> dict[str, dict[str, Tensor]]:
        # Single trunk forward. Children's own .forward is bypassed by going through
        # ._predict directly, so the trunk runs exactly once per wrapper forward.
        s_ref, z_ref = self.trunk(s, z, mask, cond)
        return {name: head._predict(s_ref, z_ref, mask) for name, head in self.children_heads.items()}
    def compute_loss_and_metrics(self, out, batch, mask_eff_by_head) -> tuple[Tensor, dict[str, Tensor]]:
        # Delegates to MultiHeadLoss (see §below) — the wrapper does no aggregation
        # itself, only routing.
```

Important: child heads are subclasses of `BaseConfidenceHead`, so each carries its own `trunk` reference from construction. The wrapper **rebinds** `child.trunk = self.trunk` at construction so all children share the wrapper's single trunk module. This means children Hydra cfg can carry any trunk (they need to; `_target_=PaeHead` requires `trunk:`); the wrapper just reassigns. Document this in the wrapper docstring.

Output of `MultiHeadConfidence.forward`: `{"plddt": {"plddt_logits": …}, "pae": {"pae_logits": …}}`.

### `MultiHeadLoss` (new in [src/proteinfoundation/confidence/losses.py](../../../src/proteinfoundation/confidence/losses.py) or `multi_head_loss.py` if it grows)

```
class MultiHeadLoss:
    """Across-head weight aggregator. Each head's own within-head ce/ev recipe is
    invoked by the head's compute_loss_and_metrics; this class only sums."""
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = {k: float(v) for k, v in weights.items()}
    def __call__(
        self,
        multi_out: dict[str, dict[str, Tensor]],
        heads: dict[str, BaseConfidenceHead],
        batch: dict[str, Tensor],
        masks_by_head: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total = torch.zeros((), device=batch["mask"].device, dtype=torch.float32)
        log_dict: dict[str, Tensor] = {}
        for name, head in heads.items():
            w = self.weights.get(name, 1.0)
            if w == 0.0:
                continue                                         # hard short-circuit
            mask_eff = masks_by_head[name]
            if mask_eff.sum() == 0:                              # all-False protection
                continue
            l_head, l_log = head.compute_loss_and_metrics(
                multi_out[name], batch, mask_eff
            )
            total = total + w * l_head
            log_dict[f"{name}/total"] = l_head
            for k, v in l_log.items():
                log_dict[f"{name}/{k}"] = v
        return total, log_dict
```

The mask-empty short-circuit AND the weight-zero short-circuit are both load-bearing: see tests 6 and 7.

### Loss and metrics functions (additions in `losses.py` and `metrics.py`)

`losses.py`:
- `masked_pae_cross_entropy(logits: (B,L,L,K), bin_labels: (B,L,L), mask: (B,L,L), label_smoothing)` — flatten to `(B*L*L, K)` then `F.cross_entropy(..., reduction="none")` and `_mask_reduce`. Already follows `masked_plddt_cross_entropy`'s pattern.
- `masked_smooth_l1_on_pae_expected_value(logits, pae_continuous, mask, bin_centers)` — analogue.
- `combined_pae_loss(logits, pae_bin_labels, pae_continuous, mask, bin_centers, ce_weight=0.9, smooth_l1_weight=0.1, label_smoothing=0.0)` — returns `(total, {"loss_ce", "loss_smooth_l1"})`. Mirrors `combined_plddt_loss` 1:1.

`metrics.py` (new):
- `pae_accuracy(logits, labels, mask)` — masked top-1 accuracy over pair positions.
- `pae_mae(logits, labels, mask, bin_centers)` — masked MAE in Å between EV and label-bin-center.
- `pae_pearson_r(pred_cont, target_cont, mask)`, `pae_spearman_r(...)` — per-protein average of pair-flattened correlations. Reuse the existing `pearson_r` / `spearman_r` by reshaping `(B, L, L) -> (B, L*L)`.
- `pae_mae_stratified_by_value(logits, labels, mask, bin_centers, buckets=((0,5),(5,15),(15,32)))` — analogue of `plddt_mae_stratified` with PAE buckets. Three buckets keeps the surface tight.
- `pae_mae_stratified_by_distance(logits, labels, mask, bin_centers, ca_coords, dist_thresholds=(8.0, 16.0))` — NEW. Distance-stratified MAE in the predicted-PAE EV vs label-EV. Three bins: `d_ij < 8`, `8 ≤ d_ij < 16`, `d_ij ≥ 16` Å. Computes `d_ij = ||CA_i - CA_j||`. Requires Cα coords on the batch. The sidecar must surface `ca_coords` to validation_step (see §6 for sourcing).
- `pae_ece(logits, labels, mask, num_bins_ece=10)` — equal-width, mirrors `expected_calibration_error`.
- `pae_ece_adaptive(logits, labels, mask, num_bins_ece=15)` — equal-mass, mirrors `expected_calibration_error_adaptive`.

The equal-width / equal-mass and Pearson / Spearman / `_logits_to_continuous` / `_labels_to_continuous` helpers in `metrics.py` are already shape-agnostic; PR-B can reuse them directly on `(B, L, L)` once flattened to `(B, L*L)`. No duplication needed — the new pair-level functions are thin shape-routers.

## 6. Module / file layout

### New files
- `src/proteinfoundation/nn/confidence/pae_head.py` — `PaeHead` class.
- `src/proteinfoundation/nn/confidence/multi_head.py` — `MultiHeadConfidence` wrapper.
- `configs/nn/confidence/pae_head.yaml` — head Hydra cfg (mirrors `plddt_head.yaml`).
- `configs/nn/confidence/multi_head.yaml` — wrapper Hydra cfg with two-child `children:` dict.
- `configs/confidence/distillation_teddymer_pae.yaml` — single-head PAE reference run config.
- `configs/confidence/distillation_teddymer_multihead.yaml` — joint pLDDT+PAE config.
- `scripts/train_confidence_teddymer_pae.sbatch` — SLURM wrapper (defaults to `--config-name=confidence/distillation_teddymer_pae`; multi-head run flips this on the command line).
- Tests (all paths under `tests/`):
  - `tests/unit/nn/confidence/test_symmetrisation_per_head.py`
  - `tests/unit/nn/confidence/test_pae_head.py`
  - `tests/unit/nn/confidence/test_multi_head_wrapper.py`
  - `tests/unit/confidence/test_pae_loss_and_metrics.py`
  - `tests/unit/confidence/test_multi_head_loss_weighting.py`
  - `tests/integration/test_pae_distillation_smoke.py`

### Touched files
- `src/proteinfoundation/nn/confidence/base.py`: (a) delete line 129 (`z = (z + z.transpose(-3, -2)) / 2.0`) — the immediately preceding `z = self.z_layer_norm(z) * pair_mask` (line 128) and the immediately following `z = z * pair_mask` (line 130) are both kept. (b) Add `expected_trunk_eval_t: float = 0.99` class attribute. (c) Add abstract `compute_loss_and_metrics` (default `raise NotImplementedError`).
- `src/proteinfoundation/nn/confidence/plddt_head.py`: implement `compute_loss_and_metrics` (move the existing per-step loss/metric logic from the Lightning module into the head). Trunk symmetrisation refactor is a no-op here (head consumes `s` only).
- `src/proteinfoundation/nn/confidence/plddt_sequence_only_head.py`: implement `compute_loss_and_metrics` (same body as `PLDDTHead`'s).
- `src/proteinfoundation/nn/confidence/__init__.py`: export `PaeHead`, `MultiHeadConfidence`.
- `src/proteinfoundation/confidence/losses.py`: add `masked_pae_cross_entropy`, `masked_smooth_l1_on_pae_expected_value`, `combined_pae_loss`, `MultiHeadLoss`.
- `src/proteinfoundation/confidence/metrics.py`: add the six pair-level functions listed in §5.
- `src/proteinfoundation/confidence/lightning_module.py`: refactor `_forward` to head-agnostic. Specifically:
  - Replace the `plddt_*`-specific access in `_forward` (lines 287-303) with a head-agnostic `mask_eff_by_head` dispatcher that reads `{name}_mask` from the batch for each head in `heads_to_drive`. For a single-head module, `heads_to_drive == {head.output_name_root: head}` (e.g. `{"plddt": head}` or `{"pae": head}`). For a multi-head module, `heads_to_drive == head.children_heads`.
  - Move loss/metric invocation from inline in `training_step` and `validation_step` to a single call `loss, log_dict = head.compute_loss_and_metrics(out, batch, mask_eff)` (single-head) or `loss, log_dict = self.multi_loss(out, head.children_heads, batch, masks_by_head)` (multi-head).
  - Pre-extract `ca_coords` on `batch["x_1"]["bb_ca"]` (trimmed to `n_orig`) and pass it through to PAE distance-stratified MAE. The metric function takes `ca_coords` as a kwarg only when called from the PAE head's metric path; the pLDDT head ignores it.
  - The reliability-diagram accumulator (currently pLDDT-specific) is gated by whether the head exposes a `reliability_diagram_logits_key` attribute (e.g. `"plddt_logits"`). For the PAE-alone head and joint runs the diagram is suppressed (or moves to a separate plddt-only path — keep simple in PR-B; suppress unless head is `PLDDTHead`).
- `src/proteinfoundation/confidence/train_confidence.py`: read `cfg.training.loss.ce_weight / smooth_l1_weight` into the head config tree instead of the Lightning module. The Lightning module's `ce_weight/smooth_l1_weight` constructor args stay (default 0.9/0.1) for backward compatibility but are passed through to the head's `compute_loss_and_metrics` only as the default when the head doesn't override. (Cleaner: pass them in the head's Hydra cfg as `loss.ce_weight / loss.ev_weight`. See §7 for the new yaml shape.)

### NOT touched
- `src/proteinfoundation/proteina.py`, `train.py`, the autoencoder, anything outside `confidence/` and `nn/confidence/`.
- `tests/regression/test_flow_matching_loss_unchanged.py` — must remain bit-identical green after the trunk refactor.

## 7. Hydra composition

### `configs/nn/confidence/pae_head.yaml`

```yaml
defaults:
  - base
  - _self_

_target_: proteinfoundation.nn.confidence.pae_head.PaeHead
token_dim: 768
pair_repr_dim: 256
num_pae_bins: 64
bin_min: 0.0
bin_max: 32.0
loss:
  ce_weight: 0.9
  ev_weight: 0.1
  label_smoothing: 0.0   # PAE labels are integer-rounded by AFDB; no smoothing needed
```

### `configs/nn/confidence/multi_head.yaml`

```yaml
defaults:
  - base
  - _self_

_target_: proteinfoundation.nn.confidence.multi_head.MultiHeadConfidence
token_dim: 768
pair_repr_dim: 256
# Children are NOT default-merged via `defaults:` (Hydra cannot easily compose a
# dict of heads from the `defaults` list). Instead, override via CLI or via
# overrides in the parent config.
children: ???     # MUST be overridden by the parent config
weights: ???      # across-head weights dict, also overridden
```

### `configs/confidence/distillation_teddymer_pae.yaml`

```yaml
# @package _global_
defaults:
  - /nn/confidence/pae_head@confidence.head
  - /dataset/unified/teddymer_with_plddt_and_pae@data
  - /training/confidence_distill@training
  - _self_

trainer:
  _target_: lightning.pytorch.Trainer
  accelerator: gpu
  devices: 1
  strategy: ddp_find_unused_parameters_false
  precision: ${training.precision}
  max_epochs: ${training.max_epochs}
  gradient_clip_val: ${training.gradient_clip_val}
  log_every_n_steps: 25
  check_val_every_n_epoch: 1
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: "val/pae/loss_ce"   # PAE head's CE logged under pae/ prefix
      mode: "min"
      save_top_k: 3
      filename: "pae-ce-{epoch:03d}-{val/pae/loss_ce:.4f}"
      auto_insert_metric_name: false
    - _target_: lightning.pytorch.callbacks.EarlyStopping
      monitor: "val/pae/loss_ce"
      mode: "min"
      patience: 10

data:
  datamodule:
    cluster_column: null
    cluster_seed: 42

seed: 42
run_name: "pae-distill-teddymer"
```

### `configs/confidence/distillation_teddymer_multihead.yaml`

```yaml
# @package _global_
defaults:
  - /nn/confidence/multi_head@confidence.head
  - /dataset/unified/teddymer_with_plddt_and_pae@data
  - /training/confidence_distill@training
  - _self_

confidence:
  head:
    # children: dict[name -> head_cfg]. Each child is a full head cfg; the
    # MultiHeadConfidence rebinds .trunk on construction so child trunk fields
    # do not need to match.
    children:
      plddt:
        _target_: proteinfoundation.nn.confidence.plddt_head.PLDDTHead
        token_dim: 768
        pair_repr_dim: 256
        num_plddt_bins: 50
        bin_min: 0.0
        bin_max: 100.0
        # within-head weights: CE vs SmoothL1(EV). Same recipe family as SwissProt.
        loss:
          ce_weight: 0.9
          ev_weight: 0.1
          label_smoothing: 0.05
      pae:
        _target_: proteinfoundation.nn.confidence.pae_head.PaeHead
        token_dim: 768
        pair_repr_dim: 256
        num_pae_bins: 64
        bin_min: 0.0
        bin_max: 32.0
        # within-head weights: independent of the across-head weights below.
        loss:
          ce_weight: 0.9
          ev_weight: 0.1
          label_smoothing: 0.0
    # across-head weights: multiply each head's total loss before summing.
    # weight: 0.0 disables a head's contribution to gradients (hard short-circuit).
    weights:
      plddt: 0.7
      pae: 0.7

trainer:
  _target_: lightning.pytorch.Trainer
  accelerator: gpu
  devices: 1
  strategy: ddp_find_unused_parameters_false
  precision: ${training.precision}
  max_epochs: ${training.max_epochs}
  gradient_clip_val: ${training.gradient_clip_val}
  log_every_n_steps: 25
  check_val_every_n_epoch: 1
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      monitor: "val/loss_total"
      mode: "min"
      save_top_k: 3
      filename: "multi-{epoch:03d}-{val/loss_total:.4f}"
      auto_insert_metric_name: false

data:
  datamodule:
    cluster_column: null
    cluster_seed: 42

seed: 42
run_name: "multi-plddt-pae-distill-teddymer"
```

The two weighting layers are **structurally** separated in this yaml: within-head `loss.{ce_weight, ev_weight}` lives under each child's cfg block; across-head `weights:` is a sibling of `children:` under `head:`. There is no ambiguity about which knob a future operator must turn for what.

## 8. Trunk refactor mechanics

### Diff at [src/proteinfoundation/nn/confidence/base.py](../../../src/proteinfoundation/nn/confidence/base.py)

```diff
@@ class ConfidenceTrunk
         s = self.s_layer_norm(s) * mask_f
         z = self.z_layer_norm(z) * pair_mask
-        z = (z + z.transpose(-3, -2)) / 2.0
-        z = z * pair_mask
+        # Symmetrisation moved to per-head _predict (CLAUDE.md option (b)).
+        # Symmetric pair-heads (PDE, ipLDDT, ipTM, future) add
+        # `z = 0.5 * (z + z.transpose(-3, -2))` as the first line of _predict.
+        # Asymmetric heads (PaeHead) consume z directly.
         return s, z
```

### Callers of `trunk.forward`

`grep -rn "self.trunk(" /mnt/storage01/home/schekmenev/projects/complexa-flex/src/` returns two hits:
1. `BaseConfidenceHead.forward` at [src/proteinfoundation/nn/confidence/base.py:191](../../../src/proteinfoundation/nn/confidence/base.py#L191) — passes returned `(s, z)` to `self._predict(s, z, mask)`. No change required: `PLDDTHead._predict` and `SequenceOnlyPLDDTHead._predict` both `del z`, so the un-symmetrised `z` they receive is unused.
2. `SequenceOnlyPLDDTHead.forward` at [src/proteinfoundation/nn/confidence/plddt_sequence_only_head.py:63](../../../src/proteinfoundation/nn/confidence/plddt_sequence_only_head.py#L63) — same situation, `_predict` discards `z`.

### Regression test guard

`tests/regression/test_flow_matching_loss_unchanged.py` is a guard on `LocalLatentsTransformer` (the `proteina.nn` module), **not** on `ConfidenceTrunk`. It will be unaffected by the trunk-symmetrisation removal — verified by reading lines 1-80 of that file. The `ConfidenceTrunk` symmetry property is currently checked at `tests/unit/confidence/test_confidence_trunk.py:70-74` (`test_z_is_symmetric_after_forward`). PR-B **must update** that test to assert the opposite (the trunk now passes `z` through un-symmetrised). The new behaviour is covered in `test_symmetrisation_per_head.py`.

## 9. Test plan (red phase first)

All tests must fail with `AttributeError` / `ImportError` / `Could not resolve` in the red phase — proving the artefact is missing — not with a setup error.

### Test 1 — `tests/regression/test_flow_matching_loss_unchanged.py` (existing; must remain green)
- **Action:** run before and after the trunk-symmetrisation deletion. Expected: green both times. Failure = abort (see §11).
- **Failure mode (sanity):** fixture sha mismatch = test setup drifted; investigate before any further work.

### Test 2 — `tests/unit/nn/confidence/test_symmetrisation_per_head.py` (new)
- **Subtest 2.1 — Trunk no longer symmetrises `z`.** Run a `ConfidenceTrunk` on random `(s, z, mask, cond)` with `dropout=0.0` and `eval()`; assert `not torch.allclose(z_out, z_out.transpose(-3, -2), atol=1e-3)` on the unmasked region. (Replaces the existing `test_z_is_symmetric_after_forward` in `test_confidence_trunk.py` — update that test in lockstep, or move it here and delete the old one with a comment pointing to the new file.)
- **Subtest 2.2 — `PLDDTHead` still produces shape-`(B, L, num_bins)` symmetric-by-construction output even on asymmetric `z`.** Because `PLDDTHead._predict` consumes `s` only, the test is degenerate; phrase it as "output is invariant to `z` symmetrisation". Run the head with `z` and with `z.transpose(-3, -2)` and assert equal logits.
- **Subtest 2.3 — `PaeHead` produces directional output.** Run on random `z` with `dropout=0.0` and `eval()`; assert `not torch.allclose(out["pae_logits"], out["pae_logits"].transpose(-3, -2), atol=1e-3)`.
- **Red-phase failure mode:** `AttributeError: module 'proteinfoundation.nn.confidence' has no attribute 'PaeHead'`.

### Test 3 — `tests/unit/nn/confidence/test_pae_head.py` (new)
- **Subtest 3.1 — Forward shape.** Random batch `(B=2, L=10)`, trunk dims 64/32; assert `head.forward(s, z, mask, cond)["pae_logits"].shape == (2, 10, 10, 64)`.
- **Subtest 3.2 — Gradients flow.** Zero-init batch, compute scalar loss `(logits ** 2).mean()`, `.backward()`; assert at least one head parameter and one trunk parameter have non-None, non-zero `.grad`.
- **Subtest 3.3 — `logits_to_expected_value`.** On a logits tensor that is all-zero (uniform softmax), `EV == mean(bin_centers) == 16.0` (since bin_centers are `[0.25, 0.75, …, 31.75]` and mean is `16.0`).
- **Red-phase failure mode:** import error on `pae_head`.

### Test 4 — `tests/unit/confidence/test_pae_loss_and_metrics.py` (new)
- **Subtest 4.1 — `masked_pae_cross_entropy` matches a hand-computed value on a `(1, 3, 3, 4)` case.** Use known logits and known bin labels; mask one cell False; assert scalar equals hand-computed `(sum of -log softmax over the 8 valid cells) / 8`.
- **Subtest 4.2 — `masked_smooth_l1_on_pae_expected_value` matches hand-computed value on the same 3×3 case.** Set logits to one-hot at known bins; EV equals known bin center; SmoothL1 reduces to L1 on cells where |error| < 1 by spec.
- **Subtest 4.3 — `combined_pae_loss` defaults sum correctly.** Asserts `total == 0.9 * loss_ce + 0.1 * loss_smooth_l1` to fp32 tolerance.
- **Subtest 4.4 — `pae_ece_adaptive` does not NaN on partial mask.** Build a `(1, 5, 5, 4)` logits tensor with a checkerboard mask; assert metric is finite.
- **Subtest 4.5 — `pae_mae_stratified_by_distance` returns three finite entries.** Build random `ca_coords (1, 5, 3)`; verify each bucket key (`d_lt8`, `d_8_16`, `d_ge16`) is in the returned dict; each value is finite.
- **Red-phase failure mode:** `ImportError: cannot import name 'combined_pae_loss' from proteinfoundation.confidence.losses`.

### Test 5 — `tests/integration/test_pae_distillation_smoke.py` (new, mark `slow`)
- **Subtest 5.1 — One Lightning training step.** Instantiate `ConfidenceDistillationModule.from_components(head=PaeHead(...), proteina=dummy_proteina)` (use the existing `dummy_proteina` fixture from `tests/integration/confidence/_fixture_loader.py`). Build a fake Teddymer batch of `B=2` dimers at `L=12`. Run `trainer.fit(module, datamodule=fake_dm)` with `fast_dev_run=True`. Assert loss is finite.
- **Subtest 5.2 — Overfit-one-batch.** Repeat 10 training steps on the same micro-batch; assert `loss[9] < 0.95 * loss[0]`.
- **Red-phase failure mode:** missing `PaeHead` import or `compute_loss_and_metrics` raises `NotImplementedError`.
- **Skipif:** mark `@pytest.mark.slow`; gate on `TEDDYMER_VIEW_ROOT` being set OR fall back to a synthetic fake datamodule (preferred; avoids host coupling).

### Test 6 — `tests/unit/nn/confidence/test_multi_head_wrapper.py` (new)
- **Subtest 6.1 — Dict-of-tensors output.** Construct `MultiHeadConfidence({"plddt": PLDDTHead(...), "pae": PaeHead(...)})`; one forward returns `{"plddt": {"plddt_logits": (B, L, 50)}, "pae": {"pae_logits": (B, L, L, 64)}}`.
- **Subtest 6.2 — Trunk-once invariant.** Monkeypatch `wrapper.trunk.forward` to a counter wrapper around the real forward; do one wrapper forward; assert counter == 1.
- **Subtest 6.3 — Per-head missing-label tolerance.** Build batch with `plddt_mask` all-True and `pae_mask` all-False; build `masks_by_head = {"plddt": all_true_mask, "pae": all_false_mask}`; invoke `MultiHeadLoss({"plddt":1.0,"pae":1.0})(out, children, batch, masks_by_head)`; assert total loss is finite (not NaN) and equals the pLDDT-only loss to within fp32 tolerance (the PAE branch is skipped via the all-False mask short-circuit).
- **Subtest 6.4 — `expected_trunk_eval_t` mismatch.** Define a fake child class `FakeChild(BaseConfidenceHead)` with `expected_trunk_eval_t = 0.5`; assert `MultiHeadConfidence(trunk=..., children={"fake": FakeChild(...)})` raises `ValueError` whose message contains `"fake"` and `"0.5"`.
- **Red-phase failure mode:** `AttributeError: module 'proteinfoundation.nn.confidence' has no attribute 'MultiHeadConfidence'`.

### Test 7 — `tests/unit/confidence/test_multi_head_loss_weighting.py` (new)
- **Subtest 7.1 — Weight-zero disables head and gradient path.** Loss `MultiHeadLoss({"plddt": 1.0, "pae": 0.0})(...)` on a batch with both masks all-True; assert (a) total equals single-head `combined_plddt_loss` result to fp32 tolerance; (b) after `total.backward()`, every `PaeHead` parameter has `.grad is None` or `.grad.abs().max() == 0`.
- **Subtest 7.2 — Both-active gradients reach both heads + trunk.** `MultiHeadLoss({"plddt": 0.7, "pae": 0.7})(...)`; after `.backward()`, at least one `PLDDTHead.logits_linear` param has non-zero grad AND at least one `PaeHead.logits_linear` param has non-zero grad AND at least one trunk param has non-zero grad.
- **Subtest 7.3 — pLDDT-zero gradients only on PAE + trunk.** `MultiHeadLoss({"plddt": 0.0, "pae": 1.0})(...)`; after `.backward()`, every `PLDDTHead.logits_linear` param has `.grad is None` or `.grad.abs().max() == 0`; PAE + trunk params have non-zero grad.
- **Subtest 7.4 — Log-dict structure.** Asserts the returned `log_dict` keys are exactly `{"plddt/total", "plddt/loss_ce", "plddt/loss_smooth_l1", "pae/total", "pae/loss_ce", "pae/loss_smooth_l1"}` and `sum(weights[k] * log_dict[f"{k}/total"]) == total` within fp32 tolerance.
- **Red-phase failure mode:** `ImportError: cannot import name 'MultiHeadLoss'`.

### Existing test maintenance
- `tests/unit/confidence/test_confidence_trunk.py:70-74` (`test_z_is_symmetric_after_forward`) MUST be updated to the opposite invariant (or moved to `test_symmetrisation_per_head.py` and deleted from the old file). Lockstep with the trunk diff.
- `tests/unit/confidence/test_config_compose.py` — if it currently asserts `cfg.training.loss.ce_weight == 0.9`, this stays. If it lacks the assertion (per PR-A reviewer punted item 7), ADD the assertion in PR-B (one-line addition, no new test file).
- `tests/unit/confidence/test_lightning_defaults.py` — same as above for the Lightning-defaults parity assertion.

## 10. Sequencing within PR-B (three slices)

### Slice 1 — Trunk refactor + regression
- **Tests first:** update `test_confidence_trunk.py::test_z_is_symmetric_after_forward` to assert un-symmetric (or move it); write `test_symmetrisation_per_head.py` subtests 2.1 + 2.2 (subtest 2.3 needs `PaeHead`, deferred to Slice 2).
- **Implementation:**
  1. `BaseConfidenceHead`: add `expected_trunk_eval_t: float = 0.99` class attribute.
  2. `BaseConfidenceHead`: add abstract `compute_loss_and_metrics` with `raise NotImplementedError`.
  3. `ConfidenceTrunk.forward`: delete line 129 only; keep line 130 (`z = z * pair_mask`).
  4. `PLDDTHead.compute_loss_and_metrics`: move the existing inline body from `lightning_module.py::training_step` and `validation_step` into the head, parameterised by within-head `ce_weight / ev_weight / label_smoothing` constructor args.
  5. `SequenceOnlyPLDDTHead.compute_loss_and_metrics`: identical body.
  6. `lightning_module.py::_forward, training_step, validation_step`: refactor to call `head.compute_loss_and_metrics(out, batch, mask_eff)`; keep the reliability-diagram gating behind `isinstance(head, PLDDTHead)` (or a `reliability_diagram_logits_key` attribute on the head).
  7. Add the missing `ce_weight == 0.9` assertion in `test_config_compose.py` and `test_lightning_defaults.py` (PR-A reviewer punted item).
- **Exit:** `tests/regression/test_flow_matching_loss_unchanged.py` green; `test_confidence_trunk.py` green (updated invariant); `test_symmetrisation_per_head.py` subtests 2.1/2.2 green; existing SwissProt-pLDDT integration tests under `tests/integration/confidence/` green (proves the head-refactor preserved behaviour); `test_config_compose.py`/`test_lightning_defaults.py` green with new assertions.

### Slice 2 — `PaeHead` + loss + metrics + single-head training entry
- **Tests first:** `test_pae_head.py` (Tests 3.x); `test_pae_loss_and_metrics.py` (Tests 4.x); `test_symmetrisation_per_head.py` subtest 2.3; `test_pae_distillation_smoke.py` (Test 5.x).
- **Implementation:**
  1. `pae_head.py`: `PaeHead` class with `_predict` returning `(B, L, L, 64)` logits, `logits_to_expected_value`, `compute_loss_and_metrics` calling `combined_pae_loss` + the PAE metrics suite.
  2. `losses.py`: `masked_pae_cross_entropy`, `masked_smooth_l1_on_pae_expected_value`, `combined_pae_loss`.
  3. `metrics.py`: `pae_accuracy`, `pae_mae`, `pae_mae_stratified_by_value`, `pae_mae_stratified_by_distance`, `pae_ece`, `pae_ece_adaptive` (the Pearson/Spearman pair-flatten wrappers can either be reused inline from `pearson_r`/`spearman_r` or thin shims).
  4. `configs/nn/confidence/pae_head.yaml`.
  5. `configs/confidence/distillation_teddymer_pae.yaml`.
  6. `scripts/train_confidence_teddymer_pae.sbatch`.
  7. `__init__.py` export of `PaeHead`.
  8. Tighten the integration test directional-PAE asymmetry check in PR-A's `test_add_pae_from_parent_afdb.py` from `max > 1e-3` to `median > 1e-3` (PR-A reviewer punted item 4).
- **Exit:** all Tests 3, 4, 5 green; `fast_dev_run=True` on `distillation_teddymer_pae.yaml` completes one step.

### Slice 3 — `MultiHeadConfidence` + `MultiHeadLoss` + multi-head training entry
- **Tests first:** `test_multi_head_wrapper.py` (Tests 6.x); `test_multi_head_loss_weighting.py` (Tests 7.x).
- **Implementation:**
  1. `multi_head.py`: `MultiHeadConfidence` class.
  2. `losses.py`: `MultiHeadLoss` class.
  3. `configs/nn/confidence/multi_head.yaml`.
  4. `configs/confidence/distillation_teddymer_multihead.yaml`.
  5. `lightning_module.py`: handle the multi-head case in `_forward / training_step / validation_step`. Detect via `isinstance(head, MultiHeadConfidence)`. The single-head path remains the default.
  6. `__init__.py` export of `MultiHeadConfidence`.
- **Exit:** all Tests 6, 7 green; `fast_dev_run=True` on `distillation_teddymer_multihead.yaml` completes one step on a fake datamodule.

## 11. Exit / abort criteria

### Exit (PR-B done)
- Tests 1–7 all green.
- `tests/regression/test_flow_matching_loss_unchanged.py` green.
- All SwissProt-pLDDT integration tests under `tests/integration/confidence/` green (proves the head-refactor + Lightning-module refactor did not regress single-head pLDDT).
- `uv run python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_teddymer_pae trainer.fast_dev_run=true trainer.devices=1` completes (one train + one val step) and emits at least one `val/pae/*` scalar to the logger.
- `uv run python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_teddymer_multihead trainer.fast_dev_run=true trainer.devices=1` completes and emits both `val/plddt/*` and `val/pae/*` scalars.
- Five-reviewer panel (§13) all approve.

### Abort (stop and rethink)
- **Regression-test break.** If `test_flow_matching_loss_unchanged.py` fails after the trunk diff, STOP — the symmetrisation removal had unintended downstream effect. Recover with the option-(a) fallback (gated `symmetrise_z: bool = True` constructor flag on `ConfidenceTrunk`).
- **PAE GPU OOM at training L.** If at `B=2, L=384` the `(B, L, L, 64)` logits + activations exceed available GPU memory at `bf16-mixed`, STOP. Mitigation order: (1) drop `B` to 1; (2) switch to gradient-accumulation × 2; (3) fall back to a bin-only output with EV computed without storing the full logits dict (heavier engineering). Logits memory budget: `2 * 384 * 384 * 64 * 2 B (bf16) ≈ 38 MB`; with attention activations ~10× that, still well under 80 GB H100. Likely fine.
- **`torch.compile` recompile storm on dynamic L.** Teddymer dimers are heterogeneously cropped; if the Lightning module enables `torch.compile` by default and recompiles every batch, STOP. Mitigation: set `compile=False` for the PR-B reference run; investigate dynamic-shape mode separately.
- **Multi-head weight-zero gradient path leaks.** If Test 7.1 / 7.3 fails because PyTorch `nn.ModuleDict` re-includes the zero-weighted child in autograd (e.g. because the child's `.forward` was called for output even though its loss was zeroed), STOP and redesign: short-circuit must skip the child's `_predict` entirely when `weight == 0.0`. (Current §5 design avoids this by short-circuiting in `MultiHeadLoss.__call__` before invoking the child's loss; only the child's `_predict` ran during the wrapper's forward. Confirm Test 7.1's gradient assertion — if the child's `_predict` output graph is reachable from `total.backward()` through some side-channel, the design is wrong.)

## 12. Verification commands

```bash
# Red phase — confirm tests fail with the right import error:
cd /mnt/storage01/home/schekmenev/projects/complexa-flex
uv run pytest \
  tests/unit/nn/confidence/test_symmetrisation_per_head.py \
  tests/unit/nn/confidence/test_pae_head.py \
  tests/unit/nn/confidence/test_multi_head_wrapper.py \
  tests/unit/confidence/test_pae_loss_and_metrics.py \
  tests/unit/confidence/test_multi_head_loss_weighting.py \
  -x -v 2>&1 | head -120

# Slice 1 green-phase check (trunk + Lightning refactor):
uv run pytest \
  tests/regression/test_flow_matching_loss_unchanged.py \
  tests/unit/confidence/test_confidence_trunk.py \
  tests/unit/nn/confidence/test_symmetrisation_per_head.py \
  tests/unit/confidence/test_config_compose.py \
  tests/unit/confidence/test_lightning_defaults.py \
  tests/integration/confidence/ \
  -x -v

# Slice 2 green-phase check (PaeHead + loss + metrics + single-head entry):
uv run pytest \
  tests/unit/nn/confidence/test_pae_head.py \
  tests/unit/confidence/test_pae_loss_and_metrics.py \
  tests/unit/datasets/test_add_pae_from_parent_afdb.py \
  -x -v
uv run python -m proteinfoundation.confidence.train_confidence \
  --config-name=confidence/distillation_teddymer_pae \
  trainer.fast_dev_run=true trainer.devices=1

# Slice 3 green-phase check (MultiHead wrapper + loss):
uv run pytest \
  tests/unit/nn/confidence/test_multi_head_wrapper.py \
  tests/unit/confidence/test_multi_head_loss_weighting.py \
  -x -v
uv run python -m proteinfoundation.confidence.train_confidence \
  --config-name=confidence/distillation_teddymer_multihead \
  trainer.fast_dev_run=true trainer.devices=1

# Full PR-B regression:
uv run pytest \
  tests/regression/ tests/unit/confidence/ tests/unit/nn/confidence/ \
  tests/integration/confidence/ tests/unit/datasets/ \
  -x -v
```

## 13. Reviewer panel (confirmed from handoff §"Reviewers for PR-B")

Mandatory:
- **code-review-debug-complexity-expert** — correctness of the trunk refactor, gradient-flow short-circuit edge cases, the `MultiHeadLoss` weight-zero invariant, log-key collision risk.

Domain-add (per handoff):
- **ml-protein-architect** — module layout (new `pae_head.py` + `multi_head.py` files; the `compute_loss_and_metrics` protocol moves loss invocation out of the Lightning module; Hydra child-instantiation pattern in `multi_head.yaml`); sidecar pattern integrity.
- **generative-protein-scientist** — PAE bin edges, loss weighting (`ce_weight / ev_weight = 0.9 / 0.1`), distance-stratified metric thresholds (`d_ij < 8`, `8 ≤ d_ij < 16`, `d_ij ≥ 16`), the choice of `pae_mae_stratified_by_value` buckets.
- **generative-flow-stochastic-math-expert** — bin-classification + EV-regression hybrid is the right discretisation for a directional pair-quantity; the SmoothL1-on-EV interaction with discrete labels does not introduce bias.
- **ml-software-pytorch-jax-expert** — interaction of the symmetrisation refactor with `torch.compile` and FSDP, regression test still passing on multi-GPU; PAE memory profile at training-scale L; deterministic gradient on weight-zero short-circuit; the strict `chain_idx` length assertion against `data.num_nodes` from PR-A (punted item 6); the `cfg.training.loss.ce_weight == 0.9` assertion in `test_config_compose.py` / `test_lightning_defaults.py` (punted item 7).

Five reviewers total. Run in parallel after each slice's green phase. All must approve before merging into `prepare_teddy`.

## 14. Risks register

| Risk | Severity | Likelihood | Mitigation |
| --- | --- | --- | --- |
| **PAE logits memory at training L.** `(B=2, L=384, L=384, K=64)` bf16 ≈ 38 MB per copy; with attention activations and gradients ×10, ~400 MB. At L=512 ≈ 800 MB. Still fits H100. | LOW | LOW | Abort criterion in §11 lists fallback path. Smoke run at `L=384` first; scale to `L=512` only if memory headroom remains. |
| **`torch.compile` recompile on dynamic L.** Teddymer dimers have heterogeneous lengths; if compile is on, every novel L triggers a recompile. | MEDIUM | MEDIUM | Default to `compile=False` in PR-B; investigate `dynamic=True` mode separately. Document in sbatch comments. |
| **Multi-head gradient flow under weight-zero short-circuit.** If `MultiHeadConfidence.forward` already ran the child's `_predict`, the child's output graph is alive; `total.backward()` would not give grad to the child (loss term zero), but autograd warnings about "unused parameters" can surface under DDP. | MEDIUM | MEDIUM | Use `strategy: ddp_find_unused_parameters_true` for the multi-head run only when any weight is 0.0. The single-head Teddymer-PAE reference run uses the default `find_unused_parameters_false`. Document. |
| **Residue-numbering frame.** Inherited from PR-A; PR-A's integration test bound the index frame, so PR-B inherits confidence — PaeHead consumes whatever PR-A emits. | LOW | LOW | Neutralised by PR-A's `test_teddymer_interface_cca_spot_check.py`. PR-B does not re-introduce. |
| **`compute_loss_and_metrics` protocol leaks Lightning into the head.** Heads now invoke loss/metric functions that historically lived in the sidecar. | MEDIUM | LOW | The protocol returns `(loss_tensor, log_dict)` — pure tensors, no Lightning import in head files. The Lightning module remains the single place that calls `self.log(...)`. |
| **Lightning-checkpoint format drift.** Adding `compute_loss_and_metrics` does not change `state_dict`; adding the `expected_trunk_eval_t` class attribute also does not (it's a class attribute, not a buffer). Existing pLDDT-on-SwissProt checkpoints continue to load. | LOW | LOW | Verified by re-running the existing pLDDT checkpoint-roundtrip integration tests (`tests/integration/confidence/test_distillation_checkpoint_roundtrip.py`) at Slice 1 exit. |
| **Pair-PAE log-key collisions** between single-head and multi-head paths. Single-head emits `val/loss_ce`; multi-head emits `val/pae/loss_ce`. | LOW | MEDIUM | Adopt the prefixed convention uniformly in `compute_loss_and_metrics`: the head's returned `log_dict` keys are unprefixed (`{"loss_ce", "loss_smooth_l1", "accuracy", ...}`); the Lightning module's single-head path emits as `train/{key}`/`val/{key}` (back-compat with the existing SwissProt run only if we re-prefix as `train/plddt/{key}` — DECISION: prefix uniformly, take the one-line break in existing dashboards). The ModelCheckpoint monitor metric in `distillation_swissprot.yaml` must be updated to `val/plddt/loss_ce` in Slice 1. Document the change in the slice-1 PR description. |
| **Reliability-diagram regression in SwissProt.** Moving `_accumulate_reliability` behind a head-side hook can change the rank-0 output path. | MEDIUM | LOW | Keep the existing `_accumulate_reliability` body in the Lightning module, gated by `isinstance(head, PLDDTHead) or isinstance(head, SequenceOnlyPLDDTHead)`. Add `tests/integration/confidence/test_validation_logging_extension.py` to the slice-1 exit checklist. |
| **In-place vs out-of-place symmetrisation in future symmetric heads.** When PDE/ipLDDT land, each must symmetrise in `_predict`. Risk: a future contributor writes `z += z.transpose(...)` mutating the wrapper's cached `z`, breaking sibling heads in the multi-head wrapper. | LOW | MEDIUM | The `MultiHeadConfidence._predict` dispatcher passes the *same* `z` object to every child. Add a docstring on `BaseConfidenceHead._predict` instructing implementers to write `z_sym = 0.5 * (z + z.transpose(-3, -2))` (new tensor, no in-place op). |
| **`MultiHeadConfidence` not registering as a meaningful checkpoint entity.** Multi-head training emits one ModelCheckpoint per `val/loss_total`; on resume, the wrapper's `state_dict` must round-trip both children's params. | LOW | LOW | `nn.ModuleDict` flattens its children's `state_dict` keys as `children_heads.{name}.{param}`; checkpoint roundtrip is standard. Add a one-step checkpoint-save / load test to Slice 3 exit if a punted PR-A item or reviewer flags it; otherwise rely on the existing checkpoint-roundtrip integration test. |

## 15. PR-A reviewer-punted items: in-scope ruling

| # | Punted item | In PR-B? | If in: which slice |
| --- | --- | --- | --- |
| 1 | Locator dict → DataFrame refactor (memory + setup time) | **OUT** | TeddymerDimerDataModule concern; PR-A code, not PR-B. File a follow-up issue. |
| 2 | Tar handle / CIF parse LRU caching (perf) | **OUT** | PR-A perf concern; defer to a perf-focused follow-up. |
| 3 | `_split_metadata` dedup against `StructureDataModule` (architecture) | **OUT** | Architecture refactor outside PR-B scope. |
| 4 | Tighten `test_add_pae_from_parent_afdb.py` directional-PAE asymmetry to `median > 1e-3` | **IN** | Slice 2 (one-line test tightening; touched while extending PAE coverage). |
| 5 | `is_interface_pair` precomputed mask on the dataset (binder) | **OUT** | Dataset-side concern; the PAE distance-stratified metric computes it at validation time from `chain_idx` and `ca_coords`, so the field is not needed for the reference run. Useful future cache. |
| 6 | Strict `chain_idx` length assertion against `data.num_nodes` (ml-pytorch-jax) | **IN** | Slice 2 — folds into `PaeHead._predict` precondition assertion (one-line `assert chain_idx.shape[-1] == s.shape[-2]` at the top of `_predict` when `chain_idx` is provided). |
| 7 | `cfg.training.loss.ce_weight == 0.9` assertion in `test_config_compose.py` / `test_lightning_defaults.py` | **IN** | Slice 1 — co-located with the Lightning-module refactor; one-line additions. |
| 8 | Transposed-index regression case in `test_add_pae_from_parent_afdb.py` | **IN** | Slice 2 (one-line subtest in the PR-A test file — touch since we're adding subtest 4 below). |

## 16. Open questions

None. The locked decisions in §2 cover every choice the user has frozen; the reviewer-punted items are ruled in §15. Plan is ready to execute.

---

**Ready to execute.** Hand to **code-review-debug-complexity-expert** for the red-phase test authoring per §9 (Slice 1 first: update `test_confidence_trunk.py::test_z_is_symmetric_after_forward` invariant + `test_symmetrisation_per_head.py` subtests 2.1/2.2 + add the `ce_weight == 0.9` assertions). Then **ml-protein-architect** drives the green phase per §10, slice by slice. After each slice's green, dispatch the five-reviewer panel in §13 in parallel. Final merge into `prepare_teddy` requires all five approvals across slices.
