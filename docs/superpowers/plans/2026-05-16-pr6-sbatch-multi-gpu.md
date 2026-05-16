# PR-6 — sbatch, multi-GPU smoke, PR-5 follow-ups, PR-4 cond-shape fix

Status: planned. Final PR of the confidence-head distillation task.
Date: 2026-05-16.

## 1. Branch and base

- Branch: `feat/confidence-sbatch-multi-gpu`
- Base: `merge_quality_graft` @ `d4a89e2`
- Worktree: `.claude/worktrees/pr6-sbatch-multi-gpu/`
- Target merge: `merge_quality_graft` (not `dev` — task delivers off the integration branch)

## 2. Goal and non-goals

**Goal.** Make the distillation run *ready to launch* on the H100 cluster, fix the 11 PR-5 follow-up items in one bundle, and pre-emptively close the PR-4 `_compute_cond` shape contract bug *on the sidecar side* so ipTM/ipAE work does not trip on it later. Deliver one `sbatch`-able script, plus the multi-GPU + concat-features integration tests that PR-5 deferred.

**Non-goals.**
- No edits to `proteina.py`, `local_latents_transformer*.py`, `feature_factory/*.py`, `nn/confidence/{base,plddt_head,projections,registry}.py`, `confidence/{losses,metrics}.py`, `datasets/transforms.py`, `product_space_flow_matcher.py` (frozen for PR-6).
- No real training run. The sbatch is a script the user invokes.
- No ipTM/ipAE/ipLDDT heads (separate research thread).
- No joint trunk+head training.
- No transfer-eval (RFdiffusion / Dayhoff) — deferred.

## 3. Files to add / edit

| Path | Op | Purpose |
|---|---|---|
| `scripts/train_confidence_swissprot.sbatch` | NEW | SLURM script, 2× h100nvl, bf16-mixed, DDP, 3-day wall time; stages trunk + AE ckpt + AFDB parquet to `/netscratch/$USER`; identical NCCL knobs to `quality-graft`'s sbatch. |
| `src/proteinfoundation/confidence/lightning_module.py` | EDIT additive | (i) rank-0 guard in `_emit_reliability_diagram`; (ii) replace per-batch stash with per-bucket accumulator + epoch-end reduction; (iii) `_pad_cond_to_n_ext` helper called from `_forward` after `_compute_cond`; (iv) class-level `_log_table_warned` flag for once-per-run logging. |
| `src/proteinfoundation/datasets/structure_data.py` | EDIT additive | `len(val) == 0` `loguru.warning` in `_cluster_aware_split` returning before downstream code crashes; rebalance hint in the message. |
| `configs/dataset/unified/afdb_monomers.yaml` | EDIT | Add `unicluster` to `columns_to_load`. |
| `configs/dataset/unified/afdb_monomers_with_plddt.yaml` | EDIT | Same (inherits, but list is overridden — append explicitly). |
| `configs/nn/confidence/plddt_sequence_only_head.yaml` | EDIT | Drop the `update_pair_repr_every_n: 1000000` override (with `n_blocks=2` only one update can fire, and `z` is zeroed pre-trunk anyway). |
| `tests/integration/confidence/test_distillation_with_concat_features.py` | NEW | Fake `Proteina` with `n_ext > n_orig`; assert head sees cond of shape `(b, n_ext, dim_cond)`. |
| `tests/integration/confidence/test_distillation_ddp_smoke.py` | NEW | `torch.distributed` gloo CPU 2-rank spawn; one `training_step` per rank; finite loss; only head params accrue grad. |
| `tests/unit/datasets/test_cluster_split.py` | EDIT | Add `test_skewed_cluster_split` (1×80 + 9×~2) and `test_empty_val_warning`. |
| `tests/integration/confidence/test_sequence_only_distillation.py` | EDIT | Rename `test_predict_handles_n_orig_less_than_n_extended` → `test_predict_zeros_padded_logits_under_partial_mask`. |
| `tests/integration/confidence/test_validation_logging_extension.py` | EDIT | Drive ≥2 val batches; assert aggregated reliability table counts == sum across batches. |

No new top-level Hydra config. No checkpoint format changes. No package-layout shifts.

## 4. Class / function signature sketches

### 4.1 Sidecar cond padding

```python
def _pad_cond_to_n_ext(
    self, cond: torch.Tensor, mask_ext: torch.Tensor
) -> torch.Tensor:
    n_orig = cond.shape[1]
    n_ext = mask_ext.shape[1]
    if n_orig == n_ext:
        return cond
    if n_orig > n_ext:
        raise ValueError(
            f"cond has more residues ({n_orig}) than mask_ext ({n_ext}); "
            "cond_factory should never extend beyond the trunk's extended mask."
        )
    pad = cond.new_zeros((cond.shape[0], n_ext - n_orig, cond.shape[2]))
    return torch.cat([cond, pad], dim=1)
```

Called once in `_forward`, between `_compute_cond` and `self.head(...)`. Zero-pad is intentional: the head's AdaLN consumes cond at padded positions, but those positions are masked out by `mask_ext` upstream of any attention and by `orig_mask & plddt_mask` downstream of the head — gradients there are irrelevant. Assertion: `cond.shape[1] == mask_ext.shape[1]` after the helper.

### 4.2 Reliability aggregator

Replace `_last_val_{logits,labels,mask}` with three `torch.Tensor` buffers shape `(num_bins_ece,)`:

- `_rd_conf_sum`, `_rd_correct_sum`, `_rd_count`.
- `on_validation_epoch_start`: zero them on `self.device`.
- `validation_step` (after computing `logits/labels/mask`): compute per-bucket `(conf_sum, correct_sum, count)` for the current batch and `+=` into the accumulators. No `.cpu()`, no per-batch stash.
- `on_validation_epoch_end`: `self.all_gather(...)` (or `self.trainer.strategy.reduce(..., reduce_op="sum")`) on each of the three buffers, divide on rank 0, build the diagram tensor `(num_bins, 3) = (mean_conf, mean_correct, total_count)`, call `_emit_reliability_diagram`.

This drops the bf16-downcast / CPU-move discussion entirely — the "stash one batch" path is gone.

### 4.3 Rank-0 guard

```python
def _emit_reliability_diagram(self, diagram: torch.Tensor) -> None:
    if self.trainer is not None and self.trainer.global_rank != 0:
        return
    ...
```

### 4.4 Once-per-run `log_table` warning

Class-level `_log_table_warned: bool = False`. On first `log_table` exception: `logger.warning(f"log_table unavailable, falling back to .npy: {type(exc).__name__}: {exc}")` and set the flag. Subsequent failures: silent fallback. Reset of the flag is **not** needed — once-per-run is the contract.

### 4.5 `_cluster_aware_split` empty-val warning

After computing `val_rows`, if `len(val_rows) == 0`: `logger.warning(f"_cluster_aware_split: empty val split with cluster_column={self.cluster_column!r}, train_split={self.train_split}, n_clusters={len(unique_clusters)}, n_rows={len(full_metadata)}; consider lowering train_split or using more clusters.")` and proceed (caller decides — Lightning will emit its own error on empty dataloader, which is the correct surface).

### 4.6 sbatch script

Structure parallels `quality-graft/scripts/train_swissprot.sbatch`:

- SBATCH header: `--job-name=complexa_confdistill`, `--partition=h100`, `--gres=gpu:h100nvl:2`, `--ntasks-per-node=2`, `--nodes=1`, `--cpus-per-task=8`, `--mem=64G`, `--time=3-00:00:00`, logs to `logs/confdistill_%j.{out,err}`.
- Env: `PROJECT_ROOT=/mnt/storage01/home/schekmenev/projects/complexa-flex`, `DATA_ROOT=/netscratch/$USER/complexa-confdistill`.
- Venv staging: tar/untar `.venv` → `$DATA_ROOT/.venv`, `VIRTUAL_ENV` + PATH export (mirror quality-graft). If `.venv` already on a shared FS reachable from compute nodes, fall back to using it directly — gate on `$STAGE_VENV` env var, default 1.
- Stage `ckpts/complexa.ckpt` + `ckpts/complexa_ae.ckpt` → `$DATA_ROOT/ckpt/`.
- Stage AFDB monomer parquet → `$DATA_ROOT/afdb_cifs/metadata.parquet`. Export `DATA_PATH=$DATA_ROOT`.
- NCCL: `NCCL_DEBUG=INFO`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`, `NCCL_ASYNC_ERROR_HANDLING=1`. Identical to the quality-graft template.
- Run: `srun --kill-on-bad-exit uv run python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_swissprot trainer.devices=2 trainer.strategy=ddp trainer.precision=bf16-mixed`. Optional override `confidence.trunk_ckpt_path=$DATA_ROOT/ckpt/complexa.ckpt` `confidence.autoencoder_ckpt_path=$DATA_ROOT/ckpt/complexa_ae.ckpt`.
- Post-run: rsync run dir from scratch back to `$PROJECT_ROOT/ckpts/runs/`. Idempotent.

## 5. Hydra config edits

- `configs/dataset/unified/afdb_monomers.yaml`: append `- unicluster` under `columns_to_load`.
- `configs/dataset/unified/afdb_monomers_with_plddt.yaml`: explicitly redeclare `columns_to_load` with `unicluster` because Hydra list-merge over a `defaults` parent replaces the list — confirm with a Hydra-compose smoke test (already covered by existing config tests). Alternative: rely on parent list and add only via `_self_` order; the explicit redeclare is safer and 5 lines.
- `configs/nn/confidence/plddt_sequence_only_head.yaml`: remove `update_pair_repr_every_n: 1000000` line.

No defaults-list change, no `_global_` key change, no compose-tree restructuring.

## 6. Tests (TDD, written first)

Order: write failing tests, then implement, then re-run.

1. **`tests/integration/confidence/test_distillation_with_concat_features.py`** — new. Build a fake `Proteina.nn` whose `forward(batch)` returns `trunk_intermediates = {"s": (b, n_ext, ds), "z": (b, n_ext, n_ext, dz), "mask": (b, n_ext), "orig_mask": (b, n_ext) with last 5 zeros, "n_orig": n_ext - 5}` and whose `cond_factory(batch)` returns `(b, n_orig, dim_cond)` (concat features NOT applied at cond stage). Construct via `ConfidenceDistillationModule.from_components`. Assert: `module._forward(batch)` runs without exception; capture cond seen by head via a monkey-patched head wrapper; `cond.shape == (b, n_ext, dim_cond)`; padded slice is exactly zero.

2. **`tests/integration/confidence/test_distillation_ddp_smoke.py`** — new. `torch.multiprocessing.spawn` 2 ranks, `init_process_group("gloo", rank=..., world_size=2)`. On each rank: build fake `Proteina` + head (CPU), wrap in `DDPStrategy` via `L.Trainer(accelerator="cpu", devices=2, strategy="ddp")` or directly via `torch.nn.parallel.DistributedDataParallel`. Run one `training_step` from a 1-batch fake loader. Assert: returned loss is finite on both ranks; `head.parameters()` have grad, `proteina.parameters()` have `None` grad on both ranks. Skip under `if not torch.distributed.is_available()`.

3. **`tests/unit/datasets/test_cluster_split.py`** — edit.
   - `test_skewed_cluster_split`: parquet with 1 cluster of 80 rows + 9 clusters of 2 rows; `train_split=0.9`; assert no cluster spans both splits, total rows preserved, train size within `[0.85, 0.95] * total` (or assert the dominant cluster lands entirely on one side — which side is deterministic given the seed).
   - `test_empty_val_warning`: cluster sizes that force empty val (`train_split=0.99` over 2 clusters of size 50 each); use `caplog`/`loguru` capture to assert the warning string contains `"empty val split"`.

4. **`tests/integration/confidence/test_sequence_only_distillation.py`** — rename only. No semantic change.

5. **`tests/integration/confidence/test_validation_logging_extension.py`** — edit. Drive ≥2 val batches with distinct conf distributions; mock `lightning_logger.log_table`; assert it is called once at epoch end with `count` column summing to total masked residues across both batches (not just the last batch).

Existing 104 tests must remain green. Specifically: PR-5's reliability-diagram test now asserts aggregated counts — update assertion in that file.

## 7. Verification commands

```
uv run pytest tests/ -v                                      # full suite
uv run pytest tests/integration/confidence -v                # confidence integration
uv run pytest tests/unit/datasets/test_cluster_split.py -v   # split unit
bash scripts/run_smoke_train.sh                              # single-GPU sanity (PR-5)
uv run python -c "from hydra import initialize_config_dir, compose; \
    import os; initialize_config_dir(os.path.abspath('configs'), version_base='1.3'); \
    cfg = compose('confidence/distillation_swissprot'); print(cfg.data.datamodule.columns_to_load)"
```

User-side (manual, after merge):

```
sbatch scripts/train_confidence_swissprot.sbatch
```

## 8. Reviewer panel

- **`code-review-debug-complexity-expert`** (mandatory): cond-pad correctness (boundaries, dtype/device), DDP-reduce semantics in the reliability aggregator, no-regression on the 104-test baseline, idempotency of sbatch staging.
- **`ml-protein-architect`**: config edits (`columns_to_load`, sequence-only YAML) don't break compose-time; module boundary still respected (sidecar-only changes); naming matches `search_*`/`evaluate_*` style.
- **`ml-software-pytorch-jax-expert`**: DDP gloo CPU test correctness, `self.all_gather` vs `strategy.reduce` choice, Lightning 2.5.x checkpoint round-trip with the new buffers (`_rd_*` should be `register_buffer` or kept off the state dict explicitly), NCCL knobs are sound for h100nvl.
- **`generative-protein-scientist`**: zero-pad of `cond` at masked positions is semantically defensible for AdaLN at currently-monomer training (it is — those positions are masked); flag if it would corrupt future ipTM training.
- **`physics-statmech-md-dft-expert`**: sbatch resource sanity (mem, cpus-per-task, time), AF2 pLDDT distillation framing still consistent with the staging path (no implicit re-scaling).

## 9. Risk register

1. **sbatch staging fails on H100 node.** Severity: high (run blocked). Likelihood: medium (first time on the complexa side). Mitigation: idempotent `[ -d $ENV_LOCAL/bin ]` guard; explicit `set -euo pipefail`; pre-flight echo of `ls $DATA_ROOT` so failure mode is obvious in `*.err`. Pre-merge: dry-run the script's data-staging block locally with `SLURM_JOB_ID=0` overridden, verify each `cp -v` lands.
2. **DDP-aggregate reliability accumulator races.** Severity: medium (silent numerical wrongness on the diagram only — loss/ECE scalars use `sync_dist=True` already). Likelihood: low if we use `self.all_gather` on rank-summed buffers. Mitigation: implement reduction with `self.trainer.strategy.reduce(t, reduce_op="sum")`; gate diagram emission on `global_rank == 0`; unit-test in the gloo-DDP smoke that the post-reduce count equals the world-summed expected count.
3. **Zero-pad `cond` corrupts head's effective output at masked positions.** Severity: low for current monomer training (PR-5 has no `n_ext > n_orig`), latent for ipTM. Likelihood: zero in PR-6 production path (concat features never fire for the monomer config); the new test exists precisely to lock this in. Mitigation: assertion that head logits at padded indices match the result under any other constant cond fill — left as a doc note in the helper, not a test (would over-constrain). If the head ever consumes cond into a position-wise residual that isn't masked, this needs revisiting before ipTM lands.
4. **Skewed-cluster-size warning fires constantly on production parquet.** Severity: low (noisy log only). Likelihood: medium given heavy-tailed UniRef-style cluster distributions. Mitigation: warning fires only on `len(val_rows) == 0`, not on imbalance. If noise is observed on first run, raise the threshold to "val < 0.5% of total" rather than 0 — defer that calibration to a follow-up if needed.
5. **Lightning 2.5.x checkpoint round-trip on the new accumulator buffers.** Severity: medium (would silently break resume). Likelihood: medium. Mitigation: register the three `_rd_*` tensors as plain attributes (NOT `register_buffer`) so they are not persisted; reset in `on_validation_epoch_start`. Add an explicit assertion in `test_distillation_checkpoint_roundtrip.py` (PR-4) that `state_dict()` does not contain `_rd_` keys.

## 10. Out of scope

- ipTM / ipAE / ipLDDT heads (separate research thread; this PR only avoids tripping on `cond` shape when they arrive).
- RFdiffusion / Dayhoff transfer eval.
- Joint trunk+head training.
- A real multi-day training run — user invokes `sbatch` post-merge.
- Architectural cleanup of `cond_factory` to emit `n_ext`-length cond natively (would touch `feature_factory.py`, which is frozen for PR-6; sidecar workaround is sufficient until ipTM).
- Bumping Lightning past 2.5.x.

Hand to `ml-protein-architect` for implementation (TDD: tests in section 6 first, then code in section 3).
