# Confidence-head distillation subsystem — full detail

A sidecar Lightning module distils AF2 per-residue pLDDT and per-pair PAE from the frozen complexa trunk into a trainable light-weight student head. **Does not modify** `proteinfoundation.proteina` or the main `train.py` entry point.

Read this file before editing anything under `src/proteinfoundation/nn/confidence/` or `src/proteinfoundation/confidence/`, or before touching a confidence-distill config or sbatch.

## Package layout + layering rule

[src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/) holds heads + trunk + registry + pure-function `_losses.py` / `_metrics.py`. [src/proteinfoundation/confidence/](../../src/proteinfoundation/confidence/) holds the Lightning module + Hydra entry point + back-compat shims; new code imports from `nn.confidence._{losses,metrics}` directly. Heads register via `@register_confidence_head(name)`. **Layering rule:** `nn/confidence/` must NOT import from `confidence/`.

## Entry point

`python -m proteinfoundation.confidence.train_confidence --config-name=confidence/<config>`. SLURM-launch via [scripts/train_confidence_swissprot.sbatch](../../scripts/train_confidence_swissprot.sbatch), [scripts/train_confidence_teddymer_pae.sbatch](../../scripts/train_confidence_teddymer_pae.sbatch), or sibling sbatches.

## Datasets

SwissProt monomers via [configs/dataset/unified/afdb_monomers_with_plddt.yaml](../../configs/dataset/unified/afdb_monomers_with_plddt.yaml) (AF2 `[0, 100]`, 50 bins of width 2, optional 30%-cluster split via `cluster_column: unicluster`). Teddymer dimers via [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml).

## Loss / metrics

Per-head recipe lives on the head (e.g. pLDDT default `0.9*masked_CE + 0.1*SmoothL1(EV)` with `ce_weight` / `ev_weight` / `label_smoothing` as Hydra `loss:` kwargs). Reduction `sum(loss * mask) / mask.sum().clamp_min(1)`. Each head implements `compute_loss_and_metrics(out, batch, mask_eff, *, stage) -> (loss, log_dict)`; the Lightning module prefixes log keys with `{train,val}/{head.output_name_root}/`. Validation logs accuracy, MAE, Pearson, Spearman, stratified MAE, equal-width / equal-mass ECE, and a per-bucket reliability diagram (rank-0 only). PaeHead also logs 10 interface metrics (i_pae, min_ipae, i_ptm, i_ptm_energy, six ipSAE variants — colabdesign-parity ports).

## Trunk hook + AdaLN

`LocalLatentsTransformer` has opt-in `expose_intermediates: bool = False`; when `True`, `nn_out["trunk_intermediates"]` carries `(s, z, local_latents, mask, orig_mask, n_orig)`. Default `False` is bit-identical to legacy (`tests/regression/test_flow_matching_loss_unchanged.py`). Sidecar reuses the trunk's `FeatureFactory` time embedder at `t = trunk_eval_t = 0.99`, zero-padded along the residue axis if concat features extend the sequence.

## Pair-repr symmetrisation is the head's job

`ConfidenceTrunk.forward` does NOT symmetrise `z`. Symmetric pair-heads (PDE, ipLDDT, ipTM) prepend `z = 0.5 * (z + z.transpose(-3, -2))` in their `_predict`; asymmetric heads (`PaeHead`) consume `z` directly. `PlddtHead` is `s`-only.

## WandB logging

Wired via [configs/logging/wandb.yaml](../../configs/logging/wandb.yaml) (`@package _global_`, project `confidence-distillation`). Compose with `- /logging/wandb@logging` and set a per-config `wandb_tags:`. Entry point builds `WandbLogger` via `_build_wandb_logger(cfg)`; honours `WANDB_MODE=disabled` and `log_wandb: false`. Top-level `run_name` is the WandB `id` and `name`, so re-launching the same `run_name` *resumes* the run. Future entry points must reuse this fragment — embedding wandb keys in training YAMLs is the anti-pattern.

## Rank-0-only loguru gate

Any Hydra-launched DDP entry point must call `_gate_loguru_to_rank0()` as the FIRST line of `main()`. Reads `LOCAL_RANK`, `NODE_RANK`, `RANK` (the last covers `torchrun`); calls `logger.remove()` if any is non-zero. Reference: [src/proteinfoundation/confidence/train_confidence.py:29-34](../../src/proteinfoundation/confidence/train_confidence.py#L29-L34).

## Multi-GPU DDP — pair `find_unused_parameters=true` with `static_graph=true`

All confidence configs ship `DDPStrategy(find_unused_parameters=true, static_graph=true)` as a structured yaml. Two failures compose without both flags: (i) frozen-trunk params don't feed the head loss (`PaeHead` uses `z`, `PlddtHead` uses `s`) → `find_unused_parameters=false` raises "parameters not used"; (ii) the head's embedded `ConfidenceTrunk` ([src/proteinfoundation/nn/confidence/base.py](../../src/proteinfoundation/nn/confidence/base.py)) uses **reentrant** `torch.utils.checkpoint` in pair-update layers, so `find_unused_parameters=true` alone makes the reducer mark the same param ready twice. The frozen complexa trunk runs under `torch.no_grad()` so its ckpts never enter backward — the reentrant-ckpt site that trips DDP is the head's trunk copy. `static_graph=true` is what lets the reducer tolerate the double-ready signal; switching to non-reentrant ckpt changes the autograd graph shape and is NOT a drop-in.

## CUDA allocator — export `PYTORCH_ALLOC_CONF=expandable_segments:True`

Reentrant ckpt in [src/proteinfoundation/nn/modules/pair_update.py](../../src/proteinfoundation/nn/modules/pair_update.py) plus Teddymer's heterogeneous L (250–550) fragments the caching allocator after ~900 steps and OOMs with multi-GiB reserved-but-unallocated. Goes alongside the NCCL block, *before* `srun python -m ...`; see [scripts/train_confidence_teddymer_pae.sbatch](../../scripts/train_confidence_teddymer_pae.sbatch). `static_graph=true` is reducer-side; `expandable_segments` is allocator-side; both required.

## Resume-from-checkpoint

Every confidence-distill config exposes `resume_ckpt_path: null`, forwarded to `trainer.fit(..., ckpt_path=...)`. Orthogonal to `resume_id` (WandB id): `resume_ckpt_path` restores Lightning state (weights + optimizer + LR + step + epoch); `resume_id` re-attaches the WandB run. Sbatch contract: `RESUME_CKPT_PATH` env var → Hydra override; usage `RESUME_CKPT_PATH=/abs/path.ckpt sbatch --export=ALL,RESUME_CKPT_PATH scripts/train_confidence_*.sbatch`. SwissProt sbatch lacks this wiring — add when next resuming SwissProt.

## Joint training via `MultiHeadConfidence` wrapper

[src/proteinfoundation/nn/confidence/multi_head.py](../../src/proteinfoundation/nn/confidence/multi_head.py), registered as `multi_head`. `nn.ModuleDict` of child heads; runs trunk once per batch and dispatches `(s, z, mask, cond, local_latents)` to every child's `_predict`. The shared `ConfidenceTrunk` projects + LN's `local_latents` (8-dim) and adds as a mask-zeroed residual to `s` at trunk entry. `MultiHeadLoss` aggregates `total = sum(w_i * L_i)`. Two Hydra weighting layers: (i) **within-head** `loss.ce_weight` / `loss.ev_weight`; (ii) **across-head** `loss_weights: {head: scalar}` (multiplies the head's total; `0.0` is a hard short-circuit). Wrapper asserts `expected_trunk_eval_t` matches and every child has a non-empty `output_name_root`. **DDP footgun:** `weight: 0.0` still runs `_predict`, so its params enter the autograd graph with no gradient → `find_unused_parameters=False` raises at backward. Escapes: remove from `children:`, or rely on `static_graph=True` (already mandatory above).

## Teddymer filter is geometry-only

Supervised pool is `interface_length > 10` at [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml). **Confidence-based pre-filters are a selection-bias antipattern**: the head's targets are per-residue pLDDT and per-pair PAE, so filtering on aggregates of those same quantities (`avg_int_plddt`, `avg_int_pae`) truncates the label distribution by the label itself. AF2-multimer, Boltz-1/-2, and Chai-1 filter on data-source quality (resolution, identity clustering, homology) but never on the model's own confidence; Boltz-2 names the antipattern. Only filter by geometric properties independent of the AF2 confidence map. `complexa_filter` is still on `dimers.parquet` but unused. Contract pinned by [tests/unit/datasets/test_teddymer_dataset_config.py::test_yaml_filter_pins_geometry_only_threshold](../../tests/unit/datasets/test_teddymer_dataset_config.py).

## LayerNorm-bias-leak invariant

Any `Sequential(Linear(d_in, d_token, bias=False), LayerNorm(d_token))` projecting an auxiliary signal into a masked sequence representation and added as a residual **must** multiply post-LN by `mask_f` before the add — even if the upstream input is zero at padded positions. At init `beta = 0` makes `LN(0) = 0` look fine, but trained drift gives `LN(0) = beta` and lands the bias in padded `s` rows, contaminating valid `(i, j)` cells via `PairReprUpdate` outer products + triangle multiplication summing over padded `k`. Reference: [src/proteinfoundation/nn/confidence/base.py:146-147](../../src/proteinfoundation/nn/confidence/base.py#L146-L147). Regression contract: [tests/integration/confidence/test_confidence_trunk_local_latents_proj.py::test_mask_multiply_guards_against_trained_layernorm_bias_leak](../../tests/integration/confidence/test_confidence_trunk_local_latents_proj.py) — writes `LN.bias = 0.3` and asserts forwards differing only at padded positions are bit-identical everywhere. Any future projection into `s` or `z` must follow `* mask` after the LN, and pin the invariant with a non-init test.
