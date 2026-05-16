# Confidence-Head Distillation in Proteina-Complexa — Enriched Task Spec

Status: **APPROVED — implementation may proceed**
Date: 2026-05-16
Branch base: `merge_quality_graft`
Authors: dispatched via `software-planning-architect`, `ml-protein-architect`, `generative-protein-scientist` subagents; synthesised by the main thread.

> This document is the synthesised output of three enrichment agents. It records the agreed design, the resolved decisions, and the sub-PR decomposition.

## 0. User-approved decisions (resolves §11)

| Q | Decision | Note |
|---|---|---|
| Q1 | **Option A — sidecar `ConfidenceDistillationModule`** | Zero edits to `proteina.py`. |
| Q2 | **`trunk_eval_t = 0.99`** | Avoids `t = 1.0` singularity. |
| Q3 | **`ckpts/complexa.ckpt`** (trunk), **`ckpts/complexa_ae.ckpt`** (autoencoder) | Loaded into the frozen `Proteina` inside the sidecar. |
| Q4 | **AF2 pLDDT scale `[0, 100]`**, 50 bins of width 2 | `plddt_to_bin(0)=0`, `plddt_to_bin(50)=25`, `plddt_to_bin(100)=49`. |
| Q5 | **`n_blocks = 4`**, `use_tri_attn = False`, `dropout = 0.1`, `n_heads = 16` | Slight up-scale from scientist's 3-block default per user. |
| Q6 | **SwissProt cluster-30% held-out only** in PR-5 | RFdiffusion / Dayhoff eval deferred to a follow-up PR. |
| Q7 | **Embed `t = 0.99` through the trunk's existing `FeatureFactory` time embedder; feed the resulting `cond` into every AdaLN inside the head's `ConfidenceTrunk`** | The trunk was trained with meaningful `t`-conditioning. We reuse the same `time_emb_*` features the trunk consumes at training time and route the embedded cond into the head's AdaLN blocks, so the head operates in the same conditioning regime the trunk was trained for. No refactor of `MultiheadAttnAndTransition`. |
| Q8 | **Confirmed.** Email `schekmenev@aithyra.at` after each PR merges and a final summary email when the whole task is done. Transport: `mail` / `sendmail` (both available on host). | No secrets, summary only, per `CLAUDE.md`. |

---

## 1. Goal

Port the trainable, light-weight *student* confidence head from
`/mnt/storage01/home/schekmenev/projects/quality-graft/` into
`/mnt/storage01/home/schekmenev/projects/complexa-flex/` (package
`proteinfoundation`).

Functional requirements (verbatim from the user):

1. Trainable light-weight confidence head, toggleable on/off via a single Hydra
   config flag.
2. Trains on per-residue AF2 pLDDT from ~240K SwissProt AF2 structures.
3. **Discard** the frozen heavy `BoltzConfidenceModule` teacher path — only
   the trainable student head is wanted.
4. The student head mimics the architecture of
   `quality_graft.models.student_head.StudentConfidenceHead` (pairformer-like
   stack reading `(s, z)` + LN + linear heads).
5. Minimum changes to existing complexa code.
6. Deliverable includes a working SLURM sbatch script.
7. Head must be built on a **base class** of confidence heads. Future heads
   (ipTM, ipAE, ipLDDT, …) will share a common light-weight trunk; only the
   final MLPs differ.
8. Workflow: TDD-via-subagents, parallel sub-PRs branched off
   `merge_quality_graft`, each going through the full
   plan → tests-first → implement → review-panel → merge cycle described in
   `CLAUDE.md`. Email `schekmenev@aithyra.at` when the task is done.

---

## 2. Non-goals

- No `BoltzConfidenceModule`, no Boltz checkpoint loading, no soft-label KL
  distillation (no teacher logits available).
- No new vendored Boltz package; we reuse complexa's existing pair-biased
  attention blocks.
- No changes to existing complexa flow-matching training, sampling, or
  evaluation paths when the head is disabled. With `confidence_head.enabled =
  False` (default), all existing configs and entry points are byte-identical
  to current `merge_quality_graft` HEAD.
- No port of `quality_graft/data/swissprot_datamodule.py` — we extend
  complexa's existing AFDB monomer dataset with a per-residue pLDDT transform.
- No joint training of trunk + head in PR-1..PR-6. Pure distillation only
  (trunk frozen). Joint training is a separate research thread, not in scope.
- No ipTM/ipAE/ipLDDT subclasses in this work — only the base class and the
  pLDDT head. The base class must be extensible to those heads cleanly.

---

## 3. Architecture decisions (agreed across agents)

### 3.1 Module layout

```
src/proteinfoundation/
  nn/confidence/
    __init__.py
    base.py           # BaseConfidenceHead, ConfidenceTrunk
    plddt_head.py     # PLDDTHead(BaseConfidenceHead)
    projections.py    # input projections from trunk dims to (target_s, target_z)
    registry.py       # build_confidence_head_from_cfg + register decorator
  confidence/
    __init__.py
    lightning_module.py   # ConfidenceDistillationModule (sidecar) — see §3.5
    losses.py             # masked_plddt_cross_entropy, plddt_to_bin, smooth_l1_on_expected
    metrics.py            # ported from quality_graft.training.metrics
    train_confidence.py   # Hydra entry point
configs/
  nn/confidence/
    base.yaml
    plddt_head.yaml
  training/
    confidence_distill.yaml
  confidence/
    distillation_swissprot.yaml   # top-level training config (composes above)
  dataset/unified/
    afdb_monomers_with_plddt.yaml
scripts/
  train_confidence_swissprot.sbatch
tests/
  unit/confidence/
    test_confidence_head_shapes.py
    test_confidence_head_masking.py
    test_confidence_head_equivariance.py
    test_plddt_binning.py
    test_metrics.py
    test_factory_registry.py
  integration/confidence/
    test_trunk_feature_extraction.py
    test_lightning_one_step.py
    test_flow_matching_loss_unchanged.py   # regression guard for default-off
```

### 3.2 Base class design

```
class ConfidenceTrunk(nn.Module):
    """Shared light-weight pair-biased stack reused by all heads.

    Built from complexa's MultiheadAttnAndTransition + PairReprUpdate +
    Transition. NOT a vendored Boltz pairformer.

    forward(s, z, mask, cond=None) -> (s_refined, z_refined)
    Both outputs LayerNormed and mask-zeroed; z symmetrised inside the trunk.
    """

class BaseConfidenceHead(nn.Module):
    output_keys: tuple[str, ...]
    bin_edges: Tensor  # registered buffer, per-head
    trunk: ConfidenceTrunk

    def forward(s_trunk, z_trunk, mask, cond=None) -> dict[str, Tensor]: ...
    def compute_loss(predictions, batch, mask) -> dict[str, Tensor]: ...
    def logits_to_expected_value(logits) -> Tensor: ...

class PLDDTHead(BaseConfidenceHead):
    output_keys = ("plddt_logits",)
    # final LN(s) -> Linear(token_dim, num_plddt_bins)
```

**Future-head hooks (must be present in the base class from day one):**

- `z` is *always* carried through and normed (do not gate behind
  `predict_pde`). ipAE / ipTM / PDE heads need it.
- `z` is symmetrised (`z + z.transpose(1,2)`) inside the trunk, not in any
  head.
- `chain_id: [b, n]` is a first-class trunk input (zero-filled for monomers).
  Required for future ipTM/ipAE/ipLDDT heads.
- Each head owns its own `bin_edges` buffer and
  `logits_to_expected_value()` — base class agnostic to bin choice.

### 3.3 Architecture defaults for `PLDDTHead`

User-approved defaults:

- `n_blocks = 4` pairformer-like blocks (matches `quality_graft` student;
  ~7-10M params).
- `n_heads = 16`.
- `use_tri_mult = True` (cheap, keeps the base class future-compatible).
- `use_tri_attn = False` (heavy, marginal for per-residue pLDDT-only).
- `dropout = 0.1` (down from `quality_graft`'s 0.2; trunk frozen, less risk
  of overfitting on 240K).
- `num_plddt_bins = 50`, bin width 2 over `[0, 100]` (AF2 scale).
- `predict_pde = False`, `predict_resolved = False` (keep toggles in the base
  class, default off).

**Justification.** Per-residue pLDDT on monomers is dominated by single-rep
context plus *local* pair geometry. Triangle attention's value is in
pair-context heads (ipTM/ipAE/PDE) — keep it off for pLDDT. The base class
admits adding `use_tri_attn=True` for those heads later.

### 3.4 Dim contract

Default **complexa-native** dims: `(token_dim=768, pair_repr_dim=256)` — the
confidence trunk runs in the same space the main trunk produces. Projection
layers exist in `projections.py` but reduce to near-identity when source and
target dims match. The registry hides the dim choice from `Proteina`. We can
flip to Boltz dims `(384, 128)` later via config (e.g. if a paper requires
that comparison) without touching code.

### 3.5 AdaLN conditioning — reuse the trunk's time embedder

The trunk was trained with **meaningful `t`-conditioning** via AdaLN —
`MultiheadAttnAndTransition`'s AdaLN modules consume a `cond` vector
produced by `FeatureFactory` from the `time_emb_*` features defined in
`proteinfoundation.nn.feature_factory.seq_cond_feats`. The right move is to
reuse that embedder when querying the head's trunk:

1. The sidecar builds `cond` once at construction (or per-batch) by passing
   `t = 0.99` through the **same** `FeatureFactory` instance the frozen
   trunk uses (or an instance configured with identical parameters), so the
   embedded conditioning is bit-compatible with the trunk's training
   regime.
2. The `ConfidenceTrunk` then forwards `(s, z, mask, cond)` into each
   `MultiheadAttnAndTransition` block exactly as the main trunk does — the
   head's pair-bias blocks consume the **same** AdaLN cond as the main
   trunk would at `t = 0.99`.

Concretely: the head's `forward(s, z, mask, cond)` accepts an external `cond`
tensor of shape `[b, n, dim_cond]`. The sidecar Lightning module computes
`cond` from `t = 0.99` via the existing time embedder and passes it both
into the frozen trunk forward *and* into the head. This keeps the head in
the same conditioning regime the trunk was trained for, requires zero
refactor of complexa's existing blocks, and avoids the previously
considered "learned constant" hack (which would have ignored a signal the
trunk was trained to use).

### 3.6 Reuse vs vendor

**Reuse complexa-native blocks**
(`proteinfoundation.nn.modules.attn_n_transition.MultiheadAttnAndTransition`,
`proteinfoundation.nn.modules.pair_update.PairReprUpdate`,
`proteinfoundation.nn.modules.seq_transition_af3.Transition`). Boltz's
`PairformerModule` is **not vendored**; it would drag a large dependency
closure (AF2 / kalign / mmCIF tooling), force a `(384, 128)` projection, and
duplicate operators we already have.

### 3.7 Trunk feature exposure

Only invasive code edit (apart from the optional `Proteina` integration):
add `expose_intermediates: bool = False` flag to `LocalLatentsTransformer`
(v1 and v2). When `True`, the forward returns an extra key:

```
nn_out["trunk_intermediates"] = {
    "s": seqs,           # [b, n_extended, token_dim]
    "z": pair_rep,       # [b, n_extended, n_extended, pair_dim]
    "mask": mask,         # [b, n_extended] bool
    "orig_mask": orig_mask, # [b, n_orig] bool
    "n_orig": n_orig,
}
```

When `False` (the default), `forward(input)` is **bit-identical** to today,
enforced by `test_flow_matching_loss_unchanged.py`. v2 hook lands in the same
PR as v1 — both must stay compatible.

### 3.8 Loss

- **Primary:** masked cross-entropy on 50 AF2-pLDDT bins.
- **Auxiliary (weight 0.1):** smooth-L1 on the bin-expected value vs
  continuous pLDDT. Calibrates the head's *expected value* — the quantity
  downstream filters and SMC reward models actually consume.
- **No** soft-KL distillation (no teacher).
- **No** focal loss, no Gaussian-smoothed CE in PR-1.
- `label_smoothing` knob exposed (default 0.0).

### 3.9 Data pipeline

- Reuse `proteinfoundation.datasets.structure_data.StructureDataModule` and
  the existing `atom37_transforms` extension hook.
- Add **one** new transform `AddPLDDTFromBFactor` (in
  `proteinfoundation.datasets.transforms`) that reads atom B-factor from
  parsed AF2 CIFs and writes:
  - `batch["plddt_residue"]` — `[n] float32` in `[0, 100]` (AF2 scale, raw
    B-factor; per-residue mean over atoms — AF2 stores the same pLDDT in
    every atom of a residue).
  - `batch["plddt_bin"]` — `[n] int64`, `floor(plddt / 2).clamp(0, 49)`.
    Bin width 2 over `[0, 100]`.
  - `batch["plddt_mask"]` — `[n] bool` (1 where valid).
- New dataset config `configs/dataset/unified/afdb_monomers_with_plddt.yaml`
  extends `afdb_monomers.yaml`, **drops the `plddt > 70` filter** (we want
  the full distribution for distillation), appends `AddPLDDTFromBFactor` to
  `atom37_transforms`. Same parquet metadata file; same staging pipeline.
- Sanity check inside `AddPLDDTFromBFactor`: assert `bfactor.max() > 1.5` on
  first batch (trip if AF2-DB ever stores normalised pLDDT instead of
  B-factor).

### 3.10 Held-out evaluation set (in PR-5)

Per user decision Q6: **SwissProt cluster-30% only** in PR-5. RFdiffusion /
Dayhoff-BackboneRef eval is a deliberate follow-up — RFdiffusion does not
ship cached per-residue AF2 pLDDT, so adding it would require generating
~100 AF2 refolds offline.

Inside PR-5:

- SwissProt held-out set: ~5-10% of the 240K corpus, held out **at the
  cluster level** (MMseqs2 30% identity) using the existing `unicluster`
  metadata column if present in the AFDB parquet, otherwise a deterministic
  hash on UniProt accession with a seed recorded in the config.

Validation metrics, beyond the standard accuracy/MAE/Pearson/Spearman:

- **ECE** over the 50-bin distribution (reliability diagram).
- **MAE / Pearson stratified by pLDDT bucket** (< 50, 50-70, 70-90, 90+) —
  bulk-vs-tail story.
- **Sequence-only control head** (diagnostic, not product) — flagged for a
  follow-up PR.

Deferred for a follow-up PR (not in PR-5):

- RFdiffusion / Dayhoff-BackboneRef refold + AF2 + held-out eval.
- CASP/CAMEO temporal split.
- Designed-binder held-out set (the deployment-distribution check).

---

## 4. Disagreement to resolve: sidecar vs monolithic Lightning module

The two architecture agents disagree on where the confidence-distillation
training loop lives. Both have working precedents; the choice is reversible
in spirit but pins different PR scopes.

### Option A — **Sidecar `ConfidenceDistillationModule`** (per `software-planning-architect`)

- New `L.LightningModule` in `proteinfoundation/confidence/lightning_module.py`.
- Owns: a *frozen* `Proteina` (loaded once from
  `pretrain_ckpt_path`) + a trainable `BaseConfidenceHead`.
- Train step: `with torch.no_grad():` run `proteina.call_nn(batch)`, extract
  intermediates, run head, compute loss.
- **No edits to `proteina.py` at all.**
- Own entry point `proteinfoundation/confidence/train_confidence.py`.

Pros: zero diff to flow-matching code paths; cleanest review surface; sbatch
is its own thing. Cons: duplicates a tiny bit of dataloader / autoencoder
plumbing.

### Option B — **Monolithic — head inside `Proteina`** (per `ml-protein-architect`)

- Optional `self.confidence_head` attribute in `Proteina.__init__`, gated by
  `cfg_exp.confidence_head.enabled` (default `None`).
- Optional loss branch in `Proteina.training_step` mixing in
  `confidence_head.compute_loss(...)` via `cfg_exp.confidence_head.loss_weight`.
- Optional freezing in `on_train_epoch_start` / `on_train_batch_start` when
  `training_mode == "pure_distillation"`.
- Uses the existing `train.py` entry point.

Pros: one Lightning module, one trainer, one checkpoint, one entry point;
"head is part of the model" is conceptually clean. Cons: invasive — entangles
two optimisers, two loss families, two training modes inside a single
`Proteina`. Risks behavioural drift on existing flow-matching configs unless
the off-branch is bit-tested in regression.

**My recommendation: Option A (sidecar).** It maximally respects constraint
5 ("minimum changes to existing complexa code") — `proteina.py` is read-only
in this work. It also keeps the new training entry point independent, so a
confidence-training job failing cannot fall back to running flow-matching.
**Question 1 in §11 asks the user to confirm.**

---

## 5. Integration surface in `Proteina`

If **Option A (sidecar):** the only edits to existing complexa code are:

1. `src/proteinfoundation/nn/local_latents_transformer.py` — add
   `expose_intermediates: bool = False` flag (§3.7).
2. `src/proteinfoundation/nn/local_latents_transformer_v2.py` — same.
3. **No edits to `proteina.py`.**

If **Option B (monolithic):** additionally:

1. `src/proteinfoundation/proteina.py` — guarded `__init__` branch, guarded
   `training_step` branch, guarded `on_train_epoch_start` /
   `on_train_batch_start` re-`eval()` hooks. New config keys
   `confidence_head.*`. Total: ~50 lines, all behind `if
   self.confidence_head is not None:` guards.

---

## 6. Hydra config tree

New configs (paths and intent):

- `configs/nn/confidence/base.yaml` — shared trunk hyperparams.
- `configs/nn/confidence/plddt_head.yaml` — pLDDT head config composing
  `base`.
- `configs/training/confidence_distill.yaml` — optimizer/scheduler/training
  mode for the head.
- `configs/confidence/distillation_swissprot.yaml` — top-level training
  config composing the head + the dataset + the new training settings.
- `configs/dataset/unified/afdb_monomers_with_plddt.yaml` — AFDB monomers
  with pLDDT transform appended.

Existing top-level configs are **not** modified. The new flag is visible only
when the user composes the new top-level config; legacy configs return
`cfg.get("confidence_head", None) is None` and the head is never
instantiated.

Suggested defaults (subject to user review):

```yaml
# configs/nn/confidence/plddt_head.yaml
defaults: [base]
_target_: proteinfoundation.nn.confidence.plddt_head.PLDDTHead
name: plddt
num_plddt_bins: 50
bin_min: 0.0
bin_max: 100.0      # AF2 scale
predict_pde: False
predict_resolved: False
loss:
  type: "ce"
  label_smoothing: 0.0
  smooth_l1_weight: 0.1
  smooth_l1_on_expected: True
trunk:
  _target_: proteinfoundation.nn.confidence.base.ConfidenceTrunk
  token_dim: 768
  pair_repr_dim: 256
  n_blocks: 4
  n_heads: 16
  use_tri_mult: True
  use_tri_attn: False
  use_qkln: True
  dropout: 0.1
  # cond is computed by the sidecar via the trunk's FeatureFactory time
  # embedder at t = 0.99 and passed in at forward() time.
  expects_external_cond: True
```

```yaml
# configs/training/confidence_distill.yaml
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
max_epochs: 50
precision: bf16-mixed
gradient_clip_val: 1.0
```

---

## 7. Sub-PR decomposition

All PRs branch off `merge_quality_graft` and target merge into
`merge_quality_graft`. Mandatory reviewer on every PR:
`code-review-debug-complexity-expert`.

| PR  | Branch | Parallelism | Scope | Reviewer panel |
|---  |---     |---         |---    |---             |
| PR-1 | `feat/confidence-trunk-internals-hook` | **PARALLEL** with PR-2 | `expose_intermediates` flag on `LocalLatentsTransformer` v1 + v2. Off-by-default regression guard. | code-review, ml-protein-architect, ml-software-pytorch-jax-expert |
| PR-2 | `feat/confidence-plddt-transform` | **PARALLEL** with PR-1 | `AddPLDDTFromBFactor` transform + `plddt_to_bin` helper + `afdb_monomers_with_plddt.yaml`. | code-review, ml-protein-architect, structural-biology-binder-expert |
| PR-3 | `feat/confidence-head-base-class` | **SEQUENTIAL-AFTER** PR-1 (uses trunk dims) | `ConfidenceTrunk`, `BaseConfidenceHead`, `PLDDTHead`, registry, head Hydra configs. No training loop yet. | code-review, ml-protein-architect, ml-software-pytorch-jax-expert, generative-protein-scientist |
| PR-4 | `feat/confidence-distillation-lightning` | **SEQUENTIAL-AFTER** PR-1, PR-2, PR-3 | Sidecar `ConfidenceDistillationModule` (Option A) or `Proteina` integration (Option B), loss + metrics, optimizer + LR schedule, Hydra entry point. | code-review, ml-protein-architect, ml-software-pytorch-jax-expert, generative-protein-scientist |
| PR-5 | `feat/confidence-heldout-eval-and-smoke` | **SEQUENTIAL-AFTER** PR-4 | Held-out eval set (§3.10) + smoke train (1000 steps on a small subset, single GPU). Memory probe at `n=512, b=2`. | code-review, ml-software-pytorch-jax-expert, ml-protein-architect, generative-protein-scientist |
| PR-6 | `feat/confidence-sbatch-multi-gpu` | **SEQUENTIAL-AFTER** PR-5 | `scripts/train_confidence_swissprot.sbatch` paralleling `quality-graft/scripts/train_swissprot.sbatch`. 2-GPU DDP bring-up, 1-epoch full run. | code-review, ml-software-pytorch-jax-expert, ml-protein-architect, physics-statmech-md-dft-expert |

Parallelism summary: **PR-1 and PR-2 run in parallel on separate branches.**
PR-3 starts as soon as PR-1 merges. PR-4 needs all three. PR-5 and PR-6 are
sequential.

Each PR follows the project's TDD workflow: plan ↦ tests written first ↦
implementation ↦ review panel ↦ revise until unanimous ↦ merge. Stuck-PR
escape per `CLAUDE.md` §"Stuck-PR escape hatch".

---

## 8. Risk register

| # | Risk | Sev | Lik | Mitigation |
|---|---|---|---|---|
| 1 | **Distribution shift** — SwissProt-monomer head asked to score Proteina-Complexa-generated complexes. Per-residue pLDDT on a monomer is not even the right quantity for a complex (no interface). | High | High | Stage the work: pLDDT now, ipLDDT/ipAE/ipTM later on AF2-multimer data. Treat pLDDT head as internal calibration target, not a binder filter. Track designed-binder held-out metric. |
| 2 | **pLDDT scale / B-factor convention mismatch.** AF2-DB stores per-residue pLDDT × 100 in B-factor; some pipelines normalise to [0,1]. Silent factor-of-100 mismatch ⇒ head trains with low loss, predicts garbage on real AF2 outputs. | High | Med | Sanity assert in `AddPLDDTFromBFactor` (first-batch `bfactor.max() > 1.5`). Unit test `test_plddt_binning.py`: `bin(0)==0, bin(50)==25, bin(100)==49`. |
| 3 | **Head learns to predict pLDDT from sequence alone.** Residue-type is a strong shortcut (Pro/Gly low, Leu high). With full structural inputs the head can still take the easy path. | High | Med | Mandatory **sequence-only control head** as a *diagnostic* in PR-5 (own short ablation). If val Spearman gap is < 0.05, structural signal is not being used. |
| 4 | **DDP unused-params** when trunk is frozen. | Med | High | Don't instantiate head when disabled (eliminates the case). In `pure_distillation` mode, the trunk has `requires_grad=False` so it's already excluded from optimiser. Safety net: `find_unused_parameters=True` only in joint mode. |
| 5 | **bf16 numerical instability** in pair ops on long sequences. Triangle multiplication is brittle in bf16. | Med | Med | Validate numerically against fp32 on a 384-residue example. If divergence > a few %, force the head to fp32 with autocast disabled. |
| 6 | **Off-by-default contract violation** — the `expose_intermediates` hook accidentally changes flow-matching behaviour. | High | Low | `test_flow_matching_loss_unchanged.py` runs one training step before vs after PR-1; bit-exact equality required. |
| 7 | **Concat-features cropping** — the trunk extends `n_orig → n_extended` for motif/target/ligand. Loss must be computed on `[:n_orig]` using `orig_mask`. | Med | Med | Pass both masks; head's `compute_loss` zeroes out concat positions. Tested in `test_confidence_head_masking.py`. |
| 8 | **Frozen-but-still-in-train-mode** — Lightning's `model.train()` flips dropout/BN-equivalents back on every epoch. | Med | High | Port the defensive `on_train_epoch_start` / `on_train_batch_start` `eval()` re-enforcement from `quality_graft.training.lightning_module`. |
| 9 | **Lightning 2.5 checkpoint round-trip** with optional head — must tolerate `confidence_head.*` keys absent in legacy ckpts. | Med | Low | `strict=False` on the head submodule; smoke test in `test_lightning_one_step.py`. Do not bump Lightning. |

---

## 9. Verification strategy

Unit tests (TDD, written first per CLAUDE.md):

- `test_confidence_head_shapes.py` — `(b=2, n=37)`, plddt logits shape
  `(2, 37, 50)`. Cover `n_blocks ∈ {1, 2, 3}` and `update_pair_repr ∈ {T, F}`.
- `test_confidence_head_masking.py` — head output on padded positions ==
  output after pad removal + re-pad. Targets Risk 7.
- `test_confidence_head_equivariance.py` — head output **invariant** under
  random rotation + translation of input coordinates (head consumes a
  rotation-equivariant trunk).
- `test_plddt_binning.py` — `plddt_to_bin(0)==0`, `plddt_to_bin(50)==25`,
  `plddt_to_bin(100)==49`. Targets Risk 2.
- `test_metrics.py` — acc/MAE/Pearson/Spearman on hand-computed tiny tensors.
- `test_factory_registry.py` — Hydra `instantiate` on `plddt_head.yaml`
  returns `PLDDTHead`. Unknown name raises.

Integration tests:

- `test_trunk_feature_extraction.py` — `expose_intermediates=True` on v1 +
  v2 returns correctly-shaped `(s, z, mask)`. Mask matches input.
- `test_lightning_one_step.py` — one-step train on a 2-protein toy batch:
  loss finite, only head params have grad (in `pure_distillation` mode),
  checkpoint round-trip equality.

Regression test:

- `test_flow_matching_loss_unchanged.py` — one step of `Proteina.training_step`
  before vs after PR-1 on a fixed seed; bit-exact equality. Targets Risk 6.

Manual smoke verification:

- PR-5: single-GPU 1000-step run, `val/plddt_accuracy > 0.10` exit
  criterion (random > uniform-50 baseline = 0.02).
- PR-6: 2-GPU 1-epoch run on full data, NCCL bring-up confirmed,
  `val/plddt_accuracy > 0.40` target.

---

## 10. Final sbatch script

Located at `scripts/train_confidence_swissprot.sbatch`. Parallels
`quality-graft/scripts/train_swissprot.sbatch`:

- `#SBATCH --partition=h100`, `--gres=gpu:h100nvl:2`, `--ntasks-per-node=2`.
- 3-day wall time, 16 G mem, 8 cpus/task.
- Stages `metadata.parquet` and trunk checkpoint to `/netscratch/$USER`.
- NCCL knobs (`NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`).
- `bf16-mixed`, DDP, `find_unused_parameters=false` in pure-distillation mode.
- Output checkpoints to `$DATA_ROOT/ckpt/confidence_runs/<run_id>/`, rsynced
  back to `ckpt/confidence_runs/` at end.
- Entry: `srun python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_swissprot ...`

---

## 11. Open questions — RESOLVED

All eight blocking questions are resolved. See §0 for the decision table.

---

## 12. Files (absolute paths)

Read-only references:

- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/student_head.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/adaptor.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/quality_graft.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/training/lightning_module.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/training/metrics.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/data/plddt_utils.py`
- `/mnt/storage01/home/schekmenev/projects/quality-graft/scripts/train_swissprot.sbatch`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/proteina.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/local_latents_transformer.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/local_latents_transformer_v2.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/modules/attn_n_transition.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/modules/pair_update.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/datasets/structure_data.py`
- `/mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/afdb_monomers.yaml`

To be created (in subsequent PRs after approval):

- `src/proteinfoundation/nn/confidence/{__init__,base,plddt_head,projections,registry}.py`
- `src/proteinfoundation/confidence/{__init__,lightning_module,losses,metrics,train_confidence}.py`
- `src/proteinfoundation/datasets/transforms.py` (extended)
- `configs/nn/confidence/{base,plddt_head}.yaml`
- `configs/training/confidence_distill.yaml`
- `configs/confidence/distillation_swissprot.yaml`
- `configs/dataset/unified/afdb_monomers_with_plddt.yaml`
- `scripts/train_confidence_swissprot.sbatch`
- `tests/unit/confidence/*.py`, `tests/integration/confidence/*.py`,
  `tests/regression/test_flow_matching_loss_unchanged.py`

To be edited:

- `src/proteinfoundation/nn/local_latents_transformer.py` (PR-1, `expose_intermediates`)
- `src/proteinfoundation/nn/local_latents_transformer_v2.py` (PR-1, same)
- `src/proteinfoundation/proteina.py` — only if **Option B** is chosen in Q1.

---

**End of spec. APPROVED for implementation as of 2026-05-16.**
