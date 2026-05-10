# 03 — Masked-token pretraining for reciprocal-space voxels

Self-supervised pretraining of a transformer on the n=32 reciprocal-space voxel
representation from `analysis/dev/coarsening_strategy_analysis.ipynb`. Per the
**recommended primary** verdict in
[`00_final_review.md`](./00_final_review.md), this plan is the lead candidate
in the 2-week dev-set bake-off.

## Framing — what the dev set is and is not

The 394-protein dev set is a **proof-of-concept and method-selection harness**,
not a representation-quality benchmark. The dev run's goal is not a useful
embedding on 394 proteins; it is to prove that the MAE objective (1) produces
a working gradient signal with the inductive biases the spec promises
(per-token CE on Wilson-E quantile bins, directional information actually
used); (2) surfaces and controls its named failure modes (radial-shortcut
leakage, trivial neighbour-copy at low mask radius, conditioning leakage,
per-head gradient starvation); (3) exhibits a scaling signature consistent
with being data-hungry at n=394.

Quoting `00_final_review.md`:

> "the bake-off measures mechanism (is the loss landscape behaving as
> designed? are the named failure modes visible and controllable? are the
> engineering primitives correct?) and predictable scaling (is the
> loss-vs-data curve still descending at n=394?), not absolute probe
> accuracy."

Outcome-shaped probe targets (SG top-1, cell-volume R², anisotropy R²) from
the previous draft are demoted to **informational read-outs** in §7 and are
explicitly *not* gate criteria. This plan and [`01_vae_voxel.md`](./01_vae_voxel.md)
compete on the same Day-14 mechanism table; the MAE is the recommended
primary and the VAE is the structured fallback per the final review.

## TL;DR

- Sparse voxel-set tokens (~2500/protein), not the dense 32 768-voxel grid.
- Vocab: **64 equal-population bins on Wilson E** + 4 special tokens, eff. 72.
  Count channel is a separate 5-bucket embedding added to the intensity embed.
- Objective: **MAE-style asymmetric masked-token modelling**. Mask 60 % in
  spatial blocks of Chebyshev radius r=2 in (h,k,l). Bidirectional encoder,
  shallow decoder, predict bin tokens at masked positions only.
- Encoder 8L × d=384 × 6h, RoPE-3D, ~16 M params. Decoder 4L × d=192, ~6 M.
- Cell, space-group, Wilson profile injected as 3 `[COND]` prefix tokens with
  per-token dropout (Wilson p=0.5; cell/SG p=0.2; see §3.3).
- Single A100 40 GB, Week 1 + Week 2 to a go/no-go decision.

Architectural choices are unchanged from the previous draft; the success
criteria are what change.

---

## 1. Tokenisation

### 1.1 Sparse set, not dense grid

n=32 → 32 768 voxels; 7.5 % occupancy. After ASU canonicalisation only the
irreducible wedge is unique — **median ~2500 nonempty voxels/protein**.
Tokenise only canonical-wedge nonempty voxels with their integer
(i, j, k) ∈ [0, 32). A separate occupancy head (§4) recovers
where-are-reflections.

Plan around `max_seq_len = 1024`. Verifying p99 ≤ 1024 on the dev set is
a Week-1 primitive (§9); if p99 > 1024, raise to 1536 with random
spatial-block sub-sampling.

### 1.2 Intensity vocabulary

- **Target**: Wilson-normalised E (η=0.11 on E vs 0.52 on |F|).
- **Bins**: **64 equal-population quantiles**. log E has std ≈ 1, range
  ±4σ, so 64 bins ≈ 0.13σ — under noise floor. Flat marginal; CE
  meaningful; "predict-mode" baseline = 1/64.
- **Edges**: fit on dev train fold; persist as `bin_edges_v1.npy`.
- **Empty voxels**: excluded from input; recovered by occupancy head.
- **High-E tail**: 63 quantile + 1 explicit top-tail bin if the top
  quantile bin spans > 1 decade in E (centric-rich SGs have heavier
  top-tail; decided Week 1).

### 1.3 Count vocabulary

5 buckets: `{1, 2, 3–4, 5–8, 9+}`. Separate learnt embedding (dim 384),
added elementwise to the intensity embed; also target for the 5-way
aux head (§4).

### 1.4 Special tokens

`[CLS]`, `[MASK]`, `[PAD]`, `[COND]` × 3 (cell / SG / Wilson). No
`[BOS]`/`[EOS]`/`[SEP]` — input is a set. Effective vocab 68, padded to 72.

### 1.6 Positional encoding — RoPE-3D

Each (i, j, k) ∈ [0, 32) encoded with rotary embedding split per axis
across head_dim. Respects translational symmetry; generalises to finer
grids (n=64) without retraining; decoupled from input order (shuffled
every step). `[CLS]` and `[COND]` get learnt positional embeddings.
Correctness is a Week-1 primitive.

---

## 2. Masking objective and order — MAE

**MAE.** AR over a scan order injects an arbitrary inductive bias on
set-shaped data; XLNet pays 2× compute for two-stream attention; BERT MLM
spends compute on trivially copy-through tokens at 7.5 % occupancy. MAE's
encoder/decoder asymmetry is the standard win on sparse spatial data.

- **Ratio**: **60 %** (vision MAE uses 75 %; reciprocal-E is more
  neighbour-correlated, so 60 % keeps signal).
- **Unit**: spatial block masking in (h, k, l). Pick centres uniformly
  among input voxels; mask within Chebyshev radius r=2 (5×5×5 cube). Stop
  at 60 % coverage. Single-voxel masking is trivially solvable by
  neighbour copy.
- **Loss positions**: CE on 64-bin softmax, **only at masked positions**.
  No encoder-side `[MASK]` tokens, so BERT-style 80/10/10 is not used.

---

## 3. Architecture

### 3.1 Encoder

| field | value |
|------|------|
| layers | 8 |
| d_model | 384 |
| heads | 6 (head_dim 64) |
| FFN | 1536 (4×) |
| activation | GELU |
| dropout | 0.1 attn / 0.1 FFN |
| norm | pre-norm RMSNorm |
| stochastic depth | 0.1 |
| positional | RoPE-3D on voxel tokens; learnt on `[CLS]`, `[COND]` |
| params | ~16 M |

Input: visible voxel tokens (intensity + count embeds) + `[CLS]` + 3 `[COND]`.

### 3.2 Decoder

| field | value |
|------|------|
| layers | 4 |
| d_model | 192 |
| heads | 6 (head_dim 32) |
| FFN | 768 |
| input | encoder outputs (384→192 linear) ∪ shared learnt `[MASK]` token at each masked position, RoPE-3D coordinates restored |
| head | linear → 64-way softmax |
| params | ~6 M |

### 3.3 Conditioning

Three `[COND]` prefix tokens:

1. `[COND_cell]` — MLP on 6 normalised cell scalars → 384.
2. `[COND_sg]` — embedding over 65 chiral SGs → 384.
3. `[COND_wilson]` — 1D-CNN on the 100-bin Wilson profile → 384.

**Leakage guard**: per-token dropout — Wilson p=0.5; cell p=0.2;
SG p=0.2. Wilson-only dropout was insufficient (per
`00_final_review.md`'s sharpest critique: a capable transformer learns
an in-network Wilson estimator from cell + SG + visible voxels).
Whether triple-dropout suffices is what M3 (§7.1) measures.

### 3.4 Padding

Pad variable-length inputs to 1024 with attention mask. >1024 voxels:
random spatial-block sub-sampling (whole blocks).

---

## 4. Training objective

1. **Intensity** (primary, weight 1.0): 64-way CE at masked positions
   with triangular soft targets — 0.8 / 0.075 / 0.025 on bin / ±1 / ±2.
2. **Occupancy** (aux, weight 0.3): binary head over the dense 32³ grid;
   sample 4 096 random (i,j,k) per protein per step; small MLP on
   RoPE-encoded coordinate cross-attends into encoder output; BCE.
3. **Count** (aux, weight 0.2): 5-way CE at masked positions.

**Per-head grad-norm logging is on from step 1.** Named failure mode
(occupancy aux starving intensity) — diagnosable on dev.

---

## 5. Training recipe

| field | value |
|------|------|
| optimiser | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| lr | 3e-4 peak, linear warmup 2k → cosine to 1e-5 |
| weight decay | 0.05 (none on biases / norms / embeddings) |
| batch | 256 proteins (grad-accum to fit) |
| grad clip | 1.0 |
| precision | bf16 mixed, fp32 master |
| epochs (dev) | 200 (~75 k steps) |
| EMA | 0.999 |

**Splits**: protein-level CD-HIT 30 % cluster split, 80/10/10. Dev val is
narrow (~20 clusters); val metrics carry bootstrap CIs. Known limitation
of the harness, not a defect.

**Curriculum**: mask ratio 0.4 → 0.6 over the first 10 % of steps;
block radius r=1 → r=2 on the same ramp.

---

## 6. Embedding extraction (informational at dev scale)

Attention pool over encoder outputs with one learnable query
(Set-Transformer / Perceiver style), 384-dim. Downstream consumes the
*full token sequence* via cross-attention; the pooled vector exists for
dev probes only.

---

## 7. Evaluation — two-tier

### 7.1 Mechanism diagnostics (gate the go/no-go decision)

These are the metrics that decide whether the MAE goes to full-PDB scale.
They are derived from the bake-off framing in
[`00_final_review.md`](./00_final_review.md) §"Two-week empirical bake-off
plan".

**M1. Loss descent shape.** Val masked-CE descends and saturates within
the 200-epoch budget; no divergence; no first-25-ep plateau. Concretely:
by epoch 25, val CE has dropped ≥ 30 % from the log(64)=4.16 random floor
toward the irreducible floor set by adjacent-bin smoothing alone.

**M2. Anisotropic-vs-radial reconstruction control — the headline
diagnostic for this method.** Compute masked-token reconstruction Pearson R
on E across val masked positions, then the same Pearson R on a
**radially-shuffled control**: for each masked voxel replace its target
with a random voxel drawn from the same `d*` shell (binning into the 100
Wilson-profile shells), preserving radial profile but destroying directional
content. The model is *not* refit; only the *targets* against which R is
computed change. With identical mask placements in both passes, the
encoder sees the same visible context; only the masked-position *targets*
differ — which isolates directional vs radial content.

The MAE is exhibiting its promised inductive bias iff

`R(E) − R(radially_shuffled_E) ≥ 0.10`.

A radial-decay-only model achieves equal R on both. **The margin is the
diagnostic; the absolute value is not.** M2 < 0.10 means the MAE is
failing on its core promise.

**M3. Conditioning-leakage gap dynamics.** Adversarial linear probe:
predict per-voxel Wilson-Σ(d*) from the attention-pooled embedding with
all three `[COND]` tokens hidden. Gate: leakage R² < 0.5 *with* the §3.3
dropout settings. Equally important: the **dynamics** of the gap between
the full-encoder probe (cell volume, SG, anisotropy, `[COND]` visible) and
the `[COND]`-only probe (no encoder, MLP on the 3 cond tokens). A
non-zero, non-decreasing gap that grows with training is the signature
of the encoder learning reflection content beyond conditioning; a flat or
shrinking gap signals uncontrolled leakage.

**M4. Named failure-mode surfacing.** Each named failure mode must be
*observable* and *controllable* by a sweep knob (Day 11–12, §9):
mask-via-positional-encoding leakage (ablate RoPE-3D vs learnt-3D);
shortcut-via-Wilson (`[COND_wilson]` dropout ∈ {0.0, 0.5, 0.8}, M3 must
respond); per-head gradient starvation (log per-head grad norms; if
occupancy dominates intensity by >3× for ≥10 epochs, re-tune 0.3 — the
diagnostic *fires*, not that we land first-pass); r=1 vs r=2 vs r=3
block masking (val CE must move; r=1 trivial, r=3 hard).

**M5. Scaling-curve slope at n=394.** Re-run the chosen MAE configuration
on n ∈ {100, 200, 394} proteins (subsets nested). Plot val masked-token CE
vs `log(n)`. Fit a 3-point trend. Pass: monotonic improvement with negative
second derivative, slope at n=394 not yet flat. A still-descending curve
licenses scaling to 150 k.

### 7.2 Informational read-outs (not gate criteria)

Frozen attention-pool linear probes for SG top-1, log cell-volume R²,
d_min R², anisotropy ratio R²; reconstruction NRMSE (predicted-bin
centroid vs ground-truth E). Interesting if high; the previous draft's
thresholds (SG top-1 ≥ 0.50, cell-volume R² ≥ 0.85, anisotropy R² ≥ 0.60)
are **not the bar** at dev scale — chasing them risks selecting a method
that overfits the dev probes rather than scales.

---

## 8. Risks and mitigations — diagnosable failure modes

Each row pairs a named failure with the dev-set instrument that surfaces
it and the knob that controls it.

| failure mode | diagnostic | mitigation knob |
|------|------|------|
| Bin edges don't transfer to full PDB | Refit at scale; `bin_edges_vN.npy` versioned. | versioned artefact |
| Sparse-set ordering bias | Shuffle ablation: invert input order, no metric should move. | RoPE-3D + per-step shuffle |
| MAE too easy (neighbour-copy) | Block-radius r∈{1,2,3} sweep — M4. r=1 trivial, r=2 hard. | radius r=2; curriculum r=1→r=2 |
| Wilson-profile leakage | M3 (adversarial probe) + M2 (radial shuffle) — both must respond to `[COND_wilson]` dropout. | Wilson 0.5; cell/SG 0.2 |
| `[COND]`-only shortcut | M3 gap dynamics. | conditioning-dropout sweep |
| Friedel duplicates | Dataloader unit test (dev set is merged). | dedupe before voxellising |
| Train/val homologue leak | CD-HIT 30 % cluster split. | manifest persisted |
| Seq-len tail | p99 verified Week 1; per-epoch truncation rate logged. | `max_seq_len=1024`, raise to 1536 |
| High-E tail mis-quantised | Top-bin E width inspected Week 1. | 63 quantile + 1 top-tail if span > 1 decade |
| Per-head grad starvation (occupancy dominates intensity) | Per-head grad-norm logging — M4. | re-tune 0.3 if dominance > 3× for ≥10 ep |
| ASU canonicalisation byte-instability | Week-1 primitive (random SG op + recanonicalise no-op). | pin `gemmi`; unit test |

---

## 9. Implementation plan — Week 1 + Week 2 to go/no-go

### Week 1 — engineering primitives, tokeniser, first descent

Days 1–2 — shared canonical-ASU + Wilson-Σ-with-shrinkage + side-channel
pipeline (also used by [`01_vae_voxel.md`](./01_vae_voxel.md)). Verify
sequence-length tail (p99 ≤ 1024).

Day 3 — MAE tokeniser; persist `bin_edges_v1.npy`. Verify bin marginals
uniform on train (≤ 1 %), val (≤ 3 %).

Day 4 — CD-HIT 30 % cluster split, persisted. The narrow dev val (~20
clusters) is expected and disclosed; all val metrics carry bootstrap CIs.

Day 5 — untrained-baseline probes (random-init MAE + `[COND]`-only) to set
the floor for the M3 gap diagnostic.

**Engineering-primitives checklist (gates Week 2 — quoted from
[`00_final_review.md`](./00_final_review.md) §"Two-week empirical bake-off
plan", Week 1 checklist):**

- [ ] ASU canonicalisation is byte-stable under random non-identity SG
      operations on all 394 dev proteins.
- [ ] Sparse-COO loader returns deterministic token order under a fixed
      seed; RoPE-3D keys are stable under shuffle.
- [ ] Wilson-Σ shrinkage estimator stabilises bin assignments under
      bootstrapped reflection subsampling on the small-protein subset
      (KL ≤ 0.05 between bootstrap replicates).
- [ ] Symmetry-augmentation orbit is a faithful sampler of the SG:
      applying `g` then `g⁻¹` is a no-op on tokens and on side-channels.
- [ ] Wilson-E quantile bin marginals are uniform on train (≤ 1 %) and
      val (≤ 3 %).
- [ ] Empirical voxel-distance E–E correlation at r=2 ≤ 0.3 on the dev
      set (block radius is sized to correlation length; was Week-1 open
      question 4 in the previous draft).

Week-2 work does not start until every box is checked.

### Week 2 — train, surface failure modes, decide

Days 6–8 — train MAE 200 epochs at §5 hyperparameters. Log per-step:
masked-CE, top-1/top-5 (informational), attention-pool variance, **per-head
grad norms**, per-shell reconstruction Pearson R on E.

Day 9 — **M2** on the EMA checkpoint.
Day 10 — **M3** at EMA + epoch-100 midpoint to capture gap dynamics.
Days 11–12 — **M4 sweep**: `[COND_wilson]` dropout ∈ {0.0, 0.5, 0.8};
cell+SG dropout ∈ {0.0, 0.2}; block radius r ∈ {1, 2, 3}; mask ratio ∈
{0.45, 0.60, 0.75}. Three runs each, 100 epochs.
Day 13 — **M5 scaling**: chosen config on n ∈ {100, 200, 394} (subsets
nested), fixed compute (80 epochs at full size, scaled to equalise
gradient steps). Plot val masked-CE and val Pearson R vs `log(n)`.
Day 14 — **cross-method table** (§10) against the parallel VAE run.

### Go / no-go (single sentence — verbatim from `00_final_review.md`)

The MAE primary track is a go for full-PDB scale if and only if
(i) the engineering-primitives checklist is fully ticked,
(ii) the masked-token CE descends and saturates within the 200-epoch
budget without diverging or plateauing in the first 25 epochs (M1),
(iii) the anisotropic-vs-radial reconstruction margin is ≥ 0.10 (M2),
(iv) the conditioning-leakage R² can be pushed below 0.5 by the dropout
sweep (M3),
(v) at least one of the named failure modes is observed *and* controlled
by a sweep knob (M4),
(vi) the n ∈ {100, 200, 394} scaling curve on val masked-CE is
monotonically improving with the trend at n=394 not yet flat (M5).

A bake-off fail (M1 plateaus, M2 < 0.10, M3 uncontrollable, or M5 flat) is
a Case A or Case B per `00_final_review.md` §"If the bake-off fails": Case
A pivots to the VAE secondary as the new primary; Case B (mechanism
ambiguous, curve still descending) proceeds to a cheap full-PDB
discriminator run.

---

## 10. Day-14 cross-method comparison table

The MAE and VAE plans report into the **same table shape** on Day 14, so
the comparison is apples-to-apples. Both plans contain this table; values
are filled at the end of Week 2.

| diagnostic | MAE result | VAE result | pass criterion |
|------|------|------|------|
| M1. Loss descent shape (first 25 ep CE drop fraction; final-epoch slope) | | | descends, no first-25-ep plateau |
| M2. Anisotropic-vs-radial margin (R(E) − R(radially_shuffled), val) | | | ≥ 0.10 |
| M3a. Conditioning-leakage R² (Wilson-Σ from embedding, `[COND]` hidden) | | | < 0.5 achievable by dropout sweep |
| M3b. `[COND]`-gap dynamics (full-encoder probe minus `[COND]`-only probe, slope vs epoch) | | | non-zero, non-decreasing |
| M4. Named failure modes surfaced and controlled (count / total) | | | ≥ 1 surfaced *and* knob-controlled |
| M5. Scaling-curve slope at n=394 (Δ val-CE per Δ log n) | | | < 0 (still descending) |
| Reconstruction Pearson R on E (val, masked positions) | | | informational |
| Cell-volume linear-probe R² (informational) | | | informational |
| SG top-1 (informational) | | | informational |
| Anisotropy-ratio linear-probe R² (informational) | | | informational |

For the VAE, the M2 row uses the VAE-equivalent: per-voxel reconstruction
Pearson R on E vs the same on radially-shuffled targets. For the VAE, the
M4 row prioritises the FiLM decorative-vs-effective test and posterior
collapse (see [`01_vae_voxel.md`](./01_vae_voxel.md) §7.1). The M3 and M5
rows are computed identically across plans.

The Day-14 deliverable is this table, filled, plus the single-sentence
go/no-go for each method, plus a recommendation of one primary to scale and
one secondary to scale alongside per `00_final_review.md`.

---

## 11. Open questions

1. Is the radial-shuffle control sufficient? It destroys within-shell
   directional content but preserves radial decay; plan a per-shell
   cosine shuffle at full scale, not on dev.
2. Is triple-dropout (Wilson 0.5, cell/SG 0.2) sufficient, or does
   full-PDB need a learned adversarial head? M3 dynamics indicates.
3. Count-head ablation: in M4 sweep.
4. Two-stage MAE → VAE-with-MAE-init (opportunity 2 of
   `00_final_review.md`): out of scope for the bake-off.
