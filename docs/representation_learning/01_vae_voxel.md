# 01 — VAE for Reciprocal-Space Voxel Grids

**Status:** design spec, ready for implementation.
**Scope:** VAE on the per-protein-normalised n=32 voxel grid + side-channels
from `analysis/dev/coarsening_strategy_analysis.ipynb`, producing a compact
latent. Per the **recommended secondary** verdict in
[`00_final_review.md`](./00_final_review.md), this is the structured
fallback in the 2-week bake-off — runs in parallel with the MAE
([`03_masked_lm.md`](./03_masked_lm.md)) on the shared pipeline, on equal
footing as a method-selection candidate, with the MAE as recommended primary.

Analysis findings (η ≈ 0.11 on Wilson-E at n=32; p99 PCA eigenvalue ratio
17.6; ~88 % of representatives outside canonical reciprocal ASU; Wilson
normalisation 10–75× stronger on E than `|F|`) are constraints.

---

## Framing — what the dev set is and is not

The 394-protein dev set is a **proof-of-concept and method-selection
harness**, not a representation-quality benchmark. The previous draft set
R-factor and probe targets that read as if 394 proteins were the bench;
those targets are demoted to **informational read-outs** in §5.2. The
VAE-on-dev goal is to prove the conditional-VAE machinery (FiLM,
free-bits, symmetry loss, log-normal NLL on E²) produces a working
gradient signal with the promised inductive biases, and to surface and
control its named failure modes — *not* to produce a useful 64-d
embedding on 394 proteins.

Quoting `00_final_review.md`:

> "the bake-off measures mechanism (is the loss landscape behaving as
> designed? are the named failure modes visible and controllable? are
> the engineering primitives correct?) and predictable scaling (is the
> loss-vs-data curve still descending at n=394?), not absolute probe
> accuracy."

Two named VAE failure modes the dev harness must surface:
(a) **decorative FiLM conditioning** — `z` re-encodes what conditioning
should be soaking up, indicating conditioning is going through the
motions but not doing its job;
(b) **posterior collapse** — large fractions of latent dims with KL near
zero, indicating the encoder has stopped routing information through `z`.
Both have first-class diagnostics in §5.1.

Architecture preserved from the previous draft; only success criteria
change. The "wrong shape for downstream" critique applies at full-PDB
scale and is a *deployment* concern, not a *bake-off* concern.

---

## 1. Data pipeline

### 1.1 Input tensor

Dense `(C, n, n, n)` float32 grid, `n=32`, `C=3`:

| channel | content | rationale |
|---|---|---|
| 0 | `mean E²` over voxel (Wilson-normalised `\|F\|²`) | scale-free target |
| 1 | `log(1 + count)` | density / occupancy proxy |
| 2 | `occupancy mask ∈ {0, 1}` | hard mask; supervises occupancy head |

On-disk format stays sparse-COO (`mean log|F|²`, `count`) at
~19 KiB/protein. The loader scatters COO → dense `(2, 32, 32, 32)`,
converts channel 0 to `mean E²` via the per-protein Wilson profile
(§1.5), and derives `log1p(count)` and the binary mask. Dense training
storage is `32³ × 3 × 4 B ≈ 384 KiB` — trivial. Sparse-conv kernels buy
nothing at ~7.5 % occupancy; **use dense conv, keep on-disk sparse**.
(The transformer-on-set route is the MAE's remit; the VAE keeps dense
conv because its reconstruction signal is a sharper falsifiability
probe than the MAE's CE-on-bins.)

### 1.2 Side-channels

- **`cell_metric ∈ R⁶`** — unique entries of `G = AAᵀ` in Å² (log-scaled
  diagonal, raw off-diagonal, standardised over train).
- **`sg_token ∈ {0,…,64}`** — index into 65 chiral SGs → embedding `R^{16}`.
- **`wilson_profile ∈ R^{100}`** — 100-bin radial profile of
  `⟨|F|²⟩(d*)`, log-standardised; small MLP inside the encoder.

### 1.3 ASU canonicalisation

Voxellise **after** mapping every reflection to the reciprocal ASU via
`gemmi.ReciprocalAsu(sg).to_asu(hkl, ops)`. Deduplicate
post-canonicalisation (E² averaged, counts summed). Bounding box in
(h,k,l) post-canonicalisation, voxellised at n=32. Use full bounding box
(wedge shape varies across SGs, breaks 3D-CNN translation equivariance).
Precompute `bool[32,32,32]` ASU mask per SG.

ASU canonicalisation byte-stability is a **Week-1 primitive** (§7),
shared with the MAE pipeline.

### 1.4 Augmentations

Exact symmetries of the reciprocal lattice with the protein's SG:

1. **Friedel flip** `(h,k,l) → (-h,-k,-l)`, p=0.5.
2. **Point-group rotations**; sample uniformly (identity included).
3. **Lattice-vector permutation** only when cell edges agree to 0.5 %.

Augmentations transform `G ← R G Rᵀ` so conditioning stays consistent
(runtime assertion required). No Gaussian noise on E², no random crops.

### 1.5 Wilson normalisation — target

Voxel target is **`E²`** (η=0.11 on E vs 0.52 on |F|; radial decay
already captured by Wilson side-channel — storing it twice wastes
capacity and biases toward radial cues).

Robust per-protein Σ estimator with shrinkage toward the global mean
profile (weight `1/(1+n_shell/50)`): bin `d*` into 30 shells (≥ 50
reflections each, merging low-count shells), compute `⟨|F|²⟩`, smooth
3-point MA, interpolate. `E² = |F|²/Σ(d*)`, clipped at 99.9 percentile.
Bootstrap stability is a Week-1 primitive (§7).

---

## 2. Model architecture

### 2.1 Encoder — 3D CNN

`32³` is small; vanilla 3D CNN with FiLM is right.

- **Stem**: `Conv3d(3 → 64, k=3, p=1)`, GroupNorm(8), SiLU.
- **Block 1–3** (32³→16³→8³→4³): two `ResBlock3D(C)` + strided
  `Conv3d(C → 2C, k=4, s=2, p=1)`, channels 64→128→256→384.
- **Head**: global avg pool → `R^{384}` → two `Linear(384 → 64)` for
  `μ, log σ`.

`ResBlock3D` = `Conv3d-GN-SiLU-Conv3d-GN` + skip; FiLM `(γ,β)` between
second GN and skip-add. ~6 M params.

### 2.2 Latent dimension — 64

**64**. 32 underfits high-anisotropy tail (p99 ratio 17.6); 128 overfits
394 examples. (Full-PDB may want 128; out of scope here.)

### 2.3 Decoder — mirror 3D CNN

- `Linear(64 + d_ctx → 384 × 4³)`, reshape `(384, 4, 4, 4)`.
- Three transposed-conv stages (384→256→128→64), each preceded by two
  `ResBlock3D` + FiLM, ending at `(64, 32, 32, 32)`.
- **Occupancy head**: `Conv3d(64 → 1, k=1)`, sigmoid per voxel.
- **Amplitude head**: `Conv3d(64 → 2, k=1)` → `(μ_E2, log σ_E2)`,
  log-normal NLL per voxel.

### 2.4 Conditioning — FiLM at every block + bottleneck concat

`c ∈ R^{128} = MLP_cell(cell_metric) ⊕ Embed_sg(sg) ⊕ MLP_wilson(profile)`
(64 + 16 + 48). Produces FiLM `(γ_l, β_l)` for every ResBlock in encoder
and decoder, AND concat at decoder bottleneck (context fed twice —
standard collapse fix). Conditioning dropout p=0.1 (CFG-style).

**Whether FiLM is "doing its job" is a first-class diagnostic, not an
assumption** — see §5.1 (M3).

---

## 3. Loss

### 3.1 Reconstruction

- **Occupancy**: per-voxel BCE on `count > 0`, positive class weight ~12.5×.
- **Amplitude** (occupied only): voxel-mean E² is ~log-normal by CLT;
  `L_amp = mean_occupied(0.5 ((log E²_target − μ_E2)/σ_E2)² + log σ_E2)`.

### 3.2 KL — β-VAE with warmup + free bits

Prior `N(0, I)` on `R^{64}`. Per-dim free bits
`L_KL = Σ_d max(λ, KL_d(q||p)), λ = 0.5 nats/dim`. β schedule: linear
warmup `β: 0 → 1` over first 5 epochs, constant thereafter. **β-warmup +
free bits are the named mitigations for posterior collapse**; per-dim KL
logged from step 1.

### 3.3 Symmetry loss

`L_sym = mean_voxel((D(E(g·x), c) − g⁻¹·D(E(x), c))²)`, sampling random
non-identity `g` from the SG point group per element. Weight 0.1.

### 3.4 Total loss

`L = L_occ + L_amp + β(t)·L_KL + 0.1·L_sym + 0.01·(c·z)²`. The
orthogonalisation term is a *soft* mitigation for conditioning leakage;
it does not replace the dev-set diagnostic (§5.1).

---

## 4. Training recipe

- **Optimiser**: AdamW, lr 3e-4, wd 0.01, β=(0.9, 0.95).
- **Schedule**: cosine to 1e-5 over 150 epochs (dev), 1k-step warmup.
  30 % of the MAE compute budget per the final review.
- **Batch**: 256 on one 24 GiB GPU.
- **Precision**: bf16 mixed, fp32 master. **Grad clip**: 1.0.
  **EMA** 0.9995 at eval.

### 4.1 Splits

Shared CD-HIT 30 % cluster split with the MAE for apples-to-apples. Dev
val ~20 clusters; bootstrap CIs throughout. Known harness limitation,
not a defect.

### 4.2 Dataloading

LMDB cache; augmentations on GPU post-batch; precompute per-SG rotation
matrices. 8 workers, pinned memory, prefetch 4.

---

## 5. Evaluation — two-tier

### 5.1 Mechanism diagnostics (gate the go/no-go decision)

These are the metrics that decide whether the VAE goes to full-PDB scale.
They are derived from the bake-off framing in
[`00_final_review.md`](./00_final_review.md) §"Two-week empirical bake-off
plan" and §"VAE-side mechanism test".

**M1. Loss descent shape.** L_amp and L_KL descend monotonically after
β-warmup (epoch 5); no diverging KL; no β-warmup-induced L_amp spike
that fails to recover by epoch 25. Concretely: val L_amp at epoch 25 is
≥ 30 % below its epoch-5 post-warmup baseline.

**M2. Anisotropic-vs-radial reconstruction control — the headline
diagnostic that the VAE has learnt directional content, not just radial
profile.** Per-voxel reconstruction Pearson R on E across val occupied
voxels, then the same R on a **radially-shuffled control**: for each
occupied voxel, replace target with a random voxel from the same `d*`
shell (binning into the 100 Wilson-profile shells); this preserves
radial profile but destroys directional pattern. Encoder sees the same
input both passes; only the *targets* against which R is computed
differ.

VAE exhibits the promised inductive bias iff

`R(E) − R(radially_shuffled_E) ≥ 0.10`.

Radial-decay-only model achieves equal R on both (the Wilson
side-channel already encodes radial decay). **The margin is the
diagnostic; its absolute value is not.** Same M2 shape as the MAE's;
the difference is per-voxel reconstructed E vs masked-token bin
centroid.

**M3. FiLM decorative-vs-effective test — the headline diagnostic that
conditioning is doing its job.** Train two variants ~3k steps each,
identical hyperparameters except:

- **A (with-cell)**: cell to FiLM in encoder and decoder + bottleneck
  concat (the spec).
- **B (decoder-blind)**: cell withheld from decoder (FiLM in encoder
  only; no bottleneck concat). Encoder still sees cell so it *can* route
  it into `z`; decoder cannot read it off the side-channel.

Linear-probe cell-volume from `μ` for each. Proof that FiLM is doing its
job:

`R²(B) − R²(A) ≥ 0.15`.

When conditioning is decorative, both probes succeed equally because `z`
carries cell either way; when conditioning is *effective* (A), `z` is
freed of cell because FiLM absorbs it, so probe on A is *worse*. A gap
below 0.05 means FiLM is decorative — method-fail signature, surfaceable
in 3 days on the dev set, exactly as the harness is meant to do.
(Source: `00_final_review.md` §"VAE-side mechanism test".)

**M4. Posterior collapse — the second VAE failure mode.** Track per-dim
KL across training. Two signatures:

- per-dim collapse: any dim with `KL < 0.01 nats` for ≥ 30 consecutive
  epochs;
- active-units fraction at end: fraction with `KL > 0.1 nats`. Healthy
  VAE under λ=0.5 free-bits should have active-units ≥ 0.5 (≥ 32 of 64).

Pass: active-units ≥ 0.5 at end; no dim collapsed > 30 epochs
post-warmup. M4 fail is fixable by raising λ to 0.75 or extending β
warmup; the *diagnostic firing* is the deliverable.

**M5. Conditioning-leakage gap dynamics.** Adversarial probe: predict
Wilson-Σ from `μ` with side-channels hidden. Gate: leakage R² < 0.5 with
cond-dropout 0.1 + orthogonalisation. Track the **gap** between
full-encoder probes (cond visible) and side-channel-only probes through
training; non-zero non-decreasing gap = `z` carries reflection content
beyond conditioning.

**M6. Scaling-curve slope at n=394.** Re-run chosen config on
n ∈ {100, 200, 394} (subsets nested). Plot val L_amp and val
reconstruction R on E vs `log(n)`. Pass: monotonic, negative second
derivative, slope at n=394 not yet flat — licenses scaling.

### 5.2 Informational read-outs (not gate criteria)

R-factor and Pearson R on `|F|`; linear probes on cell volume, `d_min`,
SG top-1/top-5, anisotropy; cluster purity ARI/NMI; generation
diversity at fixed `c`. Useful — R-factor is a sharper *physical*
falsifiability probe than the MAE's CE-on-bins (the reason the final
review keeps the VAE around at all) — but **none gates** at dev scale.
The previous "R-factor ≤ 0.45 or abort" threshold survives only as a
**sanity floor**: a run with R-factor ≥ 0.55 on val (worse than crude
binning) is diagnostically broken in a way the M-series should already
have caught, and the diagnosis is the deliverable.

---

## 6. Risks and mitigations — diagnosable failure modes

Each row pairs a named failure with the dev-set instrument that surfaces
it and the knob that controls it.

| failure mode | diagnostic | mitigation knob |
|------|------|------|
| **Posterior collapse** | M4 (per-dim KL, active-units) | free-bits λ=0.5 → 0.75 on fail; β-warmup 5 ep |
| **Decorative FiLM** | M3 (A vs B probe gap) | cond-dropout 0.1; `(c·z)²` weight 0.01, raise if marginal |
| Conditioning leakage into `z` | M5 (adversarial probe + gap dynamics) | `(c·z)²` weight; cond-dropout sweep |
| Sparsity collapse (decoder all-empty) | occupancy F1 from ep 1; abort if < 0.5 after 5 ep | positive class weight ~12.5× |
| Per-protein norm breaks cross-protein comparability | documented contract | cell-tensor conditioning (deployment constraint, not fix) |
| Anisotropy/cell leakage into `z` | M5 + M3 combined | `(c·z)²` regulariser |
| Noisy Wilson Σ on small proteins | Week-1 bootstrap KL ≤ 0.05 | shrinkage `1/(1+n_shell/50)` |
| Augmentation/cell mismatch | runtime assertion `R G Rᵀ` to 1e-5 | rotate `G` with augmentation |
| ASU canonicalisation byte-instability | Week-1 primitive | pin `gemmi`; unit test |
| Six-term loss mis-calibration | per-term loss + grad-norm logging from step 1 | re-tune weights only after named-failure surfaces |

---

## 7. Implementation plan — Week 1 + Week 2 to go/no-go

### Week 1 — primitives, baseline, first descent

Days 1–2 — **shared with the MAE plan**: canonical-ASU + Wilson-Σ-shrinkage
+ side-channel pipeline. Voxellisation reproduces analysis η/NRMSE on a
10-protein subset to within 5 %. LMDB cache.

Days 3–4 — encoder + decoder + occupancy + amplitude heads. Smoke run with
conditioning zeroed; verify L_amp descends and occupancy F1 > 0.5 by ep 5.
Shared CD-HIT 30 % split.

Day 5 — wire FiLM, free-bits, β-warmup, symmetry loss, orthogonalisation.
20-ep smoke run; verify per-dim KL logging, all loss terms descending.

**Engineering-primitives checklist (gates Week 2 — from
[`00_final_review.md`](./00_final_review.md) §"Two-week empirical bake-off
plan"):**

- [ ] ASU canonicalisation byte-stable under random non-identity SG ops
      on all 394 dev proteins.
- [ ] Sparse-COO loader deterministic; voxel scatter bit-reproducible.
- [ ] Wilson-Σ shrinkage stable under bootstrap subsampling on small
      proteins (KL ≤ 0.05).
- [ ] Symmetry-orbit faithful: `g` then `g⁻¹` no-op on grid and on
      `G ← R G Rᵀ`.
- [ ] Per-dim KL logging wired from step 1.
- [ ] Occupancy F1 logged; abort if F1 < 0.5 after 5 ep is testable.

Week-2 work does not start until every box is checked.

### Week 2 — train, surface failure modes, decide

Days 6–8 — train VAE 150 epochs at §4 hyperparameters. Log per-step:
L_amp, L_occ, β·L_KL, L_sym, `(c·z)²`, per-dim KL, occupancy F1, per-shell
Pearson R on E, per-head grad norms. Checkpoint every 25 ep.

Day 9 — **M2** on EMA.
Day 10 — **M3 FiLM decorative-vs-effective test**. Variant B
(decoder-blind) ~3k steps; linear-probe cell volume from `μ` on both;
report `R²(B) − R²(A)`.
Day 11 — **M4 posterior-collapse audit**. Per-dim KL heatmap; active-units
at ep 25, 75, 150. If a dim collapsed ≥30 ep, raise λ to 0.75 in a 50-ep
retry — goal is to *control*, not avoid surfacing.
Day 12 — **M5** at ep 50, 100, 150 to capture gap dynamics.
Day 13 — **M6 scaling**: chosen config on n ∈ {100, 200, 394} fixed
compute; val L_amp and val Pearson R on E vs `log(n)`.
Day 14 — **cross-method table** (§8) against the parallel MAE run.

### Go / no-go (single sentence, parallel to MAE's)

The VAE secondary track is a go for full-PDB scale iff
(i) primitives checklist fully ticked,
(ii) L_amp and L_KL descend post-warmup, no first-25-ep plateau (M1),
(iii) anisotropic-vs-radial margin ≥ 0.10 (M2),
(iv) FiLM gap `R²(B) − R²(A) ≥ 0.15` (M3),
(v) posterior healthy: active-units ≥ 0.5 at end, no dim collapsed >30 ep
post-warmup, or λ-raise controls it (M4),
(vi) conditioning-leakage R² < 0.5 achievable (M5),
(vii) n ∈ {100, 200, 394} scaling curve monotonic with slope at n=394
not flat (M6).

Fail (M2 < 0.10, M3 ≤ 0.05, M4 uncontrollable, M6 flat) is Case A or B
per `00_final_review.md` §"If the bake-off fails": Case A — VAE broken
on this data (if MAE *also* Case-A-fails, pause representation learning
per the final review); Case B — ambiguous, descending curve, proceed to
a cheap full-PDB discriminator run.

---

## 8. Day-14 cross-method comparison table

Both plans report into the same table on Day 14; comparison is
apples-to-apples.

| diagnostic | MAE result | VAE result | pass criterion |
|------|------|------|------|
| M1. Loss descent shape (first-25-ep drop fraction; final-ep slope) | | | descends, no first-25-ep plateau |
| M2. Anisotropic-vs-radial margin (R(E) − R(radially_shuffled), val) | | | ≥ 0.10 |
| M3a. Conditioning-leakage R² (Wilson-Σ from embed, cond hidden) | | | < 0.5 achievable by dropout sweep |
| M3b. Cond-gap dynamics (full-encoder − cond-only probe slope vs ep) | | | non-zero, non-decreasing |
| M4. Named failure modes surfaced and controlled (count / total) | | | ≥ 1 surfaced *and* knob-controlled |
| M5. Scaling-curve slope at n=394 (Δ val-loss / Δ log n) | | | < 0 |
| Reconstruction Pearson R on E (val) | | | informational |
| Cell-volume linear-probe R² | | | informational |
| SG top-1 | | | informational |
| Anisotropy-ratio linear-probe R² | | | informational |

MAE M1 = val masked-CE descent. VAE M1 = val L_amp descent (post-warmup).
MAE M2 = per-masked-position bin-centroid Pearson R vs radial shuffle.
VAE M2 = per-voxel reconstructed-E Pearson R vs radial shuffle. M3/M4/M5
computed identically across plans. The VAE's method-specific FiLM
decorative-vs-effective (M3 in §5.1) and posterior-collapse (M4 in §5.1)
diagnostics feed into the **M4 row** of this shared table — the VAE
must surface both and demonstrate at least one is knob-controlled.

Day-14 deliverable: this filled table + per-method go/no-go + a
recommendation of primary + secondary to scale per `00_final_review.md`.

---

## 9. Open questions

1. Log-normal NLL adequacy vs centric/acentric mixture: likely negligible
   at n=32; one ablation post-bake-off, not before.
2. Does the symmetry loss earn its weight? If `L_sym → 0` under
   augmentations alone, it is decorative; tracked Week 2, not gating.
3. Optimal `n` ∈ {24, 32, 40}: out of scope for the bake-off.
4. Two-stage MAE → VAE-with-MAE-init (opportunity 2,
   `00_final_review.md`): out of scope; revisit if both pass.
5. At PDB scale, is full cell+SG+Wilson conditioning still needed?
   Ablate at full scale, not on dev.
