# Independent cross-check of the bake-off plans (MAE and VAE)

## Reviewer's standpoint

I read `03_masked_lm.md` and `01_vae_voxel.md` end-to-end before opening
`00_final_review.md`. I then sanity-checked the numerical anchors against the
artifact parquets in `analysis/dev/artifacts/` (`voxel_per_protein.parquet`,
`review_wilson.parquet`, `asu_reduction.parquet`, `per_protein_meta.parquet`)
and the voxellisation code in `analysis/dev/build_notebook.py`. I formed
my position on the two plans before reading the previous review, and I
agree with parts of it independently while disagreeing with several
specifics. I am writing this as an adversarial cross-check, not a
seconding.

---

## MAE plan (`03_masked_lm.md`)

### What works

1. **The right inductive bias for the data.** A sparse-set transformer with
   per-token CE on Wilson-E quantile bins is the only loss formulation
   among the three that puts a unit of supervision on every signal-bearing
   voxel without spending FLOPs on empty space. Equal-population
   quantisation forces the predict-mode baseline to 1/64 ≈ 1.6 %, which
   makes the loss number physically interpretable; adjacent-bin label
   smoothing (0.8 / 0.075 / 0.025) correctly treats the bins as ordinal.
   This is correct and unusual — most token-LM specs in the protein space
   one-hot ordered targets and pay for it later.

2. **Block masking with curriculum r=1 → r=2.** Single-voxel masking on a
   sparse occupancy map is solvable by neighbour-copy; the spec sees this
   and addresses it. The empirical reciprocal-space correlation length
   set by inverse particle size at n=32 is ~1–3 voxels, so r=2 is the
   right radius. Curriculum from r=1 keeps the early loss informative.

3. **Conditioning as `[COND]` prefix tokens with per-token dropout.** The
   triple-dropout (Wilson 0.5 / cell 0.2 / SG 0.2) is the right
   architectural pattern for a transformer; FiLM in the VAE is more
   restrictive. M3a (adversarial leakage probe) and M3b (gap dynamics)
   are the right diagnostics to back this up.

4. **RoPE-3D over input shuffled per step.** Decouples set semantics from
   token order; Week-1 shuffle ablation is the right primitive.

5. **The Day-14 mechanism table is correctly framed as a method-selection
   instrument, not a leaderboard.** The plan demotes outcome metrics
   (SG top-1, cell-volume R²) to informational and gates only on
   mechanism (M1–M5). This is the harder, correct stance.

### What does not work or is undercooked

1. **The token count is wrong by 10×.** The plan asserts "median ~250
   nonempty voxels per protein" after canonical-ASU reduction, and sizes
   `max_seq_len = 1024` against this. The artifact `voxel_per_protein.parquet`
   says median 2475 nonempty voxels at n=32 (mean 3325, p90 4798, p95
   7721, p99 9115, max 9449). The voxellisation in
   `build_notebook.py` (lines 800–820) bins reflections in the **bounding
   box of all reflections, in the original Miller frame**, not the ASU
   wedge. `asu_reduction.parquet` shows uniform ASU reduction factor of
   1.0 — i.e. the data is already merged, with one reflection per ASU
   orbit, so canonicalising a *merged* dataset cannot reduce the count
   further. To collapse 2475 nonempty voxels down to ~250 you would need
   the canonical wedge bbox to be 1/10 the volume of the merged-data
   bbox AND the model to tolerate the resulting 30–40 % occupancy in the
   wedge; neither is shown by the analysis. Sequence length p99 will not
   be ≤ 1024 — it will be > 8000 — and the contingency in §1.1 ("raise to
   1536 with random spatial-block sub-sampling") is order-of-magnitude
   short. This is the single largest correctness defect in the bake-off
   spec and it cascades into compute, GPU memory, and curriculum
   feasibility.

   **Fix.** Before Day 1, run a literal canonicalised-wedge voxellisation on
   the dev set, write the actual nonempty-voxel quantile table to a
   parquet, and re-size `max_seq_len` against measured p99. If p99 stays
   high (and it will), the engineer must either (a) raise n to give a
   useful wedge bbox and accept higher seq-len, (b) drop n to ~24 to
   halve nonempty voxels and budget seq-len ≈ 1500–2000, or (c)
   restructure the model around chunked attention. The plan as written
   sails into Day 6 with a 4× seq-len underestimate. The p99 ≤ 1024
   primitive in §9 will fail and Week-1 will gridlock unless the model
   sizing is rethought.

2. **M2's anisotropic-vs-radial margin is *not* method-discriminative
   between MAE and VAE the way the plan claims.** With Wilson E as the
   target and the radial profile already on a side-channel and dropped
   out at p=0.5, *any* method that conditions on visible voxel content at
   all will produce R(E) > R(radially_shuffled_E). The radial-shuffle
   replaces the target by a same-shell randomly drawn voxel; even a
   trivial "predict the bin centroid of a near neighbour" model beats the
   shuffle by ≥ 0.10 because near-neighbour voxels of *the same protein*
   share much more than the radial profile (they share the *anisotropic*
   profile through the cell-shape modulation). 0.10 is too easy; passing
   M2 ≥ 0.10 is consistent with a model that is essentially copying
   visible neighbour voxel content forward, which is exactly the failure
   mode r=2 block-masking was supposed to defeat.

   **Fix.** Define the M2 margin against a *visible-neighbour-aware*
   control, not just a radial control:
   - Control 1 (current): radial shuffle. Floor.
   - Control 2 (new, mandatory): "context-only" — for each masked
     position, take the bin-centroid prediction of the **nearest visible
     voxel** (Chebyshev distance on the integer (h,k,l) lattice). This is
     what neighbour-copy would do.
   - The MAE has earned its inductive bias iff R(E) − R(context-only) ≥
     0.05 *and* R(E) − R(radially_shuffled) ≥ 0.10. The first margin is
     the discriminative one against neighbour-copy; the second is the
     discriminative one against radial-only. Both numbers should be in
     the Day-14 table.

3. **Mask-via-RoPE leakage is named but the test is shallow.** §8 lists
   "ablate RoPE-3D vs learnt-3D" as the diagnostic. RoPE-3D over a
   sparse voxel set may give the model a positional-shortcut to the
   masked location only if the mask token's position is encoded in a way
   that lets the encoder localise the mask inside its visible-token
   attention. But MAE-style encoders don't see mask tokens (the spec
   honours this). The actual leakage path here is in the **decoder**:
   the decoder receives a learnt `[MASK]` token at each masked position
   *with RoPE-3D coordinates restored* (§3.2). The decoder thus knows the
   exact (h,k,l) of every position it has to predict. That is by design
   for MAE, but it means the bin-prediction task on masked positions is
   conditioned on full positional information about the mask cluster —
   so the decoder may learn a "Wilson-Σ-from-coordinate" shortcut purely
   from its `[MASK]` token RoPE keys plus the visible Wilson side-channel
   it gets from the encoder. The plan does not test this.

   **Fix.** Add a Week-2 ablation that **scrambles the decoder's RoPE
   keys** at masked positions to a random voxel within the same r=2
   block. If val masked-CE moves by < 5 %, the decoder is using
   coordinate information beyond the block neighbourhood — and that is
   the in-network Wilson estimator that triple-dropout was supposed to
   suppress. Make this a named M4 mode, not a footnote.

4. **CD-HIT 30 % at n=394 with 49 SGs is statistically broken for the
   informational probes.** `per_protein_meta.parquet` shows 49 distinct
   SGs, 15 singletons, 26 with ≤ 3 proteins. After an 80/10/10 cluster
   split many SGs in val will not appear in train at all. Probing SG
   top-1 in this regime is pseudoscience: the headline number depends on
   which singleton SGs land where. The plan caveats with "bootstrap
   CIs"; bootstrap CIs do not rescue a probe whose support is
   structurally absent from train. The probe should be SG top-K within
   the train-supported SGs, evaluated on the val proteins whose SG is in
   train, with the held-out-SG fraction reported separately. Same applies
   for the cell-volume R² probe — train cell volumes will be a biased
   sample.

   **Fix.** Add a "support coverage" line to the Day-14 informational
   read-outs: "Fraction of val proteins whose SG and (binned) cell volume
   appear in train". Report informational probes only on the supported
   subset; report the unsupported-SG probe as a separate, low-signal
   number. This costs nothing and prevents reading too much into a
   broken probe.

5. **`[COND]`-prefix design encodes only one Wilson scalar per
   resolution shell, but the dropout regime does not match.** Wilson
   profile dropout p=0.5 means half of training steps lack any radial
   conditioning; cell + SG dropout 0.2 is ~10× weaker. But cell + SG
   together imply the entire reciprocal lattice geometry up to the
   anisotropic ADP — i.e. a sufficiently capable encoder reads off d_min
   and a coarse Σ envelope from cell + SG alone. Asymmetric dropout (0.5
   vs 0.2) bakes in an asymmetric leakage path, then M3 measures it as
   if the imbalance were a finding rather than a setting. The
   dropout-knob sweep (§9 Day 11–12) covers cell+SG dropout ∈ {0.0, 0.2}
   only — it does not include the cell-or-SG-dropped-while-Wilson-kept
   counterfactual that would isolate which side-channel dominates the
   leak.

   **Fix.** Add **independent** dropout rates for cell and SG in the
   sweep: cell ∈ {0.0, 0.2, 0.5}, SG ∈ {0.0, 0.2, 0.5}, Wilson ∈ {0.0,
   0.5, 0.8}. This is 27 settings; cut to a 3×3 fractional design (e.g.
   3-way Latin square). The point is not to find a winner; the point is
   to attribute which of the three is the dominant leakage carrier. M3a
   alone does not separate them.

6. **The auxiliary-occupancy head is bolted on with ill-defined
   semantics.** §4 says: small MLP on RoPE-encoded (i,j,k) coordinates
   cross-attends into encoder output → BCE on 4096 random voxels per
   protein per step. The 4096 voxels per protein per step at 7.5 %
   occupancy means ≈ 307 positives and ≈ 3789 negatives — a 12.3:1
   imbalance, no positive class weighting given. With weight 0.3 on the
   total loss and tens of thousands of zero-target voxels per batch, the
   gradient direction will dominate the early-training signal and pull
   the encoder toward a "predict empty everywhere" local basin. Per-head
   grad-norm logging surfaces this *if* the engineer reads the right
   row, but the spec's mitigation is "re-tune 0.3" — that is a
   parameter, not a primitive.

   **Fix.** Drop the occupancy head from v1. The marginal benefit of
   recovering full-grid occupancy is small at dev scale; the
   downstream-relevant occupancy is implicitly available from the input
   sparse set itself plus the symmetry-completed wedge. If the team
   insists, weight 0.3 should be log-balanced against intensity-CE
   gradient norm and ramped from 0 over 5 epochs. M4 should explicitly
   report the **ratio of occupancy-head to intensity-head gradient norms
   at epoch 25** as a passable threshold (e.g. ratio in [0.3, 3]).

### Concrete proposed changes (MAE)

- **§1.1 Tokenisation.** Replace the asserted "median ~250 nonempty
  voxels" with a measured number from a Day-1 canonicalised-wedge
  voxellisation on all 394 dev proteins. Re-size `max_seq_len` against
  measured p99. State up front whether n=32 or n=24 or chunked attention
  is the supported configuration.
- **§2 / §7.1 M2.** Add the context-only neighbour-copy control. Margin
  thresholds: R(E) − R(radial) ≥ 0.10 *and* R(E) − R(context-only) ≥
  0.05.
- **§3.2 / §8 M4.** Add the decoder-RoPE-scramble ablation as a named
  failure mode (mask-via-positional-shortcut). Pass criterion: val CE
  moves ≥ 5 % under scramble.
- **§3.3 / §9 Day 11–12.** Independent cell-vs-SG dropout in the
  conditioning-dropout sweep, not paired.
- **§4 / §9.** Drop the occupancy head from v1 *or* tie its weight ramp
  to a measured grad-norm ratio target.
- **§7.2.** Add a "support coverage" diagnostic to the informational
  probe block; report SG probe on supported-SG val subset.

---

## VAE plan (`01_vae_voxel.md`)

### What works

1. **Wilson E² as target with radial profile on a side-channel.** Honours
   the η = 0.11 vs 0.52 finding. log-normal NLL is the right CLT-justified
   v1 head; centric/acentric mixture is correctly deferred.

2. **Free bits at λ=0.5 nats/dim with linear β-warmup, plus
   conditioning-dropout p=0.1.** Two independent textbook collapse
   prevention measures, with a third (`(c·z)²` orthogonalisation at 0.01)
   stacked on top. M4 (per-dim KL with 30-epoch persistence + active-units
   ≥ 0.5) is a defensible diagnostic.

3. **Symmetry consistency loss `(D(E(g·x), c) − g⁻¹·D(E(x), c))²` at
   weight 0.1.** Pinning equivariance the augmentation pipeline only
   achieves in expectation is the right call at dev scale.

4. **M3 (FiLM decorative-vs-effective via cell-blind decoder variant).**
   This is the best diagnostic in either plan. R²(B) − R²(A) ≥ 0.15 has
   correct sign (cell-blind decoder *should* make z carry more cell, so
   linear probe on z is *better*), and the threshold 0.15 is defensible
   because cell volume is a single scalar ablation, well within probe
   reach if z carries it. Crisp and falsifiable.

### What does not work or is undercooked

1. **Six-term loss with three free weights, calibrated by feel, not by
   gradient-norm balance.** L = L_occ + L_amp + β·L_KL + 0.1·L_sym +
   0.01·(c·z)². Each term has a different per-voxel scale, BCE vs NLL
   vs L2 vs scalar-pair product. The weights are pulled from textbook
   values (0.1 sym, 0.01 ortho, λ=0.5 free bits) without empirical
   justification on this data. The risk is exactly what `00_final_review`
   flags ("fine on every term but not great on any"); the plan's
   mitigation is "re-tune weights only after named-failure surfaces",
   but with six terms the named failure may be in any of three couplings
   and the diagnostics don't distinguish. M1 is a single-number "did
   L_amp drop ≥30 % from epoch-5 baseline" — that does not separate
   "L_amp dropped because the decoder is good" from "L_amp dropped
   because L_occ dominated and pushed decoder toward predicting all
   nonzero voxels".

   **Fix.** From step 1, log per-term gradient norms at the encoder
   bottleneck. Pass criterion in M1 (additive): each term contributes ≥
   10 % and ≤ 60 % of total bottleneck-grad-norm at epoch 25 post-warmup.
   This is the right unit for "is the calibration sane".

2. **Posterior-collapse diagnostic via per-dim KL is necessary but not
   sufficient.** The plan flags any dim with KL < 0.01 nats for ≥ 30
   epochs as collapsed and tracks active-units. But there is a
   well-known intermediate failure mode — **collapse-into-conditioning**
   — where per-dim KL is finite (passes the diagnostic) but z is a
   deterministic affine of c (z = W·c + ε, with ε small). Per-dim KL
   stays positive because q(z|x) has a posterior with non-trivial
   variance, but the mutual information I(z; x | c) is near zero. The
   plan does not measure I(z; x | c).

   **Fix.** Add a third panel to M4: estimate I(z; x | c) by KL
   divergence between q(z|x, c) and the conditional prior p(z|c) where
   p(z|c) is approximated by the empirical Gaussian of z conditional on
   c-bin. Pass: per-protein conditional-KL median ≥ 0.5 nats at end of
   training. M3 (decorative-FiLM) catches one tail of this pathology;
   conditional-MI catches the other.

3. **M2 has the same false-easy-pass problem as the MAE.** Per-voxel
   reconstructed-E vs radially-shuffled control. With a dense CNN that
   sees the cell-FiLM and the Wilson side-channel, R(E) − R(radial) ≥
   0.10 is achievable by "memorise cell→anisotropy axes and lay down a
   plausible E² envelope". This is not the inductive bias the spec
   promises; it is the failure mode the (c·z)² term was added to prevent.

   **Fix.** Add a "**z-only reconstruction control**" — reconstruct E
   with the FiLM/cond pathway *zeroed* and only z + decoder weights;
   compute R(E) on val occupied voxels. Margin: R(E_with_cond) −
   R(E_z_only) ≤ 0.20. If z's reconstruction is ≪ full-cond
   reconstruction, the model has offloaded everything to FiLM. This is
   the symmetric counterpart of the cell-blind-decoder probe (M3) and
   gives a two-sided sandwich: M3 says "cond does work", z-only says "z
   also does work".

4. **`(c·z)²` orthogonalisation regulariser at 0.01 is a soft signal
   that competes with KL.** It penalises any linear correlation of z
   with c; KL penalises departure from N(0,I). These two pressures both
   want a near-isotropic z, and the (c·z)² term will be dominated by KL
   at λ=0.5 free bits. The plan keeps it at 0.01 by assumption; nothing
   in the plan tests whether 0.01 is doing anything. M5 (leakage probe)
   is the only check, but it is downstream of (c·z)² in a way that does
   not let the engineer tell whether the regulariser or β·KL is doing
   the work.

   **Fix.** Run an ablation at **0.0 vs 0.01** for the (c·z)² term as
   part of the Week-2 sweep. If M5 leakage R² is identical, the term is
   decorative and should be dropped; if it changes by ≥ 0.05, keep it
   and consider raising weight. The plan currently treats the
   regulariser as load-bearing without ever testing the load.

5. **Augmentation set generates only 4 700 effective views.** 394
   proteins × ~12 SG-orbit operations = 4 700 views. With 150 epochs at
   batch 256, the model sees every (protein, orbit) pair ≈ 100 times.
   This is the standard memorisation regime. The plan calls out
   "training-set memorisation" as the underdocumented risk
   (`00_final_review` flags it too) but the diagnostic is missing.

   **Fix.** Add an "epoch at which train-recon-R and val-recon-R diverge"
   metric. If divergence is < 25 epochs the model is memorising;
   secondary mitigation is to add a `(c, sg)`-conditional MixUp on
   E²-targets with low weight, then re-evaluate.

6. **Symmetry loss may be redundant with augmentation under
   p(g_identity) > 0.** The augmentation pipeline samples
   g uniformly including identity; the symmetry loss penalises
   `(D(E(g·x), c) − g⁻¹·D(E(x), c))²` for a sampled non-identity g.
   These two interact: if augmentation already enforces g ≈ identity in
   expectation through the encoder, the loss is constraining a
   redundant axis. The plan's open question 2 admits this; the
   diagnostic ("if `L_sym → 0` under augmentations alone") is in fact
   the right test, but it's not gated. The Week-2 logging plan does
   not include "L_sym pre-decay vs post-decay across the symmetry-loss-
   weight-zero ablation".

   **Fix.** Add a 50-epoch ablation at sym-weight 0 alongside the
   primary run. M2/M3 should be no worse, and L_sym at the end of the
   ablation should be near zero — *if* L_sym is decorative. If it isn't,
   keep the term; if it is, drop it and free a calibration knob.

### Concrete proposed changes (VAE)

- **§3 / §5.1 M1.** Add per-term gradient-norm balance check at epoch 25
  post-warmup; each of {L_occ, L_amp, β·L_KL, L_sym} contributes
  10–60 % of bottleneck grad-norm.
- **§5.1 M4.** Add conditional-MI estimator I(z; x | c) ≥ 0.5 nats as a
  second panel to posterior-collapse audit.
- **§5.1 M2.** Add z-only reconstruction control; margin R(E_full) −
  R(E_z_only) ≤ 0.20.
- **§3 / §7 Week 2.** Run the (c·z)² weight ∈ {0.0, 0.01} ablation in the
  M5 sweep.
- **§7 Week 2.** Add a sym-loss-weight ∈ {0.0, 0.1} ablation; report L_sym
  trajectory in both arms.
- **§5.1 / §6.** Add memorisation diagnostic: train-vs-val recon R
  divergence epoch.

---

## Day-14 cross-method comparison: is it apples-to-apples?

Mostly yes, but there are three asymmetries that creep in via metric
definitions.

1. **M1 ("loss descent shape") is **not** symmetric across methods.** The
   MAE M1 checks "val masked-CE drops ≥ 30 % from log(64)=4.16 floor toward
   the irreducible adjacent-bin smoothing floor". The VAE M1 checks "val
   L_amp at epoch 25 is ≥ 30 % below its epoch-5 post-warmup baseline".
   Both are 30 % drop thresholds, but on different axes: the MAE drop is
   *toward an irreducible floor* (a measurable quantity); the VAE drop is
   *from a moving baseline* (epoch-5 post-warmup, which depends on β-warmup
   schedule and how aggressive L_occ is at that point). A method-fair
   comparison should normalise both by their respective theoretical
   floors, or both should be relative-to-initialisation.

   **Fix.** Define MAE M1 as `(CE_init − CE_25) / (CE_init − CE_floor)` ≥
   0.5, and VAE M1 as `(L_init − L_25) / (L_init − L_*)` where L_* is
   estimated from a small overfit run on a 10-protein subset (the
   data-irreducible loss floor for the model class). Both are now
   "fraction of attainable descent achieved at epoch 25". Same axis.

2. **M2 anisotropic-vs-radial margin is symmetric in name but
   asymmetric in test difficulty.** The MAE computes Pearson R on
   bin-centroid prediction at masked positions only (~60 % of nonempty
   voxels per protein). The VAE computes Pearson R on per-voxel
   reconstructed E² over all occupied voxels (100 % of nonempty
   voxels). For a same-quality model, the VAE will report a higher R
   simply because it is reconstructing a denser, more correlated set.
   The 0.10 margin threshold is the same for both, but the **noise
   floor on R is different** because the sample sizes differ by
   ~1.7×. Worse, the MAE's bin-centroid is a quantised prediction, so
   R is bounded above by the bin-centroid R-vs-true-E correlation
   (around 0.95 for 64 bins on log-normal data), where the VAE's
   continuous μ_E2 is not.

   **Fix.** Compute **both** methods' M2 on the same evaluation
   set: the union of masked positions across val. For the VAE, evaluate
   reconstruction at the MAE's masked positions only. Report R (MAE) vs
   R (VAE on same positions) — and the radial-shuffle margin on those
   same positions.

3. **M5 scaling-curve slope is computed on a method-specific axis.** MAE
   plots Δ(masked-CE) vs Δ log n; VAE plots Δ(L_amp) vs Δ log n. These
   are not comparable axes — masked-CE is in nats, L_amp is in
   half-squared-log-E units plus log σ. A "slope < 0" criterion fires
   for both, but the magnitude is uninterpretable cross-method.

   **Fix.** Both methods report **two** scaling curves: their own loss
   *and* per-shell Pearson R on E (the only metric that is
   model-architecture-invariant). M5 passes if the per-shell R curve is
   monotonically improving across {100, 200, 394} for both methods.
   Also: at 3 points, fit a 2-parameter power law `R(n) = a − b·n^(-c)`
   with c held fixed across methods and compare b directly.

4. **M3 in the shared table refers to two different things in the two
   plans.** In the MAE's local §7.1, M3 = conditioning-leakage gap
   dynamics. In the VAE's local §5.1, M3 = FiLM decorative-vs-effective
   (cell-blind decoder probe). Both are renamed M5 in the VAE (which is
   the leakage probe) but the shared §10/§8 table uses the MAE's row
   names. The footnote "for the VAE, the M4 row prioritises the FiLM
   decorative-vs-effective test and posterior collapse" papers over a
   real asymmetry: the VAE has *two* method-specific failure modes (M3
   and M4 in its own naming) that are folded into a single "M4 named
   failure modes surfaced and controlled (count / total)" row. The MAE
   gets four named failure modes in its M4; the VAE gets two. The
   "≥ 1 surfaced *and* knob-controlled" pass criterion is therefore a
   weaker bar for the VAE than the MAE.

   **Fix.** Make the M4 row report `(surfaced count / named total) ≥ X`
   where X is method-specific because the named totals differ. Print
   *both* numbers in the table cell. Anything else lets a method look
   stronger by having fewer named failure modes.

---

## Substantive issues (≥ 3) you found

### Issue 1: MAE token-count is wrong by 10×, sequence budget is unworkable

**What.** §1.1 of `03_masked_lm.md` asserts ~250 nonempty voxels per
protein after canonical-ASU reduction; sets `max_seq_len = 1024`. The
artifact `voxel_per_protein.parquet` records median 2475 nonempty voxels
at n=32 over the full dev set; voxellisation in `build_notebook.py` is
done in the merged-data bbox, not the canonical wedge. ASU
canonicalisation cannot reduce the count further because
`asu_reduction.parquet` shows reduction factor 1.0 across the dev set
(the data is pre-merged).

**Why it matters.** Every Week-2 training step is sized against the
wrong sequence budget. The Week-1 primitive "p99 ≤ 1024" will fail; the
contingency in §1.1 ("raise to 1536 with random spatial-block
sub-sampling") is also wrong order-of-magnitude. Random sub-sampling
of half the visible voxels is not a benign mitigation — it deletes
most of the high-d* tail where the anisotropic signal lives, and at
that point the M2 anisotropic-vs-radial margin becomes meaningless.

**Fix.** On Day 1, run a literal canonicalised-wedge voxellisation,
write the actual nonempty quantile table to `wedge_voxel_count_v1.parquet`,
re-size `max_seq_len` against measured p99. Almost certainly this leads
to one of: (a) drop n to 24, (b) chunked / windowed attention, (c) keep
n=32 and accept seq-len ≈ 4K with FlashAttention. Pick before training
starts.

### Issue 2: M2 (anisotropic-vs-radial margin) is too easy; both methods will pass it without learning anisotropy

**What.** In both `03_masked_lm.md` §7.1 and `01_vae_voxel.md` §5.1, M2
margin ≥ 0.10 is the gate criterion for "model is using anisotropy, not
just radial decay". The radial-shuffle control replaces masked targets
with same-shell random voxels but preserves nothing about visible
neighbour structure. Any model that exploits short-range neighbour
correlations beats this — including a model that has learned only "near
voxels look similar" (an anisotropic-but-shortcut signature, not the
intended learned-from-cell anisotropy).

**Why it matters.** A model that passes M2 ≥ 0.10 may still be the
neighbour-copy failure that r=2 block-masking was specifically chosen
to defeat. M2 passing on a neighbour-copy model would falsely greenlight
the MAE for full-PDB scale-up.

**Fix.** Add a context-only neighbour-copy control on top of the
radial-shuffle control. Pass criterion: R(E) − R(radial) ≥ 0.10 *and*
R(E) − R(context-only) ≥ 0.05. Both numbers go into the Day-14 table.
Same fix applies symmetrically to the VAE.

### Issue 3: Scaling-curve confound — CD-HIT cluster split shifts between {100, 200, 394}

**What.** Both plans (MAE §9 Day 13, VAE §7 Day 13) propose nested
subsets at n ∈ {100, 200, 394}, fixed compute, fitting a 3-point trend.
The CD-HIT 30 % cluster split is shared at n=394; for n=200 and n=100
it must either be (a) recomputed (which changes which clusters land in
val), or (b) inherited as a subset of the n=394 train (which means the
n=100 subset has *much* lower cluster diversity than n=394 — random
nesting concentrates clusters by chance). The plan does not specify;
either path introduces a confound that is not "data-size effect".

**Why it matters.** A "monotonically descending loss vs log n" can be
produced purely by (i) more reflections in train, or (ii) more cluster
diversity in train, or (iii) more anisotropy diversity in train. The
plan reads the slope as a single signal but it is a mixture of three.
At n=394, slope-not-flat is the gate; if cluster diversity is the
driving variable the gate fires at n=394 only because the subset
sampling happened to add clusters at the right rate, not because the
method is data-hungry. Worse, fixing compute "scaled to equalise
gradient steps" changes the effective learning-rate-per-protein, which
also tilts the curve.

**Fix.** Specify the nested-subset construction explicitly:
- Subsets are nested *within clusters* — i.e. n=100 contains the *same
  set of clusters* as n=200 and n=394, with proteins-per-cluster
  proportional to subset size where possible. This holds cluster
  diversity constant across the 3 points.
- Report two slopes: (a) at fixed gradient steps (compute-equalised) and
  (b) at fixed epochs (data-equalised). A scalable method has both
  slopes negative and not-flat at n=394; a method whose curve is
  driven by gradient-step count fails the (b) test and is not actually
  data-hungry.
- Add a 4th anchor point at n=300 if compute permits — 3-point fits are
  noisy and the second derivative is ill-conditioned.

### Issue 4: VAE posterior-collapse audit misses collapse-into-conditioning

**What.** `01_vae_voxel.md` §5.1 M4 audits per-dim KL > 0.01 and
active-units fraction. This catches per-dim collapse to the prior. It
does not catch the more insidious failure where z = f(c) + ε
deterministically — per-dim KL is finite (passes), but I(z; x | c) is
near zero (the encoder has stopped routing protein-specific information
through z). The (c·z)² regulariser is supposed to prevent this but it
penalises *linear* correlation only.

**Why it matters.** A VAE that collapses-into-conditioning will pass
M4, will pass M3 (decorative-FiLM via cell-blind decoder, because z
still carries cell when decoder cannot see it), and will probably pass
M2 (because conditioning + decoder reconstructs E adequately) — and yet
will be downstream-useless because z carries no
beyond-conditioning information. All three diagnostics fire green and
the bake-off greenlights the wrong VAE.

**Fix.** Add a conditional-MI panel to M4: estimate I(z; x | c)
empirically by binning c, computing per-c-bin q(z|c), and reporting
mean KL(q(z|x,c) || q(z|c)) over val. Pass: ≥ 0.5 nats per protein at
end of training. This catches the missing tail.

### Issue 5: Decoder RoPE leakage in MAE is a named failure mode missing from §8

**What.** The MAE decoder receives a `[MASK]` token at each masked
position with RoPE-3D coordinates restored (§3.2). The decoder thus
knows the exact (h,k,l) of every prediction target. Combined with the
encoder's Wilson-conditioning output, the decoder can in principle
back-fit a Wilson-Σ-from-coordinate shortcut without ever using
masked-position context. The plan tests "RoPE-3D vs learnt-3D" on the
encoder side (M4) but not "scramble decoder RoPE keys at masked
positions" — the test that would isolate the decoder's coordinate
shortcut.

**Why it matters.** The point of M4 is to surface and control named
failure modes. Decoder positional-shortcut is a plausible failure of
exactly the kind dev-set is supposed to catch.

**Fix.** Add "decoder-RoPE-scramble" as a named failure mode in §8.
Mitigation knob: scramble decoder RoPE keys at masked positions to a
random voxel within the same r=2 block. Pass: val masked-CE moves ≥
5 % under scramble, indicating the decoder is using fine-grained
position information rather than only block-level.

### Issue 6: SG and cell-volume informational probes are statistically unsupported at n=394

**What.** `per_protein_meta.parquet` shows 49 distinct SGs, 15 with
exactly one protein, 26 with ≤ 3 proteins. After 80/10/10 cluster split,
many val proteins have SGs that do not appear in train. Reporting "SG
top-1" as an informational read-out without the support-coverage
caveat lets a future reader (or worse, a paper reviewer) anchor on a
brittle number.

**Why it matters.** The plans correctly demote informational probes to
non-gating, but a method-comparison reader will still glance at the
numbers. The SG-top-1 differences between MAE and VAE on the Day-14
table will be dominated by which singleton SGs got assigned to which
split — not by the methods.

**Fix.** Add a "support coverage" cell to the SG and cell-volume
informational rows: fraction of val proteins whose SG (or cell-volume
bin) is present in train. Report probe accuracy on the supported
subset and on the full set separately.

---

## Where I agree and disagree with `00_final_review.md`

### Agreements (independent confirmation)

1. **Ranking: MAE primary, VAE secondary, contrastive dropped.** I
   reached this conclusion before reading the final review. Reasoning is
   shape-of-output (token sequence consumable by cross-attention into
   protein-design backbones beats single-vector latent), and signal
   density per training item (per-token CE on every visible voxel beats
   per-voxel reconstruction with conditioning load, beats one
   InfoNCE/VICReg pair per protein per step). The final review's framing
   of this as "right unit of supervision" matches my own.

2. **Wilson-token dropout p=0.5 is necessary but insufficient.** I
   reached this from inspecting the conditioning architecture before
   reading the final review's "sharpest critique" section; the in-network
   Wilson estimator from cell + SG + visible voxels is reachable. The MAE
   plan's adoption of triple-dropout is the right response.

3. **The framing shift from outcome metrics to mechanism metrics is
   correct.** 394 proteins is too narrow to ground absolute thresholds
   on probe R²; the correct deliverable is mechanism (loss-descent
   shape, anisotropic-vs-radial margin, conditioning-leakage gap
   dynamics) and predictable scaling (loss-vs-data slope at n=394).
   Both rewritten plans inherit this correctly.

4. **The cross-cutting risk on cell-tensor leakage is real and central.**
   I would have flagged it independently. The gap-dynamics framing
   ("non-zero, non-decreasing gap that grows with training") is a
   stronger framing than "leakage R² < threshold" because thresholds at
   n=394 are noisy.

The most important independent confirmation is **the ranking**: same
order, reached independently from different starting evidence. That
matters as evidence.

### Disagreements

1. **The token count of "~250" in `00_final_review` §1 strong-side 1
   propagates the same 10× error from the MAE plan.** The final review
   says "~150 supervised tokens per protein per step" and "~250 sparse-set
   tokens", which inherits the spec author's unverified number rather
   than reading the artifact. At 60 % mask ratio and ~2475 nonempty
   voxels, the supervised-token count is ~1500, not ~150. This does not
   change the ranking conclusion — the MAE still beats the VAE on
   signal-per-item — but the *magnitude* of the difference cited in the
   final review's analysis ("MAE has ~150, VAE has ~2500, VICReg has 1")
   misrepresents the comparison. The corrected counts are MAE ~1500, VAE
   ~2500, VICReg 1, and the MAE-vs-VAE per-item-density gap is ~1.7×,
   not ~17×. The qualitative ordering is unchanged but the gap is
   smaller than the final review implies, and it strengthens the case
   for keeping the VAE as a *credible* secondary, not a long-shot
   secondary.

2. **The MAE M2 threshold of ≥ 0.10 is endorsed by the final review
   without testing it against a neighbour-copy control.** The final
   review's bake-off plan introduces M2 with the exact same threshold
   the plans inherit. As argued in Issue 2, this threshold is too easy
   in the absence of a context-only control. The final review's framing
   ("the margin is the diagnostic; its absolute value is not") is
   correct in spirit but the *threshold* is wrong because the *floor*
   the margin is measured against is wrong. This is my sharpest
   methodological disagreement with the previous review.

3. **The final review treats CD-HIT cluster split for the 3-point
   scaling-curve as solved.** The final review says (Day 13) "subsets
   nested" without specifying within-cluster nesting vs across-cluster
   nesting. This is the cluster-shift confound from Issue 3. It is not
   a correction to the previous review — the omission is in the spec —
   but the final review's endorsement of the 3-point fit at n ∈ {100,
   200, 394} as the gate criterion inherits the confound.

4. **The final review's "Case A" pivot to the VAE if MAE fails is
   under-specified.** "If the MAE Case-A fails, pivot to VAE secondary
   as primary" is the right instinct but the criterion for VAE-passes
   is not symmetric to MAE-passes — the VAE has different named failure
   modes and different mechanism diagnostics. If the MAE M2 fails (anisotropy
   margin < 0.10) and the VAE M2 passes (≥ 0.10), is the VAE actually
   better, or is the M2 test on the VAE just easier (Issue at §"Day-14
   cross-method comparison" point 2)? The final review does not
   adjudicate this case.

### Things I cannot evaluate without an empirical run

1. **Whether triple-dropout (Wilson 0.5, cell 0.2, SG 0.2) is sufficient
   to control conditioning leakage.** This is fundamentally an empirical
   question; the plan correctly defers it to M3 dynamics.

2. **Whether the dev set is too narrow for the M3-FiLM-decorative test
   to fire reliably.** The cell-blind-decoder variant requires that the
   encoder *can* learn to route cell into z when the decoder cannot
   read it; whether this happens in 3k steps on 394 proteins is
   data-driven.

3. **Whether the n=394 scaling-curve slope is actually informative or
   just noisy.** The final review claims a 3-point fit licenses
   scaling; without seeing the variance of the 3 points, I cannot
   confirm. (Issue 3 partially addresses this with a recommendation
   to add a 4th point.)

4. **Whether the irreducible CE floor for the MAE (set by adjacent-bin
   smoothing alone) is achievable in 200 epochs at 16 M parameters on
   394 proteins.** The 30 %-of-headroom drop by epoch 25 (M1) depends
   on this floor being a finite, learnable target. The plan says the
   floor is "set by adjacent-bin smoothing alone" — `1 − (0.8·log(0.8) +
   2·0.075·log(0.075) + 2·0.025·log(0.025))/log(64)` ≈ 4.16 −
   smoothed-floor; the smoothed floor is ~0.7 nats. Whether the model
   gets to 30 % of (4.16 − 0.7) = 3.5 × 0.30 = 1.04 nats below 4.16
   (i.e. CE ≤ 3.12) by epoch 25 is empirical.

---

## Final verdict

### Should the team proceed with the bake-off as currently specified?

**YES with caveats.** The framing is right, the ranking is right, and
the diagnostic skeleton (M1 loss descent, M2 anisotropic-vs-radial
margin, M3 conditioning leakage, M4 named failure modes, M5 scaling
slope) is the right shape. But three substantive defects — (1) the 10×
underestimate of the MAE sequence length, (2) the too-easy M2 threshold
without a neighbour-copy control, (3) the CD-HIT cluster nesting
ambiguity in the scaling-curve experiment — will, if uncorrected, lead
to a Day-14 table that either gridlocks Week 1 (the seq-len primitive
fails) or greenlights a method on a discriminative test that is not
discriminative. Patch these before Day 1 and the bake-off is
well-designed for its stated purpose.

### My ranking (and whether it matches the previous review)

**MAE primary, VAE secondary, contrastive dropped.** Same as the
previous reviewer. I reached this independently. The ranking is not
in question. What is in question is the magnitude of the MAE-VAE gap
(smaller than the previous review's per-item-density numbers imply,
because the MAE token count is ~1500 not ~150) and the calibration of
the bake-off thresholds (the M2 threshold is too easy as written, in a
way that is symmetric across both methods so it does not change the
ranking but does weaken the bake-off's discriminative power).

### Single most important fix the team must make before Day 1

**Verify the canonicalised-wedge nonempty-voxel quantile table on the
dev set, and re-size `max_seq_len` accordingly.** This is one Python
script (run gemmi `to_asu` on every reflection, voxellise into the
wedge bbox at n=32, count nonempty voxels per protein, write a
parquet). It costs an hour. If the result is — as I expect from the
existing artifacts — a p99 in the few thousands rather than ≤ 1024,
the model sizing, training-step cost, GPU-memory budget, and Week-1
primitive checklist all need re-calibration before training starts.
Discovering this on Day 6 wastes a week. Discovering it on Day 1
costs nothing.
