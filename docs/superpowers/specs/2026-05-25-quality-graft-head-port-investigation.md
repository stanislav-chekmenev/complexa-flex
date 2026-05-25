# Quality-graft confidence-head port — Phase 0 investigation report

**Date.** 2026-05-25
**Author.** code-review-debug-complexity-expert subagent (read-only)
**Scope.** Spec [2026-05-25-qg-head-port-and-pae-metric-correlations-design.md](2026-05-25-qg-head-port-and-pae-metric-correlations-design.md) §4 Phase 0.
**Mode.** Read-only. No file writes outside this report, no env touches.

---

## 1. Executive summary

Two threads were run.

**Thread A — Quality-graft architecture inventory.** Confirmed the adaptor / head / boltz-vendor manifest from the source. The relevant production setting is `n_attn_layers=1`, `num_heads=16` (`/mnt/storage01/home/schekmenev/projects/quality-graft/configs/model/quality_graft.yaml:14-24`); the standalone adaptor config is on `num_heads=8` and is not the one used in the smoke trajectory (`adaptor.yaml:20`). The pairformer student stack is 4 layers with `pairwise_head_width=32`, `pairwise_num_heads=4`, `dropout=0.2` (`confidence_head/student.yaml`). Loss is pure CE on hard pLDDT bins + optional KL-distill from a Boltz teacher; `lightning_module.py` configures AdamW(1e-4, wd=1e-2), 500-step linear warmup, linear decay to 5e-6, bf16, `gradient_clip_val=1.0`, accumulate 3 (`configs/training/default.yaml:1-30`). Vendored Boltz manifest is a pairformer-only slice; transitive imports give 9 files. Quality-graft consumes the same trunk intermediates complexa exposes (`s, z, local_latents, ca_coords`), but its trunk is La-Proteina's `LD1_ucond_notri_512.ckpt`, trained on AFDB monomers only.

**Thread B — Complexa 0.78-floor leak audit.** AUDIT-2 (centering), AUDIT-3 (label-into-batch), and AUDIT-1 (`local_latents` differential leak) are NO LEAK by code. AUDIT-0 (data-corpus diff) is the surviving candidate and matches the user's prior: complexa.ckpt's trunk was trained on Teddymer (~587k AFDB-derived synthetic dimers) + 45,856 PDB multimers + 78,368 PLINDER pairs in addition to the AFDB monomers La-Proteina saw. The Teddymer training set was filtered with `interface_pLDDT > 70` and `ipAE < 10` (see the README block at `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml:25-44` and the Complexa paper summary). Quality-graft's trunk has not seen any pLDDT-labelled dimer set; complexa.ckpt's trunk has, and the `s` representation at `trunk_eval_t=0.99` is already partially label-correlated. The recommended diagnostic is a single-epoch run on the existing legacy `complexa_filter==true` subset vs the geometry-only set on a fresh-init head; if the legacy filter run starts higher than 0.78, the data-corpus hypothesis is confirmed and the rewrite must not block on it.

---

## 2. Quality-graft architecture inventory

### 2.1 AdaptorModule

Source: `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/adaptor.py`.

Confirmed structure:

- **Single projection.** `nn.Sequential(LayerNorm(776), Linear(776, 384, bias=False))` with input `cat([trunk_seqs, local_latents], dim=-1)`. Built at `adaptor.py:236-239`.
- **Pair projection.** `nn.Sequential(LayerNorm(256), Linear(256, 128, bias=False))` on `trunk_pair`. Built at `adaptor.py:242-245`.
- **Distogram.** One-hot Cα–Cα distogram with `n_bins = pair_proj[1].out_features = 128`, bin limits `linspace(0.1, 3.0, n_bins - 1)` in nanometres, `torch.bucketize` then `F.one_hot`. Added to `z` as a *one-hot residual* (no learnt projection on the distogram itself). `adaptor.py:262-293`, applied at `:350`.
- **Source mode.** `"trunk"` is the production path; `"hybrid"` adds `decoder_fusion = LayerNorm + Linear(768, 768, bias=False, zero-init)` and gates a decoder-seqs fusion into `trunk_seqs` (`adaptor.py:227-232`, `:331-333`). Complexa has no decoder analogue — port the `"trunk"` mode only (consistent with the design spec §5.2).
- **Attention block(s).** `AdaptorAttentionBlock` is one `AttentionPairBias(c_s=384, c_z=128, num_heads=16, initial_norm=True)` + two single-layer MLPs `Sequential(LayerNorm, Linear(bias=False))`, both `Linear.weight` zero-initialised so the block is a near-identity residual transform at init. `adaptor.py:82-122`.
- **Production-config attention depth.** `n_attn_layers=1` per the full quality-graft assembly config at `quality_graft.yaml:21`. The standalone `adaptor.yaml:19-20` ships `n_attn_layers=1, num_heads=8`; the quality_graft.yaml override is what is loaded (Hydra precedence — last in `defaults:`). The architecture spec in the complexa design (§5.2) pins `n_attn_layers=1, num_heads=16` — matches the assembled config, ignore the dangling standalone yaml.
- **Mask discipline.** Both `s = s * mask[..., None]` and `z = z * mask[:,:,None,None] * mask[:,None,:,None]` are re-applied after the attention residual at `adaptor.py:156-157`. **Important:** the *projection* outputs (`single_proj(s)`, `pair_proj(z)`) are NOT re-mask-multiplied before the distogram add and the attention block. The LN-bias-leak invariant (`CLAUDE.md` §"LayerNorm-bias-leak invariant") would require `* mask` after each `single_proj` / `pair_proj`. This is a latent bug in quality-graft; the port to complexa must add the mask multiply on each post-LN projection.

### 2.2 ConfidenceHead body

Two heads exist; only the student is relevant for the complexa port:

- **`BoltzConfidenceHead`** (`confidence_head.py:54`): frozen, 152.7M params, loads the full 48-block `boltz.model.modules.confidence.ConfidenceModule` from `boltz1_conf.ckpt`. Bypasses MSA / atom encoder / input embedder. Not the architecture we're porting.
- **`StudentConfidenceHead`** (`student_head.py:27`): 4-layer `PairformerModule(token_s=384, token_z=128, num_blocks=4, num_heads=16, dropout=0.2, pairwise_head_width=32, pairwise_num_heads=4)` + final `LayerNorm(s)` + `Linear(s, num_plddt_bins=50)`. PDE/resolved heads default-off (`student.yaml:16-17`). `predict_pde=False` => no `final_z_norm`, no `to_pde_logits`. `forward` builds `pair_mask = mask[:,:,None] * mask[:,None,:]`, calls `self.pairformer(s, z, mask=mask, pair_mask=pair_mask)`, applies `final_s_norm`, returns `{"plddt_logits": Linear(s)}` (`student_head.py:101-150`).

For the complexa multi-head case, the design (§5.2) needs both `plddt` and `pae` readouts, so the port keeps `self.final_z_norm` and adds a `PaeHead` consuming `z` (asymmetric — `PaeHead._predict` should NOT symmetrise; this matches the current complexa convention `nn/confidence/pae_head.py`).

### 2.3 Boltz-1 dependency manifest

Computed transitive closure starting from `boltz.model.modules.trunk.PairformerModule` and `boltz.model.layers.attention.AttentionPairBias` (the two top-level imports the adaptor + student head make). Files under `/mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/`:

- `model/__init__.py` — empty (verified by `grep "from boltz\|import boltz"` returning no output).
- `model/layers/__init__.py` — empty.
- `model/layers/attention.py` — `AttentionPairBias`. Imports `boltz.model.layers.initialize`. (`attention.py:5`)
- `model/layers/initialize.py` — `final_init_`, `gating_init_`, `lecun_normal_init_`, `trunc_normal_init_`, `bias_init_zero_`, `bias_init_one_`, `glorot_uniform_init_`, `he_normal_init_`. Pure functions, no project imports.
- `model/layers/transition.py` — `Transition`. Imports `boltz.model.layers.initialize`.
- `model/layers/dropout.py` — `get_dropout_mask`. No project imports.
- `model/layers/triangular_mult.py` — `TriangleMultiplicationOutgoing`, `TriangleMultiplicationIncoming`. Imports `boltz.model.layers.initialize`.
- `model/layers/triangular_attention/__init__.py` — empty.
- `model/layers/triangular_attention/attention.py` — `TriangleAttentionStartingNode`, `TriangleAttentionEndingNode`. Imports `boltz.model.layers.triangular_attention.{primitives, utils}`.
- `model/layers/triangular_attention/primitives.py` — primitive attention building blocks (Linear, Attention, LayerNorm). Imports `boltz.model.layers.initialize` and `boltz.model.layers.triangular_attention.utils`.
- `model/layers/triangular_attention/utils.py` — utilities (`permute_final_dims`, `flatten_final_dims`, chunking helpers). Pure.
- `model/modules/__init__.py` — empty.
- `model/modules/trunk.py` — `PairformerModule`, `PairformerLayer` (+ other classes we don't need: `InputEmbedder`, `MSAModule`, `MSALayer`, `DistogramModule`). Imports `boltz.data.const` (uses only `const.chunk_size_threshold`, value 384, at trunk.py:538), `boltz.model.layers.{attention, dropout, transition, triangular_attention, triangular_mult}`, AND `boltz.model.modules.encoders.AtomAttentionEncoder` at the module top-level. **This top-level import is the one trap** — it pulls `boltz.model.modules.encoders` which pulls `boltz.model.modules.{transformers, utils}` and `boltz.data.const`. The port must either (a) split `PairformerModule` / `PairformerLayer` into a fresh file with only the required imports, or (b) vendor `encoders.py`, `transformers.py`, `utils.py` in addition (heavier but more verbatim). Recommendation (a) — surgical extraction.
- `model/modules/encoders.py` — needed only because `trunk.py` imports `AtomAttentionEncoder` at module scope. If we use option (a), this file is NOT needed in the vendor manifest.
- `data/__init__.py` — empty.
- `data/const.py` — only `chunk_size_threshold = 384` is used; the rest (chain_types, residue tables) is irrelevant. Vendor the whole module or expose a stub with just that one constant.

**Minimum vendor manifest (Option a, surgical):**

1. `community_models/boltz/__init__.py` — empty.
2. `community_models/boltz/model/__init__.py` — empty.
3. `community_models/boltz/model/layers/__init__.py` — empty.
4. `community_models/boltz/model/layers/initialize.py` (verbatim).
5. `community_models/boltz/model/layers/attention.py` — `AttentionPairBias` (verbatim, edit `import boltz.model.layers.initialize as init` → `from community_models.boltz.model.layers import initialize as init`).
6. `community_models/boltz/model/layers/transition.py` — `Transition`.
7. `community_models/boltz/model/layers/dropout.py` — `get_dropout_mask`.
8. `community_models/boltz/model/layers/triangular_mult.py` — `TriangleMultiplicationOutgoing`, `TriangleMultiplicationIncoming`.
9. `community_models/boltz/model/layers/triangular_attention/__init__.py` — empty.
10. `community_models/boltz/model/layers/triangular_attention/attention.py` — `TriangleAttentionStartingNode`, `TriangleAttentionEndingNode`.
11. `community_models/boltz/model/layers/triangular_attention/primitives.py`.
12. `community_models/boltz/model/layers/triangular_attention/utils.py`.
13. `community_models/boltz/model/modules/__init__.py` — empty.
14. `community_models/boltz/model/modules/pairformer.py` — **extracted** `PairformerModule` + `PairformerLayer` from upstream `trunk.py:424-653`. Drop the `AtomAttentionEncoder` import at top. Replace the `from boltz.data import const` with a single inlined constant `_CHUNK_SIZE_THRESHOLD = 384` and use it where `const.chunk_size_threshold` appears.
15. `community_models/boltz/README.md` — provenance (upstream repo `https://github.com/jwohlwend/boltz`, commit hash that quality-graft vendored from, MIT license).

External runtime dependency: `fairscale` (`from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper` at trunk.py:4) used only when `activation_checkpointing=True`. The complexa port should default `activation_checkpointing=False` and guard the import (`try/except ImportError` or conditional import).

Total: 12 files (excluding the two `__init__.py`s and the README) — well under the 8-file rough estimate in the design.

### 2.4 Loss / metrics recipe

Source: `/mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/training/lightning_module.py`, `metrics.py`.

- **Hard CE on bins.** `F.cross_entropy(student_logits.view(-1,50), plddt_labels.view(-1), reduction="none", ignore_index=-1)` masked-mean (`lightning_module.py:142-148`). 50 bins, equal width 1/50 over `[0, 1]` (`plddt_utils.py:9-23`: `floor(plddt * 50)`). Note: quality-graft works on AF2 pLDDT scaled to `[0, 1]`; complexa works on AF2 pLDDT in `[0, 100]` with 50 bins of width 2 (`AddPLDDTFromBFactor` / `AddPLDDTFromParentAFDB` constructor at `transforms.py:3587-3594`). Both binning schemes are equivalent up to a constant scale factor; no porting hazard.
- **Soft KL distillation (optional).** When `batch.get("plddt_logits")` is present (Boltz teacher logits), `F.kl_div(log_softmax(student/T), softmax(teacher/T)) * T**2` with `T=2.0` and convex blend `(1-alpha) CE + alpha KL`, `alpha=0.7` (`lightning_module.py:67-68`, `:151-162`). **Complexa does NOT have a precomputed teacher; this path is unused.** Port only the hard-CE path; no SmoothL1 / EV term (the existing complexa `PLDDTHead` recipe is `ce_weight=1.0, ev_weight=0.0` for the multihead config — `distillation_teddymer_multihead.yaml:36-37` — so the head's existing recipe is already CE-only, and the port is a no-op on the loss side).
- **No PAE in quality-graft.** Quality-graft only distils pLDDT. The PAE part of the port reuses complexa's existing `PaeHead` recipe (CE on PAE bins, masked-mean). PaeHead lives at `src/proteinfoundation/nn/confidence/pae_head.py` and is unchanged by quality-graft's recipe.
- **Validation metrics.** quality-graft (`metrics.py`): top-1 accuracy, MAE on continuous expected-value pLDDT, per-protein Pearson R averaged across batch, per-protein Spearman R averaged across batch (`metrics.py:43-148`). All in-Python tensor ops; the per-protein loop at `pearson_r:90-107` and `spearman_r:118-148` is O(B) but tiny. Complexa has richer metrics (`expected_calibration_error`, `expected_calibration_error_adaptive`, `plddt_mae_stratified`, reliability-diagram already on `PLDDTHead._predict` — `plddt_head.py:131-135`) — keep complexa's superset. No regression.

### 2.5 Training-loop knobs

Source: `lightning_module.py`, `configs/training/default.yaml`.

| Knob | Quality-graft | Complexa current (multihead-Teddymer) |
| --- | --- | --- |
| Optimizer | AdamW (lightning_module.py:256-261) | AdamW (lightning_module.py:471-478) |
| LR | 1e-4 (training/default.yaml:3) | (same default, lightning_module.py:105) |
| Weight decay | 1e-2 (default.yaml:4) | 1e-2 (lightning_module.py:106) |
| Betas | (0.9, 0.999) (default.yaml:5) | (0.9, 0.999) (lightning_module.py:107) |
| Warmup | 500 steps linear (default.yaml:9) | 500 steps cosine (lightning_module.py:108, `_cosine_warmup_factor:80-95`) |
| LR decay | Linear to `min_lr=5e-6` (default.yaml:10) | Cosine to `min_lr=5e-6` (lightning_module.py:109, `:80-95`) |
| EMA | None (no EMA module in lightning_module.py or quality_graft.py) | None |
| Grad clip | 1.0 (default.yaml:19) | (set via Hydra `training.gradient_clip_val`; see distillation_teddymer_multihead.yaml:66) |
| Precision | bf16 (default.yaml:18) | bf16-mixed (sbatch `trainer.precision=bf16-mixed`) |
| Accumulate grad batches | 3 (default.yaml:20) | (controlled by `training.confidence_distill` block — separate config) |
| val_check_interval | n/a — uses `check_val_every_n_epoch: 50` (default.yaml:27, with `limit_val_batches: 3.0`) | 1000 steps (`distillation_teddymer_multihead.yaml:69`) |
| DDP strategy | `auto` (default.yaml:23) | `DDPStrategy(find_unused_parameters=True, static_graph=True)` (`distillation_teddymer_multihead.yaml:60-63`) |
| Early stopping | `val/plddt_accuracy max patience=5` (default.yaml:32-36) | None (per CLAUDE.md memory `multihead_teddymer_ce_only.md`: "no early stopping (deliberate)") |
| `gradient_as_bucket_view` | not set | not set |

LR schedule diff (linear vs cosine decay) is a real shape-of-trajectory difference, but it does NOT explain the 0.78 vs 0.2 *starting* Pearson. The starting value is set by `f_head(s_init, z_init)` where the head is freshly initialised — the LR schedule has not run yet.

### 2.6 Data pipeline differences vs complexa

**Quality-graft (SwissProt monomers, `swissprot.yaml`):**

- AlphaFoldDB v4 SwissProt subset (`source_dir: /mnt/labs/shared/databases/swissprot_pdb_v4/files`), random 0.94/0.03/0.03 train/val/test (`swissprot.yaml:11`), per-residue B-factor → pLDDT extracted at load time, binned via `plddt_to_bin(plddt, 50)` — `floor(plddt * 50)`, `plddt_utils.py:22`.
- `min_length: 128, max_length: 256` (`default.yaml:13-14`).
- Format: PDB.
- The trunk consumes `coords_nm`, `coord_mask`, `mask`, `residue_type`. **No pLDDT-keyed field enters the trunk input.** `LaProteinaWrapper._ensure_batch_fields:333-369` and `_corrupt_batch_at_fixed_t:401-441` show only geometry / residue-type / fm-state keys. `batch["plddt_logits"]` (teacher) and `batch["plddt_bin"]` (label) are added by the datamodule for the loss step only.
- `trunk_eval_t=0.99` (`la_proteina_wrapper.yaml:20`; assembled-config override has `t_value=1.0` at `quality_graft.yaml:32`, but the README/spec calls 0.99; the production runs used 0.99 per `swissprot.yaml` defaults). **Not under audit (AUDIT-4 excluded by user prior).** Cross-reference only.

**Complexa (Teddymer dimers, `teddymer_with_plddt_and_pae.yaml`):**

- Teddymer dimer view (parent-AFDB per-residue pLDDT + per-pair PAE), `interface_length > 10` geometry-only filter (yaml:46, per memory `teddymer_geometry_only_filter.md`).
- Bin width 2.0, max_bins 50, scale_max 100.0 — equivalent to `floor(plddt / 2.0)` on AF2 0-100 scale.
- `CenteringTransform(center_mode="full", data_mode="bb_ca")` (yaml:75-77) → centers all atoms on the Cα-COM of ALL Cα atoms (no chain selectivity). See AUDIT-2 below.
- The trunk consumes the same geometry + residue-type + fm-state keys. No pLDDT/PAE field enters the trunk input (verified by greps in §3.1 AUDIT-3).

**Conclusion of §2.6:** the trunk-input contracts are equivalent — same `(coords_nm, residue_type, mask, fm-state)`, no label injection in either pipeline. The dataset diff is in the *upstream training data* of the frozen trunk weights (§3 AUDIT-0), not in the runtime feature flow.

---

## 3. Complexa 0.78-floor leak audit

### 3.1 Audit checklist

#### AUDIT-0 — Data-corpus diff (LEADING, ~60% of report depth)

**Verdict.** **LEAK by differential exposure** (HIGH confidence). complexa.ckpt's trunk has been trained on data that La-Proteina's trunk has not: a Teddymer dimer corpus filtered with `interface_pLDDT > 70` and `ipAE < 10`, plus 45,856 PDB multimers and 78,368 PLINDER pairs. The Teddymer filter alone establishes a positive selection on the very quantities the confidence head must predict — even though the *labels* never enter the trunk's input, the trunk's *weights* were trained to produce on-distribution outputs on a training set that was thresholded on those labels. The trunk has therefore learned features whose distribution over the held-out validation set is partly explained by the same pLDDT/PAE labels the head is being trained to predict. A linear readout on those features (which is essentially what the head produces at init for `s` — pre-stack `_predict = LayerNorm + Linear(token_dim, n_bins)`, see `plddt_head.py:67-86`) already starts well above chance.

**Evidence — corpus diff between La-Proteina and complexa.**

*La-Proteina (LD1_ucond_notri_512.ckpt) training corpus:*
- AFDB monomers only. Source: La-Proteina README (https://github.com/NVIDIA-Digital-Bio/la-proteina), quoted: "all our models are trained on subsets of the AFDB, not on the PDB". WebFetch result above. No multimer / dimer / PDB / PLINDER component.

*Proteina-Complexa (complexa.ckpt) training corpus (per [NVIDIA's complexa README at file:line], confirmed below):*
- AFDB monomers (~344,508 structures — flow-matching pretraining stage).
- **Teddymer dimers** (~587,687 valid IDs by file count) — `/mnt/storage01/home/schekmenev/projects/complexa-flex/assets/data/teddymer_valid_ids.txt` (counted via `wc -l`). The Complexa paper applies the binder-training thresholds `interface_pLDDT > 70`, `ipAE < 10`, `interface_length > 10` to construct the supervised training set (legacy `complexa_filter` column, see [teddymer_with_plddt_and_pae.yaml:25-32](file:///mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/teddymer_with_plddt_and_pae.yaml)). **The `interface_pLDDT > 70` cut is the load-bearing leak vector — it is a positive selection on the very quantity the confidence head must predict.**
- **PDB multimers** (45,856 IDs) — `/mnt/storage01/home/schekmenev/projects/complexa-flex/assets/data/pdb_multimer_ids.txt`. Filtered by resolution etc. (no pLDDT cut, since these are experimental — but the trunk learns a different `s` distribution from PDB-real vs AFDB-predicted, which is also label-correlated since real PDB structures correlate with high AFDB pLDDT in the regions AlphaFold modelled well).
- **PLINDER** (78,368 IDs) — `/mnt/storage01/home/schekmenev/projects/complexa-flex/assets/data/plinder_valid_ids.txt`. Protein-ligand for the LoRA ligand-binder variant; less relevant to the protein-binder distillation we care about, but contributes another label-correlated geometry distribution.

**Evidence — the trunk consumes the *output* of an AlphaFold-confidence-thresholded sampler.** The Complexa paper's stagewise pipeline (per the WebSearch and WebFetch results above) is:

> autoencoder pretrain on AFDB monomers → autoencoder fine-tune on PDB → flow-matching pretrain on AFDB Foldseek monomers → protein-binder training on Teddymer (filtered) + PDB multimers → small-molecule binder training on PLINDER + AFDB via LoRA.

complexa.ckpt and complexa_ae.ckpt are the protein-binder branch — they have both seen the Teddymer + PDB-multimer stage. La-Proteina's LD1 stops at "flow-matching pretrain on AFDB monomers", never reaches the binder stage. This is exactly the differential the user's prior describes.

**Mechanism — how the trunk leaks the label via pretrain.** The head's input is `s = trunk(x_t=Teddymer-dimer, t=0.99)`. The trunk was trained to denoise / predict `bb_ca` and `local_latents` on dimers selected for `interface_pLDDT > 70`. At inference (`t=0.99`, near-clean data manifold), the trunk's `s` representation at a residue `i` encodes:

1. local backbone geometry of `i` (always),
2. plus features that the *training distribution* taught the trunk to produce on residue `i`.

Because the training set was thresholded on high interface pLDDT, the residues the trunk has seen most are residues that AF2 was confident about. These have specific structural signatures (low local strain, satisfied H-bonds, low side-chain rotamer entropy) that the trunk has implicitly learned to reproduce in `s`. At eval time, a *test* dimer with low pLDDT in some region will have `s` features in that region that are out-of-distribution for the trunk's learned manifold — and crucially, *systematically different* from high-pLDDT features. A linear head reads this difference as a discriminator on the label, even at random init, because the random-init head sees a `(low-pLDDT-region-feature, high-pLDDT-region-feature)` separation that ALREADY EXISTS in `s` independent of the head's weights.

This is a *bona fide* representation prior: complexa's trunk did not "see the label", but it was *trained on the conditional distribution given the label exceeded 70*. Sampling at eval `t=0.99` from a held-out dimer with a low-pLDDT region produces an `s` that the trunk treats as somewhat out-of-distribution; the random-init head's softmax over 50 bins is dominated by the mean-bin response in the in-distribution case and shifted away in the OOD case, which is exactly the structure of the label.

**Falsification recipe (recommended diagnostic, §3.3).** Run a fresh-init head one-epoch on the *legacy* Teddymer training subset (`complexa_filter == True`, ~95% overlap with the trunk's training set) vs the *geometry-only* set. If the legacy-filter run starts at >0.78 *and* the geometry-only run starts at <0.78, the data-corpus hypothesis is confirmed: the floor is the trunk's training prior, exactly inheriting the trunk's training filter.

**What this means for PR #1.** AUDIT-0 is **not a bug to fix in the head pipeline**. It is the *signal* the head is supposed to read — the trunk has implicit knowledge of confidence, and we'd actually like the head to exploit it. The 0.78 floor is the floor under which the head cannot do worse, not the ceiling. Quality-graft starting at 0.2 means quality-graft's trunk has *less* implicit pLDDT knowledge — La-Proteina's trunk was trained only to denoise AFDB monomer coordinates and the AFDB monomer corpus itself was less thresholded on pLDDT (or thresholded differently — see "what could re-rank this" below). PR #1 should therefore proceed: the rewrite cannot fix what is structurally a feature, not a bug. The post-rewrite trajectory will likely still start at ~0.78 — if it does, that's expected; if it starts lower than expected, that's a secondary diagnostic worth running.

**What could re-rank this verdict.**

1. If La-Proteina's AFDB monomer subset is *also* `pLDDT > 70`-filtered (the LD1 readme does not specify the threshold; the paper would). Then both trunks should have a comparable label prior, and the 0.78 vs 0.2 differential would have to be explained by *something else*. Open question for §4.
2. If the diagnostic experiment in §3.3 shows the geometry-only set also starts at ~0.78, then it is not the training-filter mechanism but a more fundamental fact about complexa's trunk (e.g. PDB multimers carry strong label correlation even without an explicit cut). Verdict would stand but the mechanism would need reframing.

#### AUDIT-1 — `local_latents` carries pLDDT-correlated information (~10% of report depth, LOW probability)

**Verdict.** **NO DIFFERENTIAL LEAK** — quality-graft consumes the exact same `local_latents` channel (8-dim per-residue latent emitted by the frozen trunk's `local_latents_linear`) and observes 0.2 starting Pearson. If `local_latents` carried a label-correlated signal *independent of the trunk's training corpus*, the quality-graft run would also start above 0.5, and it does not.

**Evidence.**

- complexa's trunk emits `local_latents_out = local_latents_linear(seqs) * mask[..., None]` at `local_latents_transformer.py:325`. The head's `ConfidenceTrunk.local_latents_proj = Linear(8, 768, bias=False) + LayerNorm(768)` projects and adds it to `s` as a mask-zeroed residual at `nn/confidence/base.py:106-109, :146-147`.
- quality-graft's `LaProteinaWrapper._trunk_forward` (`la_proteina_wrapper.py:515-518`) emits the analogous tensor: `local_latents_out = self.trunk.local_latents_linear(seqs) * mask[..., None]`. The adaptor concatenates it onto `trunk_seqs` before the single-projection (`adaptor.py:343`). Quality-graft sees the same shape of signal.
- Because the *mechanism* (latents projected and added/concat into the head's `s`) is shared, any differential leak must come from differential trunk training (already covered by AUDIT-0), not from the latent path itself.

**Open subtlety.** complexa's head adds latents via a *residual* with a `Linear(8, 768)` projection at full token-dim, then applies `LayerNorm` and the head's attention stack. Quality-graft *concatenates* latents (8) onto `trunk_seqs` (768) before a `LayerNorm + Linear(776, 384)`. Different surfaces, same information content. No differential leak.

#### AUDIT-2 — `CenteringTransform(full, bb_ca)` exposes chain-level pLDDT

**Verdict.** **NO LEAK.**

**Evidence.** `CenteringTransform.__call__` at `src/proteinfoundation/datasets/transforms.py:1681-1768`. For `center_mode="full", data_mode="bb_ca"`:

- The centering mask is `graph.coord_mask[:, 1]` (line 1692) — the Cα atom mask for ALL residues across BOTH chains. No chain selection. No pLDDT or PAE weighting.
- The coords used are `graph.coords_nm[:, 1, :]` (line 1742) — Cα coordinates.
- The shift is `masked_mean = mean_w_mask(coords, centering_mask.bool(), keepdim=True)` (line 1748), an unweighted mean over valid Cα atoms.

The transform is a pure rigid translation. It cannot leak the label because (a) the label is per-residue / per-pair, not a single scalar; (b) the rigid shift is invariant to which residues are high-pLDDT vs low-pLDDT — every residue contributes equally; (c) the trunk is translation-equivariant on `bb_ca` (the COM is removed on the data side AND inside `_force_zero_com` per-step on the model side, per the yaml comment at `teddymer_with_plddt_and_pae.yaml:65`), so any residual translation is a no-op for the trunk's `s` output.

No further investigation needed.

#### AUDIT-3 — GT pLDDT / PAE labels accidentally enter a trunk feature dict

**Verdict.** **NO LEAK.**

**Evidence.**

- `AddPLDDTFromParentAFDB.__call__` (`transforms.py:3596-3629`) writes ONLY `data.plddt_residue`, `data.plddt_bin`, `data.plddt_mask`. No other field.
- `AddPAEFromParentAFDB.__call__` (`transforms.py:3657-3696`) writes ONLY `data.pae_residue_pair`, `data.pae_bin`, `data.pae_mask`. No other field.
- The trunk source (`src/proteinfoundation/nn/local_latents_transformer.py`, `src/proteinfoundation/nn/protein_transformer.py`, `src/proteinfoundation/nn/feature_factory/`) makes no reference to `plddt` or `pae` (grep returned no output across the whole `nn/` tree except inside `nn/confidence/`).
- The only readers of `batch["plddt_*"]` / `batch["pae_*"]` are the confidence heads: `PLDDTHead.compute_loss_and_metrics` at `plddt_head.py:103-104`, `PaeHead.compute_loss_and_metrics` (similarly), `PLDDTSequenceOnlyHead.compute_loss_and_metrics` at `plddt_sequence_only_head.py:124-125`. These are *post*-trunk loss readers, not feature readers.
- Confirmed by a global grep: `grep -rn "plddt\|pae" src/proteinfoundation/nn/{local_latents_transformer.py,protein_transformer.py,feature_factory/}` returns NO output.

No leak path exists in the batch-dict feature flow.

#### AUDIT-4 — `trunk_eval_t = 0.99` saturation

**EXCLUDED by user binding prior. No verdict.**

#### AUDIT-5 — Cross-reference with quality-graft

Quality-graft consumes the architecturally-equivalent set of trunk intermediates `(trunk_seqs, trunk_pair, local_latents, ca_coords)` and starts at Pearson ≈ 0.2. Complexa consumes `(s, z, mask, cond, local_latents)` and starts at ≈ 0.78.

The audits along *shared paths*:

- AUDIT-1 (`local_latents` content): shared → must be NO differential leak by quality-graft's contrary observation. Confirmed.
- AUDIT-2 (centering / COM equivariance): shared (both pipelines centre coords on Cα-COM before flow-matching processing) → NO LEAK. Confirmed.
- AUDIT-3 (label injection into batch): shared (neither pipeline injects labels into trunk-facing keys) → NO LEAK. Confirmed.

The audit along a *differential* path:

- AUDIT-0 (trunk training corpus): NOT shared. complexa.ckpt = AFDB monomers + Teddymer dimers (pLDDT-filtered) + PDB multimers + PLINDER. LD1_ucond_notri_512.ckpt = AFDB monomers only.

**Verdict (cross-reference).** **PROBABLE-CAUSE-IS-DATA-EXPOSURE-DIFF (AUDIT-0).** The shared-path audits cannot explain the differential. The non-shared-path audit (AUDIT-0) is the unique surviving candidate that satisfies the constraint "complexa shows the floor; quality-graft does not".

### 3.2 Probability ranking of root cause

1. **AUDIT-0** — data-corpus / pretrain-filter differential. **~75% probability.** Matches user's prior, matches the cross-reference constraint, the corpus diff is concretely observed (587k + 46k + 78k extra training samples, with explicit pLDDT/iPAE thresholds in the Teddymer filter), and the mechanism (selection-bias-encoded representation prior) is a well-known property of pretrained backbones. Falsifiable by §3.3.
2. **AUDIT-2** — chain-level COM leak. **~0% probability.** Code says full Cα-COM, no chain selection or pLDDT weighting. Ruled out by code inspection.
3. **AUDIT-3** — label-into-batch leak. **~0% probability.** Code says only confidence heads read pLDDT/PAE keys. Ruled out by grep.
4. **AUDIT-1** — `local_latents` differential leak. **~5% probability.** Architecturally shared with quality-graft; cannot explain the differential by itself. Residual probability reserved for the possibility that complexa's *latent_dim=8* trunk encodes more pLDDT-correlated information than La-Proteina's because complexa fine-tuned the autoencoder on PDB (introducing PDB-vs-AFDB distributional information into `local_latents` that quality-graft's AE never saw). Not a "leak" so much as another data-corpus path; should be folded into AUDIT-0 if it materialises.
5. **AUDIT-5** — cross-reference verdict re-affirms AUDIT-0. ~20% reserved for "unknown mechanism" given the limits of static analysis (e.g. an LR-schedule or precision artefact that delays gradient flow enough that we don't observe the head leaving 0.78 even though it would, given enough training — unlikely given Quality-graft's *trajectory* into 0.99, but logged for the record).

### 3.3 Recommended diagnostic experiment

**One ablation, before PR #1 lands.**

Hold the head fresh-init. Train one epoch each on two Teddymer subsets, identical in every other respect (same trunk weights, same `trunk_eval_t=0.99`, same DDP wiring, same LR schedule, same precision):

- **Run A:** the legacy `complexa_filter == True` subset (the trunk's own training distribution).
- **Run B:** the geometry-only `interface_length > 10` subset (the current production filter).

Log the *very first val Pearson R* (`val/plddt/pearson_r`) at step ~100 (after warmup completes) and at end-of-epoch-1.

Expected outcomes:

- **Run A starts at ≥ 0.78, ends higher.** Run B starts at ≤ 0.78, ends comparable. → AUDIT-0 confirmed. The 0.78 floor is a property of evaluating on the trunk's own training-distribution. No fix; PR #1 proceeds.
- **Both runs start at 0.78.** → AUDIT-0 partially refuted. The trunk's pLDDT correlation is broader than just its training-filter subset. Still consistent with the "extra-corpus exposure" story, just not localised to the legacy filter. PR #1 proceeds.
- **Both runs start near 0.2.** → AUDIT-0 refuted. Something else is going on. Re-open AUDIT-1 and look for an LR/precision/init artefact. PR #1 should pause until the cause is identified.

**Cost.** One epoch on ~600k Teddymer samples on 2×2 h100nvl takes ≤ 6 h based on the existing multihead trajectory (val every 1000 steps, current run takes ~0.5 h/1000 steps). With val every 100 steps for the first 200 steps then val every 1000 thereafter, one epoch ≤ 8 h. Two epochs = 16 h, plus a few hours for the val-frequency change to take effect and for sbatch queueing. **Recommended to run A and B in parallel on two h100 nodes** if quota permits, otherwise sequentially.

**If diagnostic is impractical.** PR #1 proceeds without it. The audit's qualitative case for AUDIT-0 is strong enough on its own to justify the design's §4.6 "PR #1 proceeds anyway if no fixable leak is found" path. The 0.78 floor was always best explained by *what the trunk has seen*, not by a head-side bug.

---

## 4. Open questions for main thread

1. **What is La-Proteina's AFDB pLDDT cutoff?** The La-Proteina README does not state the threshold. If LD1's AFDB subset is also `pLDDT > 70`-filtered, AUDIT-0's "differential" framing weakens; we'd need to dig deeper into Teddymer-vs-AFDB *dimer-vs-monomer* features. Worth a 5-minute check of the La-Proteina paper (arxiv 2507.09466) §Methods / §Data. WebFetch on the abstract page yielded nothing — the full paper PDF would need to be parsed by hand.

2. **Was complexa.ckpt trained with the legacy `complexa_filter` Teddymer subset only, or with multiple data revisions?** The repo's `assets/data/teddymer_valid_ids.txt` lists 587,687 IDs, which matches the *post-filter* count in the dataset config comment (562,089 with `interface_length > 10`, 587,687 total in `dimers.parquet` per memory `teddymer_geometry_only_filter.md`). Confirm with the user whether complexa.ckpt was trained on the legacy `complexa_filter == True` subset (≈ 510k IDs) or on the broader geometry-only set (≈ 562k).

3. **Do we have access to "complexa_filter==True" rows in the live blob?** The dataset config comment at `teddymer_with_plddt_and_pae.yaml:42` says "the `complexa_filter` column is still on `dimers.parquet` but unused by the runtime filter". The §3.3 diagnostic requires being able to flip the filter at config time. Worth confirming the column is still readable from the production blob (`/netscratch/schekmenev/teddymer_v1_blob/dimers.parquet`).

4. **Boltz-1 vendor — extract `PairformerLayer` standalone vs vendor `encoders.py` whole?** Spec §5.2 says "minimum vendored Boltz-1 slice" but does not specify which option. Recommended: extract `PairformerModule` + `PairformerLayer` into a clean `community_models/boltz/model/modules/pairformer.py` (Option a in §2.3) and drop the `AtomAttentionEncoder` import that the upstream file does at module scope. This requires a deliberate, documented edit to the verbatim-port rule.

5. **`fairscale.checkpoint_wrapper` runtime dependency.** Upstream `PairformerModule.__init__` imports it at module top-level. For complexa's port, default `activation_checkpointing=False` and guard the fairscale import. Confirm `fairscale` is in the pinned venv (`env/build_uv_env.sh`) before relying on conditional import; if absent, ship a graceful fallback ("`activation_checkpointing=True` requires fairscale; install via `uv pip install fairscale`").

6. **Mask discipline in `AdaptorModule._predict`.** Upstream `single_proj(s)` and `pair_proj(z)` outputs are NOT mask-multiplied before downstream attention (`adaptor.py:344, :347`). Per the LN-bias-leak invariant in `CLAUDE.md`, the port must add `* mask` after each post-LN projection. Confirm this is in scope for PR #1's TDD checklist.

7. **Does the user want §3.3 run before PR #1?** Design §4.6 says "If Thread B identifies no fixable leak, PR #1 proceeds anyway". My read of the evidence is that no fixable leak exists — AUDIT-0 is a property of the pretrained trunk, not a bug. The diagnostic in §3.3 is *confirmatory*, not *blocking*. Recommend PR #1 starts in parallel with the diagnostic; a single non-confirming result would NOT block PR #1, only adjust the expectations in PR #1's description.

---

**Sources (web-fetched, for AUDIT-0 corpus diff):**
- [La-Proteina arxiv abstract page](https://arxiv.org/abs/2507.09466) — confirms training on AFDB subsets only.
- [La-Proteina GitHub README](https://github.com/NVIDIA-Digital-Bio/la-proteina) — confirms "all our models are trained on subsets of the AFDB, not on the PDB"; specifies LD1 = unconditional latent diffusion, no triangular update, up to 500 residues; AE1 = corresponding autoencoder.
- [Proteina arxiv abstract](https://arxiv.org/abs/2503.00710) — Proteina backbone generator, trained on up to 21M AFDB structures with CATH fold-class conditioning.
- [Proteina-Complexa project page (NVIDIA Research)](https://research.nvidia.com/labs/genair/proteina-complexa/) — Complexa's 4-dataset training corpus (AFDB monomers, Teddymer, PDB multimers, PLINDER) and the stagewise pipeline.
- [Complexa OpenReview PDF](https://openreview.net/pdf/d0d1b6e1faa5cedf2608f66712167ccf67e079da.pdf) — Teddymer construction details (47M AFDB50 → 10M dimers → 3.5M Foldseek clusters → filtered training set with `interface_pLDDT > 70`, `ipAE < 10`, `interface_length > 10`).

---

INVESTIGATION COMPLETE.
