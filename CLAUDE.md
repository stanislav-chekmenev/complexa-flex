# CLAUDE.md

Persistent notes for Claude across conversations on this project. Capture important things from our discussions here — decisions, conventions, gotchas, and project-specific guidance that should survive across sessions.

## How to use this file
- Add entries as we discover them during conversations.
- Keep entries concise and actionable. Prefer short bullets over prose.
- Group related items under the appropriate section. Add new sections as needed.
- Remove or update entries that become stale rather than letting them rot.

## Project overview
- Goal: train a protein generative model that consumes **sequence + structure + reciprocal-space (X-ray reflection) data** jointly.
- Reflection data must be **coarsened** to fit a single protein on GPU alongside the other modalities. Dev set: 394 proteins.
- Current phase: data analysis + representation-learning method selection. **No production code yet** — repo root is scaffolding.

## Conventions & preferences
- Use [CLAUDE.md](CLAUDE.md) for cross-conversation findings (per user feedback).
- All analysis lives under [analysis/dev/](analysis/dev/). Representation-learning specs under [docs/representation_learning/](docs/representation_learning/).
- Random seed `SEED = 42` everywhere reproducibility matters.
- Auto mode is the default working style: act, don't ask, course-correct as needed.

## Architecture notes
- **Recommended primary representation**: per-protein-normalised 3D voxel grid in reciprocal space at **n=32**, channels `(mean log|F|², count)`, stored sparsely (COO). ~19 KiB/protein.
- **Side-channels**: cell tensor (6 floats), space-group token (one of 65 chiral protein groups), Wilson radial profile (~100 floats).
- Two surviving representation-learning plans, both targeted at a **2-week dev-set bake-off** as a *proof-of-concept and method-selection harness* (NOT a representation-quality benchmark):
  - [03_masked_lm.md](docs/representation_learning/03_masked_lm.md) — **MAE** primary; sparse voxel-token sequence, 64-bin equal-population vocabulary on Wilson-E, 8L×384 encoder, 4L×192 decoder, RoPE-3D, 60% mask, block masking r=2.
  - [01_vae_voxel.md](docs/representation_learning/01_vae_voxel.md) — **VAE** secondary; 4-stage 3D CNN encoder/decoder, latent 64, FiLM conditioning, log-normal NLL on E² + occupancy BCE + free-bits KL + symmetry penalty.
- Contrastive (VICReg/MoCo) was evaluated and **dropped**: lowest signal-density-per-item at 394-protein scale.
- Final review: [00_final_review.md](docs/representation_learning/00_final_review.md). Independent cross-check: [00b_independent_cross_check.md](docs/representation_learning/00b_independent_cross_check.md).

## Environment & tooling
- Project uses **uv** for Python env management (see `pyproject.toml`, `env/build_uv_env.sh`).
- Recent stack work: bumped to **torch 2.10 + CUDA 13** on data_analysis branch.
- Analysis stack: `gemmi`, `reciprocalspaceship`, `pandas`, `numpy`, `matplotlib`, `pyarrow`, `tqdm`, Jupyter.
- The notebook builder is `analysis/dev/build_notebook.py` — regenerates [coarsening_strategy_analysis.ipynb](analysis/dev/coarsening_strategy_analysis.ipynb) from a single source script.

## Data
- Dev set: 394 mmCIF structure factor files at [data/dev/structure_factors/](data/dev/structure_factors/) (`*_sf.cif`).
- Each file has `_refln.index_h/k/l`, `_refln.F_meas_au`, `_refln.F_meas_sigma_au`, `_refln.status`, plus `_cell.*` and `_symmetry.space_group_name_H-M`.
- Cached parsed reflections: [analysis/dev/artifacts/all_reflections.parquet](analysis/dev/artifacts/all_reflections.parquet) (22,576,472 rows, all 394 proteins, sentinel-filtered).
- Per-protein metadata: [analysis/dev/artifacts/per_protein_meta.parquet](analysis/dev/artifacts/per_protein_meta.parquet).
- Per-protein voxel statistics: [analysis/dev/artifacts/voxel_per_protein.parquet](analysis/dev/artifacts/voxel_per_protein.parquet).
- Anisotropy/asu/budget/truncation/strategy artifacts also under [analysis/dev/artifacts/](analysis/dev/artifacts/).

## Key empirical findings (data analysis)
- Reflection-list scale: **median 47k reflections/protein, p99 219k, max 304k**. Raw `(h,k,l,|F|,σ)` lists are 1–6 MiB/protein — too big for GPU batching.
- **Wilson normalisation matters.** Radial form-factor decay swamps anisotropic signal unless you Wilson-normalise to E. η on |F| ≈ 0.46 for radial 100-bin and ≈ 0.52 for voxel n=32; **on E** they diverge: radial → 0.003, voxel → 0.11. Voxel grid carries 10–75× more *anisotropic* signal than radial shells.
- **Anisotropy is real and large**: q-cloud PCA eigenvalue ratio median 3.8, p99 17.6. Cell-edge ratio median 1.75, p99 5.3.
- **Voxel sparsity**: median ~7.5% occupancy at n ∈ {16,20,32,48}. Sparse-COO costs ~10× less than dense. n=32 sparse fits in ~19 KiB/protein.
- **Voxel nonempty-count at n=32** (per [voxel_per_protein.parquet](analysis/dev/artifacts/voxel_per_protein.parquet)): **median 2475, mean 3325, p99 9115** — NOT ~250 (an earlier MAE spec draft cited 250; that was wrong by ~10×).
- **ASU**: deposited CIFs are merged (`n_total/n_unique = 1.000`, no Friedel duplicates), but **~88% of representatives lie outside gemmi's canonical ASU**. Canonicalising costs nothing but is mandatory for cross-protein representations.
- **Resolution truncation is a poor primary lever**: cutting at 3.0 Å keeps only 33% of reflections at 21% of Σ|F|² loss, and Σ|F|² is dominated by low-res Wilson tail so info loss is worse than the number suggests.

## Gotchas
- **Filter `_refln.status='x'`, plus `F=0 AND σ=0` placeholders, plus σ ≥ 9999 explicitly.** Status='x' canonically carries `F=0, σ=9999` but some depositors leave high-σ rows under other status codes — the explicit σ < 9999 guard is needed. ~42k sentinel rows in the dev set; treating them as zero amplitude pollutes per-protein moments.
- **Don't use the orthorhombic 1/d² shortcut.** Dataset includes hexagonal / monoclinic / triclinic cells. Use `gemmi.UnitCell.calculate_d` or `gemmi.ReflnBlock.make_d_array()` (full reciprocal metric tensor).
- **Per-protein normalisation papers over cross-protein cell-shape differences** — fine for per-protein voxel modelling but means raw indices are NOT in a canonical frame across proteins. ASU canonicalisation BEFORE voxellisation fixes this.
- **49 space groups in 394 proteins, 15 singletons, 26 with ≤ 3 examples** — informational probes (SG classification, anisotropy regression) are statistically thin on dev. Report support-coverage alongside any probe number.
- **Dev set goal is method selection and proof-of-concept, NOT representation quality.** Don't conflate dev-scale embedding quality with full-PDB target quality. Mechanism diagnostics (loss descent shape, anisotropic-vs-radial control margin, conditioning-leakage gap dynamics, scaling-curve slope) are the bar.

## Open questions / TODO
- Verify MAE `max_seq_len` against the canonicalised-wedge nonempty-voxel quantile table — the cross-check flagged the spec's `max_seq_len=1536` as ~10× too small. (Issue 1 in [00b_independent_cross_check.md](docs/representation_learning/00b_independent_cross_check.md).)
- Tighten M2 (anisotropic-vs-radial margin) with a context-only neighbour-copy control to rule out trivial shortcuts.
- Specify CD-HIT 30% within-cluster nesting for the {100, 200, 394} scaling-curve experiment; report both fixed-steps and fixed-epochs slopes.
- Add `I(z; x | c)` panel to VAE M4 (per-dim KL alone misses collapse-into-conditioning).
- Add decoder-RoPE-scramble ablation as a named MAE failure-mode diagnostic.
- Build the full-PDB pipeline (~150k entries) — required for the eventual real run regardless of which method wins the bake-off.
