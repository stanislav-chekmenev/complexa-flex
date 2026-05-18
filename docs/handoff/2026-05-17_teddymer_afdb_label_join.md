# Session handoff — Teddymer + AFDB label join for interface confidence distillation

**Date:** 2026-05-17
**Topic:** Extending confidence-head distillation from monomer pLDDT to interface rewards (pLDDT + directional PAE) using a Teddymer-derived dataset enriched with AFDB v4 labels.

## Goal

Extend the confidence-head distillation subsystem (currently trained on AFDB monomer pLDDT only, see [src/proteinfoundation/confidence/](../../src/proteinfoundation/confidence/) and [src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/)) so that it can distil **interface** confidence rewards (`i_pae`, `i_ptm`, `min_ipae`, `min_ipsae`, etc.) used by Complexa's test-time search. The plan is to train on a Teddymer-derived dataset enriched with per-residue pLDDT and per-pair PAE pulled from the AFDB v4 mirror.

## Key technical findings established this session

### 1. AFDB v4 bulk mirror — what's there
- Path: `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/` — 1,015,797 `proteome-tax_id-*_v4.tar` files, the full ~214M AFDB v4 dump.
- An existing **selection-and-annotation view** sits at `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/la_proteina_afdb_512_v1/` (344,507 La-Proteina training IDs ≤ 512 residues). Contains `view_config.yaml`, `ids.parquet`, `locator_rows.parquet` (byte offsets into the raw tars per entry, with `has_cif/has_pae/has_conf` flags), `enriched_metadata.parquet` (sequence/length/md5).
- For each AFDB entry the tar carries three gzipped members:
  - `*-model_v4.cif.gz` — Atom37 CIF, **per-residue pLDDT stored in the B-factor column** of CA atoms, scale [0, 100].
  - `*-confidence_v4.json.gz` — standalone per-residue pLDDT: `residueNumber`, `confidenceScore`, `confidenceCategory`. Length matches `modeled_seq_len`.
  - `*-predicted_aligned_error_v4.json.gz` — **full L×L PAE matrix as nested lists**, plus `max_predicted_aligned_error` (clipped to ~31.75 Å). PAE is **directional** (not symmetrised) in the raw data.

### 2. AFDB50, AFDB-Clusters, Teddymer relationships
- "AFDB50" = AFDB v4 clustered at 50% sequence identity / 80% coverage by MMseqs2 → ~2.3M cluster representatives (Barrio-Hernandez et al., Nature 2023).
- Teddymer (Complexa §3.1, App. C) is derived from **AFDB50 + TED domain annotations**. It splits multi-domain AFDB monomers into TED domains and treats spatially proximal domain pairs from the *same monomer* as synthetic "dimers." 10M raw dimers → 3.55M clusters → 587k non-singleton reps + 2.97M singletons. Complexa trained on **510,454 filtered cluster reps** (filter: `ipLDDT > 70`, `ipAE < 10`, `interface_length > 10`).

### 3. AF2 reward decomposition (Complexa rewards) — [src/proteinfoundation/rewards/alphafold2_reward.py](../../src/proteinfoundation/rewards/alphafold2_reward.py)
Walked through every reward in [alphafold2_reward.py](../../src/proteinfoundation/rewards/alphafold2_reward.py) and [alphafold2_reward_utils.py](../../src/proteinfoundation/rewards/alphafold2_reward_utils.py), grounded in ColabDesign's `af/loss.py` (path: `community_models/colabdesign/af/loss.py`). Classification:
- **pLDDT-derived**: `plddt`.
- **PAE-derived**: `pae`, `i_pae`, `min_ipae`, `i_ptm` (`1 - ipTM` with standard d0/TM kernel), `i_ptm_energy` (log-space TM kernel for smoother gradients), `min_ipsae`/`max_ipsae`/`avg_ipsae` + `_10` cutoff variants (ipSAE = AF2-PAE with hard cutoff τ and Cα-frame max-over-i reduction).
- **Distogram-head-derived** (not in AFDB JSONs): `con`, `i_con`, `helix`, `helix_binder`, `dgram_cce`.
- **Other**: `exp_res` (experimentally-resolved head; not in AFDB JSONs), `rg` (CA radius of gyration vs `2.38*L^0.365`), `nc_termini` (N–C terminus distance vs 7 Å), `alignment_bb_ca[_binder]` (Kabsch CA RMSD vs design).

**Computable from AFDB alone**: `plddt`, whole-protein `pae` (avg), `rg`, `nc_termini`. **Requires multimer prediction**: all `i_*`, `ipsae*`, `i_ptm*`. **Requires distogram head**: `con`, `i_con`, `helix*`, `dgram_cce`.

### 4. Monomer-vs-multimer labelling caveat (load-bearing)
- AFDB ships **monomer** predictions only. For domain pair (A, B) from the *same* AFDB monomer, the inter-domain `PAE[i ∈ A, j ∈ B]` block is the AF2 monomer prediction's PAE on those residue indices — **not** what AF2-multimer/AF3/Boltz would predict if (A, B) were folded as two separate chains.
- This is exactly the regime Teddymer's own `IntPlddt` column and `ipAE < 10` filter use, so back-filling AFDB labels into Teddymer is **provenance-consistent with the data Complexa actually saw**.
- However: distilled heads trained on these labels are biased relative to AF2-multimer's `ipAE` at test time. Acceptable for guidance *during Complexa generation* (same training regime); needs a calibration set of re-folded multimers if used as a literal substitute for AF2-multimer reward at test time.

### 5. Should interface rewards be distilled directly, or trained on monomers and applied to multimers?
**Distil interface rewards directly on Teddymer.** Reasons: (i) `IntPlddt` is a different label distribution from monomer pLDDT (incorporates multimer pair-stack reasoning in the Teddymer regime, even if computed from monomer JSONs); (ii) interface-PAE residue pairs span chains and have no monomer-only analogue; (iii) trunk `(s, z)` statistics at interfaces are out-of-distribution relative to monomer training; (iv) order-statistic rewards (`min_ipae`, etc.) are tail-sensitive and need to see real interface PAE distributions.

**Recommended approach for the distilled heads**: distil the dense per-residue / per-pair labels, *not* the scalar reductions. Compute `min`/`max`/`avg` reductions over the predicted dense fields at inference, so the head doesn't lock in a particular interface mask.

### 6. Pair-rep symmetrisation note (cross-reference)
[CLAUDE.md](../../CLAUDE.md) flags `z = (z + z.transpose(-3, -2)) / 2.0` at the end of `ConfidenceTrunk.forward` ([src/proteinfoundation/nn/confidence/base.py:129](../../src/proteinfoundation/nn/confidence/base.py#L129)) as needing revision when the first asymmetric head lands. Directional PAE (the seed signal in this work) is exactly that asymmetric head. **The symmetrisation must move from the trunk down into per-head `_predict` before the asymmetric PAE training run.** Two options: (a) add `symmetrise_z: bool = True` flag, (b) remove from trunk and require each `_predict` to symmetrise. Option (b) is cleaner long-term.

## Teddymer release inspected

Local copy at `~/Downloads/teddymer/` on user's laptop. Contents:
```
teddymer/
├── cluster.tsv                                    # 3.55M memId → repId rows
├── nonsingletonrep_metadata.tsv                   # 587k non-singleton cluster reps
│   # cols: DimerIndex, UniProtID, DomainPair, MemberCount,
│   # InterfaceLength, AvgIntPAE, AvgIntPlddt, IntPlddt (colon-separated per-chain)
├── dir_ted_afdb50_cath_dimerdb/                   # Foldseek DB of full 10M unclustered dimers
└── teddymer_repdb/                                # Foldseek DB of 3.5M cluster reps
    ├── teddymer_repdb (+_h, _ca, _ss, .lookup, .source, *.dbtype, *.index)
```

**`teddymer_repdb_h` is self-contained.** Format inspected from sample:
```
<dimer_index>DI_AF-<UniProtID>-F1-model_v4_TED<dd> \t CATH<cath_id>_RES<chopping>
```
Carries parent AFDB ID, TED domain index, CATH classification, and full residue chopping (including **discontinuous domains** like `RES15-30_37-122_289-300`) inline per chain. Two chains of a dimer share the `<dimer_index>DI_AF-<UniProtID>-F1` prefix.

**Join recipe** (no external TED Zenodo lookup needed):
- Parse `_h` regex: `^(\d+)DI_AF-([A-Z0-9]+)-F1-model_v4_TED(\d+)\tCATH([0-9.]+)_RES(.+)$` → per-chain table with `dimer_index`, `uniprot_id`, `parent_afdb_id = "AF-{uniprot_id}-F1"`, `ted_index`, `cath_id`, `residue_intervals: list<(lo, hi)>`.
- Group by `dimer_index` (each dimer has exactly two `_h` rows).
- Join to `nonsingletonrep_metadata.tsv` on `dimer_index` for cluster size + interface annotations + per-residue `IntPlddt`. **Assert `DomainPair` matches the two TED suffixes found in `_h`.**
- All Teddymer dimers are intra-monomer (both chains from same AFDB ID) per the paper, so one AFDB JSON lookup per dimer suffices for both pLDDT and L×L PAE (including the directional cross-domain block — the seed signal for the asymmetric PAE head).

## Architectural decision: storage layout

**Final destination** (chosen, will be requested once parquet processing is verified):
```
/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/teddymer_v1/
```

Rationale: structurally analogous to neighbouring `views/la_proteina_afdb_512_v1/`. Teddymer is a **derived view of AFDB v4** — every atom and every label recoverable from the AFDB mirror via `parent_afdb_id`. Co-locating with AFDB makes provenance explicit and lets existing data loaders that consume AFDB bulk via `locator_rows.parquet` work unchanged. Rejected: top-level `databases/teddymer_v1/` (implies separate source, false), `inventory_first/derived/` (for genuinely derived artefacts like embeddings/alt-clusterings that no longer point back to raw bytes).

Planned subtree:
```
views/teddymer_v1/
  view_config.yaml                  # provenance: Teddymer release date, AFDB inventory snapshot, processing SHA
  _raw/                             # original Teddymer release (kept for provenance)
    nonsingletonrep_metadata.tsv
    teddymer_repdb/
      teddymer_repdb_h, .index, .lookup, .source, .dbtype
  dimers.parquet                    # one row per non-singleton cluster rep
  clusters.parquet                  # full cluster.tsv as parquet (if/when needed)
  locator_rows.parquet              # join to AFDB tar offsets, per chain
```

## Current status

- User cannot write to `/mnt/labs/shared/`. Files **currently being transferred** from laptop to a home-directory staging area on the cluster. Parsing and parquet build will be done in the user's home first; the move to the shared databases tree will be arranged after the parquet view is verified.
- Staging layout under home mirrors the final destination so the eventual move is a single `mv`.
- **Minimal file set being shipped** (~300–500 MB total):
  - `nonsingletonrep_metadata.tsv`
  - `teddymer_repdb/teddymer_repdb_h`
  - `teddymer_repdb/teddymer_repdb_h.index`
  - `teddymer_repdb/teddymer_repdb.lookup`
  - `teddymer_repdb/teddymer_repdb.source`
  - `teddymer_repdb/teddymer_repdb.dbtype`
- **Excluded** (deliberately): `cluster.tsv` (singletons + non-reps, not needed), `dir_ted_afdb50_cath_dimerdb/*` (unclustered, not needed), `teddymer_repdb` + `_ca` + `_ss` Foldseek payloads (coordinates/3Di recoverable from AFDB; not needed for distillation).

## Next session — concrete next steps

1. **Confirm transfer landed** in the user's home staging directory.
2. **Parse `_h` + `nonsingletonrep_metadata.tsv` → `dimers.parquet`** with schema:
   - `dimer_id` (str, = repId), `dimer_index` (int64), `uniprot_id` (str)
   - `parent_afdb_id_A` / `_B` (str), `ted_index_A` / `_B` (int32), `cath_id_A` / `_B` (str)
   - `residue_intervals_A` / `_B` (list<struct{lo, hi}>) — preserve discontinuous domains
   - `member_count`, `interface_length`, `avg_int_pae`, `avg_int_plddt` (from metadata)
   - `int_plddt_chain_A` / `_B` (list<float>, parsed from colon-separated `IntPlddt`)
   - **Assert per row**: `DomainPair` matches the two `TED<dd>` suffixes; both chains have same `parent_afdb_id` (intra-monomer); `len(int_plddt_A) + len(int_plddt_B)` is consistent with `InterfaceLength` (determine per-chain vs total convention from a sample).
3. **Build `locator_rows.parquet`** by joining `parent_afdb_id` against AFDB master inventory at `inventory_first/inventory/manifests/batches/` (or the `la_proteina_afdb_512_v1/locator_rows.parquet` if subset coverage suffices — but Teddymer parents include longer multi-domain proteins, so probably need master inventory).
4. **Sanity-check on ~50 dimers**: pull parent CIF from AFDB tar at recorded offset, extract residues per `residue_intervals_A`, mask to interface (Cα–Cα < 8 Å across chains), compute mean B-factor pLDDT, compare to `avg_int_plddt` from metadata. Mismatches = residue-numbering bug, fix before generating training labels.
5. **Filtered subset view** `views/teddymer_v1_complexa_filter/` (or boolean column in `dimers.parquet`) for the 510k entries Complexa actually trained on: `interface_pLDDT > 70`, `ipAE < 10`, `interface_length > 10`.
6. **Dataset config + transforms**: `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` (analogue of [afdb_monomers_with_plddt.yaml](../../configs/dataset/unified/afdb_monomers_with_plddt.yaml)) plus `AddPLDDTFromParentAFDB` + `AddPAEFromParentAFDB` data transforms that load lazily from parent AFDB JSONs at dataloader time (don't cache PAE matrices to disk; they're up to 1 MB per dimer densified).
7. **Before the first asymmetric PAE training run**: move the `z = (z + z.T)/2` symmetrisation out of `ConfidenceTrunk` ([base.py:129](../../src/proteinfoundation/nn/confidence/base.py#L129)) into per-head `_predict`, per the CLAUDE.md flag. Symmetric heads (pLDDT, PDE, ipLDDT, ipTM) opt in; asymmetric PAE head opts out.

## Project conventions to keep in mind

- Work on a feature branch off `dev` (main branch); PR into `dev`.
- TDD via subagents (planner → implementer → reviewer panel). `code-review-debug-complexity-expert` mandatory on every PR; add domain agents based on what the PR touches.
- No Claude attribution in commits or PRs.
- `gh` requires `module load gh` in the same Bash invocation.
- Pinned env: uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5 <2.6.

## Reference paths

- AFDB v4 mirror: `/mnt/labs/shared/databases/afdb_v4_bulk/`
- Existing AFDB view (template for Teddymer view): `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/views/la_proteina_afdb_512_v1/`
- ColabDesign loss source: `/mnt/labs/home/schekmenev/projects/complexa-flex/community_models/colabdesign/af/loss.py`
- Confidence subsystem: [src/proteinfoundation/confidence/](../../src/proteinfoundation/confidence/), [src/proteinfoundation/nn/confidence/](../../src/proteinfoundation/nn/confidence/)
- Confidence trunk (symmetrisation to revisit): [src/proteinfoundation/nn/confidence/base.py:129](../../src/proteinfoundation/nn/confidence/base.py#L129)
- AF2 reward: [src/proteinfoundation/rewards/alphafold2_reward.py](../../src/proteinfoundation/rewards/alphafold2_reward.py), [src/proteinfoundation/rewards/alphafold2_reward_utils.py](../../src/proteinfoundation/rewards/alphafold2_reward_utils.py)
