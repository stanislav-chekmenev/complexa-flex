# PR-A — Teddymer data plumbing — implementation plan

**Date:** 2026-05-17
**Owner of execution:** ml-protein-architect (green phase); code-review-debug-complexity-expert (red phase, test author)
**Branch base:** `prepare_teddy` (do NOT branch off `dev`; inherits glob fix `48b2804`)
**Target merge branch:** `prepare_teddy` (downstream PR to `dev` after PR-B)
**Predecessor handoff:** [docs/handoff/2026-05-17_teddymer_interface_confidence_distillation.md](../../handoff/2026-05-17_teddymer_interface_confidence_distillation.md)
**Scope discipline:** PR-A only. PR-B (trunk symmetrisation refactor + `PaeHead` + `MultiHeadConfidence`) is OUT of scope.

## 1. Goal & non-goals

### Goal
Make `teddymer_with_plddt_and_pae` a Hydra-instantiable dataset that yields padded batches with per-residue pLDDT labels and directional per-residue-pair PAE labels, sourced from random-access reads into parent AFDB tars via `locator_rows.parquet`. The dataset must already be consumable by the existing pLDDT confidence sidecar (PR-B-independent smoke run) and, once PR-B lands, by the asymmetric `PaeHead` natively.

### Non-goals
- No changes to `proteinfoundation.proteina` or to the main `train.py`.
- No model code, no head class, no loss, no metric, no Lightning module changes — those are PR-B.
- No trunk symmetrisation refactor.
- No new sbatch / training entry — `distillation_teddymer_pae.yaml` and `distillation_teddymer_multihead.yaml` are PR-B.
- No move of the view from `/mnt/storage01/home/schekmenev/data/teddymer_v1/` to `/mnt/labs/shared/...`.

## 2. Locked decisions (carried from user, do not re-open)

| # | Decision | Effect on this plan |
| --- | --- | --- |
| 1 | Filter at dataset level (`complexa_filter == True`, ~510k rows) | Encoded as `filters: ["complexa_filter == True"]` in the new yaml. Datamodule already supports this at [src/proteinfoundation/datasets/structure_data.py:916](../../../src/proteinfoundation/datasets/structure_data.py#L916). |
| 2 | PAE binning = AF2 standard, 64 bins on `[0, 31.75]` Å, bin width 0.5 Å | Default constructor args on `AddPAEFromParentAFDB`; expose `pae_to_bin` helper alongside `plddt_to_bin` in `confidence/losses.py`. |
| 3 | Trunk symmetrisation removed entirely (CLAUDE.md option (b)) | PR-A's transform emits **directional** PAE — no symmetrisation. `pae_residue_pair[i, j] != pae_residue_pair[j, i]` by construction. |
| 4 | First PAE run is single-head | PR-A only ships data; both single- and multi-head configs in PR-B consume the same fields. |
| 5 | Combined loss weights `0.9 * CE + 0.1 * SmoothL1(EV)` | Not emitted in PR-A; flagged here so any reviewer comparing against an older spec uses the correct numbers. |

## 3. Context & constraints

- **View on disk (verified):** `/mnt/storage01/home/schekmenev/data/teddymer_v1/{dimers.parquet (587,687 rows), locator_rows.parquet (1,175,374 rows = 587,687 × 2, 100% coverage), view_config.yaml}`. AFDB raw at `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/`.
- **dimers.parquet has no `path` column.** Schema: `dimer_id, dimer_index, uniprot_id, parent_afdb_id, ted_index_A, ted_index_B, cath_id_A, cath_id_B, residue_intervals_A, residue_intervals_B, member_count, interface_length, avg_int_pae, avg_int_plddt, int_plddt_chain_A, int_plddt_chain_B, complexa_filter`. Both chains share `parent_afdb_id` (intra-monomer dimers).
- **locator_rows.parquet schema** (1 row per chain, 2 rows per `dimer_index`): `dimer_id, dimer_index, chain_id ∈ {A, B}, sample_id, afdb_id, uniprot_id, taxonomy_id, source_tar_relpath, source_tar_basename, source_version, bucket, has_{cif,pae,conf}, {cif,pae,conf}_member_{name,offset,size}, {cif,pae,conf}_is_gz`. Both rows for the same dimer share offsets (single AFDB tar serves both chains).
- **PAE payload verified empirically:** AFDB `*-predicted_aligned_error_v4.json.gz` decodes to `[{"predicted_aligned_error": <L×L int64>, "max_predicted_aligned_error": <int>}]`. Values are integer Å on `[0, 31]`. AF2's PAE is on `[0, 31.75]` with 0.25 Å precision in the float JSON, but AFDB rounds to integers — we use integers as-is and bin into 64 × 0.5 Å (the integer payload yields populated bins at every other index; that is a property of AFDB's storage and is not a bug).
- **Confidence payload verified:** `*-confidence_v4.json.gz` decodes to `{"residueNumber": [...], "confidenceScore": [...], "confidenceCategory": [...]}` — a per-residue pLDDT array on `[0, 100]`. This is **cheaper to read than CIF B-factors** (~2 KB vs ~80 KB) and avoids re-decoding CIF. Use it.
- **Pinned env:** uv-managed, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5,<2.6. `complexa-flex` package layout under `src/proteinfoundation/`. No parallel packages.
- **CLAUDE.md TDD discipline:** write tests first; tests must fail because the artefact is missing, not because of setup error.

## 4. Stakeholders / handoffs

- **Consumer 1 (immediate):** the existing pLDDT confidence sidecar at [src/proteinfoundation/confidence/](../../../src/proteinfoundation/confidence/) — the dataset must expose `plddt_residue`, `plddt_bin`, `plddt_mask` with the same shape & dtype contract `AddPLDDTFromBFactor` produces, so a pLDDT-on-Teddymer smoke run trains immediately and serves as the PR-A end-to-end check.
- **Consumer 2 (PR-B):** the asymmetric `PaeHead` — expects `pae_residue_pair` directional (L × L float on `[0, 31.75]`), `pae_bin` (L × L int64 on `[0, 63]`), `pae_mask` (L × L bool), and a precomputed `pae_inter_chain_block` (L_A × L_B float convenience view).
- **No external collaborators.** Wet-lab handoff and paper figures are downstream of PR-B's training run.

## 5. Data contracts (load-bearing)

All tensors are on CPU at dataset time; the dataloader collates and the trainer moves to device. `L = len(residue_intervals_A) + len(residue_intervals_B)` is the per-dimer token count (sum of resolved residues across both TED domains, post-discontinuity expansion).

| Field | dtype | shape | range / semantics | source |
| --- | --- | --- | --- | --- |
| `plddt_residue` | float32 | `[L]` | `[0, 100]`; per-residue AF2 pLDDT | `confidenceScore[r-1]` for `r` in expanded residue set from `residue_intervals_{A,B}` |
| `plddt_bin` | int64 | `[L]` | `[0, 49]`; `plddt_to_bin(plddt_residue, bin_width=2.0, num_bins=50)` | derived |
| `plddt_mask` | bool | `[L]` | `True` where the residue is resolved in the AFDB parent (always True here; AFDB confidence array is dense, but emit False for any residue whose number falls outside the AFDB sequence length to fail-safe) | derived |
| `pae_residue_pair` | float32 | `[L, L]` | `[0, 31.75]` Å, **directional** (no symmetrisation) | submatrix of AFDB `predicted_aligned_error` at the row/column indices `expanded_residues - 1` |
| `pae_bin` | int64 | `[L, L]` | `[0, 63]`; `pae_to_bin(pae_residue_pair, bin_width=0.5, num_bins=64)` | derived |
| `pae_mask` | bool | `[L, L]` | `True` everywhere both endpoints are resolved; False otherwise. NO within-chain vs cross-chain distinction at this stage (loss-side masking is PR-B's call). | derived |
| `pae_inter_chain_block` | float32 | `[L_A, L_B]` | view: `pae_residue_pair[:L_A, L_A:]` — convenience for the asymmetric head | derived |
| `chain_id` | int8 | `[L]` | `0` for chain-A residues, `1` for chain-B residues, in the same token order as `plddt_residue` | derived from `residue_intervals_A/B` expansion |

### Index frame (load-bearing — the residue-numbering bug the integration spot-check targets)

- `residue_intervals_{A,B}` are **1-indexed AFDB-parent residue numbers**, possibly discontinuous (e.g. `RES15-30_37-122` decodes to `[{lo:15, hi:30}, {lo:37, hi:122}]`).
- AFDB `confidenceScore` is **1-indexed by `residueNumber`** (the JSON includes the indices explicitly; assert `confidenceScore[i] == confidenceScore_by_residueNumber[i+1]` once at module import, then index by `r - 1`).
- AFDB `predicted_aligned_error` is an `[L_full, L_full]` matrix where row `i` and column `j` correspond to AFDB residue numbers `i+1` and `j+1`. Slice as `pae[expanded - 1, :][:, expanded - 1]` where `expanded` is the flat residue-number vector concatenated `[chain_A_residues, chain_B_residues]`.
- The integration spot-check (test #4 below) is the **only** test that binds this index frame to ground truth via `interface_length` and `avg_int_plddt` columns. If it fails, PR-A is wrong.

### NaN policy

- Treat any residue number outside `[1, L_full]` as `plddt_mask=False`, `pae_mask=False`, and zero-fill the float fields (do not emit NaN downstream).
- Treat a dimer whose parent confidence JSON has `len(confidenceScore) != L_full` (where `L_full` = `pae.shape[0]`) as a load failure → `__getitem__` returns `None` with a logged warning (existing dataloader skip path).

## 6. Module / file layout

### New files
- `src/proteinfoundation/datasets/teddymer/io.py` — pure IO helpers, no torch. Pull `pae`, `confidence`, and `cif` bytes by `(tar_path, offset, size, is_gz)`. Extends the sanity-check pattern already in [src/proteinfoundation/datasets/teddymer/sanity_check.py:41](../../../src/proteinfoundation/datasets/teddymer/sanity_check.py#L41) by adding `read_pae_from_tar(...) -> np.ndarray` and `read_cif_bytes_from_tar(...) -> bytes`. Also exposes the existing `read_confidence_from_tar`. Single source of truth.
- `src/proteinfoundation/datasets/teddymer/dataset.py` — new file. Hosts `TeddymerDimerDataset(torch.utils.data.Dataset)`. **Do not put this in `structure_data.py`** — that file is already 2k+ LOC and routes through CIF-on-disk; mixing in a tar-offset path bloats it. Subclass of nothing (composes — see §"Reuse"). `__getitem__(idx) -> Data | None` builds the dimer Data object end-to-end:
  1. Look up `dimer_index` from `dimers.iloc[idx]`.
  2. Get the two locator rows (chain A, chain B) via a pre-built `dimer_index -> (loc_A, loc_B)` dict (built once in `__init__`).
  3. Read CIF bytes once (chains A and B share `cif_member_offset` because they share the parent), parse with biotite/`load_structure` from an in-memory `BytesIO`, slice atoms whose residue numbers ∈ `expanded_residues_A ∪ expanded_residues_B`, tag chain-A vs chain-B with synthetic `chain_id` annotation, ingest into the existing `atomarray_to_atom37` pathway.
  4. Run `atomarray_transforms` (the standard dimer transforms — `CroppingTransform2`, `OpenFoldFrameTransform`, etc. — verbatim from `ted_dimers.yaml`).
  5. Run `atom37_transforms` including the new `AddPLDDTFromParentAFDB` and `AddPAEFromParentAFDB`.

  The two new transforms can stay stateless if they take `locator_path` and `afdb_proteomes_root` in their `__init__` and look up `dimer_index` from the `Data` object's `example_id` (which the dataset sets to `dimer_id`). Cleaner: the dataset stashes the locator rows in `data.teddymer_locator` (a 2-row dict) so transforms read it from the Data object rather than the parquet — avoids per-worker parquet re-reads. **Choose the in-Data-object approach** — see §"Reuse" §B.

- `tests/unit/datasets/test_add_plddt_from_parent_afdb.py` (red-phase 1)
- `tests/unit/datasets/test_add_pae_from_parent_afdb.py` (red-phase 2)
- `tests/unit/datasets/test_teddymer_dataset_config.py` (red-phase 3)
- `tests/integration/test_teddymer_interface_cca_spot_check.py` (red-phase 4, slow)

### Touched files
- `src/proteinfoundation/datasets/transforms.py` — add classes `AddPLDDTFromParentAFDB` and `AddPAEFromParentAFDB` near the existing `AddPLDDTFromBFactor` at line 3466. Same `BaseTransform` parent, same output-field schema for pLDDT (so the existing pLDDT head works unchanged on Teddymer).
- `src/proteinfoundation/confidence/losses.py` — add `pae_to_bin(pae, bin_width=0.5, num_bins=64)` directly under `plddt_to_bin`. Symmetric API. No loss / weighting changes (those live in PR-B). One-line public function.
- `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` — new Hydra config (see §7).

### NOT touched
- `src/proteinfoundation/datasets/structure_data.py` — left alone. Reasoning in §"Reuse vs new".
- `src/proteinfoundation/nn/confidence/*` — PR-B territory.
- `configs/confidence/*` — PR-B territory.
- `scripts/*.sbatch` — PR-B territory.

## 7. Hydra composition

The parent of `teddymer_with_plddt_and_pae.yaml` is **NOT** `ted_dimers.yaml`, because `ted_dimers.yaml` instantiates `StructureDataModule` with `metadata_file: ted_dimers.parquet` and `path_column: path`, neither of which match the Teddymer view (no `path` column; metadata is the new parquet).

Composition tree:

```yaml
# configs/dataset/unified/teddymer_with_plddt_and_pae.yaml
defaults:
  - _self_

datamodule:
  _target_: proteinfoundation.datasets.teddymer.dataset.TeddymerDimerDataModule
  dimers_parquet: ${oc.env:TEDDYMER_VIEW_ROOT,/mnt/storage01/home/schekmenev/data/teddymer_v1}/dimers.parquet
  locator_parquet: ${oc.env:TEDDYMER_VIEW_ROOT,/mnt/storage01/home/schekmenev/data/teddymer_v1}/locator_rows.parquet
  afdb_proteomes_root: ${oc.env:AFDB_PROTEOMES_ROOT,/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4}
  batch_size: 6
  num_workers: 16
  train_split: 0.99
  id_column: dimer_id
  pin_memory: true

  # Dataset-level filter — drops the ~77k rows where complexa_filter == False
  filters:
    - "complexa_filter == True"

  # Reuse the existing dimer atomarray pipeline; the dataset class hands these
  # the atom array sliced out of the AFDB parent.
  atomarray_transforms: []  # see §Reuse — verbatim copy of the ted_dimers list
                            # is inlined here; do not re-import via defaults
                            # because ted_dimers.yaml mixes in path-on-disk
                            # assumptions.

  atom37_transforms:
    - _target_: proteinfoundation.datasets.transforms.AddPLDDTFromParentAFDB
      bin_width: 2.0
      max_bins: 50
      scale_max: 100.0
    - _target_: proteinfoundation.datasets.transforms.AddPAEFromParentAFDB
      bin_width: 0.5
      max_bins: 64
      scale_max: 31.75
      emit_inter_chain_block: true
```

Key decisions:
- **No `defaults: [ted_dimers]`** because the parent assumes `path_column: path`. We instead inline the atomarray transforms (copy them verbatim from `ted_dimers.yaml` lines 22-77). Future refactor can extract the shared list into a Hydra group; explicitly out of scope here.
- **`filters: ["complexa_filter == True"]`** at the datamodule level — uses the existing `pandas.query`-based filter path at [src/proteinfoundation/datasets/structure_data.py:916](../../../src/proteinfoundation/datasets/structure_data.py#L916). The new `TeddymerDimerDataModule` mirrors that exact contract.
- **Env-var-driven roots** with sane defaults — keeps the config self-contained but lets sbatch override for the eventual cluster `mv` to `/mnt/labs/shared/...`.

## 8. Reuse vs new

### A. Tar-byte-offset reader
- **Reuse:** [src/proteinfoundation/datasets/teddymer/sanity_check.py:41](../../../src/proteinfoundation/datasets/teddymer/sanity_check.py#L41) `read_confidence_from_tar(tar_path, offset, size)`. This is the canonical pattern in the repo (note: `grep -rli cif_member_offset src/` returns only the locator builder and this helper — there is no other tar-offset reader in the codebase; the handoff's reference to "the la_proteina_afdb_512_v1 view uses it" was about the *view layout*, not a reader in this repo).
- **New (in `teddymer/io.py`):** `read_pae_from_tar(tar_path, offset, size, is_gz=True) -> np.ndarray[int]`, `read_cif_bytes_from_tar(...) -> bytes`. Share a private `_read_tar_member_bytes(tar_path, offset, size, is_gz) -> bytes` helper. Move the existing `read_confidence_from_tar` into this module **and re-export from `sanity_check.py` for backward compatibility** (one-line `from .io import read_confidence_from_tar`).

### B. Locator lookup
- **New:** `_LocatorIndex` lazily built in the datamodule's `setup()`: `{dimer_index: (loc_row_A, loc_row_B)}` from the locator parquet. Built once, shared across train/val datasets. Each worker pickle-copies it (memory cost: ~1175k rows × ~300 B / row ≈ 350 MB per worker — tolerable at `num_workers=16`; see §"Risks").
- **Alternative considered:** parquet random-access read per `__getitem__`. Rejected — disk seek latency ≈ 1 ms × 587k epochs would dominate.

### C. Atom-array construction
- **Reuse:** the existing `atomarray_to_atom37(atom_array, sample_id=..., atomworks_data=data)` path at [src/proteinfoundation/datasets/structure_data.py:377](../../../src/proteinfoundation/datasets/structure_data.py#L377). Call it from `TeddymerDimerDataset.__getitem__` after the atom-array is sliced.
- **Reuse:** `biotite.structure.io.pdbx` / the existing `load_structure(...)` accepts a `path` *or* a file-like. Use `BytesIO(read_cif_bytes_from_tar(...))` — adapt the wrapper if needed; this is the only spot that may force a `load_structure` signature widening. If `load_structure` insists on a path, write the CIF bytes to a `tempfile.NamedTemporaryFile` per call (cheap on `/tmp` SSD; ~50k bytes). **Decision point for the implementer:** if `load_structure` already accepts a `Path`-like and gzip-aware fileobj, prefer the in-memory route; else tempfile fallback. Both paths are equivalent for the tests.
- **New:** `slice_atom_array_by_residues(atom_array, chain_A_residues, chain_B_residues) -> atom_array` — relabel chain IDs so chain A becomes `A` and chain B becomes `B` even though they originated from the same monomer. Stash this in `teddymer/dataset.py` as a private helper (not in `transforms.py`).

### D. Datamodule
- **Compose, don't subclass:** `TeddymerDimerDataModule(L.LightningDataModule)` mirrors `StructureDataModule`'s public surface (`filters`, `columns_to_load`, `cluster_column`, `train_split`, `val_metadata_file`, `pin_memory`, `batch_size`, `num_workers`, `atom37_transforms`, `atomarray_transforms`) but overrides `setup()` to load the dimers parquet + locator parquet and instantiate `TeddymerDimerDataset` instead of `StructureDataset`. Reuse the existing `_split` / `_cluster_aware_split` logic by lifting it into a free function in `structure_data.py` and importing it. (Single-line refactor.)
- **Why not subclass:** `StructureDataModule.setup` is monolithic and tightly couples to `StructureDataset` via positional kwargs; subclassing forces method overrides that would be more fragile than composition. Subclass `L.LightningDataModule` directly.

## 9. Test plan (write tests first; red phase before any green)

All four files end up in `tests/unit/datasets/` (1, 2, 3) and `tests/integration/` (4). Fixtures: a `tmp_path`-built fake AFDB tar with synthetic CIF + confidence JSON + PAE JSON members at known byte offsets; a 4-dimer fake dimers parquet; a matching locator parquet. Pattern mirrors [tests/unit/datasets/test_teddymer_build_locator.py](../../../tests/unit/datasets/test_teddymer_build_locator.py) — shared `_write_fake_*` helpers go into `tests/unit/datasets/conftest.py` (extend if it exists; create if not).

### Test 1 — `test_add_plddt_from_parent_afdb.py`
- **Fixture:** one fake AFDB tar with one parent (`AF-FAKE0001-F1`) containing CIF + confidence members at known offsets. Known per-residue `confidenceScore = [80, 50, 0, 99, 70, ...]` for 10 residues.
- **Construct:** a `Data` object whose `example_id` corresponds to a dimer with `residue_intervals_A = [{lo:1, hi:5}]`, `residue_intervals_B = [{lo:6, hi:10}]`. Pass via a stub `teddymer_locator` attribute on the Data object.
- **Asserts:**
  1. `plddt_residue.shape == (10,)`, dtype `float32`, equals `[80, 50, 0, 99, 70, ...]`.
  2. `plddt_bin == [40, 25, 0, 49, 35, ...]` (same `plddt_to_bin` arithmetic as the existing transform).
  3. `plddt_mask` is `True` everywhere both endpoints are inside the parent sequence.
  4. **Discontinuous domain:** `residue_intervals_A = [{lo:1, hi:2}, {lo:5, hi:7}]` produces `plddt_residue` of length 5 with values from residues `[1, 2, 5, 6, 7]` (in that order).
  5. **Single-residue domain:** `residue_intervals_A = [{lo:3, hi:3}]` yields length-1 tensors.
  6. **All-zero confidence:** `confidenceScore = [0]*10` yields `plddt_residue = 0`, `plddt_bin = 0`, `plddt_mask = False`.
  7. **Out-of-bounds residue:** `residue_intervals_A = [{lo:1, hi:20}]` when parent has 10 residues → transform sets `plddt_mask[10:] = False`, zero-fills, does NOT raise.
- **Failure mode (red phase):** `AttributeError: module 'proteinfoundation.datasets.transforms' has no attribute 'AddPLDDTFromParentAFDB'`.

### Test 2 — `test_add_pae_from_parent_afdb.py`
- **Fixture:** same fake AFDB tar augmented with a `*-predicted_aligned_error_v4.json.gz` member encoding `[{"predicted_aligned_error": <10×10 int matrix>, "max_predicted_aligned_error": 31}]`. Construct the matrix so that `pae[i, j] = i + j` (mod 32) — distinguishable from its transpose.
- **Asserts:**
  1. `pae_residue_pair.shape == (10, 10)`, dtype `float32`, values match the sliced submatrix exactly.
  2. **Directional:** `pae_residue_pair[2, 5] != pae_residue_pair[5, 2]` on a chosen index pair. (No symmetrisation.)
  3. `pae_bin.shape == (10, 10)`, dtype `int64`, values match `pae_to_bin(pae_residue_pair, 0.5, 64)`.
  4. **AFDB clip ceiling round-trip:** entry `pae = 31` → `pae_bin = 62` (since `31 / 0.5 = 62`). Entry `pae = 31.75` (in a synthetic float JSON) → `pae_bin = 63`. Confirms ceiling is loss-less to bin resolution.
  5. **Inter-chain block:** `pae_inter_chain_block.shape == (L_A, L_B)`, equals `pae_residue_pair[:L_A, L_A:]` byte-for-byte. View, not copy (assert `.data_ptr()` equality, or — if PyTorch copies on slicing across non-contiguous strides — assert equality via `torch.equal`).
  6. **Discontinuous domain:** the L×L matrix indexes correctly through the gap; residues `[1, 2, 5, 6, 7]` (chain A) and `[8, 9]` (chain B) produce a 7×7 matrix whose `[0, 0]` cell is `pae[0, 0]`, `[0, 2]` cell is `pae[0, 4]`, etc.
- **Failure mode (red phase):** `AttributeError: module 'proteinfoundation.datasets.transforms' has no attribute 'AddPAEFromParentAFDB'`.

### Test 3 — `test_teddymer_dataset_config.py`
- **Fixture:** `tmp_path` setup with `dimers.parquet` (4 dimers, 3 of which have `complexa_filter=True`), `locator_rows.parquet` (8 rows), and a fake AFDB tar containing CIF + PAE + confidence for both `parent_afdb_id` values referenced.
- **Compose:** `hydra.compose(config_name="dataset/unified/teddymer_with_plddt_and_pae")` with overrides setting `dimers_parquet`, `locator_parquet`, `afdb_proteomes_root` to the `tmp_path`.
- **Asserts:**
  1. `instantiate` succeeds and returns a `TeddymerDimerDataModule`.
  2. `dm.setup("fit")`; `len(dm.train_dataset) + len(dm.val_dataset) == 3` (the `complexa_filter == False` row is dropped — proves the dataset-level filter binds).
  3. One batch from `dm.train_dataloader()` contains keys `plddt_residue, plddt_bin, plddt_mask, pae_residue_pair, pae_bin, pae_mask, pae_inter_chain_block, chain_id` with the expected dtypes and 2D/3D shapes for a batch of size 1-2.
  4. **Padding sanity:** if two dimers of different L go into the same batch, `pae_residue_pair.shape == (B, L_max, L_max)` and the padded entries have `pae_mask == False`.
- **Failure mode (red phase):** Hydra error — `Could not resolve _target_ proteinfoundation.datasets.teddymer.dataset.TeddymerDimerDataModule`.

### Test 4 — `test_teddymer_interface_cca_spot_check.py` (mark `slow`)
- **Fixture:** real view at `/mnt/storage01/home/schekmenev/data/teddymer_v1/`; sample 20 dimers using a seeded RNG from those with `complexa_filter == True`.
- **For each sampled dimer:**
  1. Instantiate the datamodule on the real parquets, fetch the matching `Data` object.
  2. Compute Cα-Cα interface set: pairs `(i ∈ chain_A_residues, j ∈ chain_B_residues)` with `dist(CA_i, CA_j) < 8 Å` in the parent AFDB CIF.
  3. Assert `len(interface_set) ≈ row['interface_length']` within `±2` residues (residue-numbering frame slop).
  4. Assert `mean(plddt_residue[interface_residues_A ∪ interface_residues_B]) ≈ row['avg_int_plddt']` within `±1.0` pLDDT unit.
  5. Assert `mean(pae_residue_pair[interface_pairs])` ≈ `row['avg_int_pae']` within `±0.5` Å (this catches a transposed-PAE bug — if rows and columns swap, this average shifts because PAE is directional).
- **Why this test exists:** the spot-check is the **only** ground-truth bind for the residue-numbering frame and the directional PAE indexing. It cannot be replaced by unit tests because unit tests construct fixtures with the same convention as the code under test. PR #8 review explicitly flagged this gap.
- **Failure mode (red phase):** `ImportError` or `Could not resolve` — file does not exist yet.
- **CI policy:** marked `@pytest.mark.slow`; runs only on demand. Acceptable wall time: ~30 s for 20 dimers.

## 10. Sequencing within PR-A (sub-PR boundaries)

Two independent slices; the handoff allows parallel work but they share a thin interface. **Recommend a single sequential PR** unless two human engineers are available — the slices touch the same files (`transforms.py`, the new `teddymer/io.py`) and would conflict if landed truly in parallel.

If sequential (default):
1. **Slice 1 — IO + transforms.** `teddymer/io.py`, `AddPLDDTFromParentAFDB`, `AddPAEFromParentAFDB`, `pae_to_bin`. Tests 1 + 2 go green.
2. **Slice 2 — Dataset + config.** `teddymer/dataset.py`, `teddymer_with_plddt_and_pae.yaml`, atomarray-transform inlining. Tests 3 + 4 go green.

If parallel (only if two implementers available, e.g. ml-protein-architect on slice 2 and ml-software-pytorch-jax-expert on slice 1):
- Define the contract first: `transforms.py` exports `AddPLDDTFromParentAFDB(locator_path, afdb_root, **kwargs).__call__(data: Data) -> Data` where `data.teddymer_locator` is `{"A": loc_row_A_dict, "B": loc_row_B_dict, "intervals_A": [...], "intervals_B": [...]}`. Both engineers agree on this stub signature before either starts.
- Slice 1 produces the transforms against this stub; slice 2 produces the dataset that sets `data.teddymer_locator` correctly.
- Merge in either order.

## 11. Exit / abort criteria

### Exit (PR-A done)
- All four tests pass (slow test passes on the real view).
- `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_swissprot dataset=unified/teddymer_with_plddt_and_pae trainer.fast_dev_run=true` completes one training step using **only the pLDDT head** on Teddymer data (the existing pLDDT head + sidecar, no PR-B changes). Loss is finite and gradients flow. This is the end-to-end smoke that proves Teddymer pLDDT labels reach the existing model.
- `tests/regression/test_flow_matching_loss_unchanged.py` still passes (no trunk change yet — must remain green).
- Four-reviewer panel (§13) all approve.

### Abort (stop and rethink before pushing through)
- **Residue-frame mismatch.** If test 4 (`interface_length` check) fails on >2 of 20 sampled dimers with `±2`-residue slop, the residue-numbering convention assumed in this plan is wrong. STOP. Re-derive against the metadata source, then re-spec.
- **PAE memory blow-up.** If `num_workers=16` × max-L `~500` × 4 bytes × 4 fields exceeds 5 GB per worker peak RSS during smoke train, the densified PAE design is wrong. Fall back to either lazy `pae_bin`-only emission (skip float `pae_residue_pair`, head re-derives midpoints from bins) or move to L*L int16. STOP, re-spec, then re-merge.
- **`load_structure` doesn't accept a fileobj or tempfile-rewrite is too slow.** If per-dimer CIF parse-from-bytes is slower than 10 ms median (limits throughput at 16 workers to ~1.6k dimers/s, marginal for 510k epoch), STOP and revisit — pre-materialising sliced dimer CIFs to disk under `/mnt/storage01/home/schekmenev/data/teddymer_v1/dimer_cifs/` is the next-cheapest design.
- **Locator memory.** If the 1.175M-row locator dict exceeds 500 MB per worker (measured via `psutil` in test 3), drop to a memory-mapped pyarrow Table keyed by `dimer_index` instead. STOP, re-spec.

## 12. Verification commands

```bash
# Red phase — confirm tests fail with the right error (missing artefact, not setup error)
cd /mnt/storage01/home/schekmenev/projects/complexa-flex
uv run pytest tests/unit/datasets/test_add_plddt_from_parent_afdb.py tests/unit/datasets/test_add_pae_from_parent_afdb.py tests/unit/datasets/test_teddymer_dataset_config.py -x -v 2>&1 | head -60

# Green phase — after each slice, expect pass:
uv run pytest tests/unit/datasets/test_add_plddt_from_parent_afdb.py tests/unit/datasets/test_add_pae_from_parent_afdb.py -x -v
uv run pytest tests/unit/datasets/test_teddymer_dataset_config.py -x -v

# Full PR-A regression
uv run pytest tests/unit/datasets/ tests/unit/teddymer/ tests/regression/test_flow_matching_loss_unchanged.py -x -v

# Slow integration spot-check (requires real view + AFDB access)
uv run pytest tests/integration/test_teddymer_interface_cca_spot_check.py -m slow -x -v

# End-to-end pLDDT-on-Teddymer smoke (proves dataset reaches the existing head)
uv run python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_swissprot \
    dataset=unified/teddymer_with_plddt_and_pae \
    trainer.fast_dev_run=true \
    trainer.devices=1
```

## 13. Reviewer panel (confirmed from handoff §"Reviewers for PR-A")

Mandatory:
- **code-review-debug-complexity-expert** — correctness, edge cases, the 1 MB/dimer × N_workers PAE-densification memory budget, off-by-one in residue indexing.

Domain-add:
- **ml-protein-architect** — module layout decision (new `teddymer/dataset.py` vs extending `structure_data.py`), Hydra composition (inline atomarray transforms vs `defaults: [ted_dimers]`), config-tree conformance.
- **structural-biology-binder-expert** — Cα-Cα `< 8 Å` is the right interface definition for Teddymer's intra-monomer synthetic dimers; `interface_length` slop of ±2 is appropriate; the PR #8 residue-numbering bug is plausibly caught by test 4.
- **ml-software-pytorch-jax-expert** — dataloader worker-fork determinism with the in-memory locator dict; pinned-memory cost of the L×L float PAE field; potential `torch.compile` recompile triggers if dynamic shapes vary too widely (probably not a PR-A concern, but call it out).

Run in parallel after each green phase. All four must approve before merging into `prepare_teddy`.

## 14. Risks register

| Risk | Severity | Likelihood | Mitigation |
| --- | --- | --- | --- |
| **Residue-numbering frame mismatch** (PR #8 carry-over): `residue_intervals_*` is 1-indexed vs `confidenceScore` 0-indexed vs `predicted_aligned_error` indexing | HIGH | MEDIUM | Test 4 (integration spot-check) binds index frame to ground truth via `interface_length` and `avg_int_plddt`. Failure on >2/20 → abort criterion. |
| **PAE memory cost per worker** (~1 MB densified per dimer × in-flight batch × num_workers) | MEDIUM | LOW for L≤512, HIGH for unbounded L | Float32 L×L at L=500 = 1 MB; at 16 workers × prefetch=2 × batch=6 ≈ 200 MB per worker; tolerable. Test 3 asserts shape; if Teddymer dimers ever exceed L=1000, fall back to int16 PAE (still under 2 MB) or bin-only emission. |
| **Worker-fork determinism**: 350 MB locator dict pickled into each fork; on Linux with fork-style start method this is COW so cost is shared, but on `spawn` it's per-worker | MEDIUM | MEDIUM | Set `multiprocessing_context="fork"` in dataloader if not default (it is the Linux default for Lightning). Add an assert in `TeddymerDimerDataModule.__init__` if `spawn` is in use, log a warning. |
| **AFDB tar IO contention** at high `num_workers` × random-access seek pattern | MEDIUM | LOW | AFDB tars are large but sit on NVMe in `/mnt/labs/shared/...`; seek + 80 KB read is ~1 ms. At 16 workers × 100 batches/s × 12 dimers = ~20 k seeks/s; well within an NVMe budget. If it bites, add an LRU file-handle cache per worker keyed by `source_tar_basename` (587 unique tars total, fits 64-handle LRU comfortably). |
| **CIF parse-from-bytes slow path** if `load_structure` forces a tempfile | LOW | MEDIUM | Tempfile on `/tmp` (tmpfs RAM disk on this host) is microseconds; only the parse cost matters and that is unchanged. Confirm at slice 2 start; fall back to pre-materialised dimer CIFs if median parse > 10 ms. |
| **`complexa_filter` column missing** from upstream parquet (e.g. if someone regenerates the view without it) | LOW | LOW | Test 3 asserts the filter actively drops a row; if the column is absent the `.query` raises clearly. |
| **Hydra `defaults: [ted_dimers]` accidentally restored** by a future config refactor | LOW | MEDIUM | The plan explicitly inlines the atomarray transforms; documented in §"Hydra composition". Comment in the yaml states "DO NOT add `ted_dimers` to defaults — it injects `path_column: path` that breaks Teddymer's tar-offset loading". |
| **PAE bin definition drift** (someone later changes `bin_width` to match a different paper's convention) | MEDIUM | LOW | `pae_to_bin` lives in `confidence/losses.py` next to `plddt_to_bin` with explicit AF2-bin-edge documentation in the docstring. PR-B's head config references the same helper. |
| **The integration test (#4) requires `/mnt/labs/shared/...` access**, may not be runnable in all environments | LOW | LOW | Marked `slow`; skipped if `AFDB_PROTEOMES_ROOT` is not set or unreadable. Document in the test's `skipif`. |

## 15. Open questions

None. Decisions 1-5 in §2 are locked. The plan is ready to execute.

---

**Ready to execute.** Hand to **code-review-debug-complexity-expert** for the red-phase test authoring (all four tests), then to **ml-protein-architect** for the green-phase implementation (slice 1 then slice 2). After all tests are green, dispatch the four-reviewer panel in §13 in parallel.
