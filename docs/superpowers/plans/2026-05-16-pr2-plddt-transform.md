# PR-2 — AF2 pLDDT data transform

## 1. Branch and base

- **Branch.** `feat/confidence-plddt-transform`
- **Base.** `merge_quality_graft`
- **Merges into.** `merge_quality_graft`
- **Parallel with.** PR-1 (file-disjoint).

## 2. Goal and non-goals

- **Goal.** Add a per-residue AF2 pLDDT extraction transform on the existing complexa data pipeline, plus a `plddt_to_bin` binning helper, plus a new Hydra dataset config.
- **Non-goals.** No head, no sidecar, no Lightning module, no loss (other than the bin helper). No edits to `proteina.py` or `local_latents_transformer*.py`.

## 3. Files

### 3.1 To add

- `src/proteinfoundation/confidence/__init__.py` (empty marker).
- `src/proteinfoundation/confidence/losses.py` — `plddt_to_bin` only in PR-2.
- `configs/dataset/unified/afdb_monomers_with_plddt.yaml`.
- `tests/unit/confidence/__init__.py`
- `tests/unit/confidence/test_plddt_binning.py`
- `tests/unit/confidence/test_add_plddt_transform.py`
- `tests/integration/confidence/__init__.py`
- `tests/integration/confidence/test_plddt_dataset_one_batch.py`
- `tests/fixtures/afdb_one_protein/synthetic_atom37.pt`

### 3.2 To edit

- `src/proteinfoundation/datasets/transforms.py` — append `AddPLDDTFromBFactor` (atom37-side `BaseTransform`).
- `src/proteinfoundation/datasets/structure_data.py` — extend `atomarray_to_atom37` to thread `b_factor` through `atom_array_to_encoding(..., extra_annotations=[..., "b_factor"])` and stash per-atom B-factor on the returned `Data` object as `data.atom_b_factor: torch.float32 [n_tokens, 37]`.

### 3.3 `AddPLDDTFromBFactor` (in `transforms.py`)

- Class: `AddPLDDTFromBFactor(BaseTransform)`.
- `__init__(ca_atom_idx=1, max_bins=50, bin_width=2.0, scale_max=100.0, sanity_check=True, warn_once=True)`.
- `__call__(graph: Data) -> Data`:
  - Read `graph.atom_b_factor: [n, 37] float32`.
  - Read `graph.coord_mask: [n, 37] bool`.
  - Per-residue pLDDT = mean of B-factor over valid atoms (AF2 stores same value across atoms).
  - `graph.plddt_residue: [n] float32` clipped to `[0, scale_max]`.
  - `graph.plddt_bin: [n] int64 = plddt_to_bin(plddt_residue, bin_width, max_bins)`.
  - `graph.plddt_mask: [n] bool = coord_mask.any(dim=-1) & (atom_b_factor.sum(dim=-1) > 0)`.
  - Sanity: once-per-rank `logger.warning` if `atom_b_factor.max() <= 1.5` (suggests normalised pLDDT bug). **Do not raise.**

### 3.4 `plddt_to_bin` (in `proteinfoundation/confidence/losses.py`)

- Signature: `plddt_to_bin(plddt: torch.Tensor | float, bin_width: float = 2.0, num_bins: int = 50) -> torch.Tensor`.
- Implementation: `(plddt / bin_width).floor().clamp(0, num_bins - 1).to(torch.int64)`.
- Docstring: "PR-2 ships only `plddt_to_bin`; cross-entropy/SmoothL1 losses arrive in PR-4."

### 3.5 New dataset config

`configs/dataset/unified/afdb_monomers_with_plddt.yaml`:

```yaml
defaults:
  - afdb_monomers

datamodule:
  filters:
    - "length < 512"
  atom37_transforms:
    - _target_: proteinfoundation.datasets.transforms.AddPLDDTFromBFactor
      bin_width: 2.0
      max_bins: 50
      scale_max: 100.0
      sanity_check: true
```

Drops the `plddt > 70` filter (full distribution wanted for distillation), appends one Hydra-instantiable transform.

## 4. Tests (TDD — write first)

### 4.1 `tests/unit/confidence/test_plddt_binning.py`

Pinned values:
- `plddt_to_bin(0.0) == 0`
- `plddt_to_bin(2.0 - 1e-6) == 0`
- `plddt_to_bin(2.0) == 1`
- `plddt_to_bin(50.0) == 25`
- `plddt_to_bin(99.999) == 49`
- `plddt_to_bin(100.0) == 49` (clamped)
- Vector: `plddt_to_bin([0, 50, 100]) == [0, 25, 49]`
- Dtype: `torch.int64`.

### 4.2 `tests/unit/confidence/test_add_plddt_transform.py`

Synthetic `Data` object with `n=5` residues, hand-set B-factors:
- `atom_b_factor = [[80]*37, [50]*37, [0]*37, [99]*37, [70]*37]` with masked atoms for residue 2.
- Assert `plddt_residue == [80.0, 50.0, 0.0, 99.0, 70.0]` (within atom-mask).
- Assert `plddt_bin == [40, 25, 0, 49, 35]`, `dtype == int64`.
- Assert `plddt_mask == [True, True, False, True, True]`.
- Sub-case: `atom_b_factor.max() == 1.0` → `caplog` shows WARNING, no crash.
- Sub-case: warn-once semantics (warning fires exactly once across two calls).

### 4.3 `tests/integration/confidence/test_plddt_dataset_one_batch.py`

- Hydra compose `configs/dataset/unified/afdb_monomers_with_plddt.yaml` pointing at the fixture.
- Use a tiny test-only loader that wraps `StructureDataset.__getitem__` to return the saved fixture `Data` (kept entirely under `tests/`).
- Pull one batch, assert keys `plddt_residue`, `plddt_bin`, `plddt_mask` present with right dtypes/ranges.

## 5. Fixture acquisition

Commit a synthetic atom37 fixture (`~10 KB`) at `tests/fixtures/afdb_one_protein/synthetic_atom37.pt`. A tiny test-only helper `tests/integration/confidence/_fixture_loader.py` wraps `StructureDataset` and bypasses CIF parsing. Hermetic; ~1 s test time.

## 6. Verification commands

1. `uv run pytest tests/unit/confidence/ -v`
2. `uv run pytest tests/integration/confidence/test_plddt_dataset_one_batch.py -v`
3. Manual Hydra `compose` smoke check on the new YAML — paste config tree into PR description.

## 7. Reviewer panel

- **`code-review-debug-complexity-expert`** (mandatory). Focus: aggregation correctness; once-per-rank under DDP; binning boundaries.
- **`ml-protein-architect`**. Focus: transform placement; Hydra `defaults` inheritance; `atom_b_factor` plumbing through `atomarray_to_atom37` does not break non-AFDB datasets.
- **`structural-biology-binder-expert`**. Focus: AF2-DB B-factor convention; mean-over-atoms is a no-op for AF2; dropping the `plddt > 70` filter is the right call for distillation.

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| R2 | pLDDT scale / B-factor convention mismatch | Once-per-rank warning; unit-test pins binning |
| R7 | Concat-features cropping (labels on `n_orig`) | Transform runs pre-model on raw `Data`; PR-4 indexes labels with `orig_mask` |
| R-NEW1 | `b_factor` plumbing breaks unrelated datasets | Reviewer audit; run existing `tests/datasets/` |
| R-NEW2 | Synthetic fixture diverges from real CIF | PR-5 smoke test on one real AF2-DB CIF |
| R-NEW3 | `plddt_mask = False` swallowed silently in PR-4 | Flag in `losses.py` docstring for PR-4 implementer |
