# Plan — Teddymer /netscratch blob repack (Option A, single PR)

Date: 2026-05-18. Author: software-planning-architect. Branch base + PR
target: `teddy_distillation_prep` (user override, NOT `dev`).

## 1. Goal & non-goals

**Goal.** Repack only the CIF + PAE + confidence byte ranges Teddymer
actually uses into a single contiguous `data.blob` (~136 GB) on
/netscratch, rewrite `locator_rows.parquet` to point at it, version the
artefacts, write md5 sidecars + a labs-side snapshot, and add a
train-start integrity gate that fails fast (before any GPU work) if the
labs source has drifted relative to the staged blob. The dataloader code
does NOT change — `io.py:_read_tar_member_bytes`
([src/proteinfoundation/datasets/teddymer/io.py:20-31](../../src/proteinfoundation/datasets/teddymer/io.py#L20-L31))
works equally on a tar and on a flat blob. Single PR.

**Non-goals.**
- No dataloader or transform refactor.
  [dataset.py:198-205](../../src/proteinfoundation/datasets/teddymer/dataset.py#L198-L205)
  remains untouched.
- No sharding (one `data.blob`, per user decision 1).
- No parallel-writer / multiprocess pack. Single-stream first; the
  handoff measured 250 MB/s single-stream cold and ~30 min total
  ([handoff §"Measured I/O"](../handoff/2026-05-18_teddymer_netscratch_blob.md)).
- No re-computation of `dimers.parquet` content. It is copied byte-verbatim.
- No retroactive change to SwissProt training paths or the pLDDT-only
  Teddymer multi-head config; the new train-start check is gated by a
  config flag that defaults OFF and is set ON only for the PAE Teddymer
  configs.

## 2. Context & constraints

- Branch: `teddy_distillation_prep` (current). PR target: same branch
  (user override). NOT `dev`.
- Env: uv-managed, Python 3.12, PyTorch 2.10. No new dependencies; md5
  via stdlib `hashlib`, parquet I/O via existing `pyarrow` / `pandas`.
- Dataloader contract:
  [io.py:20-31](../../src/proteinfoundation/datasets/teddymer/io.py#L20-L31) —
  `open + seek(offset) + read(size) + (optional) gzip.decompress`. The
  blob is exactly a concatenation of raw member byte ranges; the
  rewritten locator records dst offsets directly. `is_gz` flags are
  copied verbatim (the bytes-on-disk stayed gzipped if they were
  gzipped in the source tar).
- Source data, labs NFS (read-only):
  `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/`, 41,675 tars.
- Existing dataset view, writable:
  `/mnt/storage01/home/schekmenev/data/teddymer_v1/` —
  `dimers.parquet` (35 MB, 587,687 rows),
  `locator_rows.parquet` (73 MB, 1,175,374 rows),
  `view_config.yaml`, `_raw/`.
- Target staging dir:
  `/netscratch/schekmenev/teddymer_v1_blob/` (writable; verified).
- Hydra defaults:
  [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml:14-16](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml#L14-L16)
  reads `TEDDYMER_VIEW_ROOT` / `AFDB_PROTEOMES_ROOT` env vars; defaults
  point at labs.
- Training entry point:
  [src/proteinfoundation/confidence/train_confidence.py:36-70](../../src/proteinfoundation/confidence/train_confidence.py#L36-L70).
- SLURM launcher today:
  [scripts/train_confidence_teddymer_pae.sbatch:91-93](../../scripts/train_confidence_teddymer_pae.sbatch#L91-L93)
  leaves env vars to the user; the dataset config defaults take over.
- Existing `.md5` precedent: `/netscratch/schekmenev/monomers_minl50_maxl256_*.md5`
  (single line, 32-hex md5). We mirror this format.
- Existing CPU sbatch precedent:
  [scripts/preprocess_teddymer.sbatch](../../scripts/preprocess_teddymer.sbatch)
  — partition `cpu`, qos `normal`, 8 CPUs, 64 GB, 4 h, no GPU. Reuse
  verbatim.

## 3. Stakeholders / handoffs

- **Producer.** `ml-protein-architect` implements per this plan.
- **Consumer #1.** `scripts/train_confidence_teddymer_pae.sbatch` →
  `python -m proteinfoundation.confidence.train_confidence
  --config-name=confidence/distillation_teddymer_pae`. Contract: the
  train entry point's pre-fit integrity check passes iff (a) blob +
  locator + dimers exist at the configured root and (b) their md5s
  match the snapshot. Mismatch = abort with a sentence-long actionable
  error.
- **Consumer #2.** Future PAE/multi-head Teddymer runs reuse the same
  blob and snapshot until labs drifts; rebuilding is one sbatch away.
- **Reviewer panel.** `code-review-debug-complexity-expert` (mandatory)
  + `ml-protein-architect` (data pipeline). No PyTorch/JAX reviewer per
  user instruction.

## 4. Design

### 4.1 File inventory

**New files.**

| Path | Purpose |
| --- | --- |
| `src/proteinfoundation/datasets/teddymer/build_blob.py` | CLI + library: `plan_layout`, `repack`, `rewrite_locator`, `compute_md5`, `write_md5_sidecar`, `write_view_config`, `main`. |
| `src/proteinfoundation/datasets/teddymer/integrity.py` | Library used by training entry point: `verify_teddymer_blob_integrity(view_root, snapshot_path) -> None` (raises on mismatch). Streaming md5 with 8 MiB chunk. |
| `scripts/build_teddymer_blob.sbatch` | SLURM CPU launcher mirroring `preprocess_teddymer.sbatch`. |
| `tests/datasets/teddymer/__init__.py` | Empty package marker. |
| `tests/datasets/teddymer/test_blob_roundtrip.py` | Synthetic-tar round-trip + plan_layout unit tests. |
| `tests/datasets/teddymer/test_integrity.py` | md5-sidecar + integrity-check unit tests. |
| `tests/datasets/teddymer/test_blob_real_sample.py` | `requires_labs` round-trip on 50 random parents. |

**Edited files.**

| Path | Edit |
| --- | --- |
| `src/proteinfoundation/confidence/train_confidence.py` | Insert pre-`trainer.fit` integrity hook gated by `cfg.integrity.enabled`. |
| `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` | Switch `TEDDYMER_VIEW_ROOT` default to `/netscratch/schekmenev/teddymer_v1_blob`. Switch `AFDB_PROTEOMES_ROOT` default to the same (the blob lives there too). |
| `configs/confidence/distillation_teddymer_pae.yaml` | Add `integrity` block (`enabled: true`, `snapshot_path`). |
| `configs/confidence/distillation_teddymer_multihead.yaml` | Same `integrity` block. |
| `configs/confidence/distillation_swissprot.yaml` | Add `integrity: { enabled: false }` for symmetry (no-op). |
| `configs/confidence/distillation_swissprot_control.yaml` | Same. |
| `scripts/train_confidence_teddymer_pae.sbatch` | Update the comment block at L91-93 to record that env defaults now target /netscratch. No env-var exports added (Hydra defaults win). |
| `pyproject.toml` | Register `requires_labs` pytest marker. |

### 4.2 Public APIs

```python
# build_blob.py
def plan_layout(locator_df: pd.DataFrame) -> pd.DataFrame: ...
def repack(plan_df: pd.DataFrame, labs_root: Path, out_blob: Path,
           progress_every: int = 1000) -> dict[str, tuple[int, int, int]]: ...
def rewrite_locator(orig_locator: pd.DataFrame,
                    dst_offsets: dict[str, tuple[int, int, int]],
                    blob_relpath: str = "data.blob") -> pd.DataFrame: ...
def compute_md5(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str: ...
def write_md5_sidecar(paths: Sequence[Path], sidecar_path: Path,
                      relative_to: Path) -> None: ...
def write_view_config(out_dir: Path, *, version: str, blob_path: Path,
                      locator_path: Path, dimers_path: Path,
                      labs_root: Path, md5s: dict[str, str]) -> None: ...
def main() -> None: ...

# integrity.py
def verify_teddymer_blob_integrity(
    view_root: Path,
    snapshot_path: Path,
    *,
    files: Sequence[str] = ("data.blob", "locator_rows.parquet", "dimers.parquet"),
) -> None: ...
```

### 4.3 Data contracts

- **Plan DataFrame** from `plan_layout`. Columns: `parent_afdb_id,
  source_tar_relpath, src_cif_off, src_cif_size, src_pae_off,
  src_pae_size, src_conf_off, src_conf_size, cif_is_gz, pae_is_gz,
  conf_is_gz, dst_cif_off, dst_pae_off, dst_conf_off`. Rows are unique
  per `parent_afdb_id`. Order: sorted by `(source_tar_relpath,
  src_cif_off)`. `dst_*_off` strictly monotone non-decreasing across
  the table; each member is written contiguously as
  `cif | pae | conf` per parent, so within a parent the order is
  `dst_cif_off < dst_pae_off < dst_conf_off`.
- **Rewritten locator** preserves the schema of the original
  ([build_locator.py:27-34](../../src/proteinfoundation/datasets/teddymer/build_locator.py#L27-L34))
  exactly, except: `source_tar_relpath := "data.blob"`,
  `source_tar_basename := "data.blob"`, and the three `*_member_offset`
  columns are replaced with the dst offsets. `*_member_size`,
  `*_is_gz`, `*_member_name`, `has_*` are preserved verbatim (the bytes
  in the blob are byte-identical to the bytes in the tar, including
  gzip wrapping). Row count: 1,175,374 — unchanged. A-row and B-row of
  an intra-monomer dimer share dst offsets (they already share src
  offsets; verified in the handoff).
- **md5 sidecar.** One combined file `data.md5` at
  `/netscratch/schekmenev/teddymer_v1_blob/data.md5`, format mirrors
  GNU `md5sum`: `<32-hex>  <basename>\n` per line, three lines for
  `data.blob`, `locator_rows.parquet`, `dimers.parquet`. **Why one
  combined sidecar:** the integrity check verifies all three together
  atomically; three separate sidecars invite partial-snapshot bugs.
  The existing `.md5` precedent under /netscratch is single-file too.
- **Labs-side snapshot.** Written to
  `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`. **Why
  this location:** labs proper (`/mnt/labs/...`) is read-only for the
  user (verified `test -w` returned NOT writable); the dataset view at
  `/mnt/storage01/home/schekmenev/data/teddymer_v1/` IS writable and is
  what the handoff calls "the dataset view on labs-adjacent storage"
  ([handoff §"What is on disk"](../handoff/2026-05-18_teddymer_netscratch_blob.md)).
  This is also the path other scripts already treat as the canonical
  source-of-truth view. The repack script writes the snapshot under
  that view AND a copy alongside the blob; the integrity check reads
  the labs-side snapshot and compares against on-disk md5 of the
  /netscratch artefacts.
- **Version string.** `teddymer_v1_blob_<YYYYMMDD>_<md5(data.blob)[:8]>`.
  **Why:** date is human-readable; the 8-char prefix of the blob md5
  disambiguates same-day rebuilds and is content-addressable. Written
  into both `view_config.yaml` files (the labs-adjacent one and the
  /netscratch one). The integrity check compares versions first (cheap
  string compare); md5 only after.

### 4.4 Control flow — train-start integrity gate

In `train_confidence.py` immediately after `L.seed_everything` and
before `head = build_confidence_head_from_cfg(...)`:

```python
integrity_cfg = cfg.get("integrity", None)
if integrity_cfg is not None and bool(integrity_cfg.get("enabled", False)):
    from proteinfoundation.datasets.teddymer.integrity import (
        verify_teddymer_blob_integrity,
    )
    verify_teddymer_blob_integrity(
        view_root=Path(integrity_cfg["view_root"]),
        snapshot_path=Path(integrity_cfg["snapshot_path"]),
    )
```

**Why before head construction:** the trunk checkpoint load (inside
`ConfidenceDistillationModule.__init__`) is the next-most-expensive
step. Failing here costs ~md5-of-blob seconds (~136 GB / 1 GB/s NVMe
≈ 2 min on the H100 nodes) but saves loading the trunk + autoencoder
ckpts. **Why config-driven, not hard-coded:** SwissProt training has
no blob and must skip the check. **Default OFF** in the Lightning
module signature; **ON** only in the two Teddymer PAE configs.

Failure message:
```
Teddymer blob integrity check failed: md5 mismatch on '{filename}'.
Expected (from snapshot at {snapshot_path}): {expected}.
Got (live at {view_root}/{filename}): {actual}.
The Teddymer dataset on labs has changed, or the /netscratch blob
is stale or partially overwritten. Re-run
scripts/build_teddymer_blob.sbatch to repack.
```

Also raised if the snapshot is missing, or any of the three live files
is missing.

### 4.5 Hydra default flip

Both surfaces are flipped, per user decision 4:

- The Hydra config defaults at
  [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml:14-16](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml#L14-L16)
  switch to `/netscratch/schekmenev/teddymer_v1_blob` for both
  `TEDDYMER_VIEW_ROOT` and `AFDB_PROTEOMES_ROOT`. `dimers.parquet`,
  `locator_rows.parquet`, and `data.blob` all live under that root.
- The sbatch leaves the env vars unset (Hydra defaults win). The
  comment block at
  [scripts/train_confidence_teddymer_pae.sbatch:91-93](../../scripts/train_confidence_teddymer_pae.sbatch#L91-L93)
  is rewritten to say "Hydra defaults point at /netscratch; override
  via env if running against labs directly".

**Why config, not sbatch:** the sbatch already says env vars are
overridable; flipping the Hydra default makes the cluster path the
"happy path" and the labs path an explicit override. A user running
`python -m ...train_confidence` locally without the sbatch also picks
up /netscratch by default — which is correct, the blob is the only
acceptable training source post-merge. The sbatch comment is only
documentation.

## 5. Alternatives considered

1. **md5 of a sampled subset of the blob (e.g. 100 random 1 MiB
   windows).** Cheaper at train start (~5 s vs ~2 min) but unsound:
   labs drift could land in unsampled regions, and the failure is
   silent. Rejected. The 2-min check is paid once per training run;
   the model trains for days.
2. **Size + mtime check only.** Even cheaper, even less sound: an
   in-place rewrite of the blob keeps size and even mtime if `touch
   -r` is used. Rejected.
3. **md5 of `data.blob` only, not the parquets.** Rejected: a stale
   locator on a fresh blob still misroutes reads. Three-file md5
   covers the whole contract.
4. **Sharded blob (16 × ~9 GB).** Rejected per user decision 1. NVMe
   handles concurrent seeks fine; sharding only adds branching in
   `rewrite_locator`.
5. **Move dataloader to read tar-index files alongside a copied tar
   tree.** This is Option B from the handoff: ~21 TB copy, ~24 h.
   Rejected as already-decided.
6. **Two md5 sidecars (per-file).** Rejected (4.3): atomicity bugs.

## 6. Risk register

| # | Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Labs tar mtime/content drift between repack and train | High | Low | md5 snapshot at repack; train-start check compares snapshot vs live /netscratch. Drift detected → abort with actionable error. |
| R2 | /netscratch eviction / quota pressure removes the blob mid-training | Medium | Low | Documented in sbatch comment; integrity check fails cleanly on missing file. No auto-rebuild. User must rerun the build sbatch. |
| R3 | Partial write on crash leaves a corrupt blob | High | Medium | Write to `data.blob.tmp`, fsync, rename to `data.blob` (atomic on local FS). md5 sidecar only emitted on successful close. |
| R4 | Blob >136 GB because dedup logic miscounts intra-monomer pairs | Medium | Low | `plan_layout` asserts unique `parent_afdb_id` count == 555,791 in production mode; unit test verifies 4-row→2-parent reduction. |
| R5 | Round-trip test sample of 200 dimers misses a rare schema row | Medium | Low | `test_repack_byte_roundtrip` stratifies: 50 dimers with `complexa_filter==False`, 50 with `True`, 50 cross-tar pairs (chains A/B in different parents), 50 intra-monomer pairs. Plus `test_blob_real_sample` (labs-marked) on 50 random parents from the production locator. |
| R6 | md5 of 136 GB at train start takes minutes | Low | High (every run) | Accepted. ~2 min at 1 GB/s NVMe sequential; training run is days. Logged with elapsed-seconds line. Not gated by sampling per Alt 1. |
| R7 | A-row and B-row of an intra-monomer dimer accidentally get different dst offsets | High | Low | `rewrite_locator` looks up dst offsets by `parent_afdb_id` (single source of truth dict). Unit test `test_rewrite_locator_intra_monomer_dimer_shares_offsets` verifies both rows of such a dimer point at identical dst offsets. |
| R8 | Pyarrow rewrite drops or renames a column silently | Medium | Low | `test_rewrite_locator_preserves_row_count_and_schema` checks column set equality and row count. |
| R9 | Repack runtime > sbatch budget | Medium | Low | Time budget 2 h (4× measured headroom over ~30 min). Sbatch logs progress every 1000 parents → restart visibility. |
| R10 | Symlink / wrong mount on compute node | Medium | Low | `verify_teddymer_blob_integrity` uses `Path.resolve()` and reports the resolved path in the error message. Sbatch comment documents the expected mount. |
| R11 | Concurrent repack jobs trash the blob | Medium | Low | Use `flock` on `out_dir/.build.lock` during repack; second invocation aborts. |
| R12 | Snapshot file at labs-adjacent view is overwritten by a stale rebuild | Medium | Low | `view_config.yaml` carries `version` (date + content-hash prefix); `verify_teddymer_blob_integrity` cross-checks version string before md5. |

## 7. Milestones

### Phase 0 — Test fixtures (~30 min)

Build a synthetic mini-tar fixture in `tests/datasets/teddymer/`:
- `_fixtures/build_mini_tar.py` (test helper, not shipped as CLI) —
  generates an in-tmp-dir tar with N=4 fake "parents", each carrying
  `*-model_v4.cif.gz`, `*-predicted_aligned_error_v4.json.gz`,
  `*-confidence_v4.json.gz`. The gzipped payloads are tiny (≤1 KB).
- Synthetic locator (4 rows = 2 parents × 2 chains) recorded into a
  tiny parquet via `pyarrow`.

**Exit criterion.** Fixture imports cleanly; `tarfile` reports
expected member offsets/sizes.

**Abort.** If `tarfile`-reported offsets don't match what
`_read_tar_member_bytes` decodes on the synthetic tar, the fixture is
the wrong shape; stop and rethink.

### Phase 1 — Tests first (~2 h)

Write all of `tests/datasets/teddymer/test_blob_roundtrip.py` and
`tests/datasets/teddymer/test_integrity.py`. They MUST fail with
`ImportError` against the current `teddy_distillation_prep` HEAD
because `build_blob` and `integrity` modules do not yet exist —
record the expected import errors in commit message as evidence of
TDD discipline. `test_blob_real_sample.py` is also written, gated by
`@pytest.mark.requires_labs` so CI / synthetic runs skip it.

**Exit criterion.** `pytest tests/datasets/teddymer/ -x` fails with
exactly the import errors expected; no other test in the repo regresses.

**Abort.** If any synthetic fixture requires real labs paths to
parametrize, the test design is wrong; collapse the test into a
synthetic-only form before proceeding.

### Phase 2 — Implementation (~3 h)

Implement `build_blob.py` and `integrity.py` to make Phase 1's tests
pass. Vertical slice: `plan_layout` → `repack` → `rewrite_locator` →
`compute_md5` → `write_md5_sidecar` → `write_view_config` → `main`.
Then `verify_teddymer_blob_integrity`. Single-stream I/O. No
threading, no asyncio.

**Exit criterion.** All synthetic tests pass:
```
pytest tests/datasets/teddymer/ -k "not requires_labs"
```

**Abort.** If `plan_layout` cannot achieve `dst_*_off` monotonicity
in a single forward sweep without random-access on the source tar,
the design assumption (sort-by-src-tar amortises NFS open cost) is
broken — escalate to user before continuing.

### Phase 3 — SLURM sbatch + config flip + train-time check (~1 h)

Add `scripts/build_teddymer_blob.sbatch` (mirror
`preprocess_teddymer.sbatch`). Edit
`configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` to flip
defaults. Add `integrity` block to the two Teddymer training configs
and explicitly-disabled to the two SwissProt configs. Edit
`train_confidence.py` to call `verify_teddymer_blob_integrity` when
the flag is on. Update `pyproject.toml` to register the `requires_labs`
marker. Update the `scripts/train_confidence_teddymer_pae.sbatch`
comment block.

**Exit criterion.**
- `pytest tests/datasets/teddymer/ -k "not requires_labs"` still green.
- `python -m proteinfoundation.confidence.train_confidence
  --config-name=confidence/distillation_swissprot trainer.fast_dev_run=1`
  still runs (the integrity gate is OFF and the entry point is
  unaffected) — confirmed by reading the resolved config, not
  by launching training.
- Hydra resolves `integrity.enabled=true` for the PAE Teddymer config.

**Abort.** If `OmegaConf` chokes on the `integrity` sub-tree (e.g.
because it collides with an existing key), rename the block to
`dataset_integrity` and revisit.

### Phase 4 — Verification and the actual repack (~2 h, owner = user)

The PR ships at the end of Phase 3. The repack itself is a separate
operational step:
1. User runs `sbatch scripts/build_teddymer_blob.sbatch`.
2. On completion, user runs `pytest tests/datasets/teddymer/ -m
   requires_labs` against a small random sample (50 parents).
3. User launches `sbatch scripts/train_confidence_teddymer_pae.sbatch`
   and confirms the integrity gate logs "OK" then training starts.

**Exit criterion.** PAE distillation first step completes; throughput
in the first epoch is dominated by GPU, not I/O (effective dataloader
rate ≫ 10 MB/s baseline).

**Abort.** If first-epoch I/O is still the bottleneck (>5 ms / dimer
median), the blob layout has a different problem than NFS latency
(e.g. random-access within a 136 GB file on shared NVMe scratch is
contending with other users) — escalate, do not auto-shard.

## 8. Verification strategy

### Tests (all under `tests/datasets/teddymer/`)

| Test fn | Asserts | Fixture | Marker | Maps to risks |
|---|---|---|---|---|
| `test_plan_layout_dedups_intra_monomer_pairs` | 4-row locator → 2 plan entries; `dst_cif_off=0` for first parent | synthetic locator | — | R4, R7 |
| `test_plan_layout_groups_by_source_tar` | Plan sorted by `(source_tar_relpath, src_cif_off)` ascending | synthetic locator | — | R9 |
| `test_plan_layout_monotone_dst_offsets` | `dst_cif_off < dst_pae_off < dst_conf_off` per row; `dst_cif_off[i+1] >= dst_conf_off[i] + src_conf_size[i]` | synthetic locator | — | R3 |
| `test_repack_byte_roundtrip` | For each of the 4 synthetic parents: blob+new-locator yields byte-identical CIF/PAE/conf as old-locator+tar | synthetic tar | — | R3, R7 |
| `test_repack_writes_tmp_and_renames_atomically` | After repack, `data.blob.tmp` absent, `data.blob` present, mtime newer than tmp's deletion sentinel | synthetic tar | — | R3 |
| `test_rewrite_locator_preserves_row_count_and_schema` | Same columns (set equality), same row count, `source_tar_relpath` all `"data.blob"` | 1k-row synthetic locator | — | R8 |
| `test_rewrite_locator_intra_monomer_dimer_shares_offsets` | Dimer where A and B map to same parent → both rows have identical dst offsets | synthetic | — | R7 |
| `test_compute_md5_streams_in_chunks` | md5 of 32 MiB random buffer matches `hashlib.md5(buf).hexdigest()` regardless of `chunk_bytes=4K/8M/full` | synthetic buffer | — | R6 |
| `test_md5_sidecar_format_matches_md5sum` | `data.md5` parses as `<32-hex>  <basename>` per line, 3 lines, sorted by filename | synthetic | — | R3 |
| `test_view_config_records_version_and_md5s` | yaml has `version: teddymer_v1_blob_YYYYMMDD_<hex8>`, three md5s, labs source root, build timestamp | synthetic | — | R12 |
| `test_verify_integrity_passes_on_fresh_blob` | Build synthetic blob+snapshot, verify returns None | synthetic | — | R1 |
| `test_verify_integrity_fails_on_tampered_blob` | Mutate one byte of `data.blob` after snapshot → `IntegrityError` raised with file path and both md5s in the message | synthetic | — | R1, R3 |
| `test_verify_integrity_fails_on_missing_snapshot` | Snapshot file deleted → `FileNotFoundError` with path | synthetic | — | R10 |
| `test_verify_integrity_fails_on_version_mismatch` | view_config version differs from snapshot version → fast fail before any md5 work (assert no md5 computed via spy) | synthetic | — | R12 |
| `test_train_entry_invokes_integrity_when_enabled` | Mock `verify_teddymer_blob_integrity`, invoke `train_confidence.main` with `integrity.enabled=true trainer.fast_dev_run=1` against a tiny CPU config; assert mock called once | synthetic | — | R1 |
| `test_train_entry_skips_integrity_when_disabled` | Same, with `integrity.enabled=false`; assert mock NOT called | synthetic | — | R1 |
| `test_repack_real_sample` | Pick 50 random parents from prod locator; repack to tmp blob; for each, byte-compare CIF/PAE/conf vs labs | prod locator at `/mnt/storage01/home/schekmenev/data/teddymer_v1/locator_rows.parquet` | `requires_labs` | R1, R5 |

**Marker registration.** `pyproject.toml` `[tool.pytest.ini_options]`
adds `markers = ["requires_labs: requires /mnt/labs/... + /mnt/storage01/home/schekmenev/data/teddymer_v1/"]`.
Default test runs use `-m "not requires_labs"` (documented in plan;
no CI change needed — CI doesn't have labs mounted).

### Manual verification

- After Phase 4 build: `du -sh /netscratch/schekmenev/teddymer_v1_blob/`
  reports ~136 GB ± 5 %.
- `md5sum -c /mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`
  succeeds from inside the staged dir (sanity: snapshot is portable).
- First sbatch training run: the train log contains a line
  `Teddymer blob integrity OK (verified 3 files in <N> s, version
  teddymer_v1_blob_...)` before any GPU work.

## 9. Rollback plan

- **PR change rollback.** `git revert <merge_commit>` on
  `teddy_distillation_prep`. The dataloader is untouched; reverting
  only puts the Hydra default back to labs and removes the integrity
  hook. No checkpoint format change, no data migration needed.
- **Blob rollback.** `rm -rf /netscratch/schekmenev/teddymer_v1_blob/`.
  Then `unset TEDDYMER_VIEW_ROOT AFDB_PROTEOMES_ROOT` and export them
  to the labs paths for the training run. The dataset on labs is
  untouched throughout — labs is read-only.
- **Snapshot rollback.** `rm /mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`
  and the integrity check fails fast → training aborts. Re-run
  `scripts/build_teddymer_blob.sbatch` to regenerate.

## 10. Open questions

1. **md5 cost at train start (~2 min).** The plan accepts this. If
   the user wants a faster check, the alternative is a fast
   `(size, mtime)` precheck + sampled md5 of K random 1 MiB windows.
   Owner to confirm: accept 2 min one-shot? **Default in plan: accept.**
   This is the only place a reasonable person could disagree; flag it
   for review.
2. **Should the integrity check ALSO run from inside the dataset's
   `__init__`?** Currently checks only in the train entry. If
   somebody runs `python -m proteinfoundation.confidence.train_confidence`
   for sampling/eval in the future without the hook, they'd miss the
   check. **Default in plan: train-only.** Eval/sampling pipelines for
   PAE distillation are out of scope; revisit when they exist.
3. **`/mnt/labs/labs.md5` vs `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5`
   as snapshot.** Resolved: storage01 view, per writability check
   (labs is read-only). One snapshot location.

These are the only open items; none block implementation if the plan's
defaults are accepted.

## 11. Reviewer dispatch checklist

When the PR is opened, dispatch the two reviewers in parallel with
these prompts (no other reviewers per user instruction):

### code-review-debug-complexity-expert

```
PR: teddymer /netscratch blob repack (single PR, branch base
teddy_distillation_prep).

Scope: new build_blob.py + integrity.py modules, new
build_teddymer_blob.sbatch, new tests under tests/datasets/teddymer/,
edits to train_confidence.py + the two Teddymer Hydra configs +
the dataset YAML's env defaults.

Review focus:
1. plan_layout correctness — dedup, sort, monotone dst offsets, no
   off-by-one in summed sizes.
2. repack atomicity — tmp + fsync + rename. No partial-write leak.
3. md5 streaming chunk size and correctness under partial reads.
4. integrity.py error messages — they must name file paths and both
   md5s; no silent passes.
5. Test coverage gaps — anything in the risk register R1-R12 that
   isn't covered by a test.
6. Algorithmic complexity — repack is O(N parents); confirm no
   accidental O(N²) in plan_layout (groupby pattern is suspicious).

Approve when all of the above check out. If anything is broken,
file:line citations + severity tag (block / non-block).
```

### ml-protein-architect

```
PR: teddymer /netscratch blob repack (single PR, branch base
teddy_distillation_prep).

Review focus (data pipeline + module layout):
1. Module layering — build_blob.py + integrity.py live under
   src/proteinfoundation/datasets/teddymer/, alongside io.py /
   dataset.py / build_locator.py. Confirm the placement matches
   repo conventions; integrity.py imports must not pull in
   torch / lightning.
2. Hydra config — integrity block placement (top-level under each
   training config), default values, env-var resolver. Confirm
   the SwissProt configs are no-op (integrity.enabled=false).
3. Locator schema preservation — every column from
   build_locator.py:27-33 round-trips with only the four documented
   fields rewritten.
4. Dataloader contract — confirm nothing in dataset.py / transforms.py
   needs to change. The blob IS the tar from io.py's viewpoint.
5. Env-var flip in the dataset YAML — confirm the default change does
   NOT break existing SwissProt or other unrelated runs (the YAML
   touched is teddymer-only).
6. CPU sbatch — partition / time / memory budget vs handoff numbers.

Approve when all of the above check out.
```

Loop until both reviewers approve. If the loop does not converge in
two rounds, invoke the stuck-PR escape hatch
(CLAUDE.md "Stuck-PR escape hatch").

## 12. Definition of done

After PR merge into `teddy_distillation_prep` and Phase 4 ops step:

1. `/netscratch/schekmenev/teddymer_v1_blob/` contains: `data.blob`
   (~136 GB), `locator_rows.parquet` (1,175,374 rows pointing at
   `data.blob`), `dimers.parquet` (byte-identical to the labs-side
   one), `data.md5` (3 lines), `view_config.yaml` (with `version:` +
   md5s + build timestamp).
2. `/mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5` exists
   and matches the /netscratch sidecar.
3. `sbatch scripts/train_confidence_teddymer_pae.sbatch` succeeds past
   the integrity gate; the log emits `Teddymer blob integrity OK
   (verified 3 files in <N>s, version teddymer_v1_blob_...)` before
   any GPU work. Training first step completes.
4. `pytest tests/datasets/teddymer/ -k "not requires_labs"` is green
   on CI.
5. The `requires_labs` test suite was run once by the user against
   the production locator and was green (recorded in PR description).
6. The handoff at `docs/handoff/2026-05-18_teddymer_netscratch_blob.md`
   stays in place as historical context. Memory entry
   `teddymer_pae_distillation_shipped.md` is amended to note the blob
   landed.

## 13. Open issue for the human (before implementation can start)

The ONE thing that is not a routine assumption and could go either
way:

- **Open question 1 (Section 10) — md5 cost at train start (~2 min).**
  Plan default is "accept". If the user prefers a sampled check
  instead, the integrity module signature changes (add `mode:
  Literal["full", "sampled"]`) but no other downstream surface moves.
  Estimated extra effort if changed later: 1 h + 2 new tests.

Hand to **ml-protein-architect** for implementation under the TDD
discipline (Phase 0 → Phase 1 tests-first → Phase 2 impl → Phase 3
config + sbatch + train hook). The two-reviewer panel (Section 11)
runs at PR-open time.
