# Handoff — Teddymer /netscratch repack (Option A)

## TL;DR for the next agent

The user is about to run **single-head PAE distillation on Teddymer**, but training I/O against `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/` is too slow (~25 ms / dimer cold via `open + 3×lseek+read` on NFS). After walking through three options (A: repack to a Teddymer-only blob on /netscratch; B: copy 41,675 raw tars verbatim ~21 TB; C: stay on labs), the user committed to **Option A**: repack only the CIF + PAE + confidence byte ranges Teddymer actually uses into a single contiguous blob (~136 GB) on /netscratch, then rewrite `locator_rows.parquet` to point at it. Nothing has been built yet — this session was Q&A only. The next agent's job is to design, implement, verify, and stage that blob.

The dataloader code does **not** change. `TeddymerDimerDataset.__getitem__` already opens `afdb_proteomes_root / row['source_tar_relpath']` and does `lseek + read`; if we point those columns at a flat blob instead of a tar, the existing `io.py:_read_tar_member_bytes` works unchanged. That is the whole reason Option A is cheap.

## What is on disk (verified 2026-05-18)

- **Source data — labs NFS** at `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/`: 41,675 `proteome-tax_id-*-{0..N}_v4.tar` files, mean ~500 MB, total ~21 TB (extrapolated from a 30-tar sample). Each tar contains gzipped per-monomer CIF, PAE JSON, confidence JSON for one proteome.
- **Teddymer view** at `/mnt/storage01/home/schekmenev/data/teddymer_v1/`:
  - `dimers.parquet` — 587,687 rows; cols `dimer_id, dimer_index, uniprot_id, parent_afdb_id, ted_index_{A,B}, cath_id_{A,B}, residue_intervals_{A,B}, member_count, interface_length, avg_int_pae, avg_int_plddt, int_plddt_chain_{A,B}, complexa_filter`. `complexa_filter==True` for 510,454 dimers.
  - `locator_rows.parquet` — 1,175,374 rows = 587,687 dimers × 2 chains. Schema includes `source_tar_relpath, cif_member_{offset,size,is_gz}, pae_member_{offset,size,is_gz}, conf_member_{offset,size,is_gz}`. **555,791 unique parent monomers** (group by `(source_tar_relpath, cif_member_offset, cif_member_size)`). 41,675 unique `source_tar_relpath` values.
  - `view_config.yaml` — provenance.
  - `_raw/` — staged Teddymer release inputs.
  - `locator_rows.parquet.pre_glob_fix.bak` — old pre-fix backup, leave alone.
- **/netscratch** — `/mnt/storage01/netscratch` (= `/netscratch`), 20 TB total, ~20 TB free. Per-user subdirs exist: `/netscratch/<username>/`. No Teddymer artefacts staged yet.
- **Training entrypoint** — `scripts/train_confidence_teddymer_pae.sbatch` launches `python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_teddymer_pae`. Hydra config reads view paths from env vars `TEDDYMER_VIEW_ROOT` and `AFDB_PROTEOMES_ROOT` ([configs/dataset/unified/teddymer_with_plddt_and_pae.yaml:14-16](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml#L14-L16)). Defaults today point at labs.

## Measured I/O on this host (login node, 2026-05-18)

| Probe | Number |
|---|---|
| Sequential cold read of a fresh 1.6 GB tar from labs | 6.1 s → **~264 MB/s** sustained single-stream |
| 100 dimers, full `__getitem__`-style cold (open tar + 3 × lseek+read) | 2.67 s → **~27 ms / dimer**, ~10 MB/s effective |
| /netscratch free space | 20 TB / 20 TB |
| Dedup byte total for blob (CIF 59.4 GB + PAE 75.1 GB + conf 1.4 GB) | **~136 GB** for 555,791 unique parents |
| Mean per-parent: CIF | 104 KB |
| Mean per-parent: PAE | 132 KB |
| Mean per-parent: conf | 2.4 KB |

These numbers justify A: ~30 min one-shot repack vs ~24 h to copy 21 TB.

## What is NOT on disk yet (the gap)

The next agent must produce, in this order:

1. **A repack script** at `src/proteinfoundation/datasets/teddymer/build_blob.py` that emits `data.blob` + `locator_rows_blob.parquet`. Public CLI mirroring `build_view.py`'s argparse style.
2. **A round-trip test** at `tests/datasets/teddymer/test_blob_roundtrip.py` that sanity-checks the blob against the original tars for N=200 sampled dimers (byte-equality on gzipped CIF, decompressed-JSON equality on PAE + conf).
3. **Staged artefacts** under `/netscratch/schekmenev/teddymer_v1_blob/`:
   - `data.blob` (~136 GB)
   - `locator_rows.parquet` (rewritten — same 1,175,374 rows; `source_tar_relpath` → `data.blob`; `*_member_offset` → new dst offsets; everything else unchanged)
   - `dimers.parquet` (verbatim copy of the original)
   - `view_config.yaml` (regenerated with new paths, blob sha256, build timestamp, back-pointer to labs root)
4. **A PR** opened off `dev` that adds the script + the test and updates documentation if needed. The repack output itself is not committed (it's a 136 GB artefact; data lives outside the repo).
5. **A second PR** *might* be needed if the user wants `scripts/train_confidence_teddymer_pae.sbatch` to default-export `TEDDYMER_VIEW_ROOT=/netscratch/...` instead of leaving it to the user — confirm with the user before adding that.

## Architecture reference (mirror this code, do not rewrite)

The dataloader contract the blob must satisfy is *exactly* the one the existing tar-offset reader assumes:

- **The 6-line primitive** at [src/proteinfoundation/datasets/teddymer/io.py:20-31](../../src/proteinfoundation/datasets/teddymer/io.py#L20-L31) — `open(path,'rb') + seek(offset) + read(size) + (optional) gzip.decompress`. This works equally well on a tar and on a flat blob.
- **Where the path is built**: [dataset.py:198](../../src/proteinfoundation/datasets/teddymer/dataset.py#L198) — `tar_path = Path(self.afdb_proteomes_root) / loc_a["source_tar_relpath"]`. The new locator simply needs `source_tar_relpath == "data.blob"` and `afdb_proteomes_root` to point at the staged dir.
- **Where each member is read**: CIF at [dataset.py:200-205](../../src/proteinfoundation/datasets/teddymer/dataset.py#L200-L205); PAE and conf via the AFDB-label transforms at [transforms.py:3573-3697](../../src/proteinfoundation/datasets/transforms.py#L3573-L3697). All three eventually funnel through `io.py:_read_tar_member_bytes`. No exceptions.
- **Locator schema** — see [build_locator.py:29-33](../../src/proteinfoundation/datasets/teddymer/build_locator.py#L29-L33) for the canonical column list. The rewritten locator must preserve every column except the three `*_member_offset` columns and `source_tar_relpath`/`source_tar_basename`.

The build pipeline is laid out in the session log; the brief is:

1. Read original `locator_rows.parquet`, dedup to 555,791 unique parents.
2. Sort dedup table by `(src_tar, src_cif_off)` — amortises NFS `open()` cost. 41,675 opens, not 1.67M.
3. For each source tar: open once, walk planned jobs in offset order, `lseek + read` each (CIF, PAE, conf) byte range, write back-to-back to `data.blob`, record `out.tell()` per member as the new dst offset.
4. Build a `(parent_afdb_id) -> (dst_cif_off, dst_pae_off, dst_conf_off)` map. Walk original 1.175M-row locator, emit new parquet with dst offsets and `source_tar_relpath="data.blob"`. Both A-row and B-row of an intra-monomer dimer get the *same* dst offsets (they already share src offsets — verified, [dataset.py walkthrough during session]).
5. Byte-roundtrip test: 200 sampled dimers; for each, read CIF/PAE/conf via new locator + blob and via old locator + labs tar; assert equality. Test fails → repack invalid → throw away the blob and redo.

## PR plan

**Branch.** Off `dev` (current default base). Suggested name: `teddymer_blob_repack`. Do NOT branch off `prepare_teddy_pr_fix_bugs` (the current local branch this session was on); that branch's purpose is exhausted.

**PR scope.** Single PR adds:
- `src/proteinfoundation/datasets/teddymer/build_blob.py` (CLI + library entry points: `plan_layout`, `repack`, `rewrite_locator`).
- `tests/datasets/teddymer/test_blob_roundtrip.py` (the byte-roundtrip test described above, plus a unit test for `plan_layout` on a synthetic mini-locator that hits dedup logic).
- Optional: `scripts/build_teddymer_blob.sbatch` if the user wants SLURM-launched packing rather than running on the login node. Confirm before adding.

**Test plan (TDD — tests first per [CLAUDE.md TDD section](../../CLAUDE.md)).** Write these before the implementation:
- `test_plan_layout_dedups_intra_monomer_pairs` — feed a synthetic locator with 4 rows that map to 2 unique parents; assert the plan has 2 entries (not 4) and that dst offsets are strictly monotone.
- `test_plan_layout_groups_by_source_tar` — assert jobs sort by `(src_tar, src_cif_off)`.
- `test_repack_byte_roundtrip` — small synthetic tar fixture, run repack end-to-end, read both members back via the new locator, assert byte-equality.
- `test_repack_real_sample` — *requires labs access*; pick 50 random parents from the production locator, repack into a tmp blob, byte-compare against labs. Mark with `pytest.mark.requires_labs` (skipped in CI by default). This is the gate before the user runs the real ~30-min repack.
- `test_rewrite_locator_preserves_row_count_and_schema` — input 1.175M rows synthetic, output same row count, same columns except the offsets+tar columns.

**Implementation deliverables.**
- `plan_layout(locator_df: pd.DataFrame) -> pd.DataFrame` — dedup + sort + plan dst offsets.
- `repack(plan_df, labs_root: Path, out_path: Path) -> dict[afdb_id, (dst_cif, dst_pae, dst_conf)]` — actual I/O; report MB/s; **single-stream first**, parallel-writer optimisation only if the user complains. KISS.
- `rewrite_locator(orig_locator, dst_map, blob_relpath: str) -> pd.DataFrame`.
- `main()` argparse: `--locator`, `--labs-root`, `--out-dir`, `--sample-verify N`, `--shard-bytes` (default unsharded).

**Reviewer panel.** Per [CLAUDE.md PR-review protocol](../../CLAUDE.md):
- `code-review-debug-complexity-expert` (mandatory).
- `ml-protein-architect` (data pipeline + module layout — this PR touches `proteinfoundation/datasets/teddymer/`).
- `ml-software-pytorch-jax-expert` is *not* needed (no autograd, no GPU, no compile); skip unless the parallel-writer optimisation lands and someone wants a second pair of eyes on the multiprocessing.

Dispatch reviewers in parallel.

## Open questions for the user (resolve before coding)

1. **Single blob vs sharded.** Default plan is one `data.blob`. Want N shards (e.g. 16 × ~9 GB) preemptively, or only if a profiling pass shows a single-fd bottleneck? Lean: single blob, NVMe handles concurrent seeks fine.
2. **/netscratch staging path.** Confirm `/netscratch/schekmenev/teddymer_v1_blob/` is acceptable (or pick another subdir). Need it to be writable from the login node *and* readable from the compute nodes the sbatch script lands on.
3. **SLURM-launched repack vs login-node run.** Login node measured at 250 MB/s single-stream from labs; 136 GB → ~10 min plus per-syscall overhead → ~30 min total. Probably fine on the login node, but a `build_teddymer_blob.sbatch` is trivial to add if the user prefers.
4. **Default-flip the sbatch.** Should `scripts/train_confidence_teddymer_pae.sbatch` start exporting `TEDDYMER_VIEW_ROOT=/netscratch/...` once the blob exists, or leave it to runtime override? Lean: leave the Hydra defaults pointing at labs (the user manually exports during the sprint), then flip after the first successful blob-backed run.

## Reference paths

- Code module: [src/proteinfoundation/datasets/teddymer/](../../src/proteinfoundation/datasets/teddymer/)
- Tar-offset primitive: [src/proteinfoundation/datasets/teddymer/io.py](../../src/proteinfoundation/datasets/teddymer/io.py)
- Dataloader: [src/proteinfoundation/datasets/teddymer/dataset.py](../../src/proteinfoundation/datasets/teddymer/dataset.py)
- AFDB-label transforms: [src/proteinfoundation/datasets/transforms.py:3573-3697](../../src/proteinfoundation/datasets/transforms.py#L3573-L3697)
- Hydra dataset config: [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml)
- Hydra training config: [configs/confidence/distillation_teddymer_pae.yaml](../../configs/confidence/distillation_teddymer_pae.yaml)
- SLURM launcher: [scripts/train_confidence_teddymer_pae.sbatch](../../scripts/train_confidence_teddymer_pae.sbatch)
- Existing view: `/mnt/storage01/home/schekmenev/data/teddymer_v1/`
- Labs AFDB root: `/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4/`
- Target staging dir: `/netscratch/schekmenev/teddymer_v1_blob/` (to be created)

## Project conventions reminder

- Branch off `dev`, open PR into `dev`. **Do not** reuse `prepare_teddy_pr_fix_bugs`.
- **No Claude attribution** in commits or PR descriptions. The user authors all commits.
- For `gh`: prefix every Bash invocation with `module load gh && gh ...` (env-module, not on PATH).
- TDD discipline via subagents: plan with `software-planning-architect`, write tests first, implement with `ml-protein-architect`, review with `code-review-debug-complexity-expert` + domain experts.
- No new `.md` docs unless the user asks. This handoff is the exception.
- Storage rule: stage training data to /netscratch; don't read training I/O directly from labs. See `[[storage-topology]]` in user memory.
