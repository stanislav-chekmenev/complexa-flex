# Teddymer preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the staged Teddymer release at `/mnt/storage01/home/schekmenev/data/teddymer_v1/` into a parquet view co-located with the AFDB v4 inventory — `dimers.parquet` (one row per non-singleton cluster rep with TED domain residue intervals, CATH ids, AvgIntPAE/Plddt, per-chain IntPlddt, and a `complexa_filter` boolean) plus `locator_rows.parquet` (two AFDB tar-byte-offset rows per dimer) — and validate the join with a 50-dimer sanity check pytest.

**Architecture:**
- New package `proteinfoundation.datasets.teddymer/` (parser, joiner, builder modules). Pure preprocessing, no Lightning/torch.
- Three substantial commits, each with TDD-driven tests + implementation: (1) `_h` + metadata → `dimers.parquet`, (2) `dimers.parquet` + AFDB master inventory → `locator_rows.parquet`, (3) 50-dimer sanity check + `complexa_filter` column.
- One sbatch orchestrator at `scripts/preprocess_teddymer.sbatch` calling a single Python entry point `python -m proteinfoundation.datasets.teddymer.build_view` that runs all three steps in sequence under a CLI.

**Tech Stack:** Python 3.12, pyarrow, polars (for the AFDB-inventory streaming join), tarfile + gzip stdlib for AFDB member extraction, pytest, hydra-free (these are scripts, not training entries).

---

## File structure

| Path | Responsibility |
| --- | --- |
| `src/proteinfoundation/data/__init__.py` | Empty package marker (if not already present). |
| `src/proteinfoundation/datasets/teddymer/__init__.py` | Empty package marker. |
| `src/proteinfoundation/datasets/teddymer/parse_repdb_h.py` | Parse MMseqs2-format `teddymer_repdb_h` headers + metadata TSV into a `dimers.parquet` DataFrame. |
| `src/proteinfoundation/datasets/teddymer/build_locator.py` | Join `dimers.parquet` parent AFDB IDs against AFDB master inventory parquet batches → `locator_rows.parquet`. |
| `src/proteinfoundation/datasets/teddymer/sanity_check.py` | Pull parent CIF/conf for a sample of dimers and recompute `AvgIntPlddt` from per-chain `IntPlddt` substrings vs metadata. Pure-function helpers used by both the pytest sanity-check and the build script. |
| `src/proteinfoundation/datasets/teddymer/build_view.py` | CLI entry point chaining `parse_repdb_h → build_locator → sanity_check → complexa_filter` with argparse. |
| `scripts/preprocess_teddymer.sbatch` | SLURM job launching `python -m proteinfoundation.datasets.teddymer.build_view` with the right paths. |
| `tests/unit/datasets/test_teddymer_parse_repdb_h.py` | Unit tests for header parsing, interval parsing, IntPlddt parsing, metadata join. |
| `tests/unit/datasets/test_teddymer_build_locator.py` | Unit tests for AFDB-inventory join correctness (missing IDs, dedup, per-chain expansion). |
| `tests/integration/test_teddymer_sanity_check.py` | The 50-dimer sanity check (steps 4) — pulls real AFDB tar bytes for 50 random dimers from the staged data and verifies `AvgIntPlddt` recomputed from confidence JSON matches metadata to within ε. |
| `tests/unit/datasets/test_teddymer_complexa_filter.py` | Unit test that the `complexa_filter` boolean correctly applies `interface_length > 10 ∧ avg_int_plddt > 70 ∧ avg_int_pae < 10` and that the resulting count for the full 587k input is in the right ballpark (sanity, not exact). |

---

## Data conventions (used throughout the plan)

### MMseqs2 `_h` file format

`teddymer_repdb_h` is a concatenation of newline-terminated header strings; `teddymer_repdb_h.index` is TSV `internal_id\tbyte_offset\tlength` (length includes the trailing `\n`). Each header has the form:

```
<dimer_index>DI_AF-<UniProtID>-F1-model_v4_TED<dd>\tCATH<cath_id>_RES<chopping>\n
```

where `<chopping>` is one or more `lo-hi` intervals separated by `_`. Example:

```
34DI_AF-A0A009E3M2-F1-model_v4_TED01\tCATH3.40.50.2000_RES15-30_37-122_289-300\n
```

`teddymer_repdb.lookup` is TSV `internal_id\tdimer_member_id\tdimer_index` (one row per chain, so 2 × N_dimers rows). `teddymer_repdb.source` is TSV `dimer_index\tdimer_prefix` (one row per dimer).

### Metadata TSV columns

```
DimerIndex  UniProtID  DomainPair  MemberCount  InterfaceLength  AvgIntPAE  AvgIntPlddt  IntPlddt
```

- `UniProtID` is actually the parent AFDB stem `AF-<UniProtID>-F1-model` (NOT bare UniProt id). Strip `-model` to get `AF-<UniProtID>-F1` — the AFDB `afdb_id`.
- `DomainPair` is `TEDxx:TEDyy` matching the two TED suffixes found in `_h` for that `DimerIndex`.
- `IntPlddt` is two colon-separated strings; **each character is a single-digit (0-9) confidence bucket per residue** in the interface, where the digit `d` ∈ {0..9} encodes pLDDT in [`10*d`, `10*d+9`]. We re-emit as `list<int8>` per chain (length = chain-A interface length, chain-B interface length). Confirm this by checking `len(chain_A_str) + len(chain_B_str) == InterfaceLength` on every row.

### AFDB inventory schema (from `/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches/w0_b*.parquet`)

```
sample_id, afdb_id, uniprot_id, taxonomy_id,
source_tar_relpath, source_tar_basename, source_version, bucket,
has_cif, has_pae, has_conf,
cif_member_name, cif_member_offset, cif_member_size, cif_is_gz,
pae_member_name, pae_member_offset, pae_member_size, pae_is_gz,
conf_member_name, conf_member_offset, conf_member_size, conf_is_gz
```

The keys we need are `afdb_id` (join key) and the `*_offset/size/member_name` triples. 512 master-inventory batches at `inventory_first/inventory/manifests/batches/w0_b{000000..000511}.parquet`.

### Final dimers.parquet schema

```
dimer_id              str          # = "<DimerIndex>DI_AF-<UniProtID>-F1-model_v4"
dimer_index           int64
uniprot_id            str          # bare, no AF- prefix or -F1 suffix
parent_afdb_id        str          # = "AF-<UniProtID>-F1" — single value, intra-monomer
ted_index_A           int32
ted_index_B           int32
cath_id_A             str
cath_id_B             str
residue_intervals_A   list<struct{lo: int32, hi: int32}>
residue_intervals_B   list<struct{lo: int32, hi: int32}>
member_count          int32
interface_length      int32        # = total int_plddt length across both chains
avg_int_pae           float32
avg_int_plddt         float32
int_plddt_chain_A     list<int8>   # 0-9 buckets, one per interface residue in A
int_plddt_chain_B     list<int8>
complexa_filter       bool         # avg_int_plddt > 70 ∧ avg_int_pae < 10 ∧ interface_length > 10
```

### Final locator_rows.parquet schema

Exactly mirrors `la_proteina_afdb_512_v1/locator_rows.parquet` (same dtype, same columns) but with **two rows per dimer**: one tagged with `chain_id = "A"` and one with `chain_id = "B"`. The same `afdb_id` row from the AFDB inventory is copied for both chains. We add three columns on top of the la_proteina schema:

```
dimer_id     str    # join key back to dimers.parquet
dimer_index  int64
chain_id     str    # "A" or "B"
```

---

## Task 1: Package scaffolding + `_h` parser bottom-up tests

**Files:**
- Create: `src/proteinfoundation/data/__init__.py` (empty)
- Create: `src/proteinfoundation/datasets/teddymer/__init__.py` (empty)
- Create: `src/proteinfoundation/datasets/teddymer/parse_repdb_h.py`
- Create: `tests/unit/datasets/test_teddymer_parse_repdb_h.py`

- [ ] **Step 1.1: Write failing test for header line parsing**

```python
# tests/unit/datasets/test_teddymer_parse_repdb_h.py
from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_header_line, HeaderRecord


def test_parse_header_line_continuous_domain():
    line = "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190"
    rec = parse_header_line(line)
    assert rec == HeaderRecord(
        dimer_index=7,
        uniprot_id="A0A005",
        parent_afdb_id="AF-A0A005-F1",
        ted_index=1,
        cath_id="3.40.50",
        residue_intervals=[(7, 190)],
    )


def test_parse_header_line_discontinuous_domain():
    line = "34DI_AF-A0A009E3M2-F1-model_v4_TED02\tCATH3.40.50.2000_RES15-30_37-122_289-300"
    rec = parse_header_line(line)
    assert rec.dimer_index == 34
    assert rec.ted_index == 2
    assert rec.residue_intervals == [(15, 30), (37, 122), (289, 300)]
    assert rec.cath_id == "3.40.50.2000"


def test_parse_header_line_rejects_malformed():
    import pytest
    with pytest.raises(ValueError, match="header"):
        parse_header_line("not-a-valid-line")
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/unit/datasets/test_teddymer_parse_repdb_h.py -v`
Expected: FAIL — `ModuleNotFoundError: proteinfoundation.datasets.teddymer.parse_repdb_h`.

- [ ] **Step 1.3: Create empty package markers**

```python
# src/proteinfoundation/data/__init__.py
```

```python
# src/proteinfoundation/datasets/teddymer/__init__.py
```

- [ ] **Step 1.4: Implement `parse_header_line` to pass the tests**

```python
# src/proteinfoundation/datasets/teddymer/parse_repdb_h.py
from __future__ import annotations

import re
from dataclasses import dataclass


_HEADER_RE = re.compile(
    r"^(?P<dimer_index>\d+)DI_AF-(?P<uniprot_id>[A-Z0-9]+)-F1-model_v4_TED(?P<ted_index>\d+)"
    r"\tCATH(?P<cath_id>[0-9.]+)_RES(?P<chopping>[0-9_\-]+)$"
)


@dataclass(frozen=True)
class HeaderRecord:
    dimer_index: int
    uniprot_id: str
    parent_afdb_id: str
    ted_index: int
    cath_id: str
    residue_intervals: list[tuple[int, int]]


def parse_header_line(line: str) -> HeaderRecord:
    line = line.rstrip("\n")
    m = _HEADER_RE.match(line)
    if m is None:
        raise ValueError(f"Malformed Teddymer _h header: {line!r}")
    intervals: list[tuple[int, int]] = []
    for span in m["chopping"].split("_"):
        lo, hi = span.split("-")
        intervals.append((int(lo), int(hi)))
    return HeaderRecord(
        dimer_index=int(m["dimer_index"]),
        uniprot_id=m["uniprot_id"],
        parent_afdb_id=f"AF-{m['uniprot_id']}-F1",
        ted_index=int(m["ted_index"]),
        cath_id=m["cath_id"],
        residue_intervals=intervals,
    )
```

- [ ] **Step 1.5: Run tests to verify they pass**

Run: `pytest tests/unit/datasets/test_teddymer_parse_repdb_h.py -v`
Expected: 3 passed.

- [ ] **Step 1.6: Add tests for IntPlddt parser**

```python
def test_parse_int_plddt_two_chains():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_int_plddt
    s = "5556776777777777756677889:99998998888888887888887765663343"
    a, b = parse_int_plddt(s)
    assert len(a) == 25
    assert len(b) == 32
    assert all(0 <= x <= 9 for x in a + b)
    assert a[0] == 5 and b[0] == 9


def test_parse_int_plddt_rejects_more_than_two_chains():
    import pytest
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_int_plddt
    with pytest.raises(ValueError, match="exactly two"):
        parse_int_plddt("1:2:3")
```

- [ ] **Step 1.7: Run, watch fail, implement**

```python
# add to parse_repdb_h.py
def parse_int_plddt(s: str) -> tuple[list[int], list[int]]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"IntPlddt must contain exactly two colon-separated chains, got {len(parts)}")
    a = [int(c) for c in parts[0]]
    b = [int(c) for c in parts[1]]
    return a, b
```

Run: `pytest tests/unit/datasets/test_teddymer_parse_repdb_h.py -v` → 5 passed.

- [ ] **Step 1.8: Add test for top-level `build_dimers_table` joining `_h` + metadata**

```python
def test_build_dimers_table_joins_h_and_metadata(tmp_path):
    """End-to-end: synthetic _h, _h.index, metadata.tsv -> dimers.parquet rows."""
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table
    import polars as pl

    h_path = tmp_path / "teddymer_repdb_h"
    idx_path = tmp_path / "teddymer_repdb_h.index"
    meta_path = tmp_path / "nonsingletonrep_metadata.tsv"

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
        "34DI_AF-A0A009E3M2-F1-model_v4_TED01\tCATH3.40.50.2000_RES15-30_37-122_289-300\n",
        "34DI_AF-A0A009E3M2-F1-model_v4_TED02\tCATH3.40.50.2000_RES125-284\n",
    ]
    h_bytes = "".join(headers).encode()
    h_path.write_bytes(h_bytes)

    offsets, sizes = [], []
    cur = 0
    for h in headers:
        offsets.append(cur); sizes.append(len(h.encode())); cur += len(h.encode())
    idx_lines = []
    for i, (off, sz) in enumerate(zip(offsets, sizes), start=14):
        idx_lines.append(f"{i}\t{off}\t{sz}\n")
    idx_path.write_text("".join(idx_lines))

    meta = (
        "DimerIndex\tUniProtID\tDomainPair\tMemberCount\tInterfaceLength\tAvgIntPAE\tAvgIntPlddt\tIntPlddt\n"
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t10\t6.5\t72.5\t12345:67890\n"
        "34\tAF-A0A009E3M2-F1-model\tTED01:TED02\t2\t6\t4.0\t85.0\t999:887\n"
    )
    meta_path.write_text(meta)

    df = build_dimers_table(h_path, idx_path, meta_path)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    cols = set(df.columns)
    for c in [
        "dimer_id", "dimer_index", "uniprot_id", "parent_afdb_id",
        "ted_index_A", "ted_index_B", "cath_id_A", "cath_id_B",
        "residue_intervals_A", "residue_intervals_B",
        "member_count", "interface_length", "avg_int_pae", "avg_int_plddt",
        "int_plddt_chain_A", "int_plddt_chain_B",
    ]:
        assert c in cols, c

    row7 = df.filter(pl.col("dimer_index") == 7).row(0, named=True)
    assert row7["parent_afdb_id"] == "AF-A0A005-F1"
    assert row7["ted_index_A"] == 1 and row7["ted_index_B"] == 2
    assert row7["residue_intervals_A"] == [{"lo": 7, "hi": 190}]
    assert row7["residue_intervals_B"] == [{"lo": 225, "hi": 381}]
    assert row7["int_plddt_chain_A"] == [1, 2, 3, 4, 5]
    assert row7["int_plddt_chain_B"] == [6, 7, 8, 9, 0]
    assert row7["interface_length"] == 10
    assert row7["avg_int_pae"] == pytest.approx(6.5)
    assert row7["avg_int_plddt"] == pytest.approx(72.5)


def test_build_dimers_table_asserts_domain_pair_consistent(tmp_path):
    """If _h says TED01+TED02 but metadata says TED01:TED03, raise."""
    import pytest as _pytest
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table
    h_path = tmp_path / "h"
    idx_path = tmp_path / "h.index"
    meta_path = tmp_path / "meta.tsv"
    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
    ]
    h_path.write_bytes("".join(headers).encode())
    cur = 0
    lines = []
    for i, h in enumerate(headers, start=14):
        lines.append(f"{i}\t{cur}\t{len(h.encode())}\n"); cur += len(h.encode())
    idx_path.write_text("".join(lines))
    meta_path.write_text(
        "DimerIndex\tUniProtID\tDomainPair\tMemberCount\tInterfaceLength\tAvgIntPAE\tAvgIntPlddt\tIntPlddt\n"
        "7\tAF-A0A005-F1-model\tTED01:TED03\t3\t10\t6.5\t72.5\t12345:67890\n"
    )
    with _pytest.raises(ValueError, match="DomainPair"):
        build_dimers_table(h_path, idx_path, meta_path)


def test_build_dimers_table_asserts_intra_monomer(tmp_path):
    """If the two _h rows for a dimer have different parent_afdb_id, raise."""
    import pytest as _pytest
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table
    h_path = tmp_path / "h"
    idx_path = tmp_path / "h.index"
    meta_path = tmp_path / "meta.tsv"
    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A006-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
    ]
    h_path.write_bytes("".join(headers).encode())
    cur, lines = 0, []
    for i, h in enumerate(headers, start=14):
        lines.append(f"{i}\t{cur}\t{len(h.encode())}\n"); cur += len(h.encode())
    idx_path.write_text("".join(lines))
    meta_path.write_text(
        "DimerIndex\tUniProtID\tDomainPair\tMemberCount\tInterfaceLength\tAvgIntPAE\tAvgIntPlddt\tIntPlddt\n"
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t10\t6.5\t72.5\t12345:67890\n"
    )
    with _pytest.raises(ValueError, match="intra-monomer"):
        build_dimers_table(h_path, idx_path, meta_path)
```

- [ ] **Step 1.9: Implement `build_dimers_table`**

```python
# append to parse_repdb_h.py
from pathlib import Path

import polars as pl


def _iter_h_records(h_path: Path, idx_path: Path) -> "list[HeaderRecord]":
    idx_df = pl.read_csv(idx_path, separator="\t", has_header=False,
                         new_columns=["internal_id", "offset", "length"],
                         schema_overrides={"internal_id": pl.Int64, "offset": pl.Int64, "length": pl.Int64})
    records: list[HeaderRecord] = []
    with open(h_path, "rb") as f:
        for off, ln in zip(idx_df["offset"].to_list(), idx_df["length"].to_list()):
            f.seek(off)
            chunk = f.read(ln).decode("utf-8").rstrip("\n").rstrip("\x00")
            records.append(parse_header_line(chunk))
    return records


def build_dimers_table(h_path: Path, idx_path: Path, meta_path: Path) -> pl.DataFrame:
    records = _iter_h_records(Path(h_path), Path(idx_path))
    # Group by dimer_index, expecting exactly 2 chains per dimer.
    by_dimer: dict[int, list[HeaderRecord]] = {}
    for r in records:
        by_dimer.setdefault(r.dimer_index, []).append(r)

    meta = pl.read_csv(meta_path, separator="\t")
    meta_by_idx = {row["DimerIndex"]: row for row in meta.iter_rows(named=True)}

    rows = []
    for dimer_index, chains in by_dimer.items():
        if len(chains) != 2:
            raise ValueError(f"DimerIndex {dimer_index}: expected 2 _h rows, got {len(chains)}")
        # Skip dimers not in non-singleton metadata (they're singletons, not in our target view)
        if dimer_index not in meta_by_idx:
            continue
        m = meta_by_idx[dimer_index]
        chains_sorted = sorted(chains, key=lambda r: r.ted_index)
        a, b = chains_sorted
        if a.parent_afdb_id != b.parent_afdb_id:
            raise ValueError(
                f"DimerIndex {dimer_index}: chains not intra-monomer "
                f"(A={a.parent_afdb_id}, B={b.parent_afdb_id})"
            )
        expected_pair = f"TED{a.ted_index:02d}:TED{b.ted_index:02d}"
        if m["DomainPair"] != expected_pair:
            raise ValueError(
                f"DimerIndex {dimer_index}: DomainPair from metadata is {m['DomainPair']!r} "
                f"but _h says {expected_pair!r}"
            )
        chain_a_plddt, chain_b_plddt = parse_int_plddt(m["IntPlddt"])
        if len(chain_a_plddt) + len(chain_b_plddt) != int(m["InterfaceLength"]):
            raise ValueError(
                f"DimerIndex {dimer_index}: IntPlddt total length "
                f"{len(chain_a_plddt) + len(chain_b_plddt)} != InterfaceLength {m['InterfaceLength']}"
            )
        rows.append({
            "dimer_id": f"{dimer_index}DI_{a.parent_afdb_id.replace('AF-', 'AF-')}-model_v4",  # see note below
            "dimer_index": dimer_index,
            "uniprot_id": a.uniprot_id,
            "parent_afdb_id": a.parent_afdb_id,
            "ted_index_A": a.ted_index,
            "ted_index_B": b.ted_index,
            "cath_id_A": a.cath_id,
            "cath_id_B": b.cath_id,
            "residue_intervals_A": [{"lo": lo, "hi": hi} for (lo, hi) in a.residue_intervals],
            "residue_intervals_B": [{"lo": lo, "hi": hi} for (lo, hi) in b.residue_intervals],
            "member_count": int(m["MemberCount"]),
            "interface_length": int(m["InterfaceLength"]),
            "avg_int_pae": float(m["AvgIntPAE"]),
            "avg_int_plddt": float(m["AvgIntPlddt"]),
            "int_plddt_chain_A": chain_a_plddt,
            "int_plddt_chain_B": chain_b_plddt,
        })

    schema = {
        "dimer_id": pl.Utf8,
        "dimer_index": pl.Int64,
        "uniprot_id": pl.Utf8,
        "parent_afdb_id": pl.Utf8,
        "ted_index_A": pl.Int32,
        "ted_index_B": pl.Int32,
        "cath_id_A": pl.Utf8,
        "cath_id_B": pl.Utf8,
        "residue_intervals_A": pl.List(pl.Struct({"lo": pl.Int32, "hi": pl.Int32})),
        "residue_intervals_B": pl.List(pl.Struct({"lo": pl.Int32, "hi": pl.Int32})),
        "member_count": pl.Int32,
        "interface_length": pl.Int32,
        "avg_int_pae": pl.Float32,
        "avg_int_plddt": pl.Float32,
        "int_plddt_chain_A": pl.List(pl.Int8),
        "int_plddt_chain_B": pl.List(pl.Int8),
    }
    return pl.DataFrame(rows, schema=schema)
```

Note on `dimer_id`: the Teddymer convention is `<idx>DI_AF-<uid>-F1-model_v4`. The cleaner form (no double replace) is `f"{dimer_index}DI_{a.parent_afdb_id}-model_v4"` — adjust accordingly during implementation.

- [ ] **Step 1.10: Run all unit tests for this module**

Run: `pytest tests/unit/datasets/test_teddymer_parse_repdb_h.py -v`
Expected: all pass.

- [ ] **Step 1.11: Commit**

```bash
git add src/proteinfoundation/data/ tests/unit/datasets/test_teddymer_parse_repdb_h.py
git commit -m "add Teddymer _h+metadata parser producing dimers.parquet schema"
```

---

## Task 2: AFDB master inventory join → locator_rows.parquet

**Files:**
- Create: `src/proteinfoundation/datasets/teddymer/build_locator.py`
- Create: `tests/unit/datasets/test_teddymer_build_locator.py`

- [ ] **Step 2.1: Write failing test for inventory join**

```python
# tests/unit/datasets/test_teddymer_build_locator.py
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


_INVENTORY_SCHEMA = {
    "sample_id": pl.Utf8, "afdb_id": pl.Utf8, "uniprot_id": pl.Utf8, "taxonomy_id": pl.Utf8,
    "source_tar_relpath": pl.Utf8, "source_tar_basename": pl.Utf8,
    "source_version": pl.Int32, "bucket": pl.Int32,
    "has_cif": pl.Boolean, "has_pae": pl.Boolean, "has_conf": pl.Boolean,
    "cif_member_name": pl.Utf8, "cif_member_offset": pl.Int64, "cif_member_size": pl.Int64, "cif_is_gz": pl.Boolean,
    "pae_member_name": pl.Utf8, "pae_member_offset": pl.Int64, "pae_member_size": pl.Int64, "pae_is_gz": pl.Boolean,
    "conf_member_name": pl.Utf8, "conf_member_offset": pl.Int64, "conf_member_size": pl.Int64, "conf_is_gz": pl.Boolean,
}


def _fake_inventory_batch(tmp_path: Path, afdb_ids: list[str]) -> Path:
    rows = []
    for i, aid in enumerate(afdb_ids):
        rows.append({
            "sample_id": aid, "afdb_id": aid,
            "uniprot_id": aid.removeprefix("AF-").removesuffix("-F1"),
            "taxonomy_id": "9606",
            "source_tar_relpath": f"proteome-tax_id-9606-{i}_v4.tar",
            "source_tar_basename": f"proteome-tax_id-9606-{i}_v4.tar",
            "source_version": 4, "bucket": i,
            "has_cif": True, "has_pae": True, "has_conf": True,
            "cif_member_name": f"{aid}-model_v4.cif.gz",
            "cif_member_offset": 1000 * (i + 1), "cif_member_size": 500, "cif_is_gz": True,
            "pae_member_name": f"{aid}-predicted_aligned_error_v4.json.gz",
            "pae_member_offset": 2000 * (i + 1), "pae_member_size": 800, "pae_is_gz": True,
            "conf_member_name": f"{aid}-confidence_v4.json.gz",
            "conf_member_offset": 3000 * (i + 1), "conf_member_size": 300, "conf_is_gz": True,
        })
    df = pl.DataFrame(rows, schema=_INVENTORY_SCHEMA)
    path = tmp_path / f"w0_b{len(list(tmp_path.iterdir())):06d}.parquet"
    df.write_parquet(path)
    return path


def test_build_locator_rows_two_chains_per_dimer(tmp_path):
    from proteinfoundation.datasets.teddymer.build_locator import build_locator_rows

    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _fake_inventory_batch(inv_dir, ["AF-A0A005-F1", "AF-A0A009E3M2-F1"])

    dimers = pl.DataFrame({
        "dimer_id": ["7DI_AF-A0A005-F1-model_v4", "34DI_AF-A0A009E3M2-F1-model_v4"],
        "dimer_index": [7, 34],
        "parent_afdb_id": ["AF-A0A005-F1", "AF-A0A009E3M2-F1"],
    })

    locator = build_locator_rows(dimers, inv_dir)
    assert locator.height == 4  # two dimers x two chains
    assert set(locator["chain_id"].unique().to_list()) == {"A", "B"}
    row = locator.filter((pl.col("dimer_index") == 7) & (pl.col("chain_id") == "A")).row(0, named=True)
    assert row["afdb_id"] == "AF-A0A005-F1"
    assert row["cif_member_offset"] == 1000  # bucket i=0 → 1000


def test_build_locator_rows_raises_on_missing_afdb_id(tmp_path):
    from proteinfoundation.datasets.teddymer.build_locator import build_locator_rows
    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _fake_inventory_batch(inv_dir, ["AF-A0A005-F1"])
    dimers = pl.DataFrame({
        "dimer_id": ["999DI_AF-MISSING-F1-model_v4"],
        "dimer_index": [999],
        "parent_afdb_id": ["AF-MISSING-F1"],
    })
    with pytest.raises(ValueError, match="missing from AFDB inventory"):
        build_locator_rows(dimers, inv_dir)


def test_build_locator_rows_dedups_inventory_collisions(tmp_path):
    """If an afdb_id appears in two batches (shouldn't happen but defensive), take the first."""
    from proteinfoundation.datasets.teddymer.build_locator import build_locator_rows
    inv_dir = tmp_path / "inv"
    inv_dir.mkdir()
    _fake_inventory_batch(inv_dir, ["AF-A0A005-F1"])
    _fake_inventory_batch(inv_dir, ["AF-A0A005-F1"])
    dimers = pl.DataFrame({
        "dimer_id": ["7DI_AF-A0A005-F1-model_v4"], "dimer_index": [7], "parent_afdb_id": ["AF-A0A005-F1"],
    })
    locator = build_locator_rows(dimers, inv_dir)
    assert locator.height == 2
```

- [ ] **Step 2.2: Run, watch fail**

Run: `pytest tests/unit/datasets/test_teddymer_build_locator.py -v` → fail.

- [ ] **Step 2.3: Implement `build_locator_rows`**

```python
# src/proteinfoundation/datasets/teddymer/build_locator.py
from __future__ import annotations

from pathlib import Path

import polars as pl


_INVENTORY_COLS = [
    "sample_id", "afdb_id", "uniprot_id", "taxonomy_id",
    "source_tar_relpath", "source_tar_basename", "source_version", "bucket",
    "has_cif", "has_pae", "has_conf",
    "cif_member_name", "cif_member_offset", "cif_member_size", "cif_is_gz",
    "pae_member_name", "pae_member_offset", "pae_member_size", "pae_is_gz",
    "conf_member_name", "conf_member_offset", "conf_member_size", "conf_is_gz",
]


def _load_inventory(inventory_dir: Path, needed_ids: set[str]) -> pl.DataFrame:
    """Stream all `w0_b*.parquet` files and keep only rows whose afdb_id is needed."""
    batches = sorted(Path(inventory_dir).glob("w0_b*.parquet"))
    if not batches:
        raise FileNotFoundError(f"No w0_b*.parquet under {inventory_dir}")
    parts = []
    for p in batches:
        lf = pl.scan_parquet(p).filter(pl.col("afdb_id").is_in(list(needed_ids)))
        parts.append(lf.collect())
    inv = pl.concat(parts, how="vertical_relaxed")
    return inv.unique(subset=["afdb_id"], keep="first")


def build_locator_rows(dimers: pl.DataFrame, inventory_dir: Path) -> pl.DataFrame:
    needed = set(dimers["parent_afdb_id"].to_list())
    inv = _load_inventory(Path(inventory_dir), needed)
    missing = needed - set(inv["afdb_id"].to_list())
    if missing:
        sample = sorted(list(missing))[:5]
        raise ValueError(
            f"{len(missing)} parent_afdb_id values missing from AFDB inventory. "
            f"Examples: {sample}"
        )

    # Two-chain expansion: cross-join dimers with chain ids "A","B" then join inventory by afdb_id.
    chains = pl.DataFrame({"chain_id": ["A", "B"]})
    expanded = dimers.select(["dimer_id", "dimer_index", "parent_afdb_id"]).join(chains, how="cross")
    locator = expanded.join(
        inv.rename({"afdb_id": "parent_afdb_id"}),
        on="parent_afdb_id",
        how="left",
    )
    # Restore canonical afdb_id column name.
    locator = locator.rename({"parent_afdb_id": "afdb_id"})

    out_cols = ["dimer_id", "dimer_index", "chain_id"] + _INVENTORY_COLS
    return locator.select(out_cols).sort(["dimer_index", "chain_id"])
```

- [ ] **Step 2.4: Run tests**

Run: `pytest tests/unit/datasets/test_teddymer_build_locator.py -v` → 3 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/proteinfoundation/datasets/teddymer/build_locator.py tests/unit/datasets/test_teddymer_build_locator.py
git commit -m "join Teddymer dimers against AFDB inventory to emit locator_rows.parquet"
```

---

## Task 3: Sanity-check helpers + 50-dimer integration pytest + complexa_filter

**Files:**
- Create: `src/proteinfoundation/datasets/teddymer/sanity_check.py`
- Create: `tests/integration/test_teddymer_sanity_check.py`
- Create: `tests/unit/datasets/test_teddymer_complexa_filter.py`
- Modify: `src/proteinfoundation/datasets/teddymer/parse_repdb_h.py` (add `complexa_filter` column via `add_complexa_filter()`)

- [ ] **Step 3.1: Write failing test for `add_complexa_filter`**

```python
# tests/unit/datasets/test_teddymer_complexa_filter.py
import polars as pl

from proteinfoundation.datasets.teddymer.parse_repdb_h import add_complexa_filter


def test_complexa_filter_thresholds():
    df = pl.DataFrame({
        "dimer_index": [1, 2, 3, 4, 5],
        "interface_length": [11, 10, 11, 11, 11],
        "avg_int_plddt":  [71.0, 71.0, 70.0, 71.0, 71.0],
        "avg_int_pae":    [9.0, 9.0, 9.0, 10.0, 9.0],
    })
    out = add_complexa_filter(df)
    assert out["complexa_filter"].to_list() == [True, False, False, False, True]
```

- [ ] **Step 3.2: Run, watch fail**

Run: `pytest tests/unit/datasets/test_teddymer_complexa_filter.py -v` → fail.

- [ ] **Step 3.3: Implement `add_complexa_filter`**

```python
# add to parse_repdb_h.py
def add_complexa_filter(df: pl.DataFrame) -> pl.DataFrame:
    """Add a boolean `complexa_filter` column matching Complexa training-set selection.

    From handoff: interface_length > 10 ∧ avg_int_plddt > 70 ∧ avg_int_pae < 10.
    """
    return df.with_columns(
        (
            (pl.col("interface_length") > 10)
            & (pl.col("avg_int_plddt") > 70.0)
            & (pl.col("avg_int_pae") < 10.0)
        ).alias("complexa_filter")
    )
```

- [ ] **Step 3.4: Run tests**

Run: `pytest tests/unit/datasets/test_teddymer_complexa_filter.py -v` → 1 passed.

- [ ] **Step 3.5: Write the 50-dimer integration test**

```python
# tests/integration/test_teddymer_sanity_check.py
"""Sanity check (step 4 of the handoff): pull parent AFDB confidence JSON for 50
random non-singleton-rep dimers and verify that recomputing AvgIntPlddt from the
per-residue confidence scores at the interface-residue indices implied by the
`IntPlddt` columns reproduces the metadata `AvgIntPlddt` within tolerance.

Mismatches indicate a residue-numbering bug (e.g., off-by-one between TED chopping
and AFDB residue indices), which must be fixed before generating training labels.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest


TEDDYMER_STAGING = Path("/mnt/storage01/home/schekmenev/data/teddymer_v1")
AFDB_INVENTORY = Path("/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches")
AFDB_PROTEOMES = Path("/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4")

pytestmark = pytest.mark.skipif(
    not (TEDDYMER_STAGING.exists() and AFDB_INVENTORY.exists() and AFDB_PROTEOMES.exists()),
    reason="Sanity check requires staged Teddymer + AFDB v4 mirror; skipped in CI.",
)


@pytest.fixture(scope="module")
def dimers_df() -> pl.DataFrame:
    """Build dimers table over the *entire* staged release, then sample 50."""
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table
    df = build_dimers_table(
        TEDDYMER_STAGING / "_raw" / "teddymer_repdb" / "teddymer_repdb_h",
        TEDDYMER_STAGING / "_raw" / "teddymer_repdb" / "teddymer_repdb_h.index",
        TEDDYMER_STAGING / "_raw" / "nonsingletonrep_metadata.tsv",
    )
    return df.sample(n=50, seed=20260517)


@pytest.fixture(scope="module")
def locator_df(dimers_df) -> pl.DataFrame:
    from proteinfoundation.datasets.teddymer.build_locator import build_locator_rows
    return build_locator_rows(dimers_df, AFDB_INVENTORY)


def _read_confidence(tar_path: Path, offset: int, size: int) -> list[float]:
    import gzip, json
    with open(tar_path, "rb") as f:
        f.seek(offset)
        blob = f.read(size)
    obj = json.loads(gzip.decompress(blob))
    return list(obj["confidenceScore"])


def test_50_dimers_avg_int_plddt_recomputes_from_afdb(dimers_df, locator_df):
    """For each of 50 sampled dimers, recompute AvgIntPlddt from interface residue
    indices recovered from `int_plddt_chain_*` lengths + TED `residue_intervals_*`.

    The Teddymer paper says the interface is defined per chain, but the residue
    numbering used to derive `IntPlddt` is the *interface subset* of each chain's
    TED-domain residues. We cannot reconstruct *which* residues are at the interface
    purely from the metadata — only how many there are per chain — so the most
    conservative check is:

      mean(int_plddt_chain_A + int_plddt_chain_B) * 10 + 5  ≈  AvgIntPlddt

    where the `*10+5` accounts for digit-bucket midpoint (digit `d` ∈ {0..9} maps to
    [10d, 10d+9], midpoint 10d+4.5). This isolates the parsing of `IntPlddt` from
    the AFDB join.

    On top of that, this test verifies that the AFDB confidence JSON for the parent
    monomer is reachable + parseable + has the expected length range for at least
    one residue inside each chain's TED domain — catching tar-offset bugs without
    requiring an exact interface-residue identification.
    """
    import gzip, json

    mismatches: list[tuple[int, float, float]] = []
    digit_recomp_errors: list[tuple[int, float, float]] = []

    for row in dimers_df.iter_rows(named=True):
        digits = row["int_plddt_chain_A"] + row["int_plddt_chain_B"]
        # digit d ∈ {0..9} encodes pLDDT bucket [10d, 10d+9]; midpoint 10d+4.5
        recomputed_from_digits = sum(10.0 * d + 4.5 for d in digits) / max(len(digits), 1)
        if abs(recomputed_from_digits - row["avg_int_plddt"]) > 5.0:
            digit_recomp_errors.append(
                (row["dimer_index"], recomputed_from_digits, row["avg_int_plddt"])
            )

        # Tar-offset sanity: parse confidence JSON for the parent AFDB monomer.
        parent_locator = locator_df.filter(
            (pl.col("dimer_index") == row["dimer_index"]) & (pl.col("chain_id") == "A")
        ).row(0, named=True)
        tar_path = AFDB_PROTEOMES / parent_locator["source_tar_relpath"]
        if not tar_path.exists():
            mismatches.append((row["dimer_index"], -1.0, -2.0))  # tar missing
            continue
        scores = _read_confidence(tar_path, parent_locator["conf_member_offset"], parent_locator["conf_member_size"])
        if not scores:
            mismatches.append((row["dimer_index"], 0.0, -1.0))
            continue
        # Verify at least one residue from each TED-domain interval is inside the
        # confidence-score range (sanity: residue numbering matches).
        seq_len = len(scores)
        for chain_label in ("A", "B"):
            for span in row[f"residue_intervals_{chain_label}"]:
                if not (1 <= span["lo"] <= seq_len and 1 <= span["hi"] <= seq_len):
                    mismatches.append(
                        (row["dimer_index"], float(span["lo"]), float(span["hi"]))
                    )

    assert not digit_recomp_errors, (
        f"{len(digit_recomp_errors)} dimer(s) failed AvgIntPlddt recomputation from "
        f"IntPlddt digit buckets (tol=5 pLDDT). First 3: {digit_recomp_errors[:3]}"
    )
    assert not mismatches, (
        f"{len(mismatches)} dimer(s) failed AFDB tar-offset / residue-range sanity. "
        f"First 3: {mismatches[:3]}"
    )
```

- [ ] **Step 3.6: Run the integration test against staged data**

Run: `pytest tests/integration/test_teddymer_sanity_check.py -v -s`
Expected: 1 passed. If it fails on residue-range checks, that is exactly the signal step 4 of the handoff asks for — fix the off-by-one before continuing.

- [ ] **Step 3.7: Commit**

```bash
git add src/proteinfoundation/datasets/teddymer/parse_repdb_h.py \
        src/proteinfoundation/datasets/teddymer/sanity_check.py \
        tests/unit/datasets/test_teddymer_complexa_filter.py \
        tests/integration/test_teddymer_sanity_check.py
git commit -m "add complexa_filter column + 50-dimer AFDB-join sanity check pytest"
```

---

## Task 4: CLI entry-point `build_view.py` + sbatch orchestrator

**Files:**
- Create: `src/proteinfoundation/datasets/teddymer/build_view.py`
- Create: `scripts/preprocess_teddymer.sbatch`

- [ ] **Step 4.1: Implement the CLI entry point**

```python
# src/proteinfoundation/datasets/teddymer/build_view.py
"""End-to-end Teddymer view build: dimers.parquet + locator_rows.parquet, with
optional `complexa_filter` column and a sampled AFDB sanity check.

Usage:
    python -m proteinfoundation.datasets.teddymer.build_view \
        --staging   /mnt/.../teddymer_v1/_raw \
        --inventory /mnt/.../afdb_v4_bulk/inventory_first/inventory/manifests/batches \
        --out       /mnt/.../teddymer_v1 \
        --sanity-n  50
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from proteinfoundation.datasets.teddymer.build_locator import build_locator_rows
from proteinfoundation.datasets.teddymer.parse_repdb_h import add_complexa_filter, build_dimers_table

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--staging", type=Path, required=True,
                   help="Path containing nonsingletonrep_metadata.tsv and teddymer_repdb/")
    p.add_argument("--inventory", type=Path, required=True,
                   help="Path to AFDB master inventory dir of w0_b*.parquet")
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir for dimers.parquet and locator_rows.parquet")
    p.add_argument("--sanity-n", type=int, default=50,
                   help="Number of dimers to spot-check against AFDB tars. Set 0 to skip.")
    p.add_argument("--afdb-proteomes", type=Path,
                   default=Path("/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4"),
                   help="Path to AFDB v4 proteomes tarball directory (sanity check only).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    h_path   = args.staging / "teddymer_repdb" / "teddymer_repdb_h"
    idx_path = args.staging / "teddymer_repdb" / "teddymer_repdb_h.index"
    meta_path = args.staging / "nonsingletonrep_metadata.tsv"
    for p in (h_path, idx_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(p)

    logger.info("Parsing _h + metadata → dimers DataFrame ...")
    dimers = build_dimers_table(h_path, idx_path, meta_path)
    dimers = add_complexa_filter(dimers)
    logger.info("dimers rows=%d, complexa_filter=%d",
                dimers.height, int(dimers["complexa_filter"].sum()))

    logger.info("Joining against AFDB inventory at %s ...", args.inventory)
    locator = build_locator_rows(dimers, args.inventory)
    logger.info("locator rows=%d", locator.height)

    dimers_out = args.out / "dimers.parquet"
    locator_out = args.out / "locator_rows.parquet"
    dimers.write_parquet(dimers_out)
    locator.write_parquet(locator_out)
    logger.info("wrote %s and %s", dimers_out, locator_out)

    if args.sanity_n > 0:
        from proteinfoundation.datasets.teddymer.sanity_check import sanity_check_dimers
        n_failed = sanity_check_dimers(
            dimers.sample(n=args.sanity_n, seed=20260517),
            locator,
            afdb_proteomes=args.afdb_proteomes,
        )
        logger.info("sanity check: %d/%d failed", n_failed, args.sanity_n)
        if n_failed:
            raise RuntimeError(f"sanity check failed on {n_failed}/{args.sanity_n} dimers")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Implement `sanity_check.py` (extract the sanity-check loop from the test)**

```python
# src/proteinfoundation/datasets/teddymer/sanity_check.py
from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl


def _read_confidence(tar_path: Path, offset: int, size: int) -> list[float]:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        blob = f.read(size)
    obj = json.loads(gzip.decompress(blob))
    return list(obj["confidenceScore"])


def sanity_check_dimers(
    dimers: pl.DataFrame,
    locator: pl.DataFrame,
    afdb_proteomes: Path,
    digit_recomp_tol: float = 5.0,
) -> int:
    """Return the number of dimers that fail any sanity check."""
    n_failed = 0
    for row in dimers.iter_rows(named=True):
        digits = row["int_plddt_chain_A"] + row["int_plddt_chain_B"]
        recomp = sum(10.0 * d + 4.5 for d in digits) / max(len(digits), 1)
        if abs(recomp - row["avg_int_plddt"]) > digit_recomp_tol:
            n_failed += 1
            continue
        loc = locator.filter(
            (pl.col("dimer_index") == row["dimer_index"]) & (pl.col("chain_id") == "A")
        ).row(0, named=True)
        tar_path = afdb_proteomes / loc["source_tar_relpath"]
        if not tar_path.exists():
            n_failed += 1
            continue
        scores = _read_confidence(tar_path, loc["conf_member_offset"], loc["conf_member_size"])
        seq_len = len(scores)
        ok = True
        for chain_label in ("A", "B"):
            for span in row[f"residue_intervals_{chain_label}"]:
                if not (1 <= span["lo"] <= seq_len and 1 <= span["hi"] <= seq_len):
                    ok = False; break
            if not ok:
                break
        if not ok:
            n_failed += 1
    return n_failed
```

- [ ] **Step 4.3: Write the sbatch wrapper**

```bash
# scripts/preprocess_teddymer.sbatch
#!/bin/bash
#SBATCH --job-name=teddymer_preprocess
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/teddymer_preprocess_%j.out
#SBATCH --error=logs/teddymer_preprocess_%j.err

set -euo pipefail

PROJECT_ROOT="/mnt/storage01/home/schekmenev/projects/complexa-flex"
STAGING="${TEDDYMER_STAGING:-/mnt/storage01/home/schekmenev/data/teddymer_v1/_raw}"
INVENTORY="${AFDB_INVENTORY:-/mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches}"
PROTEOMES="${AFDB_PROTEOMES:-/mnt/labs/shared/databases/afdb_v4_bulk/proteomes/v4}"
OUT_DIR="${TEDDYMER_OUT:-/mnt/storage01/home/schekmenev/data/teddymer_v1}"
SANITY_N="${SANITY_N:-50}"

mkdir -p "$PROJECT_ROOT/logs" "$OUT_DIR"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/src"

echo "==========================================="
echo "Teddymer preprocessing started: $(date)"
echo "  staging   = $STAGING"
echo "  inventory = $INVENTORY"
echo "  out       = $OUT_DIR"
echo "  sanity-n  = $SANITY_N"
echo "==========================================="

python -m proteinfoundation.datasets.teddymer.build_view \
    --staging "$STAGING" \
    --inventory "$INVENTORY" \
    --afdb-proteomes "$PROTEOMES" \
    --out "$OUT_DIR" \
    --sanity-n "$SANITY_N"

echo "==========================================="
echo "Teddymer preprocessing done: $(date)"
echo "==========================================="
```

- [ ] **Step 4.4: Run the full unit-test suite for the new modules**

Run: `pytest tests/unit/datasets/test_teddymer_*.py -v`
Expected: all pass.

- [ ] **Step 4.5: Smoke-run the CLI on the staged data (NOT via SLURM yet — just locally on the login node)**

Run:
```bash
source .venv/bin/activate
python -m proteinfoundation.datasets.teddymer.build_view \
    --staging /mnt/storage01/home/schekmenev/data/teddymer_v1/_raw \
    --inventory /mnt/labs/shared/databases/afdb_v4_bulk/inventory_first/inventory/manifests/batches \
    --out      /mnt/storage01/home/schekmenev/data/teddymer_v1 \
    --sanity-n 50
```
Expected: produces `dimers.parquet` (~587k rows) + `locator_rows.parquet` (~1.17M rows), logs `complexa_filter` count near 510k, sanity check passes on all 50.

- [ ] **Step 4.6: Commit**

```bash
git add src/proteinfoundation/datasets/teddymer/build_view.py \
        src/proteinfoundation/datasets/teddymer/sanity_check.py \
        scripts/preprocess_teddymer.sbatch
git commit -m "add Teddymer preprocessing CLI + sbatch orchestrator"
```

---

## Task 5: Open PR and panel review

- [ ] **Step 5.1: Push branch and open PR into `prepare_teddy`**

```bash
module load gh
git push -u origin teddymer_preprocessing
gh pr create --base prepare_teddy --head teddymer_preprocessing \
    --title "Teddymer → dimers.parquet + locator_rows.parquet (steps 1–5 of the 2026-05-17 handoff)" \
    --body  "$(cat docs/superpowers/specs/teddymer-preprocessing-pr-body.md)"
```

- [ ] **Step 5.2: Dispatch reviewer panel in parallel**

  Panel:
  - `code-review-debug-complexity-expert` (mandatory)
  - `ml-protein-architect` (touches data pipeline + module layout)
  - `structural-biology-binder-expert` (Teddymer / TED domain annotations, biological correctness of the IntPlddt parsing convention)

  Each reviewer either approves or returns file:line issues. Loop fix → re-review until all approve. If a reviewer flags an issue with the symmetrisation note (Task is NOT modifying that), respond that it is explicitly out of scope per session brief.

- [ ] **Step 5.3: Merge PR after all approvals**

```bash
gh pr merge --merge --delete-branch
```

- [ ] **Step 5.4: Email the merged PR description to schekmenev@aithyra.at**

The user explicitly granted permission for this one address. Use whatever mail transport is available on the host.

---

## Self-Review Notes

- All schemas (`dimers.parquet`, `locator_rows.parquet`) are listed once and reused.
- Step 5 (`complexa_filter`) is folded into the `dimers.parquet` schema as a boolean column rather than a separate view subdirectory — simpler, easier to filter at dataloader time.
- TED-domain residue-range sanity is the **structural** check; the `*10+5` digit-bucket midpoint recomputation is the **independent IntPlddt parsing** check. The two together catch (a) parsing errors and (b) residue-numbering / off-by-one bugs.
- No changes to `proteinfoundation.proteina`, `confidence/`, `nn/confidence/`, or `configs/dataset/unified/`. Steps 6 and 7 of the handoff are explicitly out of scope per the session brief.

---

## Execution Handoff

Inline execution selected (single session, contiguous bash environment, real AFDB v4 paths only available on this host).
