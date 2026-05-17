"""Parse the Teddymer MMseqs2 `_h` header DB and the non-singleton metadata TSV
into a single ``dimers.parquet``-shaped DataFrame.

The Teddymer release ships dimer headers as a concatenation of newline-terminated
records in ``teddymer_repdb_h``, indexed by ``teddymer_repdb_h.index``
(TSV ``internal_id\\toffset\\tlength``). Each header carries the parent AFDB id,
TED domain index, CATH classification and the (possibly discontinuous) residue
chopping of one chain of a dimer. The non-singleton metadata TSV
(``nonsingletonrep_metadata.tsv``) carries the cluster-representative annotations
keyed by ``DimerIndex`` (interface length, mean intra-interface pAE / pLDDT, and
per-chain ``IntPlddt`` strings).

This module joins the two and emits the schema documented in
``docs/superpowers/plans/2026-05-17-teddymer-preprocessing.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    """Parse one MMseqs2 `_h` header line."""
    line = line.rstrip("\n").rstrip("\x00")
    m = _HEADER_RE.match(line)
    if m is None:
        raise ValueError(f"Malformed Teddymer _h header: {line!r}")
    intervals: list[tuple[int, int]] = []
    for span in m["chopping"].split("_"):
        lo_str, hi_str = span.split("-")
        intervals.append((int(lo_str), int(hi_str)))
    return HeaderRecord(
        dimer_index=int(m["dimer_index"]),
        uniprot_id=m["uniprot_id"],
        parent_afdb_id=f"AF-{m['uniprot_id']}-F1",
        ted_index=int(m["ted_index"]),
        cath_id=m["cath_id"],
        residue_intervals=intervals,
    )


def parse_int_plddt(s: str) -> tuple[list[int], list[int]]:
    """Decode the per-chain digit-bucket pLDDT string in metadata.

    The Teddymer release encodes ``IntPlddt`` as two colon-separated digit
    strings, one per chain. Each digit ``d`` is the pLDDT bucket
    ``[10*d, 10*d+10)`` for ``d in 0..8`` and ``[90, 100]`` for ``d == 9``,
    with the lower bound used to compute the metadata's ``AvgIntPlddt``
    (verified empirically against the first 8 rows of
    ``nonsingletonrep_metadata.tsv``).
    """
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"IntPlddt must contain exactly two colon-separated chains, got {len(parts)}"
        )
    a = [int(c) for c in parts[0]]
    b = [int(c) for c in parts[1]]
    return a, b


def _read_h_records(h_path: Path, idx_path: Path) -> Iterable[HeaderRecord]:
    """Stream every record in ``h_path`` using the offsets in ``idx_path``."""
    idx = pd.read_csv(
        idx_path,
        sep="\t",
        header=None,
        names=["internal_id", "offset", "length"],
        dtype={"internal_id": "int64", "offset": "int64", "length": "int64"},
    )
    internal_ids = idx["internal_id"].to_numpy()
    offsets = idx["offset"].to_numpy()
    lengths = idx["length"].to_numpy()
    with open(h_path, "rb") as f:
        for internal_id, off, ln in zip(internal_ids, offsets, lengths):
            f.seek(int(off))
            chunk = f.read(int(ln)).decode("utf-8", errors="strict")
            try:
                yield parse_header_line(chunk)
            except ValueError as exc:
                raise ValueError(
                    f"_h record internal_id={int(internal_id)} offset={int(off)}: {exc}"
                ) from exc


_DIMERS_ARROW_SCHEMA = pa.schema(
    [
        ("dimer_id", pa.string()),
        ("dimer_index", pa.int64()),
        ("uniprot_id", pa.string()),
        ("parent_afdb_id", pa.string()),
        ("ted_index_A", pa.int32()),
        ("ted_index_B", pa.int32()),
        ("cath_id_A", pa.string()),
        ("cath_id_B", pa.string()),
        (
            "residue_intervals_A",
            pa.list_(pa.struct([("lo", pa.int32()), ("hi", pa.int32())])),
        ),
        (
            "residue_intervals_B",
            pa.list_(pa.struct([("lo", pa.int32()), ("hi", pa.int32())])),
        ),
        ("member_count", pa.int32()),
        ("interface_length", pa.int32()),
        ("avg_int_pae", pa.float32()),
        ("avg_int_plddt", pa.float32()),
        ("int_plddt_chain_A", pa.list_(pa.int8())),
        ("int_plddt_chain_B", pa.list_(pa.int8())),
    ]
)


def build_dimers_table(h_path: Path, idx_path: Path, meta_path: Path) -> pd.DataFrame:
    """Build the ``dimers.parquet`` DataFrame from Teddymer raw inputs."""
    h_path = Path(h_path)
    idx_path = Path(idx_path)
    meta_path = Path(meta_path)

    by_dimer: dict[int, list[HeaderRecord]] = {}
    for rec in _read_h_records(h_path, idx_path):
        by_dimer.setdefault(rec.dimer_index, []).append(rec)

    meta = pd.read_csv(meta_path, sep="\t").set_index("DimerIndex", drop=False)

    rows: list[dict] = []
    for dimer_index, chains in by_dimer.items():
        if dimer_index not in meta.index:
            # Dimer is a singleton (or otherwise not in non-singleton reps); skip.
            continue
        if len(chains) != 2:
            raise ValueError(
                f"DimerIndex {dimer_index}: expected 2 _h rows, got {len(chains)}"
            )
        chains_sorted = sorted(chains, key=lambda r: r.ted_index)
        a, b = chains_sorted
        if a.parent_afdb_id != b.parent_afdb_id:
            raise ValueError(
                f"DimerIndex {dimer_index}: chains not intra-monomer "
                f"(A={a.parent_afdb_id}, B={b.parent_afdb_id})"
            )
        m = meta.loc[dimer_index]
        expected_pair = f"TED{a.ted_index:02d}:TED{b.ted_index:02d}"
        if m["DomainPair"] != expected_pair:
            raise ValueError(
                f"DimerIndex {dimer_index}: DomainPair in metadata is "
                f"{m['DomainPair']!r} but _h says {expected_pair!r}"
            )
        chain_a_plddt, chain_b_plddt = parse_int_plddt(m["IntPlddt"])
        total_digits = len(chain_a_plddt) + len(chain_b_plddt)
        if total_digits != int(m["InterfaceLength"]):
            raise ValueError(
                f"DimerIndex {dimer_index}: IntPlddt total length {total_digits} "
                f"!= InterfaceLength {m['InterfaceLength']}"
            )

        rows.append(
            {
                "dimer_id": f"{dimer_index}DI_{a.parent_afdb_id}-model_v4",
                "dimer_index": dimer_index,
                "uniprot_id": a.uniprot_id,
                "parent_afdb_id": a.parent_afdb_id,
                "ted_index_A": a.ted_index,
                "ted_index_B": b.ted_index,
                "cath_id_A": a.cath_id,
                "cath_id_B": b.cath_id,
                "residue_intervals_A": [
                    {"lo": lo, "hi": hi} for (lo, hi) in a.residue_intervals
                ],
                "residue_intervals_B": [
                    {"lo": lo, "hi": hi} for (lo, hi) in b.residue_intervals
                ],
                "member_count": int(m["MemberCount"]),
                "interface_length": int(m["InterfaceLength"]),
                "avg_int_pae": float(m["AvgIntPAE"]),
                "avg_int_plddt": float(m["AvgIntPlddt"]),
                "int_plddt_chain_A": chain_a_plddt,
                "int_plddt_chain_B": chain_b_plddt,
            }
        )

    return _rows_to_dimers_df(rows)


def _rows_to_dimers_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of row dicts to a DataFrame whose backing arrow table
    matches ``_DIMERS_ARROW_SCHEMA`` (preserves struct/list types through round-trip).
    """
    if not rows:
        empty_table = _DIMERS_ARROW_SCHEMA.empty_table()
        return empty_table.to_pandas()
    table = pa.Table.from_pylist(rows, schema=_DIMERS_ARROW_SCHEMA)
    return table.to_pandas()


def write_dimers_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dimers DataFrame to parquet preserving struct types."""
    table = pa.Table.from_pandas(df, schema=_dimers_schema_with_filter(df), preserve_index=False)
    pq.write_table(table, path)


def _dimers_schema_with_filter(df: pd.DataFrame) -> pa.Schema:
    fields = list(_DIMERS_ARROW_SCHEMA)
    if "complexa_filter" in df.columns:
        fields.append(pa.field("complexa_filter", pa.bool_()))
    return pa.schema(fields)


def add_complexa_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean ``complexa_filter`` column matching the Complexa training-set
    selection (§3.1 of the Complexa paper): ``interface_length > 10`` *and*
    ``avg_int_plddt > 70`` *and* ``avg_int_pae < 10``.
    """
    out = df.copy()
    out["complexa_filter"] = (
        (out["interface_length"] > 10)
        & (out["avg_int_plddt"] > 70.0)
        & (out["avg_int_pae"] < 10.0)
    )
    return out
