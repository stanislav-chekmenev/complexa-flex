"""Per-dimer sanity-check helpers used by both the build CLI and the
50-dimer integration pytest.

The check has two prongs:

1. **Parsing self-consistency.** Recompute ``AvgIntPlddt`` from the per-chain
   digit-bucket strings (``IntPlddt`` column) and compare to the metadata value.
   This catches errors in ``parse_int_plddt`` and in our schema mapping. It
   does *not* require the AFDB mirror.

2. **AFDB round-trip.** Read the parent monomer's confidence JSON straight from
   the AFDB tar at the byte offset recorded in ``locator_rows.parquet``, then
   verify the residue numbering implied by each chain's ``residue_intervals_*``
   lies inside the AFDB sequence length. This catches tar-offset bugs and
   off-by-one residue-numbering bugs.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def recompute_avg_int_plddt(chain_a: Sequence[int], chain_b: Sequence[int]) -> float:
    """Recompute ``AvgIntPlddt`` from the per-chain digit-bucket strings.

    Each digit ``d`` in [0, 9] encodes the pLDDT bucket [10*d, 10*d+10); the
    Teddymer release uses the lower bound (``10*d``) when computing the mean
    (verified against the first eight rows of ``nonsingletonrep_metadata.tsv``).
    """
    digits = list(chain_a) + list(chain_b)
    if not digits:
        return 0.0
    return sum(10.0 * d for d in digits) / len(digits)


def read_confidence_from_tar(tar_path: Path, offset: int, size: int) -> list[float]:
    """Pull a gzipped AFDB ``*-confidence_v4.json.gz`` member from ``tar_path``
    at byte ``offset`` (data offset, not header offset) and length ``size``,
    decode the JSON, and return the per-residue confidence-score list.
    """
    with open(tar_path, "rb") as f:
        f.seek(int(offset))
        blob = f.read(int(size))
    obj = json.loads(gzip.decompress(blob))
    return [float(x) for x in obj["confidenceScore"]]


def _chain_a_locator_row(locator: pd.DataFrame, dimer_index: int) -> dict:
    rows = locator[(locator["dimer_index"] == dimer_index) & (locator["chain_id"] == "A")]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one chain-A locator row for dimer {dimer_index}, "
            f"got {len(rows)}"
        )
    return rows.iloc[0].to_dict()


def sanity_check_dimers(
    dimers: pd.DataFrame,
    locator: pd.DataFrame,
    afdb_proteomes: Path,
    digit_recomp_tol: float = 0.5,
) -> list[tuple[int, str]]:
    """Run the sanity check on every row of ``dimers``.

    Returns a list of (dimer_index, reason) tuples for the failures. Empty list
    means all dimers passed.
    """
    failures: list[tuple[int, str]] = []
    for _, row in dimers.iterrows():
        dimer_index = int(row["dimer_index"])
        chain_a = list(row["int_plddt_chain_A"])
        chain_b = list(row["int_plddt_chain_B"])
        recomp = recompute_avg_int_plddt(chain_a, chain_b)
        if abs(recomp - float(row["avg_int_plddt"])) > digit_recomp_tol:
            failures.append(
                (dimer_index,
                 f"IntPlddt recomputation {recomp:.4f} vs metadata "
                 f"{float(row['avg_int_plddt']):.4f}")
            )
            continue

        loc = _chain_a_locator_row(locator, dimer_index)
        tar_path = Path(afdb_proteomes) / loc["source_tar_relpath"]
        if not tar_path.exists():
            failures.append((dimer_index, f"AFDB tar missing: {tar_path}"))
            continue
        try:
            scores = read_confidence_from_tar(
                tar_path, loc["conf_member_offset"], loc["conf_member_size"]
            )
        except Exception as exc:
            failures.append((dimer_index, f"AFDB confidence extract raised: {exc}"))
            continue
        seq_len = len(scores)
        ok = True
        bad_reason = ""
        for chain_label in ("A", "B"):
            spans = list(row[f"residue_intervals_{chain_label}"])
            for span in spans:
                lo, hi = int(span["lo"]), int(span["hi"])
                if not (1 <= lo <= seq_len and 1 <= hi <= seq_len):
                    ok = False
                    bad_reason = (
                        f"chain {chain_label} residue span [{lo}, {hi}] outside "
                        f"AFDB sequence length {seq_len}"
                    )
                    break
            if not ok:
                break
        if not ok:
            failures.append((dimer_index, bad_reason))
    return failures
