"""Produce a 30%-sequence-identity cluster column for the Teddymer dimer view.

The Teddymer split was a positional head/tail slice of ``dimers.parquet``.
Because the parquet is parent-grouped, that slice is clean by
``parent_afdb_id`` / ``uniprot_id`` but leaks at the fold / sequence-family
level: ~99.9% of validation dimers share a CATH fold with the training set,
which inflates validation pLDDT / PAE correlation on an untrained confidence
head. In the spirit of AlphaFold-Multimer redundancy reduction (which clusters the
ingested chain sequences at ~40% identity and splits by cluster; Boltz / AF3
apply their own redundancy reduction), we hold out whole sequence-identity
clusters. The threshold here is
tightened to 30% — stricter than their 40% — because the head is leak-sensitive
and Teddymer crops are intra-parent sub-domains. NB this is NOT a verbatim
port of those pipelines: they cluster the chains the model ingests, whereas we
cluster the parent monomer (see below) and map down.

Each Teddymer dimer is an intra-monomer crop of a shared AFDB parent monomer
(both chains carry the same ``uniprot_id`` and parent CIF), so we cluster the
**parent monomer sequences** once per unique ``uniprot_id`` and map the
cluster representative back onto every dimer as a new ``seqclust30`` column.
This guarantees same-parent crops never split across train/val. It does NOT
remove cross-parent shared sub-domains (two divergent parents whose cropped
domains are homologous below the 30%/80%-coverage full-chain threshold) — a
fold-level Foldseek/TM holdout would be the strict follow-up.

The output is written to ``--out`` (default ``dimers.seqclust30.parquet``
next to the input) — the source blob is never mutated in place. The config
``cluster_column: seqclust30`` is already wired; ``_split_metadata`` warns and
falls back to the positional slice if the column is absent, and
``TeddymerDimerDataModule.setup`` logs ``CLUSTER SPLIT NOT ACTIVE`` at rank 0
in that case (so a forgotten rebuild is visible, not silent).

DEPLOY RUNBOOK (the integrity gate hashes the file literally named
``dimers.parquet`` at ``TEDDYMER_VIEW_ROOT`` against ``data.md5`` and compares
``view_config.yaml:version`` — see datasets/teddymer/integrity.py — so you
CANNOT just drop the new parquet beside the old one):

  1. Build a NEW view dir (do not overwrite the live blob), e.g.
     ``/netscratch/schekmenev/teddymer_v2_blob/``; copy ``data.blob`` and
     ``locator_rows.parquet`` in, and place this script's output there
     renamed to ``dimers.parquet``.
  2. Regenerate the md5 sidecar over the three files:
     ``cd <v2_dir> && md5sum data.blob locator_rows.parquet dimers.parquet > data.md5``
  3. Bump ``view_config.yaml:version`` (the gate's fast-path compares it) and
     update its ``md5s`` block to match ``data.md5``.
  4. Stage to BOTH partitions' ``/netscratch`` (gpu and h100 are separate
     physical stores); cross-partition copies go via labs NFS.
  5. Point ``TEDDYMER_VIEW_ROOT`` at the v2 dir and confirm the run header
     logs ``CLUSTER-AWARE on 'seqclust30' ACTIVE``.

MMseqs2 is an environment module on this cluster::

    module load mmseqs2/2026.06
    .venv/bin/python scripts/cluster_teddymer_parents.py \
        --dimers /netscratch/schekmenev/teddymer_v1_blob/dimers.parquet \
        --locator /netscratch/schekmenev/teddymer_v1_blob/locator_rows.parquet \
        --afdb-root /netscratch/schekmenev/teddymer_v1_blob \
        --out /netscratch/schekmenev/teddymer_v1_blob/dimers.seqclust30.parquet \
        --num-workers 32

``cluster_sequences`` honours ``MMSEQS_EXEC``; with the module loaded the
``mmseqs`` wrapper is on PATH so no override is needed.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from proteinfoundation.datasets.teddymer.dataset import _load_atom_array_from_bytes
from proteinfoundation.datasets.teddymer.io import read_cif_bytes_from_tar
from proteinfoundation.utils.cluster_utils import cluster_sequences, df_to_fasta

OUT_COLUMN = "seqclust30"
MIN_SEQ_ID = 0.3
COVERAGE = 0.8


def _seq_from_cif_bytes(cif_bytes: bytes, chain_id: str = "A") -> str:
    """One-letter parent-monomer sequence for ``chain_id`` from CIF bytes.

    Uses ``biotite.structure.to_sequence`` over the chain's residues; the
    AFDB parent CIF stores a single monomer chain.
    """
    import biotite.structure as bs

    atom_array = _load_atom_array_from_bytes(cif_bytes)
    chain = atom_array[atom_array.chain_id == chain_id]
    if len(chain) == 0:
        chain = atom_array
    sequences, _ = bs.to_sequence(chain)
    return "".join(str(s) for s in sequences)


def _tsv_to_member_map(tsv_path: Path | str) -> dict[str, str]:
    """Parse an MMseqs2 ``*_cluster.tsv`` (``rep<TAB>member`` per line) into a
    ``{member_id: representative_id}`` mapping.
    """
    member_map: dict[str, str] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            member_map[member] = rep
    return member_map


def _assign_clusters_to_dimers(
    dimers: pd.DataFrame,
    uniprot_to_cluster: dict[str, str],
    *,
    uniprot_column: str = "uniprot_id",
    out_column: str = OUT_COLUMN,
) -> pd.DataFrame:
    """Return ``dimers`` with ``out_column`` set to each row's parent cluster.

    Raises ``KeyError`` if any dimer's parent ``uniprot_id`` is missing from
    ``uniprot_to_cluster`` (every parent must have been clustered).
    """
    out = dimers.copy()
    missing = set(out[uniprot_column].unique()) - set(uniprot_to_cluster)
    if missing:
        raise KeyError(
            f"{len(missing)} parent uniprot_id(s) absent from cluster map, e.g. "
            f"{sorted(missing)[:5]}"
        )
    out[out_column] = out[uniprot_column].map(uniprot_to_cluster)
    return out


def _parent_locator_rows(locator: pd.DataFrame) -> pd.DataFrame:
    """One CIF-locator row per unique parent ``uniprot_id`` (chain A)."""
    chain_a = locator[locator["chain_id"] == "A"]
    return chain_a.drop_duplicates(subset="uniprot_id").reset_index(drop=True)


def _read_one_sequence(args: tuple[str, str, int, int, bool]) -> tuple[str, str | None]:
    uniprot_id, tar_path, offset, size, is_gz = args
    try:
        cif_bytes = read_cif_bytes_from_tar(tar_path, offset, size, is_gz)
        return uniprot_id, _seq_from_cif_bytes(cif_bytes, chain_id="A")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"sequence extraction failed for {uniprot_id}: {exc}")
        return uniprot_id, None


def _extract_parent_sequences(
    parent_rows: pd.DataFrame,
    afdb_root: str,
    num_workers: int,
) -> pd.DataFrame:
    tasks = [
        (
            str(r["uniprot_id"]),
            str(Path(afdb_root) / r["source_tar_relpath"]),
            int(r["cif_member_offset"]),
            int(r["cif_member_size"]),
            bool(r.get("cif_is_gz", True)),
        )
        for _, r in parent_rows.iterrows()
    ]
    records: list[dict] = []
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            for uid, seq in tqdm(
                pool.imap_unordered(_read_one_sequence, tasks),
                total=len(tasks),
                desc="parent sequences",
            ):
                if seq:
                    records.append({"id": uid, "sequence": seq})
    else:
        for t in tqdm(tasks, desc="parent sequences"):
            uid, seq = _read_one_sequence(t)
            if seq:
                records.append({"id": uid, "sequence": seq})
    return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimers", required=True, type=Path)
    ap.add_argument("--locator", required=True, type=Path)
    ap.add_argument("--afdb-root", required=True, type=str)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tmp", type=Path, default=Path("./tmp/teddymer_cluster"))
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--min-seq-id", type=float, default=MIN_SEQ_ID)
    ap.add_argument("--coverage", type=float, default=COVERAGE)
    ap.add_argument(
        "--filter",
        action="append",
        default=["interface_length > 10"],
        help="pandas query applied to dimers before clustering (repeatable)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cluster only the first N unique parents (smoke testing)",
    )
    args = ap.parse_args()

    out_path = args.out or args.dimers.with_name(f"dimers.{OUT_COLUMN}.parquet")
    args.tmp.mkdir(parents=True, exist_ok=True)

    dimers = pd.read_parquet(args.dimers)
    locator = pd.read_parquet(args.locator)
    logger.info(f"loaded {len(dimers)} dimers, {len(locator)} locator rows")

    for f in args.filter:
        dimers = dimers.query(f)
    dimers = dimers.reset_index(drop=True)
    logger.info(f"{len(dimers)} dimers after filters {args.filter}")

    keep_uniprots = set(dimers["uniprot_id"].unique())
    parent_rows = _parent_locator_rows(locator)
    parent_rows = parent_rows[parent_rows["uniprot_id"].isin(keep_uniprots)].reset_index(
        drop=True
    )
    if args.limit is not None:
        parent_rows = parent_rows.iloc[: args.limit].reset_index(drop=True)
        dimers = dimers[
            dimers["uniprot_id"].isin(set(parent_rows["uniprot_id"]))
        ].reset_index(drop=True)
    logger.info(f"extracting sequences for {len(parent_rows)} unique parents")

    seq_df = _extract_parent_sequences(parent_rows, args.afdb_root, args.num_workers)
    logger.info(f"extracted {len(seq_df)} parent sequences")

    # Fail fast: a parent whose CIF read fails or yields an empty sequence is
    # dropped here, but every dimer still references it and would raise a
    # KeyError in _assign_clusters_to_dimers AFTER the (expensive) MMseqs2 run.
    # Reconcile now with an actionable message instead.
    dropped = set(parent_rows["uniprot_id"]) - set(seq_df["id"])
    if dropped:
        raise RuntimeError(
            f"{len(dropped)} parent(s) produced no sequence (CIF read failed or "
            f"empty), e.g. {sorted(dropped)[:5]}. Their dimers cannot be assigned "
            "a cluster. Fix the source tars / locator offsets before clustering."
        )

    tmp_dir = args.tmp.resolve()
    fasta_path = tmp_dir / "parents.fasta"
    df_to_fasta(seq_df, str(fasta_path))

    # The mmseqs module is an Apptainer wrapper that runs in $PWD and writes
    # its `pdb_cluster_*` outputs into the current directory; cluster_sequences
    # then shutil.move's them next to the output. Run from tmp_dir with
    # relative names so the container sees the input and the move stays on one
    # filesystem (cross-device rename otherwise fails when tmp_dir != CWD).
    cluster_out_name = "parents_cluster.fasta"
    prev_cwd = Path.cwd()
    os.chdir(tmp_dir)
    try:
        cluster_sequences(
            fasta_input_filepath="parents.fasta",
            cluster_output_filepath=cluster_out_name,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            overwrite=True,
            mmseqs_exec=os.getenv("MMSEQS_EXEC"),
        )
    finally:
        os.chdir(prev_cwd)
    tsv_path = tmp_dir / "parents_cluster.tsv"
    uniprot_to_cluster = _tsv_to_member_map(tsv_path)
    n_clusters = len(set(uniprot_to_cluster.values()))
    logger.info(
        f"{n_clusters} clusters over {len(uniprot_to_cluster)} parents "
        f"at min_seq_id={args.min_seq_id}, coverage={args.coverage}"
    )

    out = _assign_clusters_to_dimers(dimers, uniprot_to_cluster)
    out.to_parquet(out_path)
    logger.info(
        f"wrote {len(out)} dimers with `{OUT_COLUMN}` ({n_clusters} clusters) -> {out_path}"
    )


if __name__ == "__main__":
    main()
