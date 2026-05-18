"""Repack the Teddymer dimer view's CIF / PAE / confidence byte ranges
into a single contiguous ``data.blob`` plus a rewritten
``locator_rows.parquet`` that points at it.

The dataloader contract (``src/proteinfoundation/datasets/teddymer/io.py``)
is ``open + seek(offset) + read(size) + optional gzip.decompress`` — it
works equally on a tar and on a flat blob, which is the whole reason this
repack exists. After repack, ``source_tar_relpath == "data.blob"`` for
every locator row and ``afdb_proteomes_root`` is pointed at the staged
directory.

Public API:
    plan_layout(locator_df) -> plan_df
    repack(plan_df, labs_root, out_blob) -> {parent_afdb_id: (dst_cif, dst_pae, dst_conf)}
    rewrite_locator(orig_locator, dst_offsets, blob_relpath="data.blob") -> new_df
    compute_md5(path, chunk_bytes=8 MiB) -> hex
    write_md5_sidecar(paths, sidecar_path, relative_to) -> None
    write_view_config(out_dir, version=, blob_path=, locator_path=, dimers_path=,
                      labs_root=, md5s=) -> None
    main() -> None  # CLI entry

Run via SLURM (``scripts/build_teddymer_blob.sbatch``) — never on the
login node. ~30 min wall-clock to repack ~136 GB at ~250 MB/s
single-stream cold from labs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import logging
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


logger = logging.getLogger(__name__)

DEFAULT_MD5_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_BLOB_NAME = "data.blob"
DEFAULT_MD5_NAME = "data.md5"
DEFAULT_VIEW_CONFIG_NAME = "view_config.yaml"
DEFAULT_LOCATOR_NAME = "locator_rows.parquet"
DEFAULT_DIMERS_NAME = "dimers.parquet"


# ---------------------------------------------------------------------------
# plan_layout
# ---------------------------------------------------------------------------


def plan_layout(locator_df: pd.DataFrame) -> pd.DataFrame:
    """Build the repack plan: one row per unique parent monomer, sorted by
    (source_tar_relpath, src_cif_off), with monotone-non-decreasing dst
    offsets laid out as ``cif | pae | conf`` per parent.

    The plan is the single source of truth for both ``repack`` (which
    consumes ``src_*_off`` / ``dst_*_off``) and ``rewrite_locator`` (which
    builds the parent->dst-offsets dict from the same rows).
    """
    if locator_df.empty:
        raise ValueError("plan_layout: empty locator_df")

    src_cols = [
        "afdb_id",
        "source_tar_relpath",
        "cif_member_offset",
        "cif_member_size",
        "pae_member_offset",
        "pae_member_size",
        "conf_member_offset",
        "conf_member_size",
        "cif_is_gz",
        "pae_is_gz",
        "conf_is_gz",
    ]
    missing = [c for c in src_cols if c not in locator_df.columns]
    if missing:
        raise ValueError(f"plan_layout: locator_df missing columns: {missing}")

    dedup = (
        locator_df[src_cols]
        .drop_duplicates(subset=["afdb_id"], keep="first")
        .rename(
            columns={
                "afdb_id": "parent_afdb_id",
                "cif_member_offset": "src_cif_off",
                "cif_member_size": "src_cif_size",
                "pae_member_offset": "src_pae_off",
                "pae_member_size": "src_pae_size",
                "conf_member_offset": "src_conf_off",
                "conf_member_size": "src_conf_size",
            }
        )
        .sort_values(["source_tar_relpath", "src_cif_off"], kind="mergesort")
        .reset_index(drop=True)
    )

    cif_sizes = dedup["src_cif_size"].astype("int64").to_numpy()
    pae_sizes = dedup["src_pae_size"].astype("int64").to_numpy()
    conf_sizes = dedup["src_conf_size"].astype("int64").to_numpy()
    per_parent = cif_sizes + pae_sizes + conf_sizes

    dst_cif_off = per_parent.cumsum() - per_parent
    dst_pae_off = dst_cif_off + cif_sizes
    dst_conf_off = dst_pae_off + pae_sizes

    dedup["dst_cif_off"] = dst_cif_off
    dedup["dst_pae_off"] = dst_pae_off
    dedup["dst_conf_off"] = dst_conf_off

    cols = [
        "parent_afdb_id",
        "source_tar_relpath",
        "src_cif_off",
        "src_cif_size",
        "src_pae_off",
        "src_pae_size",
        "src_conf_off",
        "src_conf_size",
        "cif_is_gz",
        "pae_is_gz",
        "conf_is_gz",
        "dst_cif_off",
        "dst_pae_off",
        "dst_conf_off",
    ]
    return dedup[cols]


# ---------------------------------------------------------------------------
# repack
# ---------------------------------------------------------------------------


_BLOB_READ_CHUNK = 8 * 1024 * 1024


def _read_member_bytes(src_fh, off: int, size: int) -> bytes:
    """Read ``size`` bytes from ``src_fh`` starting at ``off``. Uses an
    explicit loop because a single ``read(N)`` on NFS can return fewer
    bytes than requested for very large N.
    """
    src_fh.seek(int(off))
    remaining = int(size)
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = src_fh.read(min(remaining, _BLOB_READ_CHUNK))
        if not chunk:
            raise OSError(
                f"short read at offset {off}: {size - remaining}/{size} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def repack(
    plan_df: pd.DataFrame,
    labs_root: Path,
    out_blob: Path,
    progress_every: int = 1000,
) -> dict[str, tuple[int, int, int]]:
    """Stream the planned member byte ranges into ``out_blob`` and return
    the ``parent_afdb_id -> (dst_cif_off, dst_pae_off, dst_conf_off)`` map.

    Atomicity: writes to ``out_blob.with_name(out_blob.name + ".tmp")``,
    fsyncs, then renames into place (atomic on local FS).
    """
    labs_root = Path(labs_root)
    out_blob = Path(out_blob)
    out_blob.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_blob.with_name(out_blob.name + ".tmp")

    dst_offsets: dict[str, tuple[int, int, int]] = {}
    current_tar = None
    src_fh = None
    t0 = time.perf_counter()

    try:
        with open(tmp_path, "wb") as out_fh:
            for i, row in enumerate(plan_df.itertuples(index=False), start=1):
                if row.source_tar_relpath != current_tar:
                    if src_fh is not None:
                        src_fh.close()
                    src_path = labs_root / row.source_tar_relpath
                    src_fh = open(src_path, "rb")
                    current_tar = row.source_tar_relpath

                cif_bytes = _read_member_bytes(src_fh, row.src_cif_off, row.src_cif_size)
                pae_bytes = _read_member_bytes(src_fh, row.src_pae_off, row.src_pae_size)
                conf_bytes = _read_member_bytes(
                    src_fh, row.src_conf_off, row.src_conf_size
                )

                dst_cif = out_fh.tell()
                out_fh.write(cif_bytes)
                dst_pae = out_fh.tell()
                out_fh.write(pae_bytes)
                dst_conf = out_fh.tell()
                out_fh.write(conf_bytes)

                if dst_cif != int(row.dst_cif_off):
                    raise AssertionError(
                        f"dst_cif drift for {row.parent_afdb_id}: "
                        f"planned {row.dst_cif_off}, actual {dst_cif}"
                    )
                dst_offsets[row.parent_afdb_id] = (dst_cif, dst_pae, dst_conf)

                if progress_every > 0 and (i % progress_every == 0):
                    elapsed = time.perf_counter() - t0
                    mb = out_fh.tell() / (1024 * 1024)
                    logger.info(
                        "repack: %d/%d parents, %.1f MB written, %.1f MB/s",
                        i,
                        len(plan_df),
                        mb,
                        mb / max(elapsed, 1e-9),
                    )

            out_fh.flush()
            os.fsync(out_fh.fileno())
    finally:
        if src_fh is not None:
            src_fh.close()

    os.replace(tmp_path, out_blob)
    return dst_offsets


# ---------------------------------------------------------------------------
# rewrite_locator
# ---------------------------------------------------------------------------


def rewrite_locator(
    orig_locator: pd.DataFrame,
    dst_offsets: dict[str, tuple[int, int, int]],
    blob_relpath: str = DEFAULT_BLOB_NAME,
) -> pd.DataFrame:
    """Rewrite ``orig_locator`` to point at ``blob_relpath`` for every row.

    Preserves the schema exactly. Only ``source_tar_relpath``,
    ``source_tar_basename``, and the three ``*_member_offset`` columns
    change. A-row and B-row of an intra-monomer dimer (same parent on both
    chains) point at identical dst offsets by construction (they're keyed
    off ``afdb_id``, which is shared).
    """
    afdb_ids = orig_locator["afdb_id"].astype(str).to_numpy()
    missing = {a for a in afdb_ids if a not in dst_offsets}
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"rewrite_locator: {len(missing)} afdb_id(s) absent from dst_offsets. "
            f"Examples: {sample}"
        )

    cif_off = [dst_offsets[a][0] for a in afdb_ids]
    pae_off = [dst_offsets[a][1] for a in afdb_ids]
    conf_off = [dst_offsets[a][2] for a in afdb_ids]

    new_df = orig_locator.copy()
    new_df["source_tar_relpath"] = blob_relpath
    new_df["source_tar_basename"] = Path(blob_relpath).name
    new_df["cif_member_offset"] = cif_off
    new_df["pae_member_offset"] = pae_off
    new_df["conf_member_offset"] = conf_off
    return new_df[list(orig_locator.columns)]


# ---------------------------------------------------------------------------
# md5 + sidecar + view_config
# ---------------------------------------------------------------------------


def compute_md5(path: Path, chunk_bytes: int = DEFAULT_MD5_CHUNK_BYTES) -> str:
    """Stream-md5 the file at ``path``. Returns the 32-char lowercase
    hexdigest. Reads in ``chunk_bytes``-sized chunks (default 8 MiB).
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be > 0, got {chunk_bytes}")
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def write_md5_sidecar(
    paths: Sequence[Path],
    sidecar_path: Path,
    relative_to: Path,
) -> None:
    """Write a GNU ``md5sum``-style sidecar: ``<32-hex>  <basename>\\n``,
    one line per path, sorted by basename. The "basename" is the path
    relative to ``relative_to`` (typically the staging out_dir) so the
    sidecar is portable across mounts.
    """
    relative_to = Path(relative_to)
    rows: list[tuple[str, str]] = []
    for p in paths:
        rel = str(Path(p).resolve().relative_to(relative_to.resolve()))
        digest = compute_md5(Path(p))
        rows.append((rel, digest))
    rows.sort(key=lambda kv: kv[0])
    Path(sidecar_path).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in rows)
    )


def parse_md5_sidecar(sidecar_path: Path) -> dict[str, str]:
    """Inverse of `write_md5_sidecar`: returns ``{basename: hexdigest}``."""
    out: dict[str, str] = {}
    for ln in Path(sidecar_path).read_text().splitlines():
        if not ln.strip():
            continue
        digest = ln[:32]
        name = ln[34:]
        out[name] = digest
    return out


def write_view_config(
    out_dir: Path,
    *,
    version: str,
    blob_path: Path,
    locator_path: Path,
    dimers_path: Path,
    labs_root: Path,
    md5s: dict[str, str],
) -> None:
    """Emit ``out_dir / view_config.yaml`` carrying the version string,
    per-file md5s, the labs source root, and a build timestamp.

    The version string is the cheap-string-compare gate the integrity
    check uses before paying for full md5 (plan §4.3).
    """
    payload = {
        "version": version,
        "blob_path": str(Path(blob_path).resolve()),
        "locator_path": str(Path(locator_path).resolve()),
        "dimers_path": str(Path(dimers_path).resolve()),
        "labs_root": str(Path(labs_root).resolve()),
        "build_timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "md5s": dict(md5s),
    }
    (Path(out_dir) / DEFAULT_VIEW_CONFIG_NAME).write_text(yaml.safe_dump(payload, sort_keys=True))


def _build_version_string(blob_md5: str) -> str:
    today = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")
    return f"teddymer_v1_blob_{today}_{blob_md5[:8]}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _acquire_lock(out_dir: Path):
    """Acquire an exclusive flock on ``out_dir/.build.lock``. Fails fast
    if a concurrent build is running (R11).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".build.lock"
    lock_fh = open(lock_path, "wb")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.close()
        raise RuntimeError(
            f"build_teddymer_blob: another build holds {lock_path}; refusing to proceed"
        ) from None
    return lock_fh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack the Teddymer view into a single contiguous data.blob."
    )
    parser.add_argument("--locator", type=Path, required=True,
                        help="Path to the source locator_rows.parquet")
    parser.add_argument("--labs-root", type=Path, required=True,
                        help="Root under which source_tar_relpath resolves")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to stage data.blob + locator + dimers + sidecar")
    parser.add_argument("--dimers", type=Path, default=None,
                        help="Path to the source dimers.parquet (defaults to "
                             "{locator parent}/dimers.parquet)")
    parser.add_argument("--snapshot-path", type=Path, required=True,
                        help="Path on labs-adjacent storage at which to write "
                             "the canonical md5 snapshot")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Log progress every N parents during repack")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    locator_src = Path(args.locator)
    labs_root = Path(args.labs_root)
    out_dir = Path(args.out_dir)
    snapshot_path = Path(args.snapshot_path)
    dimers_src = args.dimers if args.dimers is not None else locator_src.parent / DEFAULT_DIMERS_NAME

    if not locator_src.exists():
        raise FileNotFoundError(f"locator parquet not found: {locator_src}")
    if not dimers_src.exists():
        raise FileNotFoundError(f"dimers parquet not found: {dimers_src}")
    if not labs_root.exists():
        raise FileNotFoundError(f"labs root not found: {labs_root}")

    lock_fh = _acquire_lock(out_dir)
    try:
        logger.info("build_blob: loading locator from %s", locator_src)
        orig_locator = pq.read_table(locator_src).to_pandas()
        logger.info("build_blob: %d locator rows", len(orig_locator))

        plan = plan_layout(orig_locator)
        logger.info("build_blob: planned %d unique parents", len(plan))

        blob_path = out_dir / DEFAULT_BLOB_NAME
        logger.info("build_blob: repacking into %s", blob_path)
        t0 = time.perf_counter()
        dst_offsets = repack(plan, labs_root=labs_root, out_blob=blob_path,
                             progress_every=args.progress_every)
        elapsed = time.perf_counter() - t0
        size_gb = blob_path.stat().st_size / (1024 ** 3)
        logger.info(
            "build_blob: repack done in %.1fs, %.2f GB, %.1f MB/s",
            elapsed, size_gb, size_gb * 1024 / max(elapsed, 1e-9),
        )

        new_loc = rewrite_locator(orig_locator, dst_offsets)
        locator_dst = out_dir / DEFAULT_LOCATOR_NAME
        pq.write_table(pa.Table.from_pandas(new_loc, preserve_index=False), locator_dst)
        logger.info("build_blob: wrote rewritten locator to %s", locator_dst)

        dimers_dst = out_dir / DEFAULT_DIMERS_NAME
        shutil.copyfile(dimers_src, dimers_dst)
        logger.info("build_blob: copied dimers to %s", dimers_dst)

        md5s = {
            DEFAULT_BLOB_NAME: compute_md5(blob_path),
            DEFAULT_LOCATOR_NAME: compute_md5(locator_dst),
            DEFAULT_DIMERS_NAME: compute_md5(dimers_dst),
        }
        sidecar = out_dir / DEFAULT_MD5_NAME
        write_md5_sidecar(
            [blob_path, locator_dst, dimers_dst],
            sidecar_path=sidecar,
            relative_to=out_dir,
        )
        logger.info("build_blob: wrote md5 sidecar to %s", sidecar)

        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sidecar, snapshot_path)
        logger.info("build_blob: wrote labs-side snapshot to %s", snapshot_path)

        version = _build_version_string(md5s[DEFAULT_BLOB_NAME])
        write_view_config(
            out_dir,
            version=version,
            blob_path=blob_path,
            locator_path=locator_dst,
            dimers_path=dimers_dst,
            labs_root=labs_root,
            md5s=md5s,
        )
        snapshot_view = snapshot_path.parent / DEFAULT_VIEW_CONFIG_NAME
        if snapshot_view.resolve() != (out_dir / DEFAULT_VIEW_CONFIG_NAME).resolve():
            write_view_config(
                snapshot_path.parent,
                version=version,
                blob_path=blob_path,
                locator_path=locator_dst,
                dimers_path=dimers_dst,
                labs_root=labs_root,
                md5s=md5s,
            )
        logger.info("build_blob: version=%s", version)
        logger.info("build_blob: done.")
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        lock_fh.close()


if __name__ == "__main__":
    main()
