"""Train-time integrity gate for the Teddymer /netscratch blob.

Compares the md5s of the three live files at ``view_root`` —
``data.blob``, ``locator_rows.parquet``, ``dimers.parquet`` — against
the canonical snapshot at ``snapshot_path`` (typically the labs-adjacent
``data.md5`` written at repack time). Raises on mismatch BEFORE the
Lightning trainer constructs the head or loads the trunk checkpoint, so
the wasted compute on a drifted dataset is bounded by the ~2 min md5
walk.

Framework-pure: stdlib only. **Must not** import torch / lightning;
called from the training entry point inside `hydra.main`, but the gate
itself is framework-agnostic and reusable from any tool.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


_DEFAULT_FILES: tuple[str, ...] = (
    "data.blob",
    "locator_rows.parquet",
    "dimers.parquet",
)
_DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
_VIEW_CONFIG_NAME = "view_config.yaml"


class IntegrityError(RuntimeError):
    """Raised when the live md5s diverge from the snapshot."""


def _stream_md5(path: Path, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _read_view_config_version(view_config_path: Path) -> str:
    """Load ``view_config.yaml`` and return its ``version`` string.

    Raises ``IntegrityError`` if the file is missing or has no ``version``
    key. Used by the cheap version-string fast-path in
    ``verify_teddymer_blob_integrity`` (plan section 4.3 + R12).
    """
    view_config_path = Path(view_config_path)
    if not view_config_path.exists():
        raise IntegrityError(
            f"Teddymer view_config.yaml missing: {view_config_path}"
        )
    cfg = yaml.safe_load(view_config_path.read_text())
    if not isinstance(cfg, dict) or "version" not in cfg:
        raise IntegrityError(
            f"Teddymer view_config.yaml has no 'version' key: {view_config_path}"
        )
    return str(cfg["version"])


def _parse_md5_sidecar(path: Path) -> dict[str, str]:
    """Parse a GNU md5sum-format file: ``<32-hex>  <basename>\\n`` per line."""
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        digest, _, name = ln.partition("  ")
        if len(digest) != 32 or not name:
            raise IntegrityError(
                f"malformed md5 sidecar line in {path}: {ln!r}"
            )
        out[name] = digest
    return out


def verify_teddymer_blob_integrity(
    view_root: Path,
    snapshot_path: Path,
    *,
    files: Sequence[str] = _DEFAULT_FILES,
) -> None:
    """Verify the md5s of the live files at ``view_root`` match the
    snapshot at ``snapshot_path``. Returns ``None`` on success; raises
    ``IntegrityError`` (or ``FileNotFoundError`` for missing inputs) on
    failure.

    Streaming md5; no full-file slurp into RAM. ~2 min on a 136 GB blob
    at ~1 GB/s NVMe sequential.
    """
    view_root = Path(view_root).resolve()
    snapshot_path = Path(snapshot_path).resolve()

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Teddymer integrity snapshot missing: {snapshot_path}")

    view_version = _read_view_config_version(view_root / _VIEW_CONFIG_NAME)
    snapshot_version = _read_view_config_version(
        snapshot_path.parent / _VIEW_CONFIG_NAME
    )
    if view_version != snapshot_version:
        raise IntegrityError(
            f"Teddymer view_config.yaml version mismatch: "
            f"view {view_version!r} (at {view_root / _VIEW_CONFIG_NAME}) "
            f"!= snapshot {snapshot_version!r} "
            f"(at {snapshot_path.parent / _VIEW_CONFIG_NAME}). "
            f"Re-run scripts/build_teddymer_blob.sbatch to refresh /netscratch."
        )

    expected = _parse_md5_sidecar(snapshot_path)

    t0 = time.perf_counter()
    for name in files:
        live_path = view_root / name
        if not live_path.exists():
            raise FileNotFoundError(
                f"Teddymer view missing file {name!r} under {view_root} "
                f"(resolved: {live_path})"
            )
        if name not in expected:
            raise IntegrityError(
                f"Snapshot {snapshot_path} has no entry for '{name}'; "
                f"snapshot entries: {sorted(expected.keys())}"
            )

        live_md5 = _stream_md5(live_path)
        if live_md5 != expected[name]:
            raise IntegrityError(
                f"Teddymer blob integrity check failed: md5 mismatch on '{name}'.\n"
                f"Expected (from snapshot at {snapshot_path}): {expected[name]}.\n"
                f"Got (live at {view_root}/{name}): {live_md5}.\n"
                f"The Teddymer dataset on labs has changed, or the /netscratch blob\n"
                f"is stale or partially overwritten. Re-run\n"
                f"scripts/build_teddymer_blob.sbatch to repack."
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Teddymer blob integrity OK (version=%s, verified %d files in %.1fs)",
        view_version,
        len(files),
        elapsed,
    )
