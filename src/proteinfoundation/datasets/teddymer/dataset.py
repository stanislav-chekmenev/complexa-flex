"""Lightning DataModule and Dataset for the Teddymer dimer view.

The Teddymer view stores ``dimers.parquet`` and ``locator_rows.parquet``
side-by-side under a single view directory; each dimer's two chains are
synthetic crops of a shared AFDB parent monomer (intra-monomer dimers).
``__getitem__`` reads the CIF, the confidence JSON and the PAE JSON
directly from the parent AFDB tar at the byte offsets recorded in
``locator_rows.parquet``, slices atoms by the per-chain residue-interval
spans in ``dimers.parquet``, and routes the result through the standard
``atomarray_to_atom37`` + atom37-transforms pipeline.
"""

from __future__ import annotations

import io
import random
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.utils.data import DataLoader, Dataset

import proteinfoundation.patches.atomworks_patches  # noqa: F401

from proteinfoundation.datasets.structure_data import (
    _ensure_atomworks_annotations,
    atomarray_to_atom37,
    make_collate_fn,
)
from proteinfoundation.datasets.teddymer.io import read_cif_bytes_from_tar
from proteinfoundation.datasets.transforms import Data


def _teddymer_worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else 0
    np.random.seed(base % (2**32))
    random.seed(base)


def _split_metadata(
    metadata: pd.DataFrame,
    train_split: float,
    cluster_column: str | None,
    cluster_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train / val, holding out whole clusters when requested.

    With ``cluster_column`` set and present, shuffle the unique cluster ids
    under ``cluster_seed`` and greedily fill train up to ``train_split`` of the
    rows, assigning the remaining clusters to val — so no cluster spans both
    splits. Falls back to a positional row-order slice when ``cluster_column``
    is ``None`` or absent. Guard parity with
    ``StructureDataModule._cluster_aware_split``: warn (do not raise) on a
    missing column and on an empty val split.
    """
    if cluster_column is not None:
        if cluster_column not in metadata.columns:
            logger.warning(
                f"cluster_column={cluster_column!r} requested but absent from "
                f"metadata columns {list(metadata.columns)}; falling back to "
                "row-order split."
            )
        else:
            rng = np.random.default_rng(cluster_seed)
            cluster_ids = metadata[cluster_column].to_numpy()
            unique = np.unique(cluster_ids)
            shuffled = unique.copy()
            rng.shuffle(shuffled)
            target_train = int(len(metadata) * train_split)
            cluster_to_rows = {c: np.where(cluster_ids == c)[0] for c in shuffled}
            train_rows: list[int] = []
            train_clusters: set = set()
            for c in shuffled:
                if len(train_rows) >= target_train:
                    break
                train_rows.extend(int(i) for i in cluster_to_rows[c])
                train_clusters.add(c)
            val_rows = [
                int(i)
                for c in shuffled
                if c not in train_clusters
                for i in cluster_to_rows[c]
            ]
            if len(val_rows) == 0:
                logger.warning(
                    "_split_metadata: empty val split with "
                    f"cluster_column={cluster_column!r}, train_split={train_split}, "
                    f"n_clusters={len(unique)}, n_rows={len(metadata)}; all clusters "
                    "fell on the train side."
                )
            return (
                metadata.iloc[train_rows].reset_index(drop=True),
                metadata.iloc[val_rows].reset_index(drop=True),
            )
    n_train = int(len(metadata) * train_split)
    return (
        metadata.iloc[:n_train].reset_index(drop=True),
        metadata.iloc[n_train:].reset_index(drop=True),
    )


def _load_atom_array_from_bytes(cif_bytes: bytes):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    buffer = io.StringIO(cif_bytes.decode())
    cif_file = CIFFile.read(buffer)
    block = cif_file.block
    atom_site = block["atom_site"]
    n_atoms = len(atom_site["id"].as_array())
    if "pdbx_PDB_model_num" not in atom_site:
        atom_site["pdbx_PDB_model_num"] = np.ones(n_atoms, dtype=np.int32)
    if "pdbx_PDB_ins_code" not in atom_site:
        atom_site["pdbx_PDB_ins_code"] = np.full(n_atoms, "", dtype="<U1")
    if "label_alt_id" not in atom_site:
        atom_site["label_alt_id"] = np.full(n_atoms, ".", dtype="<U1")
    if "label_seq_id" not in atom_site and "auth_seq_id" in atom_site:
        atom_site["label_seq_id"] = atom_site["auth_seq_id"].as_array()
    if "label_asym_id" not in atom_site and "auth_asym_id" in atom_site:
        atom_site["label_asym_id"] = atom_site["auth_asym_id"].as_array()
    if "label_comp_id" not in atom_site and "auth_comp_id" in atom_site:
        atom_site["label_comp_id"] = atom_site["auth_comp_id"].as_array()
    atom_array = get_structure(
        cif_file, model=1, extra_fields=["b_factor", "occupancy"]
    )
    if not hasattr(atom_array, "occupancy") or atom_array.occupancy is None:
        atom_array.set_annotation(
            "occupancy", np.ones(len(atom_array), dtype=np.float32)
        )
    if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
        atom_array.set_annotation(
            "b_factor", np.zeros(len(atom_array), dtype=np.float32)
        )
    return atom_array


def _intervals_to_residue_set(intervals: list[dict] | None) -> list[int]:
    if not intervals:
        return []
    out: list[int] = []
    for span in intervals:
        out.extend(range(int(span["lo"]), int(span["hi"]) + 1))
    return out


def _slice_dimer_atoms(atom_array, residues_a: list[int], residues_b: list[int]):
    res_id = np.asarray(atom_array.res_id)
    set_a = set(residues_a)
    set_b = set(residues_b)
    mask_a = np.array([int(r) in set_a for r in res_id], dtype=bool)
    mask_b = np.array([int(r) in set_b for r in res_id], dtype=bool)

    atoms_a = atom_array[mask_a]
    atoms_b = atom_array[mask_b]
    atoms_a.chain_id = np.array(["A"] * len(atoms_a), dtype=atoms_a.chain_id.dtype)
    atoms_b.chain_id = np.array(["B"] * len(atoms_b), dtype=atoms_b.chain_id.dtype)
    return atoms_a + atoms_b


class TeddymerDimerDataset(Dataset):
    """Per-dimer dataset reading parent AFDB tars at byte offsets.

    The locator-row pair for each dimer is pre-indexed in ``__init__`` so
    that ``__getitem__`` is a single dict lookup plus three random-access
    file reads (CIF, confidence, PAE).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        locator: pd.DataFrame,
        afdb_proteomes_root: str,
        atomarray_transforms: list[Callable] | None = None,
        atom37_transforms: list[Callable] | None = None,
        id_column: str = "dimer_id",
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.afdb_proteomes_root = str(afdb_proteomes_root)
        self.atomarray_transforms = atomarray_transforms or []
        self.atom37_transforms = atom37_transforms or []
        self.id_column = id_column

        self._locator_by_dimer: dict[int, tuple[dict, dict]] = {}
        for _, row in locator.iterrows():
            di = int(row["dimer_index"])
            existing = self._locator_by_dimer.get(di, (None, None))
            row_dict = row.to_dict()
            if str(row["chain_id"]) == "A":
                self._locator_by_dimer[di] = (row_dict, existing[1])
            else:
                self._locator_by_dimer[di] = (existing[0], row_dict)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Data | None:
        row = self.metadata.iloc[idx]
        dimer_index = int(row["dimer_index"])
        sample_id = str(row.get(self.id_column, dimer_index))

        loc_pair = self._locator_by_dimer.get(dimer_index)
        if loc_pair is None or loc_pair[0] is None or loc_pair[1] is None:
            logger.warning(
                f"TeddymerDimerDataset: missing locator pair for dimer_index={dimer_index}"
            )
            return None
        loc_a, loc_b = loc_pair

        intervals_a = list(row["residue_intervals_A"]) if row["residue_intervals_A"] is not None else []
        intervals_b = list(row["residue_intervals_B"]) if row["residue_intervals_B"] is not None else []
        residues_a = _intervals_to_residue_set(intervals_a)
        residues_b = _intervals_to_residue_set(intervals_b)

        tar_path = Path(self.afdb_proteomes_root) / loc_a["source_tar_relpath"]
        try:
            cif_bytes = read_cif_bytes_from_tar(
                tar_path,
                int(loc_a["cif_member_offset"]),
                int(loc_a["cif_member_size"]),
                bool(loc_a.get("cif_is_gz", True)),
            )
            atom_array = _load_atom_array_from_bytes(cif_bytes)
        except Exception as exc:
            logger.warning(
                f"TeddymerDimerDataset: CIF load failed for dimer_index={dimer_index}: {exc}"
            )
            return None

        try:
            atom_array = _slice_dimer_atoms(atom_array, residues_a, residues_b)
            if len(atom_array) == 0:
                logger.warning(
                    f"TeddymerDimerDataset: empty atom slice for dimer_index={dimer_index}"
                )
                return None
            atom_array = _ensure_atomworks_annotations(atom_array)
            for tf in self.atomarray_transforms:
                atom_array = tf(atom_array)
        except Exception as exc:
            logger.warning(
                f"TeddymerDimerDataset: atomarray transform failed for "
                f"dimer_index={dimer_index}: {exc}"
            )
            return None

        try:
            data = atomarray_to_atom37(atom_array, sample_id=sample_id)
        except Exception as exc:
            logger.warning(
                f"TeddymerDimerDataset: atomarray_to_atom37 failed for "
                f"dimer_index={dimer_index}: {exc}"
            )
            return None

        data.example_id = sample_id
        data.dimer_index = dimer_index
        data.teddymer_locator = {
            "A": loc_a,
            "B": loc_b,
            "intervals_A": intervals_a,
            "intervals_B": intervals_b,
            "afdb_proteomes_root": self.afdb_proteomes_root,
        }

        try:
            for tf in self.atom37_transforms:
                data = tf(data)
        except Exception as exc:
            logger.warning(
                f"TeddymerDimerDataset: atom37 transform failed for "
                f"dimer_index={dimer_index}: {exc}"
            )
            return None

        data.chain_idx = data.chains.to(torch.int8)

        return data


class TeddymerDimerDataModule(L.LightningDataModule):
    """Lightning DataModule for the Teddymer dimer view.

    The composition mirrors ``StructureDataModule``'s public surface but
    drives a tar-offset random-access read path instead of CIF-on-disk.
    """

    def __init__(
        self,
        dimers_parquet: str,
        locator_parquet: str,
        afdb_proteomes_root: str,
        batch_size: int = 6,
        num_workers: int = 16,
        atomarray_transforms: list[Callable | dict] | None = None,
        atom37_transforms: list[Callable | dict] | None = None,
        train_split: float = 0.99,
        id_column: str = "dimer_id",
        pin_memory: bool = True,
        filters: list[str] | None = None,
        cluster_column: str | None = None,
        cluster_seed: int = 42,
        pad_max_total_tokens: int | None = None,
        pad_group_priority: list[str] | None = None,
        drop_last: bool = True,
        seed: int = 0,
        **_unused: Any,
    ):
        super().__init__()
        self.dimers_parquet = dimers_parquet
        self.locator_parquet = locator_parquet
        self.afdb_proteomes_root = afdb_proteomes_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.atomarray_transforms = atomarray_transforms or []
        self.atom37_transforms = atom37_transforms or []
        self.train_split = train_split
        self.id_column = id_column
        self.pin_memory = pin_memory
        self.filters = filters
        self.cluster_column = cluster_column
        self.cluster_seed = int(cluster_seed)
        self.pad_max_total_tokens = pad_max_total_tokens
        self.pad_group_priority = pad_group_priority
        self.drop_last = drop_last
        self.seed = int(seed)

        self.train_dataset: TeddymerDimerDataset | None = None
        self.val_dataset: TeddymerDimerDataset | None = None

    def _instantiate_transforms(
        self, transforms: list[Callable | dict]
    ) -> list[Callable]:
        out: list[Callable] = []
        for t in transforms:
            if isinstance(t, dict) and "_target_" in t:
                out.append(hydra.utils.instantiate(t))
            elif hasattr(t, "_target_"):
                out.append(hydra.utils.instantiate(t))
            elif callable(t):
                out.append(t)
        return out

    def _log_split_provenance(
        self, train_meta: pd.DataFrame, val_meta: pd.DataFrame
    ) -> None:
        """Emit a rank-0 line stating whether the cluster-aware split was used.

        `_split_metadata` warns + silently falls back to a positional slice
        when `cluster_column` is requested but absent — and a positional split
        passes every test and trains without error, so a forgotten blob rebuild
        would silently reinstate the homology leak this split exists to remove.
        This makes the outcome a positive, rank-0 line in the run header rather
        than a buried warning: a degenerate fallback prints CLUSTER SPLIT NOT
        ACTIVE.
        """
        col = self.cluster_column
        if col is None:
            logger.info(f"Teddymer split: positional (cluster_column=None); "
                     f"train={len(train_meta)} val={len(val_meta)}")
            return
        present = col in train_meta.columns and col in val_meta.columns
        disjoint = False
        n_clusters = 0
        if present:
            tc = set(train_meta[col].unique())
            vc = set(val_meta[col].unique())
            disjoint = tc.isdisjoint(vc)
            n_clusters = len(tc | vc)
        if present and disjoint:
            logger.info(
                f"Teddymer split: CLUSTER-AWARE on {col!r} ACTIVE — "
                f"{n_clusters} disjoint clusters, train={len(train_meta)} "
                f"val={len(val_meta)}"
            )
        else:
            reason = "column absent from parquet" if not present else "clusters not disjoint"
            logger.info(
                f"Teddymer split: CLUSTER SPLIT NOT ACTIVE — cluster_column={col!r} "
                f"requested but {reason}; fell back to POSITIONAL slice "
                f"(homology leak risk). Rebuild the blob with `{col}` before trusting val."
            )

    def setup(self, stage: str | None = None):
        if self.num_workers > 0:
            start_method = mp.get_start_method(allow_none=True)
            if start_method is not None and start_method != "fork":
                warnings.warn(
                    f"TeddymerDimerDataModule expects 'fork' multiprocessing context "
                    f"(locator dict ~2 GB; spawn pickles per worker -> OOM at num_workers>=8). "
                    f"Active start method: {start_method}.",
                    RuntimeWarning,
                )

        dimers = pd.read_parquet(self.dimers_parquet)
        locator = pd.read_parquet(self.locator_parquet)

        if self.filters:
            for f in self.filters:
                dimers = dimers.query(f)
            dimers = dimers.reset_index(drop=True)

        train_meta, val_meta = _split_metadata(
            dimers,
            train_split=self.train_split,
            cluster_column=self.cluster_column,
            cluster_seed=self.cluster_seed,
        )
        self._log_split_provenance(train_meta, val_meta)

        atomarray_tfs = self._instantiate_transforms(self.atomarray_transforms)
        atom37_tfs = self._instantiate_transforms(self.atom37_transforms)

        self.train_dataset = TeddymerDimerDataset(
            metadata=train_meta,
            locator=locator,
            afdb_proteomes_root=self.afdb_proteomes_root,
            atomarray_transforms=atomarray_tfs,
            atom37_transforms=atom37_tfs,
            id_column=self.id_column,
        )
        if len(val_meta) > 0:
            self.val_dataset = TeddymerDimerDataset(
                metadata=val_meta,
                locator=locator,
                afdb_proteomes_root=self.afdb_proteomes_root,
                atomarray_transforms=atomarray_tfs,
                atom37_transforms=atom37_tfs,
                id_column=self.id_column,
            )

    def _collate_fn(self):
        return make_collate_fn(
            pad_max_total_tokens=self.pad_max_total_tokens,
            pad_group_priority=self.pad_group_priority,
        )

    def _dataloader_kwargs(self) -> dict:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        kwargs: dict = {
            "generator": generator,
            "worker_init_fn": _teddymer_worker_init_fn,
        }
        if self.num_workers > 0:
            kwargs["multiprocessing_context"] = "fork"
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            **self._dataloader_kwargs(),
        )
