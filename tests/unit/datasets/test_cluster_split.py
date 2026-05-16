"""Cluster-aware split tests for `StructureDataModule`.

The split must (a) assign whole clusters to train / val so no cluster
spans both splits, (b) be deterministic under a fixed seed and different
under different seeds, (c) fall through to the row-order baseline when
`cluster_column=None`, (d) log a warning + fall through when the column
is missing from the parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from proteinfoundation.datasets.structure_data import StructureDataModule


def _make_metadata(tmp_path: Path, *, with_cluster: bool = True) -> Path:
    rows = []
    for cluster_id in range(10):
        for j in range(10):
            row = {
                "id": f"id_{cluster_id}_{j}",
                "path": f"file_{cluster_id}_{j}.pdb",
            }
            if with_cluster:
                row["unicluster"] = f"cluster_{cluster_id}"
            rows.append(row)
    df = pd.DataFrame(rows)
    out = tmp_path / "metadata.parquet"
    df.to_parquet(out)
    return out


class _SplitProbe(StructureDataModule):
    """Skip StructureDataset construction to exercise the split logic only."""

    def setup(self, stage=None):
        full_metadata = pd.read_parquet(self.metadata_file)
        if self.filters:
            for f in self.filters:
                full_metadata = full_metadata.query(f)
            full_metadata = full_metadata.reset_index(drop=True)
        self._train_meta, self._val_meta = self._split(full_metadata)


def _split_dm(metadata: Path, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    kwargs.setdefault("train_split", 0.9)
    dm = _SplitProbe(metadata_file=str(metadata), **kwargs)
    dm.setup()
    return dm._train_meta, dm._val_meta


def test_cluster_column_none_preserves_row_order(tmp_path: Path) -> None:
    meta_path = _make_metadata(tmp_path, with_cluster=True)
    train, val = _split_dm(meta_path, cluster_column=None)
    full = pd.read_parquet(meta_path).reset_index(drop=True)
    n_train = int(len(full) * 0.9)
    expected_train = full.iloc[:n_train].reset_index(drop=True)
    expected_val = full.iloc[n_train:].reset_index(drop=True)
    assert train["id"].tolist() == expected_train["id"].tolist()
    assert val["id"].tolist() == expected_val["id"].tolist()


def test_cluster_aware_split_keeps_clusters_disjoint(tmp_path: Path) -> None:
    meta_path = _make_metadata(tmp_path, with_cluster=True)
    train, val = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=42)
    assert len(train) > 0
    assert len(val) > 0
    train_clusters = set(train["unicluster"].unique())
    val_clusters = set(val["unicluster"].unique())
    assert train_clusters.isdisjoint(val_clusters)
    assert train_clusters | val_clusters == {f"cluster_{i}" for i in range(10)}


def test_cluster_split_changes_with_seed(tmp_path: Path) -> None:
    meta_path = _make_metadata(tmp_path, with_cluster=True)
    _, val0 = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=0)
    _, val1 = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=1)
    assert set(val0["unicluster"].unique()) != set(val1["unicluster"].unique())


def test_missing_column_falls_back_with_warning(tmp_path: Path, caplog) -> None:
    meta_path = _make_metadata(tmp_path, with_cluster=False)
    from loguru import logger as loguru_logger

    msgs: list[str] = []
    sink_id = loguru_logger.add(lambda msg: msgs.append(str(msg)), level="WARNING")
    try:
        train, val = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=42)
    finally:
        loguru_logger.remove(sink_id)

    full = pd.read_parquet(meta_path).reset_index(drop=True)
    n_train = int(len(full) * 0.9)
    assert train["id"].tolist() == full.iloc[:n_train]["id"].tolist()
    assert val["id"].tolist() == full.iloc[n_train:]["id"].tolist()
    assert any("unicluster" in m for m in msgs)


def _make_skewed_metadata(tmp_path: Path) -> Path:
    rows = []
    for j in range(80):
        rows.append(
            {
                "id": f"id_big_{j}",
                "path": f"file_big_{j}.pdb",
                "unicluster": "cluster_big",
            }
        )
    for cluster_id in range(9):
        for j in range(2):
            rows.append(
                {
                    "id": f"id_small_{cluster_id}_{j}",
                    "path": f"file_small_{cluster_id}_{j}.pdb",
                    "unicluster": f"cluster_small_{cluster_id}",
                }
            )
    df = pd.DataFrame(rows)
    out = tmp_path / "skewed_metadata.parquet"
    df.to_parquet(out)
    return out


def test_skewed_cluster_split(tmp_path: Path) -> None:
    """One dominant cluster of 80 rows + nine tiny clusters of 2 rows.

    Whole clusters must land on one side; total preserved; the dominant
    cluster lands deterministically given the seed.
    """
    meta_path = _make_skewed_metadata(tmp_path)
    train, val = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=42)
    full = pd.read_parquet(meta_path)
    assert len(train) + len(val) == len(full)
    train_clusters = set(train["unicluster"].unique())
    val_clusters = set(val["unicluster"].unique())
    assert train_clusters.isdisjoint(val_clusters)
    train2, val2 = _split_dm(meta_path, cluster_column="unicluster", cluster_seed=42)
    assert train["id"].tolist() == train2["id"].tolist()
    assert val["id"].tolist() == val2["id"].tolist()


def _make_two_cluster_metadata(tmp_path: Path) -> Path:
    rows = []
    for cluster_id in range(2):
        for j in range(50):
            rows.append(
                {
                    "id": f"id_{cluster_id}_{j}",
                    "path": f"file_{cluster_id}_{j}.pdb",
                    "unicluster": f"cluster_{cluster_id}",
                }
            )
    df = pd.DataFrame(rows)
    out = tmp_path / "two_cluster_metadata.parquet"
    df.to_parquet(out)
    return out


def test_empty_val_warning(tmp_path: Path) -> None:
    """High train_split over few clusters can produce an empty val split.

    The split must emit a loguru warning naming the cluster column,
    train_split, n_clusters, n_rows and proceed (Lightning surfaces the
    empty-dataloader error downstream).
    """
    meta_path = _make_two_cluster_metadata(tmp_path)
    from loguru import logger as loguru_logger

    msgs: list[str] = []
    sink_id = loguru_logger.add(lambda msg: msgs.append(str(msg)), level="WARNING")
    try:
        train, val = _split_dm(
            meta_path, cluster_column="unicluster", cluster_seed=42, train_split=0.99
        )
    finally:
        loguru_logger.remove(sink_id)

    assert len(val) == 0
    assert any("empty val split" in m for m in msgs), (
        f"expected empty-val warning, got messages: {msgs}"
    )
    joined = " ".join(msgs)
    assert "unicluster" in joined
    assert "train_split" in joined or "0.99" in joined
