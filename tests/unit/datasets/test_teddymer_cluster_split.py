"""Cluster-aware split tests for the Teddymer `_split_metadata` helper.

Mirrors `tests/unit/datasets/test_cluster_split.py` (which covers the
`StructureDataModule` split) for the Teddymer reimplementation. The split
must (a) assign whole clusters to train / val so no cluster spans both
splits, (b) be deterministic under a fixed seed and differ across seeds,
(c) fall through to the row-order baseline when `cluster_column=None`,
(d) warn + fall through when the requested column is absent, and (e) warn
when the split yields an empty val set.
"""

from __future__ import annotations

import pandas as pd

from proteinfoundation.datasets.teddymer.dataset import (
    TeddymerDimerDataModule,
    _split_metadata,
)


def _make_metadata(*, with_cluster: bool = True) -> pd.DataFrame:
    rows = []
    for cluster_id in range(10):
        for j in range(10):
            row = {"dimer_id": f"id_{cluster_id}_{j}", "dimer_index": cluster_id * 10 + j}
            if with_cluster:
                row["seqclust30"] = f"cluster_{cluster_id}"
            rows.append(row)
    return pd.DataFrame(rows)


def _capture_warnings_at(level: str = "WARNING"):
    from loguru import logger as loguru_logger

    msgs: list[str] = []
    sink_id = loguru_logger.add(lambda msg: msgs.append(str(msg)), level=level)
    return loguru_logger, sink_id, msgs


def _capture_warnings():
    return _capture_warnings_at("WARNING")


def test_cluster_column_none_preserves_row_order() -> None:
    df = _make_metadata(with_cluster=True)
    train, val = _split_metadata(df, train_split=0.9, cluster_column=None, cluster_seed=42)
    n_train = int(len(df) * 0.9)
    assert train["dimer_id"].tolist() == df.iloc[:n_train]["dimer_id"].tolist()
    assert val["dimer_id"].tolist() == df.iloc[n_train:]["dimer_id"].tolist()


def test_cluster_aware_split_keeps_clusters_disjoint() -> None:
    df = _make_metadata(with_cluster=True)
    train, val = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=42)
    assert len(train) > 0
    assert len(val) > 0
    train_clusters = set(train["seqclust30"].unique())
    val_clusters = set(val["seqclust30"].unique())
    assert train_clusters.isdisjoint(val_clusters)
    assert train_clusters | val_clusters == {f"cluster_{i}" for i in range(10)}
    assert len(train) + len(val) == len(df)


def test_cluster_split_changes_with_seed() -> None:
    df = _make_metadata(with_cluster=True)
    _, val0 = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=0)
    _, val1 = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=1)
    assert set(val0["seqclust30"].unique()) != set(val1["seqclust30"].unique())


def test_cluster_split_is_deterministic() -> None:
    df = _make_metadata(with_cluster=True)
    t0, v0 = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=7)
    t1, v1 = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=7)
    assert t0["dimer_id"].tolist() == t1["dimer_id"].tolist()
    assert v0["dimer_id"].tolist() == v1["dimer_id"].tolist()


def test_missing_column_falls_back_with_warning() -> None:
    df = _make_metadata(with_cluster=False)
    loguru_logger, sink_id, msgs = _capture_warnings()
    try:
        train, val = _split_metadata(
            df, train_split=0.9, cluster_column="seqclust30", cluster_seed=42
        )
    finally:
        loguru_logger.remove(sink_id)
    n_train = int(len(df) * 0.9)
    assert train["dimer_id"].tolist() == df.iloc[:n_train]["dimer_id"].tolist()
    assert val["dimer_id"].tolist() == df.iloc[n_train:]["dimer_id"].tolist()
    assert any("seqclust30" in m for m in msgs)


def _provenance_dm(cluster_column: str | None) -> TeddymerDimerDataModule:
    return TeddymerDimerDataModule(
        dimers_parquet="x",
        locator_parquet="x",
        afdb_proteomes_root="x",
        cluster_column=cluster_column,
        cluster_seed=0,
    )


def test_provenance_logs_cluster_active_when_disjoint() -> None:
    df = _make_metadata(with_cluster=True)
    train, val = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=0)
    dm = _provenance_dm("seqclust30")
    loguru_logger, sink_id, msgs = _capture_warnings_at("INFO")
    try:
        dm._log_split_provenance(train, val)
    finally:
        loguru_logger.remove(sink_id)
    joined = " ".join(msgs)
    assert "CLUSTER-AWARE" in joined and "ACTIVE" in joined
    assert "NOT ACTIVE" not in joined


def test_provenance_logs_not_active_on_positional_fallback() -> None:
    """When cluster_column is requested but absent, the split silently
    falls back to a positional slice; the provenance line must shout that
    the cluster split is NOT active so the leak risk is visible at rank 0.
    """
    df = _make_metadata(with_cluster=False)
    train, val = _split_metadata(df, train_split=0.9, cluster_column="seqclust30", cluster_seed=0)
    dm = _provenance_dm("seqclust30")
    loguru_logger, sink_id, msgs = _capture_warnings_at("INFO")
    try:
        dm._log_split_provenance(train, val)
    finally:
        loguru_logger.remove(sink_id)
    joined = " ".join(msgs)
    assert "CLUSTER SPLIT NOT ACTIVE" in joined
    assert "POSITIONAL" in joined


def test_empty_val_warning() -> None:
    """High train_split over few clusters can produce an empty val split.

    The split must warn (naming the column + train_split) and proceed;
    Lightning surfaces the empty-dataloader behaviour downstream.
    """
    rows = []
    for cluster_id in range(2):
        for j in range(50):
            rows.append(
                {
                    "dimer_id": f"id_{cluster_id}_{j}",
                    "dimer_index": cluster_id * 50 + j,
                    "seqclust30": f"cluster_{cluster_id}",
                }
            )
    df = pd.DataFrame(rows)
    loguru_logger, sink_id, msgs = _capture_warnings()
    try:
        train, val = _split_metadata(
            df, train_split=0.99, cluster_column="seqclust30", cluster_seed=42
        )
    finally:
        loguru_logger.remove(sink_id)
    assert len(val) == 0
    joined = " ".join(msgs)
    assert any("empty val split" in m for m in msgs), f"got: {msgs}"
    assert "seqclust30" in joined
    assert "train_split" in joined or "0.99" in joined
