"""Unit tests for BestOfNSearch iteration-unique metadata tags.

The best-of-N loop runs many predict batches on the SAME conditioning batch, so a
tag keyed only on (sample_index, replica) collides across iterations. Downstream
joins (plot_successes_vs_gpu_hours) merge on metadata_tag, so a colliding tag fans
out. These tests pin that two consecutive ``search()`` calls on the same batch
produce DISJOINT tag sets, and that the tag encodes the iteration index.

CPU-only. Run: .venv/bin/python -m pytest tests/unit/search/test_best_of_n_tags_unique.py
"""

from __future__ import annotations

import re

import torch
from omegaconf import OmegaConf

from proteinfoundation.search.best_of_n_search import BestOfNSearch


class _FakeSinglePass:
    """Returns a chunk-sized final dict with placeholder tags (overwritten by BoN)."""

    def search(self, chunk: dict) -> dict:
        n = chunk["mask"].shape[0]
        return {
            "lookahead": None,
            "final": {
                "coors": torch.zeros(n, 4, 3),
                "mask": chunk["mask"].clone(),
                "metadata_tag": ["placeholder"] * n,
            },
        }


def _make_search(replicas: int, max_batch_size: int | None) -> BestOfNSearch:
    inst = BestOfNSearch.__new__(BestOfNSearch)
    inst.inf_cfg = OmegaConf.create(
        {"search": {"best_of_n": {"replicas": replicas}, "max_batch_size": max_batch_size}}
    )
    inst.single_pass = _FakeSinglePass()
    inst._iteration = 0
    return inst


def _batch(nsamples: int, nres: int = 5) -> dict:
    return {"mask": torch.ones(nsamples, nres, dtype=torch.bool)}


def test_two_search_calls_produce_disjoint_tags():
    search = _make_search(replicas=3, max_batch_size=None)
    batch = _batch(nsamples=2)

    tags_it0 = set(search.search(batch)["final"]["metadata_tag"])
    tags_it1 = set(search.search(batch)["final"]["metadata_tag"])

    assert len(tags_it0) == 6
    assert len(tags_it1) == 6
    assert tags_it0.isdisjoint(tags_it1)


def test_tag_encodes_iteration_sample_replica():
    search = _make_search(replicas=2, max_batch_size=None)
    batch = _batch(nsamples=2)

    it0 = search.search(batch)["final"]["metadata_tag"]
    it1 = search.search(batch)["final"]["metadata_tag"]

    pattern = re.compile(r"^bon_it(\d+)_orig(\d+)_r(\d+)$")
    for tag in it0:
        assert pattern.match(tag), tag
    assert {pattern.match(t).group(1) for t in it0} == {"0"}
    assert {pattern.match(t).group(1) for t in it1} == {"1"}
    # sample/replica coordinates are re-emitted each iteration, only the iter differs.
    coords_it0 = {(m.group(2), m.group(3)) for t in it0 if (m := pattern.match(t))}
    coords_it1 = {(m.group(2), m.group(3)) for t in it1 if (m := pattern.match(t))}
    assert coords_it0 == coords_it1 == {("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")}


def test_disjoint_tags_survive_chunking():
    search = _make_search(replicas=4, max_batch_size=3)
    batch = _batch(nsamples=2)  # total 8 > max_batch_size 3 -> chunked

    tags_it0 = search.search(batch)["final"]["metadata_tag"]
    tags_it1 = search.search(batch)["final"]["metadata_tag"]

    assert len(tags_it0) == 8
    assert len(set(tags_it0)) == 8
    assert set(tags_it0).isdisjoint(set(tags_it1))
