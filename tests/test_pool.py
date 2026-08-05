"""Pooling logic tests — pure, no ML/DB/API needed."""

from __future__ import annotations

from manifold.eval.pool import (
    PoolResult,
    coverage_summary,
    leave_one_out,
    union_pool,
)


def _sample() -> PoolResult:
    # 3 systems that fail differently on 2 queries.
    per_system = {
        "dense": {"q0": ["a", "b", "c"], "q1": ["x", "y"]},
        "bm25": {"q0": ["c", "d"], "q1": ["y", "z"]},       # d only bm25 finds on q0
        "rerank": {"q0": ["a", "e"], "q1": ["x"]},          # e only rerank finds on q0
    }
    return union_pool(per_system, depth=3)


def test_union_dedups_and_tracks_contributors():
    r = _sample()
    # q0 union = a,b,c,d,e ; a found by dense+rerank, c by dense+bm25
    assert r.pool("q0") == ["a", "b", "c", "d", "e"]
    assert r.contributors["q0"]["a"] == ["dense", "rerank"]
    assert r.contributors["q0"]["c"] == ["bm25", "dense"]
    assert r.contributors["q0"]["d"] == ["bm25"]


def test_depth_truncation():
    per_system = {"s": {"q0": ["a", "b", "c", "d"]}}
    r = union_pool(per_system, depth=2)
    assert r.pool("q0") == ["a", "b"]   # only top-2 pooled


def test_leave_one_out_counts_unique_finds():
    r = _sample()
    loo = leave_one_out(r)
    # q0 uniques: b(dense), d(bm25), e(rerank); q1 uniques: z(bm25) [x,y shared]
    assert loo["dense"] == 1     # b
    assert loo["bm25"] == 2      # d, z
    assert loo["rerank"] == 1    # e


def test_coverage_summary_shape():
    s = coverage_summary(_sample())
    assert s["num_queries"] == 2
    assert s["systems"] == ["bm25", "dense", "rerank"]
    assert s["max_pool_size"] == 5          # q0 has a,b,c,d,e
    assert s["unique_contribution_per_system"]["bm25"] == 2


def test_pool_roundtrip(tmp_path):
    r = _sample()
    path = str(tmp_path / "pool.json")
    r.save(path)
    back = PoolResult.load(path)
    assert back.depth == r.depth
    assert back.systems == r.systems
    assert back.pool("q0") == r.pool("q0")
    assert back.contributors["q0"]["c"] == ["bm25", "dense"]
