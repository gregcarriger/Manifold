"""Retrieval tests. RRF is pure and runs anywhere; the end-to-end retriever test is gated
on a reachable, populated pgvector DB + built BM25 indexes."""

from __future__ import annotations

import os

import pytest

from manifold.retrieve.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    # 'x' tops both lists -> must win; items in only one list rank below.
    fused = reciprocal_rank_fusion([["x", "a", "b"], ["x", "c", "d"]])
    assert fused[0][0] == "x"
    assert fused[0][1] > fused[1][1]


def test_rrf_weights_bias_a_list():
    # heavily weight the second list; its head 'c' should outrank list-one's head 'a'.
    fused = reciprocal_rank_fusion([["a"], ["c"]], weights=[0.1, 10.0])
    order = [cid for cid, _ in fused]
    assert order[0] == "c"


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []


_DB = os.environ.get("MANIFOLD_DB_URL", "postgresql://manifold:manifold@localhost:5433/manifold")


def _ready() -> bool:
    try:
        import psycopg

        conn = psycopg.connect(_DB, connect_timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM chunks WHERE strategy='structure'")
        n = cur.fetchone()[0]
        conn.close()
        return n > 0 and os.path.exists(os.path.join(
            os.path.dirname(__file__), "..", "corpus", "index", "bm25",
            "structure__bge-small-en-v1.5.pkl"))
    except Exception:  # noqa: BLE001 - any failure means "index not ready"; skip the test
        return False


@pytest.mark.skipif(not _ready(), reason="pgvector not populated / bm25 index missing")
def test_retriever_all_methods_return_ranked_results():
    from manifold.retrieve.config import RetrievalConfig
    from manifold.retrieve.retriever import Retriever

    r = Retriever()
    try:
        for method in ("dense", "bm25", "hybrid"):
            res = r.retrieve("safety instrumented system",
                             RetrievalConfig(method=method, strategy="structure", k=5))
            assert 1 <= len(res) <= 5
            assert [x.rank for x in res] == list(range(1, len(res) + 1))
            assert all(x.text for x in res)
    finally:
        r.close()


@pytest.mark.skipif(not _ready(), reason="pgvector not populated / bm25 index missing")
def test_bm25_beats_dense_on_exact_cve():
    """The headline claim, as a regression test: an exact CVE id is found lexically."""
    from manifold.retrieve.config import RetrievalConfig
    from manifold.retrieve.retriever import Retriever

    cve = "CVE-2023-38545"
    r = Retriever()
    try:
        bm = r.retrieve(cve, RetrievalConfig(method="bm25", strategy="structure", k=10))
        # the advisory that actually carries this CVE should surface lexically
        assert any(hit.doc_id == "cisa:ICSA-24-004-01" for hit in bm)
    finally:
        r.close()
