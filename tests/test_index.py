"""Index-layer tests. BM25 tokenizer tests run anywhere; the pgvector round-trip is
gated on a reachable database (skipped in CI without one)."""

from __future__ import annotations

import os

import pytest

from manifold.index import bm25


def test_tokenizer_preserves_ics_identifiers():
    toks = bm25.tokenize("CVE-2024-38545 affects Rockwell 1756-L8x over Modbus/TCP.")
    # Identifiers must survive as single tokens — that is where lexical beats dense.
    assert "cve-2024-38545" in toks
    assert "1756-l8x" in toks
    assert "modbus/tcp" in toks


def test_bm25_ranks_exact_identifier_first():
    ids = ["a", "b", "c"]
    texts = [
        "General discussion of buffer overflow weaknesses in controllers.",
        "Advisory for CVE-2024-38545 out-of-bounds write in the device firmware.",
        "Network segmentation guidance for industrial control systems.",
    ]
    idx = bm25.build(ids, texts)
    top = idx.search("CVE-2024-38545", k=1)
    assert top[0][0] == "b" and top[0][1] > 0


_DB = os.environ.get("MANIFOLD_DB_URL", "postgresql://manifold:manifold@localhost:5433/manifold")


def _db_reachable() -> bool:
    try:
        import psycopg

        psycopg.connect(_DB, connect_timeout=2).close()
        return True
    except Exception:  # noqa: BLE001 - any failure (import, refused, timeout) means "no DB"
        return False


@pytest.mark.skipif(not _db_reachable(), reason="no pgvector DB reachable")
def test_pgvector_roundtrip():
    import numpy as np

    from manifold.index import store
    from manifold.index.embedder import DIM
    from manifold.schema import Chunk

    conn = store.connect(_DB)
    store.init_schema(conn)
    store.clear(conn, "_test", "m")
    chunks = [
        Chunk(chunk_id=f"t{i}", doc_id="d", source="cisa", doc_type="advisory",
              strategy="_test", title=f"c{i}", text=f"chunk {i}")
        for i in range(3)
    ]
    # orthonormal-ish vectors so nearest neighbor is deterministic
    vecs = np.eye(3, DIM, dtype="float32")
    # NB: upsert reads chunk.strategy; force our test strategy.
    for c in chunks:
        c.strategy = "_test"
    store.upsert(conn, chunks, vecs, "m")
    assert store.count(conn, "_test", "m") == 3
    hits = store.search(conn, "_test", "m", vecs[1], k=1)
    assert hits[0]["chunk_id"] == "t1"
    store.clear(conn, "_test", "m")
    conn.close()
