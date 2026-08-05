"""pgvector-backed vector store.

Design choices and why they are the right default here:

* **Exact search by default.** The corpus is ~11K–22K vectors per chunk set — small enough
  that exact cosine (``ORDER BY embedding <=> q``) runs in milliseconds. For a *retrieval
  quality* benchmark that is the right default: an ANN index would fold approximation
  error into the numbers we are trying to measure. An optional HNSW index (`create_hnsw`)
  is provided as the documented scale path.
* **One table, keyed by (strategy, model, chunk_id).** Both chunk sets and future embedding
  models coexist; every query filters by strategy+model, so results are always comparable.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from ..schema import Chunk
from .embedder import DIM

DEFAULT_DSN = "postgresql://manifold:manifold@localhost:5433/manifold"

_DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    strategy       TEXT NOT NULL,
    model          TEXT NOT NULL,
    chunk_id       TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    source         TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    title          TEXT,
    url            TEXT,
    section_path   JSONB,
    metadata       JSONB,
    text           TEXT NOT NULL,
    token_estimate INT,
    embedding      vector({DIM}),
    PRIMARY KEY (strategy, model, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (strategy, model);
"""


def get_dsn() -> str:
    return os.environ.get("MANIFOLD_DB_URL", DEFAULT_DSN)


def connect(dsn: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or get_dsn())
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()


def clear(conn: psycopg.Connection, strategy: str, model: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE strategy=%s AND model=%s", (strategy, model))
    conn.commit()


def upsert(conn: psycopg.Connection, chunks: list[Chunk], embeddings, model: str,
           batch_size: int = 1000) -> int:
    """Insert/replace chunk rows with their embeddings."""
    import json

    sql = """
        INSERT INTO chunks (strategy, model, chunk_id, doc_id, source, doc_type,
                            title, url, section_path, metadata, text, token_estimate, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (strategy, model, chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding, text = EXCLUDED.text, metadata = EXCLUDED.metadata
    """
    n = 0
    with conn.cursor() as cur:
        batch = []
        for c, emb in zip(chunks, embeddings):
            batch.append((
                c.strategy, model, c.chunk_id, c.doc_id, c.source, c.doc_type,
                c.title, c.url, json.dumps(c.section_path), json.dumps(c.metadata),
                c.text, c.token_estimate, emb,
            ))
            if len(batch) >= batch_size:
                cur.executemany(sql, batch)
                n += len(batch)
                batch = []
        if batch:
            cur.executemany(sql, batch)
            n += len(batch)
    conn.commit()
    return n


def create_hnsw(conn: psycopg.Connection) -> None:
    """Optional ANN index — the scale path. Not used by the exact-search benchmark."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()


def search(conn: psycopg.Connection, strategy: str, model: str, query_vec, k: int = 10,
           source: str | None = None) -> list[dict[str, Any]]:
    """Exact cosine nearest-neighbor search, returning ranked chunk rows + scores."""
    where = "strategy=%s AND model=%s"
    if source:
        where += " AND source=%s"
    sql = f"""
        SELECT chunk_id, doc_id, source, doc_type, title, url, text,
               1 - (embedding <=> %s) AS cosine_sim
        FROM chunks WHERE {where}
        ORDER BY embedding <=> %s LIMIT %s
    """
    # Params must follow placeholder order in the SQL text: SELECT's query_vec, then the
    # WHERE filters, then ORDER BY's query_vec, then the LIMIT.
    params: list[Any] = [query_vec, strategy, model]
    if source:
        params.append(source)
    params.extend([query_vec, k])
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_by_ids(conn: psycopg.Connection, strategy: str, model: str,
                 chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return {chunk_id: row} for the given ids (used to enrich BM25/fused results)."""
    if not chunk_ids:
        return {}
    sql = """
        SELECT chunk_id, doc_id, source, doc_type, title, url, text
        FROM chunks WHERE strategy=%s AND model=%s AND chunk_id = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (strategy, model, list(chunk_ids)))
        cols = [d.name for d in cur.description]
        return {row[0]: dict(zip(cols, row)) for row in cur.fetchall()}


def count(conn: psycopg.Connection, strategy: str, model: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE strategy=%s AND model=%s",
                    (strategy, model))
        return cur.fetchone()[0]
