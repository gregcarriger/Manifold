"""Indexing orchestrator: embed a chunk set into pgvector and build its BM25 index.

Usage:
    python -m manifold.index.build --strategy structure     # index one chunk set
    python -m manifold.index.build --strategy all            # both
    python -m manifold.index.build --strategy structure --smoke "modbus command injection"

Reports counts and per-stage timings, and runs a dense + BM25 smoke query so a bad index
is obvious immediately.
"""

from __future__ import annotations

import argparse
import os
import time

from ..schema import read_chunks_jsonl
from . import bm25, store
from .embedder import MODEL_NAME, Embedder

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CHUNKS_DIR = os.path.join(_ROOT, "corpus", "chunks")
BM25_DIR = os.path.join(_ROOT, "corpus", "index", "bm25")

_MODEL_SLUG = MODEL_NAME.split("/")[-1]


def index_strategy(
    strategy: str, embedder: Embedder, smoke: str | None, chunks_dir: str = CHUNKS_DIR
) -> dict:
    path = os.path.join(chunks_dir, f"{strategy}.jsonl")
    chunks = list(read_chunks_jsonl(path))
    print(f"[{strategy}] loaded {len(chunks)} chunks (device={embedder.device})")

    t0 = time.time()
    embeddings = embedder.embed_passages([c.text for c in chunks])
    t_embed = time.time() - t0

    conn = store.connect()
    store.init_schema(conn)
    store.clear(conn, strategy, MODEL_NAME)
    t0 = time.time()
    n = store.upsert(conn, chunks, embeddings, MODEL_NAME)
    t_upsert = time.time() - t0

    t0 = time.time()
    os.makedirs(BM25_DIR, exist_ok=True)
    idx = bm25.build([c.chunk_id for c in chunks], [c.text for c in chunks])
    idx.save(os.path.join(BM25_DIR, f"{strategy}__{_MODEL_SLUG}.pkl"))
    t_bm25 = time.time() - t0

    print(f"[{strategy}] embed {t_embed:.1f}s | upsert {n} rows {t_upsert:.1f}s | "
          f"bm25 {t_bm25:.1f}s | in pgvector: {store.count(conn, strategy, MODEL_NAME)}")

    if smoke:
        print(f"\n  smoke query: {smoke!r}")
        qv = embedder.embed_query(smoke)
        print("  -- dense (cosine) top 5 --")
        for r in store.search(conn, strategy, MODEL_NAME, qv, k=5):
            print(f"     {r['cosine_sim']:.3f}  {r['chunk_id']:32} {r['title'][:45]}")
        print("  -- bm25 top 5 --")
        for cid, sc in idx.search(smoke, k=5):
            print(f"     {sc:6.2f}  {cid}")
    conn.close()
    return {"strategy": strategy, "chunks": n, "embed_s": round(t_embed, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dense + BM25 indexes for a chunk set.")
    ap.add_argument("--strategy", choices=["fixed", "structure", "all"], default="all")
    ap.add_argument("--smoke", default=None, help="run a smoke query after indexing")
    ap.add_argument("--chunks-dir", default=CHUNKS_DIR, help="chunk input dir (default: corpus/chunks)")
    args = ap.parse_args()

    embedder = Embedder()
    strategies = ["fixed", "structure"] if args.strategy == "all" else [args.strategy]
    for s in strategies:
        index_strategy(s, embedder, args.smoke, args.chunks_dir)


if __name__ == "__main__":
    main()
