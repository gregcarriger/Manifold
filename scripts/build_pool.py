#!/usr/bin/env python3
"""Build a system-agnostic candidate pool for the public gold set (Phase-4 prerequisite).

Runs a *diverse* fleet of retrievers over the PUBLIC chunk set, unions their top-_K_ doc IDs
per query, and writes ``corpus/goldset/pool.json`` — the set of documents a human/LLM then
judges. Judging the union (rather than one retriever's candidates) is what keeps the qrels
from favoring the retriever that generated them. See ``POOLING.md`` for the methodology.

Why a standalone in-memory script (not the pgvector path):
  * The production ``chunks`` table fixes the embedding column at 384-dim (bge-small); these
    pooling models are 768–1024-dim, so re-indexing the whole corpus per model is wasteful.
    Pooling only needs candidates for ~50 queries against ~33K chunks — trivially in RAM.
  * It reads ``corpus/chunks/*.jsonl`` (the committed *public* chunk set) directly, so the
    public pool can never be contaminated by the licensed Dragos/IEC augmented index that may
    be loaded in pgvector. (Guarded below — refuses a *_local chunk path unless --allow-local.)

Sequence:  eval.generate --stage queries   ->   THIS SCRIPT   ->   eval.generate --stage judge

Usage:
    python scripts/build_pool.py                       # default diverse fleet + BM25
    python scripts/build_pool.py --with-anchor         # add Qwen3-Embedding-4B (heavy)
    python scripts/build_pool.py --with-reranker       # add a cross-encoder pass
    python scripts/build_pool.py --models e5-base,gte-large,bm25
    python scripts/build_pool.py --light               # bge-small only (smoke test)

Local & free — no API key. First run downloads the selected models from HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.goldset import GoldSet
from manifold.eval.metrics import chunks_to_docs
from manifold.eval.pool import (
    InMemoryDense,
    ModelSpec,
    coverage_summary,
    union_pool,
)
from manifold.schema import read_chunks_jsonl

# --- the diversity fleet ---------------------------------------------------------------
# Chosen from *different labs / objectives* so their errors are uncorrelated (POOLING.md).
# Prefixes are each family's intended asymmetric encoding, so every model pools at its best.
_BGE_Q = "Represent this sentence for searching relevant passages: "

REGISTRY: dict[str, ModelSpec] = {
    "bge-small":  ModelSpec("bge-small", "BAAI/bge-small-en-v1.5", query_prefix=_BGE_Q),
    "bge-base":   ModelSpec("bge-base", "BAAI/bge-base-en-v1.5", query_prefix=_BGE_Q),
    "e5-base":    ModelSpec("e5-base", "intfloat/e5-base-v2",
                            query_prefix="query: ", passage_prefix="passage: "),
    "gte-large":  ModelSpec("gte-large", "Alibaba-NLP/gte-large-en-v1.5",
                            trust_remote_code=True),
    "nomic":      ModelSpec("nomic", "nomic-ai/nomic-embed-text-v1.5",
                            query_prefix="search_query: ", passage_prefix="search_document: ",
                            trust_remote_code=True),
    "arctic":     ModelSpec("arctic", "Snowflake/snowflake-arctic-embed-l",
                            query_prefix=_BGE_Q),
    # Tier-2 anchor (heavy, ~8 GB) — opt in with --with-anchor.
    "qwen3-4b":   ModelSpec("qwen3-4b", "Qwen/Qwen3-Embedding-4B",
                            query_prefix="Instruct: Given a web search query, retrieve "
                                         "relevant passages that answer the query\nQuery: ",
                            trust_remote_code=True),
}
DEFAULT_FLEET = ["bge-base", "e5-base", "gte-large", "nomic", "arctic", "bm25"]
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _load_chunks(path: str, allow_local: bool):
    if ("local" in os.path.basename(os.path.dirname(path)) or "local" in path) \
            and not allow_local:
        raise SystemExit(
            f"[build_pool] refusing to pool from what looks like a local/licensed chunk set:\n"
            f"    {path}\n"
            f"The PUBLIC gold set must be built from public chunks only. Pass --allow-local "
            f"to override (the resulting pool is a private, unpublishable artifact)."
        )
    chunk_ids, doc_ids, texts = [], [], []
    for c in read_chunks_jsonl(path):
        chunk_ids.append(c.chunk_id)
        doc_ids.append(c.doc_id)
        texts.append(c.text)
    if not chunk_ids:
        raise SystemExit(f"[build_pool] no chunks in {path}")
    return chunk_ids, doc_ids, texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the system-agnostic candidate pool.")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"),
                    help="gold-set dir containing queries.jsonl; pool.json is written here")
    ap.add_argument("--strategy", default="structure",
                    help="public chunk set to pool over (finer chunks -> wider doc coverage)")
    ap.add_argument("--chunks", default=None, help="explicit chunk jsonl (overrides --strategy)")
    ap.add_argument("--depth", type=int, default=100, help="top-K doc IDs per system to pool")
    ap.add_argument("--models", default=None,
                    help="comma list of registry keys (default: the diverse fleet)")
    ap.add_argument("--with-anchor", action="store_true", help="add Qwen3-Embedding-4B (heavy)")
    ap.add_argument("--with-reranker", action="store_true",
                    help=f"add a {RERANKER_MODEL} pass over the union")
    ap.add_argument("--light", action="store_true", help="bge-small only (smoke test)")
    ap.add_argument("--include-unanswerable", action="store_true",
                    help="also pool unanswerable queries (to confirm nothing relevant surfaces)")
    ap.add_argument("--device", default=None, help="force a device (cpu/mps/cuda); default auto")
    ap.add_argument("--allow-local", action="store_true", help="permit a *_local chunk set")
    ap.add_argument("--from-runs", action="store_true",
                    help="skip encoding: re-derive the pool at --depth from cached pool_runs.json "
                         "(instant; use to change judging depth without re-encoding the corpus)")
    args = ap.parse_args()

    runs_path = os.path.join(args.gold, "pool_runs.json")
    out_path = os.path.join(args.gold, "pool.json")

    # Fast path: re-truncate cached per-system runs to a new depth, no models loaded.
    if args.from_runs:
        if not os.path.exists(runs_path):
            raise SystemExit(f"[build_pool] --from-runs needs {runs_path} (run an encode first).")
        with open(runs_path, encoding="utf-8") as f:
            cache = json.load(f)
        if args.depth > cache.get("depth", 0):
            raise SystemExit(f"[build_pool] cached runs only go to depth {cache['depth']}; "
                             f"requested {args.depth}. Re-encode at a deeper --depth.")
        result = union_pool(cache["runs"], args.depth)
        result.save(out_path)
        _report(result, out_path, skipped=[])
        return

    chunks_path = args.chunks or os.path.join(_ROOT, "corpus", "chunks", f"{args.strategy}.jsonl")

    gold = GoldSet.load(args.gold)
    queries = [q for q in gold.queries
               if args.include_unanswerable or q.query_type != "unanswerable"]
    if not queries:
        raise SystemExit(f"[build_pool] no queries in {args.gold}. Run "
                         f"`eval.generate --stage queries` first.")

    if args.light:
        fleet = ["bge-small"]
    elif args.models:
        fleet = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        fleet = list(DEFAULT_FLEET)
        if args.with_anchor:
            fleet.append("qwen3-4b")

    dense_keys = [m for m in fleet if m != "bm25"]
    use_bm25 = "bm25" in fleet
    unknown = [m for m in dense_keys if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"[build_pool] unknown models {unknown}; known: {sorted(REGISTRY)}")

    print(f"[build_pool] chunks={os.path.relpath(chunks_path, _ROOT)}  queries={len(queries)}  "
          f"depth={args.depth}\n[build_pool] fleet={fleet}"
          f"{' +reranker' if args.with_reranker else ''}")
    chunk_ids, doc_ids, texts = _load_chunks(chunks_path, args.allow_local)
    print(f"[build_pool] loaded {len(chunk_ids)} chunks "
          f"({len(set(doc_ids))} unique docs)")

    per_system: dict[str, dict[str, list[str]]] = {}

    # --- dense retrievers -------------------------------------------------------------
    # Resilient: a model that fails to download/load/encode is skipped with a warning rather
    # than aborting the whole run (and discarding the models that already succeeded). A pool
    # from a diverse subset is still valid — it just has one fewer contributor.
    skipped: list[str] = []
    for key in dense_keys:
        spec = REGISTRY[key]
        print(f"[build_pool] encoding corpus with {spec.name} ({spec.model_name}) ...")
        try:
            index = InMemoryDense.build(spec, doc_ids, texts, device=args.device)
            runs = {q.qid: index.search_docs(q.text, args.depth) for q in queries}
            per_system[spec.name] = runs
            del index  # free the embedding matrix before loading the next model
        except Exception as e:  # noqa: BLE001 — any load/encode failure: skip this model
            skipped.append(spec.name)
            print(f"[build_pool] WARNING: skipping {spec.name} ({spec.model_name}): "
                  f"{type(e).__name__}: {str(e)[:200]}")

    # --- BM25 (highest-diversity contributor) -----------------------------------------
    if use_bm25:
        from manifold.index import bm25 as bm25mod
        print("[build_pool] fitting BM25 over the public chunks ...")
        bm = bm25mod.build(chunk_ids, texts)
        cid_to_doc = dict(zip(chunk_ids, doc_ids))
        runs = {}
        for q in queries:
            hits = bm.search(q.text, k=args.depth * 8)  # over-fetch chunks, collapse to docs
            runs[q.qid] = chunks_to_docs([cid_to_doc[cid] for cid, _ in hits])[:args.depth]
        per_system["bm25"] = runs

    # --- optional cross-encoder over the union ----------------------------------------
    if args.with_reranker:
        from manifold.retrieve.reranker import Reranker
        print(f"[build_pool] reranking the union with {RERANKER_MODEL} ...")
        # representative chunk per doc = its longest chunk (most content for the cross-encoder)
        doc_repr: dict[str, str] = {}
        for did, txt in zip(doc_ids, texts):
            if len(txt) > len(doc_repr.get(did, "")):
                doc_repr[did] = txt
        rr = Reranker(RERANKER_MODEL, device=args.device)
        prelim = union_pool(per_system, args.depth)
        runs = {}
        for q in queries:
            cands = prelim.pool(q.qid)
            if not cands:
                runs[q.qid] = []
                continue
            scores = rr.score(q.text, [doc_repr[d] for d in cands])
            ranked = [d for d, _ in sorted(zip(cands, scores), key=lambda kv: kv[1],
                                           reverse=True)]
            runs[q.qid] = ranked[:args.depth]
        per_system["reranker"] = runs

    if not per_system:
        raise SystemExit("[build_pool] every retriever failed — no pool built. See warnings above.")

    # Cache per-system ranked runs so the judging depth can be re-tuned later without
    # re-encoding the corpus (`--from-runs --depth N`). Encoding is the expensive part.
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump({"depth": args.depth, "systems": sorted(per_system), "runs": per_system}, f)
    print(f"[build_pool] cached per-system runs -> {os.path.relpath(runs_path, _ROOT)} "
          f"(re-depth later with --from-runs)")

    result = union_pool(per_system, args.depth)
    result.save(out_path)
    _report(result, out_path, skipped)


def _report(result, out_path: str, skipped: list[str]) -> None:
    summ = coverage_summary(result)
    print(f"\n[build_pool] pool saved -> {os.path.relpath(out_path, _ROOT)}")
    print(f"  systems used       : {result.systems}"
          + (f"  (SKIPPED: {skipped})" if skipped else ""))
    print(f"  queries pooled     : {summ['num_queries']}")
    print(f"  docs to judge      : {summ['total_pooled_docs']} "
          f"(mean {summ['mean_pool_size']}/query, range {summ['min_pool_size']}"
          f"-{summ['max_pool_size']})")
    print("  unique contribution per system (docs only that system found):")
    for s, n in sorted(summ["unique_contribution_per_system"].items(), key=lambda kv: -kv[1]):
        print(f"    {s:12} {n}")
    print("\nNEXT: judge the pool -> `python -m manifold.eval.generate --stage judge`")


if __name__ == "__main__":
    main()
