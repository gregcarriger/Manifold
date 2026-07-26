"""Run the retrieval benchmark: score every config in the matrix against the gold set.

The matrix is (chunking strategy × retrieval method × rerank). For each config we retrieve
for every answerable gold query, collapse the chunk ranking to doc-level, and average the
IR metrics. Results print as a table and save to corpus/goldset/results.json.

Run:  python -m manifold.eval.run
      python -m manifold.eval.run --only-verified   # score only human-verified queries
"""

from __future__ import annotations

import argparse
import json
import os

from ..index.bm25 import Bm25Index
from ..index.embedder import MODEL_NAME
from ..retrieve.config import RetrievalConfig
from ..retrieve.retriever import Retriever
from .goldset import GoldSet
from .metrics import chunks_to_docs, evaluate_run

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
GOLD_DIR = os.path.join(_ROOT, "corpus", "goldset")
GOLD_DIR_LOCAL = os.path.join(_ROOT, "corpus", "goldset_local")
BM25_DIR = os.path.join(_ROOT, "corpus", "index", "bm25")
_MODEL_SLUG = MODEL_NAME.split("/")[-1]

# The full comparison matrix.
STRATEGIES = ("fixed", "structure")
METHODS = ("dense", "bm25", "hybrid")

# Sources that may appear in a *published* benchmark. Licensed local-only sources (dragos,
# iec62443) must never be in the index behind a public results table.
PUBLIC_SOURCES = {"nist", "cisa", "mitre"}


def _bm25_extraneous(retriever: Retriever, strategy: str) -> int:
    """Count BM25-indexed chunk_ids absent from `strategy`'s *public* pgvector rows.

    The BM25 pickle is a separate artifact from the table, so rebuilding one does not
    rebuild the other — they have drifted apart in practice. Comparing against the table
    rather than parsing chunk_id prefixes keeps this free of a doc-id -> source mapping
    (`nist-800-82r3:` chunks come from source `nist`, `mitre-ics:` from `mitre`).
    """
    path = os.path.join(BM25_DIR, f"{strategy}__{_MODEL_SLUG}.pkl")
    if not os.path.exists(path):
        return 0  # bm25/hybrid will fail later with a clearer, path-specific error
    indexed = set(Bm25Index.load(path).chunk_ids)
    with retriever.conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM chunks WHERE strategy=%s AND source = ANY(%s)",
                    (strategy, sorted(PUBLIC_SOURCES)))
        public = {r[0] for r in cur.fetchall()}
    return len(indexed - public)


def _assert_public_index(retriever: Retriever, allow_local: bool,
                         strategies: tuple[str, ...] = STRATEGIES) -> None:
    """Guard: refuse to publish numbers computed over a licensed-augmented index.

    Both halves of the index are checked, because either alone can carry licensed text into
    a published row: the shared pgvector `chunks` table (dense, plus the enrichment behind
    bm25/hybrid) and the per-strategy BM25 pickle (bm25/hybrid candidates).

    The check is scoped to the strategies this run actually scores, so a licensed set
    preserved under its own strategy label -- e.g. `structure_local`, the convention this
    project uses -- does not block a public run. Rebuild a contaminated arm from the
    public corpus (`python -m manifold.index.build --strategy all`) or pass --allow-local
    for a private run.
    """
    scored = set(strategies)
    with retriever.conn.cursor() as cur:
        cur.execute("SELECT strategy, source, count(*) FROM chunks GROUP BY 1, 2 ORDER BY 1, 2")
        rows = cur.fetchall()
    contaminating = {f"{st}/{src}": n for st, src, n in rows
                     if src not in PUBLIC_SOURCES and st in scored}
    quarantined = {f"{st}/{src}": n for st, src, n in rows
                   if src not in PUBLIC_SOURCES and st not in scored}
    for strategy in sorted(scored):
        extraneous = _bm25_extraneous(retriever, strategy)
        if extraneous:
            contaminating[f"{strategy}/bm25-index"] = extraneous

    if contaminating and not allow_local:
        raise SystemExit(
            "[run] REFUSING to score: the index contains non-public content in a scored "
            f"strategy {contaminating}.\nA published table must be computed over the PUBLIC "
            "corpus only. Rebuild it:\n    python -m manifold.index.build --strategy all\n"
            "…from a checkout whose corpus/normalized/documents.jsonl has no licensed sources, "
            "or pass --allow-local to run a private (unpublishable) eval."
        )
    if contaminating:
        print(f"[run] WARNING: scoring over a licensed-augmented index {contaminating}; "
              "results are PRIVATE and must not be published.")
    if quarantined:
        print(f"[run] note: licensed content present under non-scored strategies "
              f"{quarantined} — not read by this run.")


def _default_gold_dir() -> str:
    """Pick the gold set to score when --gold isn't given.

    Mirrors the ingest `documents.local.jsonl` convention: a clean public checkout
    has only corpus/goldset, but a licensed-source run also produces the gitignored
    corpus/goldset_local. Prefer the local set only when the public one has no
    queries yet — so public reproduction stays deterministic while local dev "just
    works" without remembering --gold.
    """
    def _has_queries(d: str) -> bool:
        return os.path.exists(os.path.join(d, "queries.jsonl"))

    if _has_queries(GOLD_DIR_LOCAL) and not _has_queries(GOLD_DIR):
        return GOLD_DIR_LOCAL
    return GOLD_DIR


def _run_config(retriever: Retriever, gold: GoldSet, cfg: RetrievalConfig,
                qids: list[str]) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for qid in qids:
        q = next(x for x in gold.queries if x.qid == qid)
        results = retriever.retrieve(q.text, cfg)
        run[qid] = chunks_to_docs([r.doc_id for r in results])
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the retrieval matrix against the gold set.")
    ap.add_argument("--retrieve-k", type=int, default=30,
                    help="chunks fetched before doc-collapse (needs >= k for recall@k)")
    ap.add_argument("--only-verified", action="store_true")
    ap.add_argument("--gold", default=None,
                    help="gold-set dir (default: corpus/goldset, or corpus/goldset_local "
                         "if only that one has queries)")
    ap.add_argument("--allow-local", action="store_true",
                    help="permit scoring over a licensed-augmented index (private run only)")
    args = ap.parse_args()

    gold_dir = args.gold or _default_gold_dir()
    print(f"gold set: {os.path.relpath(gold_dir, _ROOT)}")
    gold = GoldSet.load(gold_dir)
    answerable = gold.answerable()
    if args.only_verified:
        answerable = [q for q in answerable if q.verified]
    qids = [q.qid for q in answerable]
    if not qids:
        raise SystemExit("No answerable queries found. Generate/verify the gold set first.")
    print(f"scoring {len(qids)} answerable queries "
          f"({'verified only' if args.only_verified else 'all drafts'})")

    retriever = Retriever()
    all_results: dict[str, dict] = {}
    try:
        _assert_public_index(retriever, args.allow_local)
        for strategy in STRATEGIES:
            for method in METHODS:
                for rerank in (False, True):
                    cfg = RetrievalConfig(method=method, strategy=strategy, rerank=rerank,
                                          k=args.retrieve_k, candidate_k=args.retrieve_k)
                    label = cfg.label()
                    run = _run_config(retriever, gold, cfg, qids)
                    metrics = evaluate_run(run, gold.qrels)
                    all_results[label] = metrics
    finally:
        retriever.close()

    # Results table, sorted by nDCG@10. bpref + judged@10 sit next to nDCG so a reader can see,
    # per config, whether the pool actually covered what it retrieved (POOLING.md).
    cols = ["ndcg@10", "bpref", "judged@10", "recall@10", "mrr", "precision@5"]
    rows = sorted(all_results.items(), key=lambda kv: kv[1].get("ndcg@10", 0), reverse=True)
    w = max(len(lbl) for lbl in all_results) + 2
    print("\n" + "config".ljust(w) + "".join(c.rjust(12) for c in cols))
    print("-" * (w + 12 * len(cols)))
    for label, m in rows:
        print(label.ljust(w) + "".join(f"{m.get(c, 0):.4f}".rjust(12) for c in cols))

    # Pool-coverage / leave-one-out summary — the evidence the gold set is system-agnostic.
    pool_summary = _pool_report(gold_dir)

    out = {"n_queries": len(qids), "verified_only": args.only_verified, "results": all_results}
    if pool_summary:
        out["pool"] = pool_summary
    with open(os.path.join(gold_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(gold_dir, 'results.json')}")


def _pool_report(gold_dir: str) -> dict | None:
    """Print + return the pool coverage / leave-one-out summary if a pool.json is present."""
    pool_path = os.path.join(gold_dir, "pool.json")
    if not os.path.exists(pool_path):
        print("\n[run] no pool.json — this gold set was not built by pooling; bpref/judged@k "
              "read as single-retriever coverage. Public runs should pool (see POOLING.md).")
        return None
    from .pool import PoolResult, coverage_summary

    summ = coverage_summary(PoolResult.load(pool_path))
    print(f"\npool: depth={summ['depth']}  systems={summ['systems']}  "
          f"queries={summ['num_queries']}  docs judged={summ['total_pooled_docs']} "
          f"(mean {summ['mean_pool_size']}/query)")
    print("leave-one-out unique contribution (docs only that system pooled):")
    for s, n in sorted(summ["unique_contribution_per_system"].items(), key=lambda kv: -kv[1]):
        print(f"    {s:12} {n}")
    return summ


if __name__ == "__main__":
    main()
