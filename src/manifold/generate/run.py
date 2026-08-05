"""Batch grounded generation over the gold set → answers.jsonl.

Produces one grounded, cited answer per gold-set query so Phase 6 (safety-weighted
groundedness) has a fixed answer set to score. Also prints an aggregate grounding report,
including whether the model correctly abstains on the unanswerable queries — a cheap but
meaningful sanity check of the grounding discipline.

Run:  python -m manifold.generate.run              # all gold queries
      python -m manifold.generate.run --only-verified
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

from ..eval.goldset import GoldSet
from ..retrieve.config import RetrievalConfig
from ..retrieve.retriever import Retriever
from .answer import MODEL, generate_answer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
GOLD_DIR = os.path.join(_ROOT, "corpus", "goldset")
OUT_DIR = os.path.join(_ROOT, "corpus", "answers")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate grounded answers for every gold query.")
    ap.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    ap.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--only-verified", action="store_true")
    ap.add_argument("--gold", default=GOLD_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    import anthropic

    gold = GoldSet.load(args.gold)
    queries = gold.queries
    if args.only_verified:
        queries = [q for q in queries if q.verified]
    if not queries:
        raise SystemExit("No gold queries found. Generate/verify the gold set first.")

    cfg = RetrievalConfig(method=args.method, strategy=args.strategy,
                          rerank=not args.no_rerank, k=args.k, candidate_k=40)
    retriever = Retriever()
    client = anthropic.Anthropic()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "answers.jsonl")

    records = []
    print(f"[generate] model={MODEL}  {len(queries)} queries  retrieval={cfg.label()}")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for q in queries:
                ans = generate_answer(retriever, client, q.text, cfg, max_context=args.k)
                rec = {"qid": q.qid, "query_type": q.query_type, **ans.to_dict()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append((q, ans))
                flag = "ABSTAINED" if ans.abstained else f"cited={ans.cited_fraction:.0%}"
                print(f"  {q.qid} [{q.query_type:11}] {flag}  {len(ans.uncited_claims)} uncited")
    finally:
        retriever.close()

    # Aggregate grounding report.
    ans_q = [(q, a) for q, a in records if q.query_type != "unanswerable"]
    unans = [(q, a) for q, a in records if q.query_type == "unanswerable"]
    report = {
        "model": MODEL,
        "retrieval": cfg.label(),
        "n_answers": len(records),
        "answerable_mean_cited_fraction":
            round(mean([a.cited_fraction for _, a in ans_q]), 4) if ans_q else None,
        "answerable_with_uncited_claims": sum(1 for _, a in ans_q if a.uncited_claims),
        "unanswerable_correctly_abstained":
            f"{sum(1 for _, a in unans if a.abstained)}/{len(unans)}" if unans else "n/a",
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {len(records)} answers -> {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
