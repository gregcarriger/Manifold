#!/usr/bin/env python3
"""Export pooled candidates for unjudged queries as task files a Claude Code subagent can grade.

This is the free judge route: instead of spending Anthropic API credits, the relevance grading
is done by Claude Code subagents (which run on the session's own auth, no API key). This
script writes one JSON task file per unjudged answerable query — {query, candidates
with title+snippet} — for an agent to read; the agent writes a results file that
``cc_judge_ingest.py`` merges back into ``qrels.tsv``. Same 0/1/2 grades, same output, $0 API.

Only queries with an EMPTY qrels row are exported, so this resumes cleanly alongside any
queries already judged (by the API path or a prior agent run).

Run:  python scripts/cc_judge_export.py            # -> <scratchpad>/judge_tasks/*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.goldset import GoldSet
from manifold.eval.pool import PoolResult
from manifold.schema import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Export unjudged-query candidates for CC judging.")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"))
    ap.add_argument("--normalized", default=os.path.join(_ROOT, "corpus", "normalized",
                                                         "documents.jsonl"))
    ap.add_argument("--out", required=True, help="task-file output dir (use the scratchpad)")
    ap.add_argument("--snippet", type=int, default=900, help="chars of doc text per candidate")
    args = ap.parse_args()

    gold = GoldSet.load(args.gold)
    pool = PoolResult.load(os.path.join(args.gold, "pool.json"))
    by_id = {d.doc_id: d for d in read_jsonl(args.normalized)}
    os.makedirs(args.out, exist_ok=True)

    unjudged = [q for q in gold.queries
                if q.query_type != "unanswerable" and not gold.qrels.get(q.qid)]
    n_docs = 0
    for q in unjudged:
        cand = set(pool.pool(q.qid))
        if q.seed_doc_id:
            cand.add(q.seed_doc_id)
        candidates = []
        for doc_id in sorted(cand):
            d = by_id.get(doc_id)
            if not d:
                continue
            text = d.text[:args.snippet] + (" …" if len(d.text) > args.snippet else "")
            candidates.append({"doc_id": doc_id, "title": d.title, "source": d.source,
                               "snippet": text})
        n_docs += len(candidates)
        with open(os.path.join(args.out, f"{q.qid}.json"), "w", encoding="utf-8") as f:
            json.dump({"qid": q.qid, "query": q.text, "query_type": q.query_type,
                       "candidates": candidates}, f, ensure_ascii=False, indent=2)

    print(f"[cc_judge_export] {len(unjudged)} unjudged queries, {n_docs} candidate docs "
          f"-> {args.out}")
    print("queries:", ", ".join(q.qid for q in unjudged))


if __name__ == "__main__":
    main()
