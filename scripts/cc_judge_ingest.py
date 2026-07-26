#!/usr/bin/env python3
"""Merge Claude Code subagent judgments back into the gold set's qrels.

Reads result files written by the judging subagents — one JSON per query, shaped
``{"qid": "...", "judgments": {"<doc_id>": 0|1|2, ...}}`` — validates them against the
exported task files (so a doc_id the agent invented or dropped is caught), writes the grades
into ``qrels.tsv`` (keeping rel=0 entries, which the bias-robust metrics need), and regenerates
``REVIEW.md``. Idempotent and resumable: a query already present in qrels is left untouched
unless --overwrite.

Run:  python scripts/cc_judge_ingest.py --results <scratchpad>/judge_results \
                                        --tasks   <scratchpad>/judge_tasks
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.generate import write_review_file
from manifold.eval.goldset import GoldSet
from manifold.schema import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest CC subagent judgments into qrels.")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"))
    ap.add_argument("--normalized", default=os.path.join(_ROOT, "corpus", "normalized",
                                                         "documents.jsonl"))
    ap.add_argument("--results", required=True, help="dir of per-query result JSON files")
    ap.add_argument("--tasks", required=True, help="dir of the exported task files (for checks)")
    ap.add_argument("--overwrite", action="store_true", help="re-ingest already-judged queries")
    args = ap.parse_args()

    gold = GoldSet.load(args.gold)
    by_id = {d.doc_id: d for d in read_jsonl(args.normalized)}

    ingested = 0
    skipped_existing = 0
    for fn in sorted(os.listdir(args.results)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(args.results, fn), encoding="utf-8") as f:
            res = json.load(f)
        qid = res.get("qid") or fn[:-5]
        if gold.qrels.get(qid) and not args.overwrite:
            skipped_existing += 1
            continue
        # cross-check against the exported candidate set for this query
        task_path = os.path.join(args.tasks, f"{qid}.json")
        allowed = None
        if os.path.exists(task_path):
            with open(task_path, encoding="utf-8") as f:
                allowed = {c["doc_id"] for c in json.load(f)["candidates"]}
        rels: dict[str, int] = {}
        for doc_id, rel in res.get("judgments", {}).items():
            if allowed is not None and doc_id not in allowed:
                print(f"  [warn] {qid}: ignoring unknown doc_id {doc_id}")
                continue
            try:
                r = int(rel)
            except (TypeError, ValueError):
                print(f"  [warn] {qid}: non-int grade {rel!r} for {doc_id}, treating as 0")
                r = 0
            rels[doc_id] = max(0, min(2, r))
        gold.qrels[qid] = rels
        ingested += 1
        n_rel = sum(1 for r in rels.values() if r > 0)
        print(f"  {qid}: judged {len(rels):3} → {n_rel} relevant")

    gold.save(args.gold)
    write_review_file(gold, by_id, args.gold)
    judged = sum(1 for q in gold.queries
                 if q.query_type != "unanswerable" and gold.qrels.get(q.qid))
    print(f"\n[cc_judge_ingest] ingested {ingested} queries "
          f"({skipped_existing} already judged, left as-is).")
    print(f"[cc_judge_ingest] answerable queries now judged: {judged}")
    print(gold.summary())


if __name__ == "__main__":
    main()
