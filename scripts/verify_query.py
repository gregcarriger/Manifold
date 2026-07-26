#!/usr/bin/env python3
"""Record human sign-off on gold-set queries — flips `verified` in queries.jsonl, safely.

The gold-set human-review pass is a **binary per-query sign-off**: a query is accepted
(`verified=true`) ONLY if its drafted relevance labels in qrels.tsv are essentially correct
as-is. You do **NOT** hand-edit the grades in qrels.tsv — editing the labels would fabricate
ground truth and destroy the very signal the benchmark measures (how good the auto-judge is).
If a query's labels are wrong, leave it `verified=false` (rejected → flagged for a later
relabel, not deleted).

This script loads the gold set and re-saves it, so qrels.tsv round-trips **unchanged** — only
the `verified` flag in queries.jsonl moves.

Examples:
    python scripts/verify_query.py --status                 # show verification progress
    python scripts/verify_query.py q000                     # accept q000
    python scripts/verify_query.py q000 q001 u000           # accept several
    python scripts/verify_query.py q002 --reject            # explicitly mark rejected (false)
    python scripts/verify_query.py q000 --show              # accept, then print q000's labels
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.goldset import GoldSet


def _status(gold: GoldSet) -> None:
    ans = [q for q in gold.queries if q.query_type != "unanswerable"]
    una = [q for q in gold.queries if q.query_type == "unanswerable"]
    va = sum(1 for q in ans if q.verified)
    vu = sum(1 for q in una if q.verified)
    print(f"accepted (verified): {va}/{len(ans)} answerable, {vu}/{len(una)} unanswerable "
          f"(target: >=50 answerable)")
    rejected = [q.qid for q in gold.queries if q.reviewed and not q.verified]
    pending = [q.qid for q in gold.queries if not q.reviewed]
    print(f"rejected (needs relabel) ({len(rejected)}): {' '.join(rejected) or '-'}")
    print(f"not yet reviewed ({len(pending)}): {' '.join(pending) or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Flip `verified` on gold-set queries.")
    ap.add_argument("qids", nargs="*", help="query ids to sign off (e.g. q000 q001)")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"))
    ap.add_argument("--reject", action="store_true",
                    help="mark reviewed but NOT accepted (verified=false) — needs later relabel")
    ap.add_argument("--note", default="", help="reason to store on the query (esp. for --reject)")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--show", action="store_true", help="after flipping, print each qid's labels")
    args = ap.parse_args()

    gold = GoldSet.load(args.gold)
    by_qid = {q.qid: q for q in gold.queries}

    if args.status or not args.qids:
        _status(gold)
        return

    accept = not args.reject
    unknown = [q for q in args.qids if q not in by_qid]
    if unknown:
        raise SystemExit(f"[verify] unknown qids: {unknown}")

    for qid in args.qids:
        q = by_qid[qid]
        q.reviewed = True          # a human has now looked at it
        q.verified = accept        # accepted only if labels are correct as-is
        if args.note:
            q.review_note = args.note
        rels = gold.qrels.get(qid, {})
        n_rel = sum(1 for r in rels.values() if r > 0)
        verdict = "ACCEPTED" if accept else "REJECTED (relabel later)"
        print(f"  {qid}: {verdict}  ({n_rel} relevant / {len(rels)} judged)"
              + (f"  note: {args.note}" if args.note else ""))
        if args.show:
            for doc_id, rel in sorted(rels.items(), key=lambda kv: -kv[1]):
                if rel > 0:
                    print(f"      rel={rel}  {doc_id}")

    gold.save(args.gold)  # qrels.tsv round-trips unchanged; only queries.jsonl's verified moves
    print()
    _status(gold)


if __name__ == "__main__":
    main()
