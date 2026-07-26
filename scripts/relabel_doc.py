#!/usr/bin/env python3
"""Correct a single drafted relevance grade in qrels — the deliberate relabel pass.

The accept/reject sign-off (`verify_query.py`) never touches grades. This tool is the SEPARATE
relabel step: it changes one (qid, doc_id) grade at a time (or a reviewed batch), and appends
every change to ``corpus/goldset/relabel_log.tsv`` so the human corrections are auditable and
reversible — not silent hand-editing. Use it to demote confirmed boilerplate false-positives
(rel=1 → rel=0); the rel=0 entry is kept (judged-nonrelevant), which the bias-robust metrics
(bpref/judged@k) actively want.

Only doc_ids ALREADY judged for that query can be relabeled (you can't invent a new judgment
here — that would be fabricating a candidate the pool never surfaced).

Examples:
    python scripts/relabel_doc.py q041 cisa:ICSA-13-011-03 0 --note "boilerplate FP"
    python scripts/relabel_doc.py --batch relabel.tsv --note "boilerplate demotion pass"
      # relabel.tsv lines:  <qid>\\t<doc_id>\\t<new_rel>   (optional 4th col: per-row note)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.goldset import GoldSet


def _apply(gold: GoldSet, qid: str, doc_id: str, new_rel: int) -> tuple[bool, str]:
    q = next((x for x in gold.queries if x.qid == qid), None)
    if q is None:
        return False, f"unknown qid {qid}"
    rels = gold.qrels.get(qid, {})
    if doc_id not in rels:
        return False, f"{doc_id} was not judged for {qid} (can't relabel an unjudged doc)"
    if new_rel not in (0, 1, 2):
        return False, f"grade must be 0/1/2, got {new_rel}"
    old = rels[doc_id]
    rels[doc_id] = new_rel
    return True, f"{old}->{new_rel}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel drafted qrels grades (audited).")
    ap.add_argument("qid", nargs="?")
    ap.add_argument("doc_id", nargs="?")
    ap.add_argument("new_rel", nargs="?", type=int)
    ap.add_argument("--batch", help="TSV of <qid>\\t<doc_id>\\t<rel>[\\t<note>] rows")
    ap.add_argument("--note", default="", help="reason recorded in the audit log")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"))
    args = ap.parse_args()

    gold = GoldSet.load(args.gold)

    edits: list[tuple[str, str, int, str]] = []
    if args.batch:
        with open(args.batch, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or not parts[0].strip():
                    continue
                note = parts[3] if len(parts) > 3 else args.note
                edits.append((parts[0], parts[1], int(parts[2]), note))
    elif args.qid and args.doc_id and args.new_rel is not None:
        edits.append((args.qid, args.doc_id, args.new_rel, args.note))
    else:
        raise SystemExit("give <qid> <doc_id> <new_rel>, or --batch FILE")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    log_path = os.path.join(args.gold, "relabel_log.tsv")
    applied = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for qid, doc_id, new_rel, note in edits:
            ok, msg = _apply(gold, qid, doc_id, new_rel)
            if not ok:
                print(f"  SKIP {qid} {doc_id}: {msg}")
                continue
            old = int(msg.split("->")[0])
            log.write(f"{stamp}\t{qid}\t{doc_id}\t{old}\t{new_rel}\t{note}\n")
            applied += 1
            print(f"  {qid} {doc_id}: rel {msg}")

    gold.save(args.gold)
    print(f"\n[relabel] applied {applied}/{len(edits)} edits -> qrels.tsv "
          f"(audit: {os.path.relpath(log_path, _ROOT)})")


if __name__ == "__main__":
    main()
