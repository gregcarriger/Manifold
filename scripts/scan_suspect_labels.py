#!/usr/bin/env python3
"""Flag likely boilerplate over-labels in the drafted qrels — a review aid, not ground truth.

The auto-judge systematically over-labels CISA advisories as rel=1 when the advisory's only
tie to the query is the generic "CISA recommends defense-in-depth / recommended practices"
boilerplate every advisory carries. This script finds those suspects so the human review can
target them instead of hunting one query at a time.

Heuristic (transparent, deliberately simple):
  * Look at CISA advisories graded rel>0 that are NOT the query's seed doc.
  * Split each advisory into its *substantive* head (product / risk / vuln text, up to the
    first boilerplate marker) and discard the generic recommendations tail.
  * Tokenize (identifier-aware) and drop domain-generic stopwords, so only *distinctive* terms
    count. If the query shares ~no distinctive terms with the substantive head, the rel=1 is
    almost certainly a boilerplate match → SUSPECT.

Output is advisory only — it changes nothing. Use it to decide which queries to --reject.

Run:  python scripts/scan_suspect_labels.py
      python scripts/scan_suspect_labels.py --max-overlap 1 --include-rel2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from manifold.eval.goldset import GoldSet
from manifold.index.bm25 import tokenize
from manifold.schema import read_jsonl

# Phrases that mark the start of the generic recommendations tail in CISA advisories.
BOILERPLATE_MARKERS = [
    "cisa reminds organizations",
    "cisa also provides a section for control systems security recommended practices",
    "defense-in-depth",
    "minimize network exposure",
    "locate control system networks",
    "when remote access is required",
    "recommends users take defensive measures",
    "perform proper impact analysis",
    "no known public exploitation",
    "additional mitigation guidance and recommended practices",
]

# Domain-generic terms shared by nearly every OT/ICS security doc — remove so only
# DISTINCTIVE query/doc overlap counts (otherwise "security/system/network" match everything).
DOMAIN_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are", "be", "with",
    "that", "this", "by", "as", "at", "from", "it", "its", "can", "could", "which", "what",
    "how", "their", "they", "these", "use", "used", "using", "provide", "provides",
    "system", "systems", "security", "control", "cyber", "cybersecurity", "cisa", "ics", "ot",
    "network", "networks", "device", "devices", "attack", "attacker", "attacks",
    "vulnerability", "vulnerabilities", "practice", "practices", "recommended", "organization",
    "organizations", "industrial", "operational", "technology", "successful", "exploitation",
    "allow", "impact", "risk", "evaluation", "affected", "product", "products",
    # Query scaffolding: words that phrase almost any security question but carry no topical
    # discrimination. Without these, an irrelevant advisory banks "free" matches on terms like
    # access/controls/should and clears the max-overlap threshold. Measured on q006: the Hitachi
    # boilerplate cluster scored overlap {access, controls, should} = 3 and escaped the filter,
    # while the true answer (Siemens SCALANCE telnet DoS) scored on telnet/siemens/switches.
    "should", "our", "we", "us", "your", "you", "implement", "implementing", "implemented",
    "measure", "measures", "protect", "protecting", "protection", "access", "controls",
    "remote", "service", "services", "align", "aligns", "guideline", "guidelines", "targeting",
    "target", "prevent", "preventing", "ensure", "help", "need", "needs", "consider",
    "specific", "common", "key", "main", "relevant", "appropriate", "additional", "various",
    "apply", "address", "addresses", "reduce", "mitigate", "mitigation", "mitigations",
    "environment", "environments", "guidance", "requirements", "capabilities",
}


def _sentences(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", s.strip()) for s in re.split(r"(?<=[.!?])\s+", text)]


def repeated_sentences(docs, min_df: int) -> set[str]:
    """Sentences repeated across >= min_df advisories — boilerplate by measurement.

    The hand-curated BOILERPLATE_MARKERS above only catch CISA's *trailing* recommendations
    block. Vendor-authored boilerplate (ABB/Hitachi/Siemens/Schneider "recommended practices",
    corporate marketing, CVSS vector strings) is *interleaved* with substantive text — an ABB
    advisory puts the generic removable-media line in "Mitigating factors", ABOVE a real
    "Workarounds" section — so truncating at a marker would discard genuine content. Removing
    high-document-frequency sentences instead is both safer and self-maintaining as the corpus
    grows. Example: "Portable computers and removable storage media should be carefully scanned
    for viruses before they are connected to a control system." appears in 63 advisories and
    made every one of them look relevant to any removable-media query.
    """
    df: Counter[str] = Counter()
    for d in docs:
        if d.source != "cisa":
            continue
        df.update({s for s in _sentences(d.text) if 40 < len(s) < 260})
    return {s for s, n in df.items() if n >= min_df}


def _substantive_terms(text: str, boiler: frozenset[str] = frozenset()) -> set[str]:
    low = text.lower()
    cut = len(text)
    for m in BOILERPLATE_MARKERS:
        i = low.find(m)
        if i != -1:
            cut = min(cut, i)
    head = text[:cut]
    if boiler:
        head = " ".join(s for s in _sentences(head) if s not in boiler)
    return {t for t in tokenize(head) if t not in DOMAIN_STOP and len(t) > 2}


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag suspected boilerplate over-labels.")
    ap.add_argument("--gold", default=os.path.join(_ROOT, "corpus", "goldset"))
    ap.add_argument("--normalized", default=os.path.join(_ROOT, "corpus", "normalized",
                                                         "documents.jsonl"))
    ap.add_argument("--max-overlap", type=int, default=1,
                    help="flag if <= this many distinctive query terms appear in the doc head")
    ap.add_argument("--include-rel2", action="store_true", help="also scrutinize rel=2 labels")
    ap.add_argument("--min-df", type=int, default=20,
                    help="treat a sentence repeated across >= this many CISA advisories as "
                         "boilerplate and strip it before measuring distinctive overlap "
                         "(0 disables; catches vendor boilerplate the marker list misses)")
    ap.add_argument("--emit-batch", metavar="PATH",
                    help="write ONLY the clear-cut (zero-overlap) rel=1 CISA suspects as a "
                         "relabel_doc --batch TSV (demote to 0); skips any query it would leave "
                         "with 0 relevant docs (those need manual review, not auto-demotion)")
    args = ap.parse_args()

    gold = GoldSet.load(args.gold)
    by_id = {d.doc_id: d for d in read_jsonl(args.normalized)}
    boiler = frozenset()
    if args.min_df > 0:
        boiler = frozenset(repeated_sentences(by_id.values(), args.min_df))
        print(f"[scan] {len(boiler)} repeated sentences (df >= {args.min_df}) treated as "
              f"boilerplate and stripped before overlap scoring")
    grades = {1, 2} if args.include_rel2 else {1}

    total_suspect = 0
    queries_hit = 0
    batch_rows: list[str] = []
    gutted: list[str] = []
    for q in gold.queries:
        if q.query_type == "unanswerable":
            continue
        qterms = {t for t in tokenize(q.text) if t not in DOMAIN_STOP and len(t) > 2}
        all_rels = gold.qrels.get(q.qid, {})
        n_relevant = sum(1 for r in all_rels.values() if r > 0)
        suspects = []
        for doc_id, rel in all_rels.items():
            if rel not in grades or doc_id == q.seed_doc_id:
                continue
            d = by_id.get(doc_id)
            if not d or d.source != "cisa":
                continue
            overlap = qterms & _substantive_terms(d.text, boiler)
            if len(overlap) <= args.max_overlap:
                suspects.append((doc_id, rel, d.title, sorted(overlap)))
        if not suspects:
            continue
        queries_hit += 1
        total_suspect += len(suspects)
        flag = " [REVIEWED]" if q.reviewed else ""
        print(f"\n{q.qid} [{q.query_type}]{flag}: {q.text[:75]}")
        for doc_id, rel, title, overlap in sorted(suspects, key=lambda s: len(s[3])):
            ov = ("shared: " + ",".join(overlap)) if overlap else "shared: (none)"
            print(f"    rel={rel}  {doc_id}  — {title[:45]:45}  {ov}")

        if args.emit_batch:
            # only the unambiguous zero-overlap rel=1 cases are auto-demotion candidates
            clear = [s for s in suspects if not s[3] and s[1] == 1]
            if not clear:
                continue
            if n_relevant - len(clear) < 1:
                gutted.append(f"{q.qid} (would drop to 0 relevant; {len(clear)} clear FPs)")
                continue
            for doc_id, _rel, _title, _ov in clear:
                batch_rows.append(f"{q.qid}\t{doc_id}\t0\tboilerplate FP (zero distinctive "
                                  f"overlap; auto-flagged)")

    print(f"\n[scan] {total_suspect} suspect labels across {queries_hit} queries "
          f"(grades {sorted(grades)}, max-overlap {args.max_overlap}).")
    if args.emit_batch:
        with open(args.emit_batch, "w", encoding="utf-8") as f:
            f.write("\n".join(batch_rows) + ("\n" if batch_rows else ""))
        print(f"[scan] wrote {len(batch_rows)} clear-cut demotions -> {args.emit_batch}")
        print("       apply with: python scripts/relabel_doc.py --batch "
              f"{args.emit_batch}")
        if gutted:
            print(f"[scan] SKIPPED {len(gutted)} queries that would drop to 0 relevant "
                  "(review/reject manually, do NOT auto-demote):")
            for g in gutted:
                print(f"         {g}")
    else:
        print("These are CANDIDATES to check in REVIEW.md — the scan changes nothing. "
              "Reject a query with: python scripts/verify_query.py <qid> --reject --note '...'")


if __name__ == "__main__":
    main()
