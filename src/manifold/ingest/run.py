"""Ingestion orchestrator: normalize every source into one JSONL corpus.

Usage:
    python -m manifold.ingest.run                # all sources, default paths
    python -m manifold.ingest.run --source cisa  # one source only

Output: ``corpus/normalized/documents.jsonl`` plus a ``corpus/normalized/stats.json``
summary. This normalized corpus is the single input for every downstream stage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from ..schema import Document, write_jsonl
from . import cisa, dragos, iec62443, mitre, nist

# Repo-root-relative default locations.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
RAW = os.path.join(_ROOT, "corpus", "raw")
OUT_DIR = os.path.join(_ROOT, "corpus", "normalized")

# The committed, public corpus. Local-only sources must never land here.
PUBLIC_OUT = os.path.join(OUT_DIR, "documents.jsonl")
# Gitignored corpus produced when a licensed local-only source contributes.
LOCAL_OUT = os.path.join(OUT_DIR, "documents.local.jsonl")
LOCAL_ONLY_SOURCES = {"iec62443", "dragos"}


def _load_source(name: str) -> list[Document]:
    if name == "nist":
        return list(nist.load(os.path.join(RAW, "nist", "NIST.SP.800-82r3.pdf")))
    if name == "cisa":
        return list(cisa.load(os.path.join(RAW, "cisa")))
    if name == "mitre":
        return list(mitre.load(os.path.join(RAW, "mitre", "ics-attack.json")))
    if name == "iec62443":
        return list(iec62443.load(os.path.join(RAW, "iec62443")))
    if name == "dragos":
        return list(dragos.load(os.path.join(RAW, "dragos")))
    raise ValueError(f"unknown source: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize OT/ICS sources into a JSONL corpus.")
    ap.add_argument(
        "--source",
        choices=["nist", "cisa", "mitre", "iec62443", "dragos", "all"],
        default="all",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output path (default: documents.jsonl, or documents.local.jsonl if a "
        "licensed local-only source contributes — keeps the public corpus clean)",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_sources = ["nist", "cisa", "mitre", "iec62443", "dragos"]
    sources = all_sources if args.source == "all" else [args.source]

    all_docs: list[Document] = []
    per_source: dict[str, int] = {}
    for name in sources:
        if name == "iec62443" and not iec62443.is_available(os.path.join(RAW, "iec62443")):
            print(f"[{name}] no licensed manifest present — skipped (public corpus only)")
            per_source[name] = 0
            continue
        if name == "dragos" and not dragos.is_available(os.path.join(RAW, "dragos")):
            print(f"[{name}] no licensed WorldView data present — skipped (public corpus only)")
            per_source[name] = 0
            continue
        t0 = time.time()
        docs = _load_source(name)
        all_docs.extend(docs)
        per_source[name] = len(docs)
        print(f"[{name}] {len(docs)} documents in {time.time() - t0:.1f}s")

    # Integrity: no duplicate doc_ids.
    ids = Counter(d.doc_id for d in all_docs)
    dupes = {k: v for k, v in ids.items() if v > 1}
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate doc_ids, e.g. {list(dupes.items())[:3]}")

    # Guardrail: if a licensed local-only source contributed and the user didn't force an
    # explicit path, write to the gitignored local corpus so the committed public
    # documents.jsonl can never be silently overwritten with proprietary content.
    out = args.out
    if out is None:
        contributed_local = any(per_source.get(s, 0) > 0 for s in LOCAL_ONLY_SOURCES)
        out = LOCAL_OUT if contributed_local else PUBLIC_OUT
        if contributed_local:
            print(f"NOTE: licensed local-only source present — writing augmented corpus to "
                  f"{os.path.basename(out)} (gitignored), leaving the public corpus untouched.")

    n = write_jsonl(all_docs, out)

    # Corpus statistics.
    tok = [d.token_estimate for d in all_docs]
    by_type = Counter(d.doc_type for d in all_docs)
    stats = {
        "documents": n,
        "per_source": per_source,
        "by_doc_type": dict(by_type),
        "tokens_est_total": sum(tok),
        "tokens_est_mean": round(sum(tok) / len(tok), 1) if tok else 0,
        "tokens_est_min": min(tok) if tok else 0,
        "tokens_est_max": max(tok) if tok else 0,
        "duplicate_doc_ids": len(dupes),
    }
    # Keep stats alongside their corpus so a non-public run never clobbers the public
    # stats.json: public -> stats.json, local-augmented -> stats.local.json, any explicit
    # custom --out -> a sibling <out>.stats.json.
    if out == PUBLIC_OUT:
        stats_path = os.path.join(OUT_DIR, "stats.json")
    elif out == LOCAL_OUT:
        stats_path = os.path.join(OUT_DIR, "stats.local.json")
    else:
        stats_path = os.path.splitext(out)[0] + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote {n} documents -> {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
