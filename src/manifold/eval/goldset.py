"""Gold-set data model and BEIR/TREC-compatible persistence.

Files (under corpus/goldset/):
  queries.jsonl   — one GoldQuery per line
  qrels.tsv       — TREC format: `qid  0  doc_id  relevance` (tab-separated, the `0` is the
                    conventional unused iteration column). Empty for unanswerable queries.

Relevance is judged at the document level so one gold set scores every chunking strategy.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# Query types — a spread of retrieval difficulty, plus the credibility tell.
QUERY_TYPES = ("lookup", "multi_hop", "synthesis", "unanswerable")


@dataclass
class GoldQuery:
    qid: str
    text: str
    query_type: str  # one of QUERY_TYPES
    seed_doc_id: str | None = None  # doc the query was generated from (answerable only)
    note: str = ""  # free-text rationale / provenance
    verified: bool = False  # accepted: drafted qrels are correct as-is (scored by eval.run)
    reviewed: bool = False  # a human has looked at it (so reject != not-yet-reviewed)
    review_note: str = ""   # why rejected / what to fix in a later relabel pass

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoldSet:
    queries: list[GoldQuery] = field(default_factory=list)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)  # qid -> {doc_id: rel}

    # -- persistence -----------------------------------------------------------
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(q.to_dict(), ensure_ascii=False) + "\n" for q in self.queries)
        with open(os.path.join(out_dir, "qrels.tsv"), "w", encoding="utf-8") as f:
            for qid, rels in self.qrels.items():
                f.writelines(f"{qid}\t0\t{doc_id}\t{rel}\n" for doc_id, rel in rels.items())

    @classmethod
    def load(cls, out_dir: str) -> GoldSet:
        queries: list[GoldQuery] = []
        qpath = os.path.join(out_dir, "queries.jsonl")
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(GoldQuery(**json.loads(line)))
        qrels: dict[str, dict[str, int]] = {q.qid: {} for q in queries}
        rpath = os.path.join(out_dir, "qrels.tsv")
        if os.path.exists(rpath):
            with open(rpath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 4:
                        qid, _, doc_id, rel = parts
                        qrels.setdefault(qid, {})[doc_id] = int(rel)
        return cls(queries=queries, qrels=qrels)

    # -- convenience -----------------------------------------------------------
    def answerable(self) -> list[GoldQuery]:
        return [q for q in self.queries if any(r > 0 for r in self.qrels.get(q.qid, {}).values())]

    def unanswerable(self) -> list[GoldQuery]:
        return [q for q in self.queries if q.query_type == "unanswerable"]

    def summary(self) -> dict:
        from collections import Counter

        by_type = Counter(q.query_type for q in self.queries)
        return {
            "total_queries": len(self.queries),
            "by_type": dict(by_type),
            "answerable": len(self.answerable()),
            "unanswerable": len(self.unanswerable()),
            "verified": sum(1 for q in self.queries if q.verified),
            "reviewed": sum(1 for q in self.queries if q.reviewed),
            "rejected": sum(1 for q in self.queries if q.reviewed and not q.verified),
            "total_relevance_judgments": sum(len(v) for v in self.qrels.values()),
        }
