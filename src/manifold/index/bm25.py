"""BM25 lexical index (rank-bm25 / Okapi BM25).

A real BM25 rather than Postgres ``ts_rank`` — because the whole point of the hybrid-search
comparison is honest lexical scoring on identifier-dense ICS text. The tokenizer is chosen
deliberately: it keeps CVE ids, model numbers, and protocol names intact
(``CVE-2024-38545``, ``1756-L8x``, ``Modbus/TCP`` → single tokens) instead of shattering
them on punctuation, since those exact-match tokens are exactly where lexical beats dense.

The fitted index is small; it is pickled per (strategy, model) so retrieval doesn't refit
on every run.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

# Keep alphanumerics plus the punctuation that binds identifiers together.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Bm25Index:
    chunk_ids: list[str]
    bm25: BM25Okapi

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in ranked]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: str) -> Bm25Index:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(chunk_ids=d["chunk_ids"], bm25=d["bm25"])


def build(chunk_ids: list[str], texts: list[str]) -> Bm25Index:
    corpus = [tokenize(t) for t in texts]
    return Bm25Index(chunk_ids=chunk_ids, bm25=BM25Okapi(corpus))
