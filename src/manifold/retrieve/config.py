"""Retrieval configuration and result types.

A ``RetrievalConfig`` fully determines a retrieval run, so the Phase-4 benchmark can sweep
the (chunking strategy × method × rerank) matrix just by enumerating configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..index.embedder import MODEL_NAME


@dataclass
class RetrievalConfig:
    method: str = "hybrid"          # 'dense' | 'bm25' | 'hybrid'
    strategy: str = "structure"     # which chunk set: 'fixed' | 'structure'
    k: int = 10                     # final number of results returned

    candidate_k: int = 50           # per-retriever depth before fusion/rerank
    rrf_k: int = 60                 # RRF rank constant
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model: str = MODEL_NAME         # embedding model (part of the index key)

    def label(self) -> str:
        base = f"{self.method}"
        if self.rerank:
            base += "+rerank"
        return f"{base}[{self.strategy}]"


@dataclass
class RetrievalResult:
    rank: int
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    title: str
    url: str
    text: str
    score: float
    scores: dict[str, Any] = field(default_factory=dict)  # component scores (dense/bm25/rrf/rerank)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)
