"""Cross-encoder reranker.

A bi-encoder (bge-small) scores query and passage independently, which is fast but blunt.
A cross-encoder reads the (query, passage) pair jointly and is far more precise — too slow
to run over the whole corpus, but ideal as a second stage over the top ~50 candidates.

Default: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — small (~80 MB), fast on CPU/MPS, and a
well-established reranking baseline. Swappable for a heavier reranker (e.g. bge-reranker)
via config without touching call sites.
"""

from __future__ import annotations

from functools import cached_property


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None):
        self.model_name = model_name
        self._device = device

    @cached_property
    def _model(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device=self._device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]
