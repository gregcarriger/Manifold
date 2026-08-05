"""Dense embeddings via BAAI/bge-small-en-v1.5.

bge-small is chosen for reproducibility: it is small (384-dim, ~130 MB), fully local, and
deterministic, so anyone can rebuild identical vectors without an API key or per-call cost.
It follows the bge convention of prepending a retrieval instruction to *queries* but not to
passages — asymmetric encoding that measurably helps retrieval. Embeddings are L2-normalized
so cosine similarity reduces to an inner product (and pgvector's ``<=>`` cosine distance is
exact).
"""

from __future__ import annotations

from functools import cached_property

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
# The instruction bge recommends for retrieval queries (passages get none).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self._device = device

    @cached_property
    def _model(self):
        # Imported lazily so the package is importable without the ML stack installed.
        from sentence_transformers import SentenceTransformer

        # device=None lets sentence-transformers pick MPS (Apple GPU) / CUDA / CPU.
        return SentenceTransformer(self.model_name, device=self._device)

    @property
    def device(self) -> str:
        return str(self._model.device)

    def embed_passages(self, texts: list[str], batch_size: int = 64):
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
            convert_to_numpy=True,
        )

    def embed_queries(self, texts: list[str], batch_size: int = 64):
        return self._model.encode(
            [QUERY_INSTRUCTION + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_query(self, text: str):
        return self.embed_queries([text])[0]
