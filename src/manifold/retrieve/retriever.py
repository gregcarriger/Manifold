"""The retrieval API: one ``retrieve(query, config)`` entry point behind every config.

Pipeline per method:
    dense  -> pgvector exact cosine top candidate_k
    bm25   -> BM25 top candidate_k (enriched from pgvector)
    hybrid -> dense list + bm25 list fused by RRF
then (optionally) a cross-encoder reranks the candidates, and the top ``k`` are returned.

Heavy resources (embedder, BM25 indexes, reranker) are lazy and cached on the instance, so
the Phase-4 benchmark can reuse one Retriever across a whole config sweep.
"""

from __future__ import annotations

import os

from ..index import bm25 as bm25mod
from ..index import store
from ..index.embedder import Embedder
from .config import RetrievalConfig, RetrievalResult
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
BM25_DIR = os.path.join(_ROOT, "corpus", "index", "bm25")


def _model_slug(model: str) -> str:
    return model.split("/")[-1]


class Retriever:
    def __init__(self, conn=None, embedder: Embedder | None = None):
        self.conn = conn or store.connect()
        self.embedder = embedder or Embedder()
        self._bm25: dict[tuple[str, str], bm25mod.Bm25Index] = {}
        self._rerankers: dict[str, Reranker] = {}

    # -- lazy resource loaders -------------------------------------------------
    def _bm25_index(self, strategy: str, model: str) -> bm25mod.Bm25Index:
        key = (strategy, model)
        if key not in self._bm25:
            path = os.path.join(BM25_DIR, f"{strategy}__{_model_slug(model)}.pkl")
            self._bm25[key] = bm25mod.Bm25Index.load(path)
        return self._bm25[key]

    def _reranker(self, model_name: str) -> Reranker:
        if model_name not in self._rerankers:
            self._rerankers[model_name] = Reranker(model_name)
        return self._rerankers[model_name]

    # -- per-method candidate generation --------------------------------------
    def _dense(self, query: str, cfg: RetrievalConfig, n: int):
        qv = self.embedder.embed_query(query)
        rows = store.search(self.conn, cfg.strategy, cfg.model, qv, k=n)
        return rows  # ordered; each row has cosine_sim

    def _bm25_search(self, query: str, cfg: RetrievalConfig, n: int):
        return self._bm25_index(cfg.strategy, cfg.model).search(query, k=n)  # [(id, score)]

    # -- public API ------------------------------------------------------------
    def retrieve(self, query: str, cfg: RetrievalConfig | None = None) -> list[RetrievalResult]:
        cfg = cfg or RetrievalConfig()
        # over-fetch when a reranker will re-order the candidates
        n = max(cfg.candidate_k, cfg.k) if (cfg.rerank or cfg.method == "hybrid") else cfg.k

        component: dict[str, dict] = {}  # chunk_id -> component scores
        if cfg.method == "dense":
            dense = self._dense(query, cfg, n)
            ordered_ids = [r["chunk_id"] for r in dense]
            for r in dense:
                component[r["chunk_id"]] = {"dense": r["cosine_sim"]}
        elif cfg.method == "bm25":
            hits = self._bm25_search(query, cfg, n)
            ordered_ids = [cid for cid, _ in hits]
            for cid, s in hits:
                component[cid] = {"bm25": s}
        elif cfg.method == "hybrid":
            dense = self._dense(query, cfg, cfg.candidate_k)
            hits = self._bm25_search(query, cfg, cfg.candidate_k)
            for r in dense:
                component.setdefault(r["chunk_id"], {})["dense"] = r["cosine_sim"]
            for cid, s in hits:
                component.setdefault(cid, {})["bm25"] = s
            fused = reciprocal_rank_fusion(
                [[r["chunk_id"] for r in dense], [cid for cid, _ in hits]],
                weights=[cfg.dense_weight, cfg.bm25_weight],
                rrf_k=cfg.rrf_k,
            )
            ordered_ids = [cid for cid, _ in fused]
            for cid, s in fused:
                component[cid]["rrf"] = s
        else:
            raise ValueError(f"unknown method: {cfg.method}")

        # enrich to full rows (dense already has them, but fetch uniformly for simplicity)
        rows = store.fetch_by_ids(self.conn, cfg.strategy, cfg.model, ordered_ids)

        if cfg.rerank:
            cand_ids = ordered_ids[: cfg.candidate_k]
            scores = self._reranker(cfg.reranker_model).score(
                query, [rows[cid]["text"] for cid in cand_ids if cid in rows]
            )
            valid = [cid for cid in cand_ids if cid in rows]
            for cid, sc in zip(valid, scores):
                component[cid]["rerank"] = sc
            ordered_ids = [cid for cid, _ in sorted(
                zip(valid, scores), key=lambda kv: kv[1], reverse=True)]

        # assemble final ranked results
        results: list[RetrievalResult] = []
        for i, cid in enumerate(ordered_ids[: cfg.k], start=1):
            row = rows.get(cid)
            if not row:
                continue
            comp = component.get(cid, {})
            primary = comp.get("rerank", comp.get("rrf", comp.get("dense", comp.get("bm25", 0.0))))
            results.append(RetrievalResult(
                rank=i, chunk_id=cid, doc_id=row["doc_id"], source=row["source"],
                doc_type=row["doc_type"], title=row["title"] or "", url=row["url"] or "",
                text=row["text"], score=float(primary), scores=comp,
            ))
        return results

    def close(self) -> None:
        self.conn.close()
