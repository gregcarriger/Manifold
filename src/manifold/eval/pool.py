"""Depth-_k_ pooling across diverse retrievers — the system-agnostic gold-set builder.

Retrieval metrics score any *unjudged* document as non-relevant, so a gold set is only fair
if its judged set is not tied to one retriever's view of the corpus. TREC/BEIR avoid this by
**depth-_k_ pooling**: run many retrievers that fail differently, union their top-_k_
candidates, and judge that union. See ``POOLING.md`` for the full rationale and model plan.

This module is split into two layers:

* **Pure pooling logic** (``union_pool``, ``PoolResult``, ``leave_one_out``,
  ``coverage_summary``) — no ML/numpy dependency, fully unit-testable. Operates on
  already-doc-level ranked lists per system, so it is agnostic to how candidates were found.
* **In-memory dense retrieval** (``InMemoryDense``) — encodes the corpus with one
  sentence-transformers model and ranks docs by cosine, entirely in RAM. This is how
  ``scripts/build_pool.py`` runs a diverse embedder fleet over the ~33K public chunks without
  re-indexing pgvector per model (the production ``chunks`` table is fixed at 384-dim
  bge-small). numpy / sentence-transformers are imported lazily so importing this module —
  and the pure logic — costs nothing.

Pooling retrievers only *build* the gold set; they are **not** published benchmark configs.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

from .metrics import chunks_to_docs

# ---------------------------------------------------------------------------------------
# Pure pooling logic (no ML dependencies)
# ---------------------------------------------------------------------------------------

# per_system: {system_name: {qid: [doc_id, ...ranked best-first]}}
PerSystem = dict[str, dict[str, list[str]]]


@dataclass
class PoolResult:
    """The judged-candidate universe for a query set, with per-doc provenance.

    ``contributors[qid][doc_id]`` is the sorted list of systems that surfaced that doc within
    depth — the provenance that makes the leave-one-out bias check possible.
    """

    depth: int
    systems: list[str]
    contributors: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def pool(self, qid: str) -> list[str]:
        """Sorted doc_ids to judge for a query (deterministic ordering)."""
        return sorted(self.contributors.get(qid, {}))

    @property
    def qids(self) -> list[str]:
        return sorted(self.contributors)

    # -- persistence -----------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "depth": self.depth,
            "systems": self.systems,
            "queries": self.contributors,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> PoolResult:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(depth=d["depth"], systems=d["systems"], contributors=d["queries"])


def union_pool(per_system: PerSystem, depth: int) -> PoolResult:
    """Union each system's top-``depth`` doc_ids per query, tracking which systems found each.

    Input lists are assumed already collapsed to doc level (best chunk rank per doc); use
    ``metrics.chunks_to_docs`` before calling if you have chunk-level rankings.
    """
    systems = sorted(per_system)
    all_qids = sorted({q for sysruns in per_system.values() for q in sysruns})
    contributors: dict[str, dict[str, list[str]]] = {}
    for qid in all_qids:
        contrib: dict[str, list[str]] = {}
        for sysname in systems:
            for doc_id in per_system[sysname].get(qid, [])[:depth]:
                who = contrib.setdefault(doc_id, [])
                if sysname not in who:
                    who.append(sysname)
        contributors[qid] = contrib
    return PoolResult(depth=depth, systems=systems, contributors=contributors)


def leave_one_out(result: PoolResult) -> dict[str, int]:
    """Count docs that *only* one system contributed (summed across queries).

    A large count for a system means it pulled in relevant candidates no one else did — direct
    evidence the pool is not biased toward any single retriever (e.g. hybrid). A near-zero
    count means that system is redundant for pooling.
    """
    counts: dict[str, int] = {s: 0 for s in result.systems}
    for contrib in result.contributors.values():
        for who in contrib.values():
            if len(who) == 1:
                counts[who[0]] += 1
    return counts


def coverage_summary(result: PoolResult) -> dict:
    """Aggregate pool statistics for the reporting section."""
    sizes = [len(result.contributors[q]) for q in result.qids]
    per_system = Counter()
    for contrib in result.contributors.values():
        for who in contrib.values():
            for s in who:
                per_system[s] += 1
    n = len(sizes)
    return {
        "depth": result.depth,
        "systems": result.systems,
        "num_queries": n,
        "total_pooled_docs": sum(sizes),
        "mean_pool_size": round(sum(sizes) / n, 2) if n else 0.0,
        "min_pool_size": min(sizes) if sizes else 0,
        "max_pool_size": max(sizes) if sizes else 0,
        "docs_contributed_per_system": dict(per_system),
        "unique_contribution_per_system": leave_one_out(result),
    }


# ---------------------------------------------------------------------------------------
# In-memory dense retrieval (one model, no pgvector) — used by scripts/build_pool.py
# ---------------------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """A pooling retriever's identity + the asymmetric prefixes its family expects.

    Query/passage prefixes matter for retrieval quality and differ by family (bge uses a query
    instruction, e5/nomic use ``query:``/``passage:`` style markers, gte/arctic use none). They
    are declared per model in ``scripts/build_pool.py`` so the pool reflects each model at its
    intended operating point rather than a lowest-common-denominator encoding.
    """

    name: str          # short pool label, e.g. "e5-base"
    model_name: str     # HF id, e.g. "intfloat/e5-base-v2"
    query_prefix: str = ""
    passage_prefix: str = ""
    trust_remote_code: bool = False


class InMemoryDense:
    """Encode a corpus with one model and rank docs by exact cosine, all in RAM."""

    def __init__(self, spec: ModelSpec, chunk_doc_ids: list[str], embeddings):
        self.spec = spec
        self.chunk_doc_ids = chunk_doc_ids  # parallel to embeddings rows
        self._emb = embeddings              # (n_chunks, dim), L2-normalized float32
        self._model = None

    @property
    def name(self) -> str:
        return self.spec.name

    @classmethod
    def build(cls, spec: ModelSpec, chunk_doc_ids: list[str], texts: list[str],
              device: str | None = None, batch_size: int = 64) -> InMemoryDense:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            spec.model_name, device=device, trust_remote_code=spec.trust_remote_code
        )
        passages = [spec.passage_prefix + t for t in texts] if spec.passage_prefix else texts
        emb = model.encode(
            passages, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        )
        inst = cls(spec, chunk_doc_ids, emb)
        inst._model = model  # reuse for query encoding
        return inst

    def search_docs(self, query: str, k: int, over_fetch: int = 8) -> list[str]:
        """Return the top-``k`` doc_ids for a query (chunk hits collapsed to parent docs)."""
        import numpy as np

        q = (self.spec.query_prefix + query) if self.spec.query_prefix else query
        qv = self._model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._emb @ qv
        # over-fetch chunks so that, after collapsing to docs, we still have >= k distinct docs
        n_chunks = min(len(sims), max(k * over_fetch, k))
        top = np.argpartition(-sims, n_chunks - 1)[:n_chunks]
        top = top[np.argsort(-sims[top])]
        ranked_doc_ids = chunks_to_docs([self.chunk_doc_ids[i] for i in top])
        return ranked_doc_ids[:k]
