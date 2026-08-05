"""Information-retrieval metrics, implemented transparently.

These are deliberately hand-written rather than pulled from a library: for a benchmark
artifact, the metric definitions should be inspectable and testable, and this avoids a
fragile native dependency (pytrec_eval) on Python 3.14. The qrels format we emit is still
BEIR/TREC-compatible, so anyone who prefers `pytrec_eval` can run the same gold set through
it and get the same numbers.

All metrics operate on a single query: a ranked list of doc_ids (best first) and a qrels
mapping ``{doc_id: relevance}`` where relevance is a non-negative int (0 = not relevant,
1 = relevant, 2 = highly relevant). Relevance is judged at the **document** level so the
same gold set scores every chunking strategy fairly (retrieved chunks are collapsed to
their parent doc before scoring — see ``chunks_to_docs``).

**Judged vs. unjudged.** A doc *present* in ``qrels`` (any value, including 0) was pooled and
judged; a doc *absent* from ``qrels`` was never judged. This distinction only matters once
the gold set is built by depth-_k_ pooling (see ``POOLING.md``): the pool defines the judged
set, so judged-nonrelevant docs are stored explicitly as ``rel=0``. The rank-based metrics
(precision/recall/nDCG/MRR) treat any non-positive relevance as non-relevant, so they are
unchanged by the presence of ``0`` entries. Two metrics are pool-aware:

* ``judged_at_k`` — the fraction of a config's top-_k_ that was actually judged (the "holes"
  metric). A low value means the pool did not cover what this config retrieves, so its nDCG
  is optimistic-by-omission and should be read with caution.
* ``bpref`` — a rank metric that only uses judged docs, so it is robust to incomplete
  judgments (Buckley & Voorhees, SIGIR 2004). Reported next to nDCG@10; where they disagree,
  the pool has holes for that config.
"""

from __future__ import annotations

import math
from statistics import mean


def chunks_to_docs(ranked_chunk_doc_ids: list[str]) -> list[str]:
    """Collapse a chunk-level ranking to a doc-level ranking, keeping best (first) rank
    per doc and preserving order. A doc's rank is that of its highest-ranked chunk."""
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranked_chunk_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def precision_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for d in topk if qrels.get(d, 0) > 0)
    return hits / k


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    total_relevant = sum(1 for r in qrels.values() if r > 0)
    if total_relevant == 0:
        return 0.0  # undefined for unanswerable queries; caller should exclude these
    topk = ranked[:k]
    hits = sum(1 for d in topk if qrels.get(d, 0) > 0)
    return hits / total_relevant


def reciprocal_rank(ranked: list[str], qrels: dict[str, int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    total = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        rel = qrels.get(d, 0)
        if rel > 0:
            total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    ideal = sorted(qrels.values(), reverse=True)
    idcg = sum((2**rel - 1) / math.log2(i + 1)
               for i, rel in enumerate(ideal[:k], start=1) if rel > 0)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, qrels, k) / idcg


# --- pool-aware (bias-robust) metrics ---------------------------------------------------


def judged_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Fraction of the top-k that was pooled and judged (present in qrels, incl. rel=0).

    This is *pool coverage* for a single config: 1.0 means every doc it surfaced in the top-k
    was judged, so its nDCG rests on complete judgments; a lower value flags "holes" where the
    pool never saw what this config retrieves. Meaningful only when qrels stores judged-
    nonrelevant (0) entries — otherwise every top-k hit that isn't relevant reads as a hole.
    """
    topk = ranked[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in qrels) / len(topk)


def bpref(ranked: list[str], qrels: dict[str, int]) -> float:
    """Binary preference (Buckley & Voorhees 2004) — a rank metric robust to unjudged docs.

    Only *judged* documents (present in ``qrels``) participate: unjudged docs in the ranking
    are skipped entirely, so a config is neither rewarded nor punished for surfacing docs the
    pool never covered. This is the property that makes bpref the honest companion to nDCG on
    a pooled gold set.

    bpref = (1/R) · Σ_r [ 1 − min(n_above_r, D) / D ],  D = min(R, N)

    where R = #judged-relevant, N = #judged-nonrelevant, the sum is over relevant docs that
    appear in the ranking, and ``n_above_r`` is the count of judged-nonrelevant docs ranked
    above r. When N = 0 (no nonrelevant judged) every retrieved relevant doc scores 1.0.
    """
    relevant = {d for d, r in qrels.items() if r > 0}
    nonrel = {d for d, r in qrels.items() if r <= 0}
    R = len(relevant)
    if R == 0:
        return 0.0
    N = len(nonrel)
    denom = min(R, N)
    total = 0.0
    n_above = 0  # judged-nonrelevant seen so far while walking the ranking
    for d in ranked:
        if d in relevant:
            total += 1.0 if denom == 0 else (1.0 - min(n_above, denom) / denom)
        elif d in nonrel:
            n_above += 1
    return total / R


# --- aggregation over a query set -------------------------------------------------------

_DEFAULT_KS = (1, 3, 5, 10)


def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = _DEFAULT_KS,
) -> dict[str, float]:
    """Average metrics over all *answerable* queries (those with ≥1 relevant doc in qrels).

    ``run`` maps qid -> ranked doc_ids. ``qrels`` maps qid -> {doc_id: relevance}.
    Unanswerable queries (empty relevant set) are excluded here and measured separately.
    """
    answerable = [q for q in qrels if any(r > 0 for r in qrels[q].values())]
    if not answerable:
        return {}

    scores: dict[str, list[float]] = {}
    for qid in answerable:
        ranked = run.get(qid, [])
        rel = qrels[qid]
        for k in ks:
            scores.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, rel, k))
            scores.setdefault(f"precision@{k}", []).append(precision_at_k(ranked, rel, k))
            scores.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, rel, k))
            scores.setdefault(f"judged@{k}", []).append(judged_at_k(ranked, rel, k))
        scores.setdefault("mrr", []).append(reciprocal_rank(ranked, rel))
        scores.setdefault("bpref", []).append(bpref(ranked, rel))

    out = {m: round(mean(v), 4) for m, v in scores.items()}
    out["num_queries"] = len(answerable)
    return out
