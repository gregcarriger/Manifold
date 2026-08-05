"""Reciprocal Rank Fusion (RRF).

RRF combines ranked lists using only rank position, not raw scores — which is exactly what
we want here: dense cosine similarities and BM25 scores live on incomparable scales, so
fusing by rank avoids the brittle score-normalization step that sinks many hybrid setups.

    rrf_score(d) = sum_i  weight_i * 1 / (rrf_k + rank_i(d))

with rank 1-based. Cormack et al. (2009) use rrf_k=60; it damps the influence of very deep
ranks so the head of each list dominates.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists into one ranking. Returns [(id, rrf_score)] desc."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += w * (1.0 / (rrf_k + rank))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
