"""Metric and gold-set tests — pure, no API or DB needed."""

from __future__ import annotations

import math

from manifold.eval import metrics as M
from manifold.eval.goldset import GoldQuery, GoldSet


def test_chunks_to_docs_dedup_keeps_best_rank():
    assert M.chunks_to_docs(["a", "a", "b", "c", "b"]) == ["a", "b", "c"]


def test_precision_recall_basic():
    ranked = ["d1", "d2", "d3", "d4"]
    qrels = {"d1": 2, "d3": 1}  # 2 relevant total
    assert M.precision_at_k(ranked, qrels, 4) == 0.5      # 2 hits / 4
    assert M.recall_at_k(ranked, qrels, 2) == 0.5         # d1 in top2, d3 not -> 1/2
    assert M.recall_at_k(ranked, qrels, 4) == 1.0         # both found


def test_mrr_first_relevant_rank():
    assert M.reciprocal_rank(["x", "y", "d1"], {"d1": 1}) == 1 / 3
    assert M.reciprocal_rank(["d1"], {"d1": 1}) == 1.0
    assert M.reciprocal_rank(["x", "y"], {"d1": 1}) == 0.0


def test_ndcg_perfect_is_one():
    qrels = {"a": 2, "b": 1}
    assert M.ndcg_at_k(["a", "b"], qrels, 2) == 1.0


def test_ndcg_matches_hand_calc():
    # single relevant doc at rank 2, rel=1: DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    qrels = {"a": 1}
    expected = (2**1 - 1) / math.log2(3)
    assert abs(M.ndcg_at_k(["x", "a", "y"], qrels, 3) - expected) < 1e-9


def test_judged_at_k_counts_pool_coverage():
    # qrels carries a judged-nonrelevant (0) entry: b and d are judged, x is not (a hole).
    qrels = {"b": 2, "d": 0}
    assert M.judged_at_k(["b", "x", "d", "y"], qrels, 4) == 0.5   # b,d judged / 4
    assert M.judged_at_k(["b", "d"], qrels, 4) == 1.0             # both judged
    assert M.judged_at_k([], qrels, 4) == 0.0


def test_bpref_hand_calc():
    # R=2 relevant (r1,r2), N=1 nonrelevant (n1). D=min(2,1)=1.
    qrels = {"r1": 1, "r2": 2, "n1": 0}
    # r1 above n1, r2 below n1 -> (1 + 0)/2 = 0.5
    assert abs(M.bpref(["r1", "n1", "r2"], qrels) - 0.5) < 1e-9
    # all relevant above nonrelevant -> 1.0
    assert M.bpref(["r1", "r2", "n1"], qrels) == 1.0
    # unjudged docs are skipped, not penalized -> still 1.0
    assert M.bpref(["r1", "unjudged", "r2", "n1"], qrels) == 1.0


def test_bpref_no_nonrelevant_is_full_credit():
    qrels = {"r1": 1, "r2": 1}  # N=0
    assert M.bpref(["r1", "r2"], qrels) == 1.0
    assert M.bpref(["r1"], qrels) == 0.5   # only 1 of 2 relevant retrieved


def test_bpref_no_relevant_is_zero():
    assert M.bpref(["a", "b"], {"a": 0}) == 0.0


def test_evaluate_run_reports_bpref_and_judged():
    run = {"q0": ["a", "hole", "b"]}
    qrels = {"q0": {"a": 2, "b": 1, "c": 0}}  # c judged-nonrelevant, not retrieved
    out = M.evaluate_run(run, qrels, ks=(3,))
    assert "bpref" in out and "judged@3" in out
    # a,b judged & retrieved; 'hole' unjudged -> 2/3 judged coverage (rounded to 4dp)
    assert out["judged@3"] == round(2 / 3, 4)


def test_evaluate_run_excludes_unanswerable():
    run = {"q0": ["a", "b"], "u0": ["c"]}
    qrels = {"q0": {"a": 1}, "u0": {}}  # u0 unanswerable
    out = M.evaluate_run(run, qrels, ks=(1, 2))
    assert out["num_queries"] == 1          # only q0 counted
    assert out["recall@1"] == 1.0


def test_goldset_roundtrip(tmp_path):
    gs = GoldSet(
        queries=[
            GoldQuery(qid="q0", text="how to segment OT networks", query_type="lookup",
                      seed_doc_id="nist:1", verified=True),
            GoldQuery(qid="u0", text="how to reset my gmail password",
                      query_type="unanswerable"),
        ],
        qrels={"q0": {"nist:1": 2, "nist:2": 1}, "u0": {}},
    )
    gs.save(str(tmp_path))
    back = GoldSet.load(str(tmp_path))
    assert back.qrels["q0"] == {"nist:1": 2, "nist:2": 1}
    assert back.qrels["u0"] == {}
    assert back.queries[0].verified is True
    assert len(back.answerable()) == 1
    assert len(back.unanswerable()) == 1
