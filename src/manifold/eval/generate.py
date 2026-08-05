"""Draft the gold set with Claude, for human verification afterward.

The public path is three staged steps so the qrels are not biased toward the retriever that
built them (see POOLING.md):

  1. ``--stage queries`` — generate realistic OT-security questions from stratified seed docs
     (plus a batch of unanswerable / out-of-corpus questions). No judging yet.
  2. ``scripts/build_pool.py`` — union top-k candidates across a *diverse* retriever fleet
     into ``pool.json`` (the system-agnostic judged set).
  3. ``--stage judge`` — grade every pooled candidate 0/1/2, storing rel=0 explicitly so
     ``bpref`` / ``judged@k`` can measure pool coverage. Emits ``REVIEW.md`` for humans.

  ``--stage all`` keeps the legacy single-retriever draft (one hybrid retriever generates and
  judges its own candidates). That path bakes in pooling bias, so it is for the licensed,
  local-only Dragos/IEC eval — never the public gold set.

Everything Claude produces is a *draft*: `verified=False` until a human reviews it via the
generated review file. Model defaults to claude-sonnet-5 (override with MANIFOLD_JUDGE_MODEL).

Run:  python -m manifold.eval.generate --stage queries --n-answerable 60 --n-unanswerable 15
      python scripts/build_pool.py
      python -m manifold.eval.generate --stage judge
Requires ANTHROPIC_API_KEY (or an `ant auth login` profile). Stage 'queries'/'judge' need no
DB; the legacy 'all' path needs built indexes (Phase 2).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import defaultdict

from pydantic import BaseModel

from ..llm import client as llm_client
from ..retrieve.config import RetrievalConfig
from ..retrieve.retriever import Retriever
from ..schema import read_jsonl
from .goldset import GoldQuery, GoldSet

# Sonnet 5 by default — near-Opus judgment at ~1/3 the cost, which matters for the
# hundreds of judge calls a draft run makes. Override with MANIFOLD_JUDGE_MODEL, e.g.
# `MANIFOLD_JUDGE_MODEL=claude-opus-4-8` for a final calibration pass on a verified subset.
MODEL = os.environ.get("MANIFOLD_JUDGE_MODEL", "claude-sonnet-5")

# Adaptive thinking sharpens relevance judgments, but not every model supports it
# (e.g. Haiku 4.5 rejects it). Enable it only where available so a cheap judge still runs.
_THINKING_KW = {} if "haiku" in MODEL.lower() else {"thinking": {"type": "adaptive"}}

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
NORMALIZED = os.path.join(_ROOT, "corpus", "normalized", "documents.jsonl")
OUT_DIR = os.path.join(_ROOT, "corpus", "goldset")

_CORPUS_DESC = (
    "OT/ICS (operational technology / industrial control systems) cybersecurity: "
    "NIST SP 800-82r3 (OT security guidance), CISA ICS advisories (vendor product "
    "vulnerabilities, CVEs, remediations), and MITRE ATT&CK for ICS (adversary "
    "techniques, mitigations, assets targeting industrial systems)."
)


class GeneratedQuery(BaseModel):
    query: str
    query_type: str  # lookup | multi_hop | synthesis
    rationale: str


class RelevanceJudgment(BaseModel):
    relevance: int  # 0 not relevant, 1 relevant, 2 highly relevant / directly answers
    reason: str


class UnanswerableBatch(BaseModel):
    questions: list[str]


def _snippet(text: str, limit: int = 1500) -> str:
    return text[:limit] + (" …" if len(text) > limit else "")


def _parse(client, output_format, prompt: str, max_tokens: int = 1024,
           thinking: bool = False, retries: int = 4):
    """Structured `messages.parse` with retry/backoff. Returns the parsed object, or None if
    every attempt failed — so one transient API error (rate limit, a 400 'grammar compilation
    timed out', a network blip) skips a single item instead of killing the whole run."""
    kw = _THINKING_KW if thinking else {}
    last = None
    for attempt in range(retries):
        try:
            resp = client.messages.parse(
                model=MODEL, max_tokens=max_tokens, **kw,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_format,
            )
            return resp.parsed_output
        except Exception as e:  # noqa: BLE001 — retry any API/parse error, then give up
            last = e
            if attempt < retries - 1:
                time.sleep(min(2 ** (attempt + 1), 20))
    print(f"[generate] WARNING: parse failed after {retries} tries "
          f"({type(last).__name__}: {str(last)[:140]}) — skipping this item")
    return None


def _gen_query(client, doc, query_type: str) -> GeneratedQuery | None:
    guidance = {
        "lookup": "a specific factual question answered directly by this document (a CVE "
                  "detail, a control, a single technique)",
        "multi_hop": "a question that requires connecting this document with related ones "
                     "(e.g. a technique and its mitigation, or a vulnerability and the "
                     "guidance that addresses it)",
        "synthesis": "a broader question this document contributes to but does not fully "
                     "answer alone (e.g. defense-in-depth for a class of systems)",
    }[query_type]
    prompt = (
        f"You are building a retrieval benchmark for {_CORPUS_DESC}\n\n"
        f"Here is one source document:\n\nTITLE: {doc.title}\nSOURCE: {doc.source}\n\n"
        f"{_snippet(doc.text)}\n\n"
        f"Write ONE realistic question an OT security practitioner would actually ask, of "
        f"type '{query_type}': {guidance}. The question must be answerable using this "
        f"corpus and must NOT name the document, CVE id, or technique id verbatim if a "
        f"practitioner wouldn't know it in advance. Keep it natural and specific."
    )
    return _parse(client, GeneratedQuery, prompt, max_tokens=1024)


def _judge(client, query: str, doc) -> RelevanceJudgment | None:
    prompt = (
        f"Query: {query!r}\n\n"
        f"Candidate document (TITLE: {doc.title}, SOURCE: {doc.source}):\n"
        f"{_snippet(doc.text)}\n\n"
        "Rate how well this document helps answer the query:\n"
        "2 = directly and substantially answers it\n"
        "1 = relevant / partially helpful\n"
        "0 = not relevant\n"
        "Judge only on the content shown."
    )
    return _parse(client, RelevanceJudgment, prompt, max_tokens=1024, thinking=True)


def _gen_unanswerable(client, n: int) -> list[str]:
    prompt = (
        f"This corpus covers ONLY: {_CORPUS_DESC}\n\n"
        f"Write {n} plausible-sounding questions that a user might ask but that this corpus "
        f"CANNOT answer — either adjacent-but-out-of-scope (IT security, general cloud, "
        f"specific products/CVEs not in ICS advisories) or requiring info not present. "
        f"They should look answerable at a glance. Return the list."
    )
    out = _parse(client, UnanswerableBatch, prompt, max_tokens=2048)
    return out.questions if out else []


def _stratified_seeds(docs: list, n: int) -> list[tuple]:
    """Pick n (doc, query_type) seeds spread across sources and query types."""
    by_source: dict[str, list] = defaultdict(list)
    for d in docs:
        if d.token_estimate >= 60:  # skip near-empty stubs
            by_source[d.source].append(d)
    types = ["lookup", "lookup", "multi_hop", "synthesis"]  # weight toward lookups
    # iec62443/dragos included only when present locally (they never ship publicly, so
    # any query they seed belongs to a private, separately-reported gold set).
    sources = [s for s in ("cisa", "nist", "mitre", "iec62443", "dragos") if by_source[s]]
    seeds = []
    for i in range(n):
        src = sources[i % len(sources)]
        seeds.append((random.choice(by_source[src]), types[i % len(types)]))
    return seeds


def _stage_queries(client, args, docs) -> None:
    """Generate queries only (no judging) — the input to depth-k pooling."""
    gold = GoldSet()
    seeds = _stratified_seeds(docs, args.n_answerable)
    for i, (seed_doc, qtype) in enumerate(seeds):
        gq = _gen_query(client, seed_doc, qtype)
        if not gq:
            continue
        qid = f"q{i:03d}"
        gold.queries.append(GoldQuery(qid=qid, text=gq.query, query_type=gq.query_type,
                                      seed_doc_id=seed_doc.doc_id, note=gq.rationale))
        gold.qrels[qid] = {}  # judged later, after pooling
        print(f"  {qid} [{gq.query_type:9}] {gq.query[:70]}")
    for j, q in enumerate(_gen_unanswerable(client, args.n_unanswerable)):
        qid = f"u{j:03d}"
        gold.queries.append(GoldQuery(qid=qid, text=q, query_type="unanswerable"))
        gold.qrels[qid] = {}
        print(f"  {qid} [unanswerable] {q[:70]}")
    gold.save(args.out)
    print(f"\n[generate] {len(gold.queries)} queries saved to {args.out}")
    print("NEXT: build the candidate pool -> `python scripts/build_pool.py`, "
          "then `--stage judge`.")


def _stage_judge(client, args, by_id) -> None:
    """Judge the pooled candidate union — the bias-robust qrels (stores rel=0 explicitly)."""
    from .pool import PoolResult

    gold = GoldSet.load(args.out)
    pool_path = args.pool or os.path.join(args.out, "pool.json")
    if not os.path.exists(pool_path):
        raise SystemExit(f"[generate] no pool at {pool_path}. Run scripts/build_pool.py first "
                         "(pooling must precede judging — see POOLING.md).")
    pool = PoolResult.load(pool_path)
    print(f"[generate] judging pool {os.path.basename(pool_path)} (depth={pool.depth}, "
          f"systems={pool.systems}) with {MODEL}")

    # Resume is per-(query, doc), not per-query: any doc that already carries a grade is never
    # re-judged. Two consequences that matter —
    #   1. restartable, as before (a transient API failure costs only the unjudged docs);
    #   2. the pool can DEEPEN later. Re-derive at a larger depth (`build_pool.py --from-runs
    #      --depth N`) and re-run this stage: only the newly pooled docs cost tokens. Existing
    #      grades — including human corrections from scripts/relabel_doc.py — are merged, never
    #      overwritten. Pool depth is a cost/coverage dial (see POOLING.md), so growing it
    #      incrementally has to be cheap or it never happens.
    n_existing = sum(len(rels) for rels in gold.qrels.values() if rels)
    if n_existing:
        print(f"[generate] resuming — keeping {n_existing} existing judgments; judging only "
              f"pooled docs that have no grade yet")

    for q in gold.queries:
        if q.query_type == "unanswerable" and q.qid not in pool.contributors:
            gold.qrels.setdefault(q.qid, {})
            continue
        # candidate set = pooled docs, plus the seed doc so the generating doc is never a hole
        cand = set(pool.pool(q.qid))
        if q.seed_doc_id:
            cand.add(q.seed_doc_id)
        rels: dict[str, int] = dict(gold.qrels.get(q.qid) or {})
        todo = sorted(cand - set(rels))
        if not todo:
            continue
        for doc_id in todo:
            doc = by_id.get(doc_id)
            if not doc:
                continue
            j = _judge(client, q.text, doc)
            if j is not None:
                rels[doc_id] = j.relevance  # keep 0s: they are judged-nonrelevant, not holes
        gold.qrels[q.qid] = rels
        gold.save(args.out)  # incremental: persist after each query so a crash never loses work
        n_rel = sum(1 for r in rels.values() if r > 0)
        print(f"  {q.qid} [{q.query_type:9}] +{len(todo):3} new → {len(rels):3} judged, "
              f"{n_rel} relevant | {q.text[:44]}")

    gold.save(args.out)
    write_review_file(gold, by_id, args.out)
    print(f"\n[generate] judged gold set saved to {args.out}")
    print(gold.summary())
    print(f"NEXT: verify {os.path.join(args.out, 'REVIEW.md')}, correct qrels.tsv, "
          "flip verified=true per query. Do NOT edit qrels after verification.")


def _stage_all(client, args, docs, by_id) -> None:
    """Legacy single-retriever draft (local quick path). NOT for the public gold set —
    it bakes in pooling bias (one hybrid retriever's candidates). Kept for the licensed,
    local-only Dragos/IEC eval where a full pool is overkill. Public set: queries→pool→judge."""
    retriever = Retriever()
    gold = GoldSet()
    print(f"[generate] STAGE=all (single-retriever draft; biased — local use only)  "
          f"answerable={args.n_answerable}")
    seeds = _stratified_seeds(docs, args.n_answerable)
    for i, (seed_doc, qtype) in enumerate(seeds):
        gq = _gen_query(client, seed_doc, qtype)
        if not gq:
            continue
        qid = f"q{i:03d}"
        cfg = RetrievalConfig(method="hybrid", strategy=args.strategy, k=args.candidate_k,
                              candidate_k=args.candidate_k)
        results = retriever.retrieve(gq.query, cfg)
        cand_doc_ids = {r.doc_id for r in results} | {seed_doc.doc_id}
        rels: dict[str, int] = {}
        for doc_id in cand_doc_ids:
            doc = by_id.get(doc_id)
            if not doc:
                continue
            j = _judge(client, gq.query, doc)
            if j and j.relevance > 0:
                rels[doc_id] = j.relevance
        if seed_doc.doc_id not in rels:
            rels[seed_doc.doc_id] = 1
        gold.queries.append(GoldQuery(qid=qid, text=gq.query, query_type=gq.query_type,
                                      seed_doc_id=seed_doc.doc_id, note=gq.rationale))
        gold.qrels[qid] = rels
        print(f"  {qid} [{gq.query_type:9}] {len(rels)} relevant  | {gq.query[:60]}")
    for j, q in enumerate(_gen_unanswerable(client, args.n_unanswerable)):
        qid = f"u{j:03d}"
        gold.queries.append(GoldQuery(qid=qid, text=q, query_type="unanswerable"))
        gold.qrels[qid] = {}
        print(f"  {qid} [unanswerable] {q[:60]}")
    retriever.close()
    gold.save(args.out)
    write_review_file(gold, by_id, args.out)
    print(f"\n[generate] draft saved to {args.out}")
    print(gold.summary())


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft the retrieval gold set with Claude.")
    ap.add_argument("--stage", choices=("queries", "judge", "all"), default="queries",
                    help="queries: generate questions only (then pool, then judge). "
                         "judge: label the pooled union (bias-robust, public path). "
                         "all: legacy single-retriever draft (biased — local-only).")
    ap.add_argument("--n-answerable", type=int, default=40)
    ap.add_argument("--n-unanswerable", type=int, default=12)
    ap.add_argument("--strategy", default="structure", help="chunk set for the legacy 'all' path")
    ap.add_argument("--candidate-k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--allow-session-auth", action="store_true",
                    help="permit inherited Claude Code session credentials when no "
                         "ANTHROPIC_API_KEY is set (bills this session's account; prefer "
                         "scripts/cc_judge_export.py for a $0 run)")
    ap.add_argument("--pool", default=None,
                    help="pool.json to judge (default: <out>/pool.json); --stage judge only")
    ap.add_argument("--normalized", default=NORMALIZED,
                    help="corpus to seed/judge from (default public; pass documents.local.jsonl "
                         "for the Dragos-augmented, local-only gold set)")
    args = ap.parse_args()


    random.seed(args.seed)
    docs = list(read_jsonl(args.normalized))
    by_id = {d.doc_id: d for d in docs}
    client = llm_client(args.allow_session_auth)

    if args.stage == "queries":
        _stage_queries(client, args, docs)
    elif args.stage == "judge":
        _stage_judge(client, args, by_id)
    else:
        _stage_all(client, args, docs, by_id)


def write_review_file(gold: GoldSet, by_id: dict, out_dir: str) -> None:
    """Emit a human-readable review doc pairing each query with its judged docs."""
    lines = ["# Gold-set review", "",
             ("Draft generated by Claude. Verify each query and its relevance labels, then "
              "correct `qrels.tsv` and flip `verified` to true in `queries.jsonl`."), ""]
    for q in gold.queries:
        lines.append(f"## {q.qid} — `{q.query_type}`")
        lines.append(f"**Query:** {q.text}")
        if q.note:
            lines.append(f"_Rationale:_ {q.note}")
        rels = gold.qrels.get(q.qid, {})
        relevant = {d: r for d, r in rels.items() if r > 0}
        nonrel = [d for d, r in rels.items() if r <= 0]
        if not relevant:
            lines.append("_Relevant docs:_ (none — unanswerable or all-pooled-nonrelevant)")
        else:
            lines.append("_Relevant docs:_")
            for doc_id, rel in sorted(relevant.items(), key=lambda kv: -kv[1]):
                d = by_id.get(doc_id)
                title = d.title if d else "(missing)"
                lines.append(f"- **rel={rel}** `{doc_id}` — {title}")
        if nonrel:
            # Judged-nonrelevant (pooled but rel=0): scan for a wrongly-dismissed doc.
            lines.append(f"_Judged non-relevant ({len(nonrel)}):_ "
                         + ", ".join(f"`{d}`" for d in sorted(nonrel)))
        lines.append("")
    with open(os.path.join(out_dir, "REVIEW.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
