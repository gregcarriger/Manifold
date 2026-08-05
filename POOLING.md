# Pooling — building a system-agnostic gold set

**Status: implemented.** The pooling pipeline is built: pure logic + bias-robust metrics in
[`src/manifold/eval/pool.py`](src/manifold/eval/pool.py) and
[`metrics.py`](src/manifold/eval/metrics.py) (unit-tested in `tests/test_pool.py`,
`tests/test_eval.py`), the diverse-fleet builder in
[`scripts/build_pool.py`](scripts/build_pool.py), the staged gold-set drafter
(`eval.generate --stage queries|judge`), and pool-coverage / bpref / leave-one-out reporting
in [`eval/run.py`](src/manifold/eval/run.py). **Not yet run at scale:** no ≥50-query public
pool has been built or human-verified — that is the next step (needs the model downloads +
the judge key). The methodology below is what the code now does.

This note captures how Manifold builds its public retrieval gold set so the qrels are
not biased toward the retriever that generated them. It is the single most important
methodology fix before the benchmark table can be published, because the whole pitch is
*the first public OT/ICS retrieval benchmark* — a biased gold set would quietly invalidate
every number in the results table.

## The problem: pooling bias

Retrieval metrics follow one rule that quietly does a lot of damage: **any document that was
never judged is scored as non-relevant.** Today `eval.generate` fetches candidates with a
single (hybrid) retriever, has the judge grade only those, and then the same hybrid config is
one of the 12 configs being scored. Consequences:

- A relevant document that only *dense* or only *BM25* (or an external model) would surface
  never enters the pool, so it is never judged, so it counts against whichever config finds
  it. Competitors are penalized for finding real answers.
- The pool-builder (hybrid) is structurally advantaged: it can only be scored on documents it
  itself surfaced.

This is a well-known IR failure mode. TREC and BEIR avoid it with **depth-_k_ pooling across
many diverse systems**.

## The fix: depth-_k_ pooling from systems that fail differently

For each gold query:

1. Retrieve the top-_K_ (e.g. `K = 100` at doc level) from **many** retrievers.
2. **Union + dedup** the candidate sets. Manifold judges at the document level, so the union
   is over parent `doc_id`s — clean and cheap.
3. Judge the union (LLM 0/1/2 → human-verify). This is the qrels.
4. Score all 12 configs against that shared, system-agnostic qrels.

**Guiding principle — diversity beats individual quality.** The goal is a *complete* pool, so
you want retrievers whose errors are uncorrelated. Three sizes of the same model (all BGE)
make correlated mistakes and barely widen the pool. Models from different labs, trained on
different data with different objectives, surface genuinely different documents. **BM25 is the
most valuable single contributor** here — lexical retrieval fails completely differently from
every dense model, especially on the CVE/identifier queries this corpus is full of.

## Local model plan (built and run on an M3 / 32 GB Mac, no API key)

You do **not** need to re-index the whole corpus into pgvector per model. The production
`chunks` table fixes the embedding column at 384 dims (`bge-small`), while these models are
768–4096 dims. Pooling only needs candidates for the ~50 gold queries against ~33K chunks —
33K × 1024 float32 ≈ 135 MB in RAM. So pooling is a **one-off script** that encodes the corpus
per model with `sentence-transformers` + in-memory cosine (or FAISS), dumps top-100 doc IDs
per query, and unions them. No schema change, no risk to the production retrieval path.

### Tier 1 — the diversity pool (this is the fix). Pick 4–5 from *different families*:

| Model | Family / lab | Dim | Role in the pool |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | BGE (already have `bge-small`) | 768 | familiar baseline |
| `intfloat/e5-base-v2` | E5 / Microsoft | 768 | different objective → different errors |
| `Alibaba-NLP/gte-large-en-v1.5` | GTE / Alibaba | 1024 | long-context, distinct lineage |
| `nomic-ai/nomic-embed-text-v1.5` | Nomic | 768 | 8k context, Matryoshka, different data |
| `Snowflake/snowflake-arctic-embed-l` | Snowflake | 1024 | retrieval-tuned, another family |
| **BM25** (already indexed) | lexical | — | **highest-diversity contributor** — keep it in |

Each encodes 33K chunks in minutes on MPS and uses < 3 GB.

### Tier 2 — one current-SOTA anchor (the credibility signal). Add one:

| Model | Notes |
|---|---|
| `Qwen/Qwen3-Embedding-4B` | 2025 top-of-MTEB open embedder; ~8 GB; fast enough; best signal-per-cost |
| `intfloat/e5-mistral-7b-instruct` | 7B LLM-embedder, ~14 GB fp16 (fits) or quantized; strong but slow to encode |

### Rerankers — add pool diversity + a signal upgrade

Keep `cross-encoder/ms-marco-MiniLM-L-6-v2`; add **`BAAI/bge-reranker-v2-m3`** and optionally
**`Qwen/Qwen3-Reranker-0.6B`**. A cross-encoder over a *broad* candidate set surfaces
different documents than any bi-encoder, so it earns a place in the pool.

### The judge (defines relevance — use the strongest, not the most diverse)

Diversity is for *retrievers*; the judge sets ground truth, so it should be the best labeler
available. Two defensible paths, both human-verified:

- **Best signal:** Claude Sonnet/Opus as judge (hundreds of cheap 0/1/2 calls), report κ.
- **Free / reproducible (ROADMAP stretch goal):** a local judge on 32 GB via Ollama —
  `qwen2.5:32b` (Q4 ≈ 18–20 GB), `qwen3:30b-a3b` (MoE, fast), or `gpt-oss:20b` (2025 open
  weights). Human-verify regardless — that is the credibility tell, not the model.

## Reporting that proves the bias was handled

Publishing these alongside nDCG@10 is what turns "we have a benchmark" into "we have a
benchmark whose gold set is demonstrably system-agnostic":

1. **Pool coverage / judged fraction** — for each config, what fraction of its top-10 was
   actually judged (the "holes" metric).
2. **Leave-one-out unique contribution** — for each system, how many relevant docs *only it*
   found. This directly demonstrates the pool is not hybrid-biased.
3. **A bias-robust metric next to nDCG** — report **bpref** or **judged@10** (both tolerate
   incomplete judgments) alongside nDCG@10, and note where they disagree.
4. **State the pool depth _K_** and cite the methodology (Cormack et al. 2009 for RRF;
   TREC/BEIR for depth-_k_ pooling).

## What was actually built and judged (state, not plan)

The committed pool is **depth 15 across 3 systems** — `bge-base`, `e5-base`, `bm25` — over 61
answerable queries, mean 34.4 docs/query, yielding **2,075 judgments** (147 rel=2, 851 rel=1,
1,077 explicit rel=0), all graded by **`claude-haiku-4-5`** with adaptive thinking disabled
(Haiku rejects the parameter, so `generate.py` drops it). Full detail:
[`corpus/goldset/PROVENANCE.md`](corpus/goldset/PROVENANCE.md) — re-judging must use the same
model or grades stop being comparable across queries.

**Why 15 and not 50 or 100.** Depth is a cost dial, not a quality knob: every extra pooled doc
is a judge call. Judging runs on a single self-funded API account, and depth 50 exceeded that
budget, so the pool was derived at 15 from
runs cached at 50. This is a documented trade-off, not an oversight — but it has a measured cost,
below. State the depth _K_ whenever results are published.

**Measured cost of shallow pooling.** On q007 (personnel security), `nist-800-82r3:6.2.8`
"Personnel Security" — the single most on-point document in the corpus — was retrieved by *all
three* systems at ranks 28 / 46 / 49 and excluded purely by the depth-15 cutoff.
`nist-800-82r3:6.2.2` (Awareness and Training) sat at bm25 rank 18. A doc that three independent
systems agree on, just outside the cutoff, is the clearest possible signal that _K_ is too small.

**Known recall holes (absent even at depth 50 — documented, not patched).**

| Query | Missing doc | Why it matters |
|---|---|---|
| q007 | `nist-800-82r3:258` (F.7.2 Awareness & Training – AT) | control family for the training clause |
| q007 | `nist-800-82r3:5.3.4` (Regulatory Requirements) | directly answers the query's regulatory clause; cites NERC CIP-005 |
| q075 | `mitre-ics:T0895` (Autorun Image) | states the actual countermeasure ("AutoRun or AutoPlay are disabled in many operating systems configurations") |

**Holes stay holes.** `scripts/relabel_doc.py` refuses to grade a doc the pool never surfaced,
because that fabricates a candidate and breaks pool-coverage accounting — bpref and judged@k
assume the judged set is a subset of the pool. A `T0895` patch was applied during review and then
reverted for this reason. bpref ignores unjudged docs; nDCG treats them as non-relevant, so
report both and note the disagreement.

**Deepening later is cheap.** Judge resume is per-`(query, doc)`, so raising the depth only costs
the newly pooled docs, and existing grades — including human relabels — merge rather than being
overwritten:

```bash
python scripts/build_pool.py --from-runs --depth 30   # free: no re-encoding, runs cached to 50
python -m manifold.eval.generate --stage judge        # judges ONLY docs with no grade yet
```

Cost from 15 → 30 is ~1,968 new judgments (~0.9× the existing volume); 15 → 50 is ~4,537 (~2.1×).
Raising the depth invalidates prior human sign-off on any query that gains docs, so clear
`verified` on those queries and re-review — cheaper at 6 verified than at 50.

## Human verification, and what it catches

Verification is a **binary per-query sign-off** (`scripts/verify_query.py`) plus a *separate*,
audited relabel pass (`scripts/relabel_doc.py` → append-only `relabel_log.tsv`). Progress so far:
**6 of 61 answerable accepted, 3 rejected, 93 grade corrections logged.** Recurring findings:

- **Boilerplate false positives dominate.** An advisory's only tie to a query is often generic
  recommendation text. One vendor sentence — *"Portable computers and removable storage media
  should be carefully scanned for viruses before they are connected to a control system"* —
  appears verbatim in **63 of 3,829 advisories**. `scripts/scan_suspect_labels.py` flags these:
  it truncates CISA's trailing recommendations block by marker, *and* strips any sentence
  repeated across ≥20 advisories (`--min-df`), because vendor boilerplate is *interleaved* with
  substantive text — an ABB advisory puts the generic line in "Mitigating factors", above a real
  "Workarounds" section — so marker truncation alone would discard genuine content. Its
  `DOMAIN_STOP` set also excludes query-scaffolding words (*should, our, access, controls,
  protect*), without which an irrelevant advisory banks free matches and evades the threshold.
- **Judge the query text before the labels, with the seed withheld.** A query generated *from* a
  seed doc can look precise merely by reusing that doc's wording. Two rejections were query
  defects that no relabelling could fix — one asserted a cross-source mapping absent from the
  corpus, another was under-specified. Treat a **diffuse pool as evidence about the query**: if
  three independent systems spray across unrelated sections, the query is vague.
- **Seeds can be under-graded.** One query's seed — the only doc matching every discriminator —
  sat at rel=1 while a generic chapter held rel=2, so a retriever finding the right answer scored
  *below* one finding boilerplate. Check the seed's grade explicitly.
- **Stub seeds make bad queries.** A seed whose entire body is a cross-reference list ("Addresses
  techniques: …") admits no natural question that uniquely targets it. A thin seed is fine when it
  asserts something substantive or names a distinctive entity; it fails when it is pure pointers.

## Judging without an API key (the $0 route)

Relevance grading does not require API credit. Export the pooled candidates, let Claude Code
subagents grade them on the session's own auth, then merge:

```bash
python scripts/cc_judge_export.py --out <scratchpad>/judge_tasks
# subagents read the task files and write <scratchpad>/judge_results/*.json
python scripts/cc_judge_ingest.py --results <scratchpad>/judge_results --tasks <scratchpad>/judge_tasks
```

Same 0/1/2 grades, same `qrels.tsv`, validated against the task files so an invented or dropped
`doc_id` is caught. Only queries with an empty qrels row are exported, so it resumes cleanly.
(Note: like the API judge before it was made incremental, the exporter is per-*query*; deepening
the pool for already-judged queries needs the same per-doc treatment.)

**Credential guard.** The API path refuses to run when no `ANTHROPIC_API_KEY` is set but Claude
Code / gateway variables are present, because `anthropic.Anthropic()` would silently bill
whatever account is behind the session and stamp the gold set with judgments that cannot be
reproduced from a clean checkout. That happened once; those judgments were reverted. Opt in
deliberately with
`--allow-session-auth` / `MANIFOLD_ALLOW_SESSION_AUTH=1`, or prefer the route above. See
[`src/manifold/llm.py`](src/manifold/llm.py).

## Recommended sequence (now the actual commands)

```bash
# 1. Draft queries only (no judging yet). Needs ANTHROPIC_API_KEY; no DB.
python -m manifold.eval.generate --stage queries --n-answerable 60 --n-unanswerable 15

# 2. Pool candidates across the diverse fleet + BM25 (+ optional anchor/reranker).
#    Reads the PUBLIC chunk set in-memory; writes corpus/goldset/pool.json. No key, no DB.
python scripts/build_pool.py --with-anchor --with-reranker

# 3. Judge the pooled union 0/1/2 (stores rel=0 explicitly), emit REVIEW.md. Needs key.
python -m manifold.eval.generate --stage judge

# 4. Human-verify REVIEW.md -> correct qrels.tsv -> flip verified=true per query (>=50).
#    Then score the 12-config matrix (guards against a licensed-augmented index):
python -m manifold.index.build --strategy all      # rebuild index from the PUBLIC corpus first
python -m manifold.eval.run --only-verified        # prints nDCG + bpref + judged@10 + pool LOO
```

Notes:
- **`scripts/build_pool.py` refuses a `*_local` chunk path** (needs `--allow-local`), and
  **`eval.run` refuses to score if the pgvector index contains non-public sources** (dragos /
  iec62443) unless `--allow-local`. Both guard the public artifact against contamination.
- The pooling retrievers only *build* the gold set — they are **not** published benchmark
  configs, so adding them costs nothing in scope and only widens coverage.
- Fleet is configurable: `--models e5-base,gte-large,bm25`, `--light` (bge-small only, for a
  smoke test), `--with-anchor` (Qwen3-Embedding-4B), `--with-reranker` (bge-reranker-v2-m3).
