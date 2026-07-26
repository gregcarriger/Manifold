# Manifold

**A public retrieval benchmark and grounded-RAG pipeline for OT/ICS security documentation.**

Retrieval-augmented generation is only as good as its retrieval — yet in the
operational-technology / industrial-control-systems security domain there is *no public
benchmark that measures retrieval quality at all*. Every cybersecurity LLM benchmark
(CyberMetric, SECURE, CTIBench, …) tests knowledge with multiple-choice questions; none
publish query→relevant-passage judgments over a corpus, and none target OT/ICS. Manifold
builds that missing benchmark on copyright-clean public sources, and ships a working RAG
pipeline whose answer faithfulness is weighted by the safety-consequence of getting OT
guidance wrong.

> Status: **active build — truthseeking, not yet benchmarked.** The full pipeline
> (ingest → chunk → index → retrieve, plus grounded generation and the metrics harness) is
> built and unit-tested. **No benchmark numbers are published yet:** there is no results
> table below the fold — only the harness that will produce one.
>
> A **public gold-set draft now exists and is mid human-verification**: 76 queries
> (30 lookup / 15 multi-hop / 16 synthesis / 15 unanswerable), pooled across a 3-system fleet
> and judged into **2,075 relevance judgments** (147 rel=2, 851 rel=1, 1,077 explicit rel=0).
> Human review is the bottleneck and the honest number is small: **6 of 61 answerable queries
> verified, 3 rejected** (target ≥50). Verification is turning up substantial judge
> over-labelling — 93 logged grade corrections so far — so treat the unverified rows as drafts,
> not ground truth. `eval.run` scores `verified` queries only, so no table can be produced yet.
> Phase 6 (safety-weighted groundedness) is designed but not implemented. See
> [ROADMAP.md](ROADMAP.md) for per-phase status and [POOLING.md](POOLING.md) for gold-set
> methodology, pool depth, and known recall holes.

## The corpus (public + copyright-clean)

| Source | Content | Docs |
|---|---|---|
| **NIST SP 800-82r3** | Guide to OT Security — structure-aware sections (public domain) | 273 |
| **CISA ICS advisories** | CSAF 2.0 machine-readable advisories, 2010–2026 (TLP:WHITE) | 3,829 |
| **MITRE ATT&CK for ICS** | Techniques, mitigations, software, groups, assets (STIX 2.1) | 212 |
| **IEC 62443** | *Optional, licensed, local-only — never committed* | 0 (public build) |

**4,314 normalized documents, ~4.8M tokens.** All sources normalize into one
[`Document`](src/manifold/schema.py) schema and land in `corpus/normalized/documents.jsonl`.

## What makes it different (and where each claim honestly stands)

1. **The first public OT/ICS *retrieval* gold set** — BEIR/TREC-compatible qrels with
   recall@k / precision@k / MRR / nDCG, including unanswerable / out-of-corpus queries.
   *Status: the generator + metrics harness are built and tested. A 76-query public gold set
   is drafted and pooled (2,075 judgments committed), but human verification is only 6/61
   answerable queries in, so no numbers are published yet.*
2. **Safety-weighted, human-calibrated groundedness** — wrong OT guidance is dangerous, so
   faithfulness is weighted by claim severity and calibrated against human labels. *Status:
   Phase 6, designed but not built. Today only a lightweight cited-fraction / abstention
   heuristic exists (see [`generate/answer.py`](src/manifold/generate/answer.py)); there is
   no claim decomposition, no severity taxonomy, and no judge↔human agreement number yet.
   This is the second contribution and it lives **in this repo** (see the note on `judge`
   below), not a separate project.*
3. **Honest configuration comparisons** — ≥2 chunking strategies × dense / BM25 / hybrid /
   rerank, with negative results reported, not hidden. *Status: the 12-config matrix runner
   exists; it produces no numbers until a verified gold set exists to score against.*

> **Truthseeking note.** This README describes the artifact honestly, including what is
> *not* done. Until a verified gold set and a committed `results.json` exist, treat every
> quantitative claim about which retrieval config "wins" as a hypothesis, not a result. The
> two side-by-side query anecdotes below are illustrations locked as regression tests, not
> aggregate evidence.

## Architecture

```mermaid
flowchart LR
    subgraph Sources
      N[NIST SP 800-82r3]
      C[CISA ICS advisories]
      M[MITRE ATT&CK ICS]
      I[IEC 62443 - licensed, optional]
    end
    Sources --> ING[Ingest + normalize<br/>unified Document schema]
    ING --> CH[Chunking<br/>fixed vs structure-aware]
    CH --> IDX[(pgvector + BM25)]
    IDX --> R[Retrieve<br/>dense / sparse / hybrid RRF + rerank]
    R --> BENCH{{Retrieval benchmark<br/>recall@k · nDCG · MRR}}
    R --> GEN[Grounded generation<br/>Claude + citation enforcement]
    GEN --> GND{{Safety-weighted groundedness<br/>judge vs. human}}
```

## Reproduce Phase 0

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
# raw sources are pulled by scripts/pull_corpus.py (kept out of git); then:
python -m manifold.ingest.run
# -> corpus/normalized/documents.jsonl + stats.json
```

## Running the LLM stages (and what they cost)

Ingestion, chunking, embedding (BAAI/bge-small-en-v1.5), and reranking (ms-marco-MiniLM)
all run **locally and free** on Apple Silicon — no API key. Only three stages call an LLM,
each with a `MANIFOLD_*_MODEL` override:

| Stage | Env var | Default |
|---|---|---|
| Gold-set draft + relevance judging (`eval.generate`) | `MANIFOLD_JUDGE_MODEL` | `claude-sonnet-5` |
| Grounded generation (`generate.answer` / `generate.run`) | `MANIFOLD_GEN_MODEL` | `claude-opus-4-8` |

The default mix (Sonnet judge + Opus generation) is roughly **$10–12 for one clean pass**,
~$30–50 with iteration. To run the whole pipeline cheaply — a full pass fits inside a new
account's trial credit — point both at Haiku:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
MANIFOLD_JUDGE_MODEL=claude-haiku-4-5 MANIFOLD_GEN_MODEL=claude-haiku-4-5 \
  python -m manifold.eval.generate --n-answerable 40 --n-unanswerable 12
```

The judge does hundreds of cheap 0/1/2 relevance calls that a human verifies anyway, so
Haiku is a fine choice there; reserve Opus/Sonnet for the visible generation and a final
calibration subset.

**No API key? Judge for $0.** `scripts/cc_judge_export.py` writes one task file per unjudged
query; Claude Code subagents grade them on the session's own auth; `scripts/cc_judge_ingest.py`
validates and merges the grades back into `qrels.tsv`. Same 0/1/2 grades, same output, no spend.

**Credential guard.** The LLM stages refuse to run when no `ANTHROPIC_API_KEY` is set *but*
Claude Code / gateway credentials (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDECODE`)
are present in the environment — because `anthropic.Anthropic()` would silently spend the
account behind the session and stamp the artifact with judgments nobody can reproduce from a
clean checkout. See [`src/manifold/llm.py`](src/manifold/llm.py). Override deliberately with
`--allow-session-auth` (or `MANIFOLD_ALLOW_SESSION_AUTH=1`), which warns loudly; prefer the
free judge route above.

## Related work — and how Manifold stays in its lane

- **[Interlock](https://github.com/gregcarriger/Interlock)** — a red-team harness and
  action-boundary validator that stops an OT/ICS agent from being *talked into an unsafe
  action* (prompt injection, tool-call ALLOW/CONFIRM/DENY, provenance / operating-envelope /
  authorization checks on writes). Interlock guards what an agent **does**. Manifold is the
  complementary side: it measures whether what an agent **retrieves and says** is grounded in
  the corpus. There is no functional overlap — Interlock owns runtime action safety; Manifold
  owns retrieval quality and answer faithfulness. Manifold does **not** implement, and should
  not grow into, prompt-injection defense or a tool-call boundary.
- **`judge` (now merged here, not a separate repo).** The safety-weighted, human-calibrated
  groundedness metric was previously framed as a spin-off LLMOps eval project. It is now a
  first-class part of Manifold — Phase 6, living in this repository. Answer-faithfulness
  scoring is squarely within Manifold's "measure what the RAG system says" mandate, so there
  is no reason to split it out.

## Honest limitations (what a reviewer should know)

These are real, current constraints — documented rather than hidden, per the project's
truthseeking stance. Several are tracked as work items in [ROADMAP.md](ROADMAP.md).

- **No published results.** `corpus/goldset/` now holds a drafted, pooled, judged gold set
  (`queries.jsonl`, `qrels.tsv`, `pool.json`, `pool_runs.json`, `REVIEW.md`, `relabel_log.tsv`),
  but `corpus/answers/` is empty and no `results.json` exists. `eval.run` scores only `verified`
  queries (6 of 61), so the benchmark table this project is *about* has not been produced.
- **The draft judge was `claude-haiku-4-5` with adaptive thinking disabled.** Haiku was a
  deliberate cost choice (judging is self-funded), but `generate.py` drops the
  adaptive-thinking parameter for Haiku because the model rejects it — so all 2,075 draft grades
  came from the cheapest judge without its judgment-sharpening feature, which plausibly explains
  much of the over-labelling below. Provenance, including reverted batches, is recorded in
  [`corpus/goldset/PROVENANCE.md`](corpus/goldset/PROVENANCE.md).
- **The gold set is the bottleneck, and the judge over-labels.** Human review of the drafted
  qrels is finding systematic false positives, chiefly CISA advisories whose only tie to a query
  is boilerplate. One vendor sentence — "Portable computers and removable storage media should
  be carefully scanned for viruses before they are connected to a control system" — appears
  verbatim in **63 advisories** and made every one of them look relevant to any removable-media
  query. `scripts/scan_suspect_labels.py` flags these automatically; corrections go through
  `scripts/relabel_doc.py` into an append-only `relabel_log.tsv` (93 edits so far). Two of nine
  reviewed queries were rejected outright as unanswerable-as-posed rather than relabelled.
- **Gold-set pooling bias — addressed in code and now run, but shallow.** A single-retriever
  gold set lets a document no retriever surfaces stay unjudged (scored non-relevant) and
  structurally favors the retriever that built it. The fix is built and has been executed:
  depth-_k_ pooling across a diverse retriever fleet via `scripts/build_pool.py`, a staged
  `queries → pool → judge` drafter, and bias-robust reporting (bpref, judged@k pool coverage,
  leave-one-out) in `eval.run`. **The committed pool is depth 15 over 3 systems** (bge-base,
  e5-base, bm25), mean 34.4 docs/query — deliberately shallow, because judging cost is bounded
  by a single self-funded API account. Per-system runs are cached to depth 50, so the pool can be
  deepened without re-encoding.
  **This shallowness demonstrably costs recall:** for one query, `nist-800-82r3:6.2.8`
  ("Personnel Security") was ranked by *all three* systems (28/46/49) and excluded purely by the
  depth-15 cutoff. Two further answer-bearing docs are absent even at depth 50. Known holes are
  documented rather than hand-patched — adding a qrel no system retrieved would break pool-
  coverage accounting. Methodology, depth rationale, and the hole list: [POOLING.md](POOLING.md).
- **Doc-level qrels blunt the chunking comparison.** Collapsing chunk rankings to parent docs
  makes one gold set fair across strategies, but means the metric cannot reward a chunk that
  *pinpoints* the right passage over a blob that merely *contains* it — which is the whole
  point of comparing `fixed` vs `structure`.
- **Corpus imbalance.** ~89% of documents are CISA advisories. A "hybrid wins" result could
  be an artifact of advisory/identifier-lookup dominance rather than a general finding;
  report per-query-type breakdowns, not just corpus-wide means.
- **Token counts are a `chars/4` proxy** everywhere. All chunk-size statistics and the
  fixed-vs-structure size comparison rest on this estimate, not a real tokenizer.
- **Reproducibility depends on a frozen corpus.** Raw CISA/MITRE sources are pulled live and
  drift over time; "deterministic rebuild" holds only against a pinned snapshot/hash, which
  is not yet committed.

## License

MIT for the code. Corpus sources retain their own terms (NIST: public domain; CISA:
TLP:WHITE; MITRE ATT&CK: [terms](https://attack.mitre.org/resources/legal-and-branding/terms-of-use/)).
IEC 62443 is never redistributed here.
