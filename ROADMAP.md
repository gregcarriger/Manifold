# Manifold — Product Roadmap

**Product in one sentence:** the first public *retrieval* benchmark for OT/ICS security
documentation, delivered alongside a grounded RAG pipeline whose faithfulness is weighted
by the safety-consequence of getting OT guidance wrong.

**Why it exists:** every existing cybersecurity LLM benchmark (CyberMetric, SECURE,
CTIBench, ...) tests *knowledge* via multiple choice. None publish query→relevant-passage
judgments to measure retrieval quality over a corpus, and none exist for OT/ICS at all.
Manifold fills that gap and doubles as a working, measurable RAG system.

**Product goals (what "done" means at the top level):**
1. A reproducible, BEIR-compatible retrieval benchmark anyone can run and cite.
2. Honest, published numbers: recall@k / precision@k / MRR / nDCG across chunking and
   retrieval configurations — including where hybrid *loses*.
3. A grounded generator that cites its sources and refuses when context is insufficient.
4. A safety-weighted, human-calibrated groundedness metric — built **in this repo** as
   Phase 6 (previously planned as a standalone `judge` project; now merged here, since
   answer-faithfulness scoring is core to Manifold's mandate and shares its corpus, gold
   set, and generation stage).
5. A README that lets a reviewer understand the contribution above the fold.

---

## Phase 0 — Corpus & normalization ✅ DONE

**Goal:** one clean, unified document stream from heterogeneous public sources.

- Sources pulled: NIST SP 800-82r3 (PDF, public domain), CISA ICS advisories (3,829 CSAF
  2.0 JSON, TLP:WHITE), MITRE ATT&CK for ICS (STIX 2.1). IEC 62443 stub for licensed
  local use (never committed).
- Unified `Document` schema (`src/manifold/schema.py`); per-source loaders normalize into
  it (`src/manifold/ingest/`). Structure-aware NIST sectioning via PDF outline; CISA
  boilerplate stripped; MITRE relationships resolved.
- **Result:** 4,314 documents, 0 duplicate ids, ~4.8M tokens, median 537 tok/doc →
  `corpus/normalized/documents.jsonl` + `stats.json`.

**Done when:** ✅ normalized corpus builds deterministically and passes integrity checks.

---

## Phase 1 — Chunking (comparison built in from day one) ✅ DONE

**Goal:** turn documents into retrieval passages via ≥2 deliberately different strategies
so the benchmark can measure which wins on this corpus.

- **`fixed`** — structure-agnostic overlapping token windows (the baseline everyone ships).
- **`structure`** — respects normalized boundaries (CISA per-CVE/`##` blocks, NIST outline
  sections, whole MITRE objects), prepends a context header to each chunk, no overlap;
  tiny blocks merged greedily, oversized blocks split on paragraph/sentence boundaries.
- Shared offset-preserving splitter (`chunk/splitter.py`) so any quality difference is
  attributable to boundary choice, not low-level splitting. `max_tokens` is a hard cap.
- Every `Chunk` keeps `chunk_id`, parent `doc_id`, `char_start/char_end` offsets,
  `section_path`, citable `url`, and carried-forward filter metadata — provenance survives
  chunking (verified: 100% of offsets reconstruct the source body).

**Result:** `fixed` = 11,077 chunks (median 565 tok, 2.6/doc); `structure` = 22,138
chunks (median 180 tok, 5.1/doc). Meaningful difference confirmed: a multi-CVE advisory
is 1 blob under `fixed` but 1 chunk-per-CVE under `structure`. Outputs in
`corpus/chunks/{fixed,structure}.jsonl` + `stats.json`; 6 passing tests in
`tests/test_chunking.py`.

**Done when:** ✅ both strategies emit JSONL + a comparison stats report.

**Note for later:** a semantic/recursive third arm and a real tokenizer (swap `n_tok`)
are easy follow-ons if the benchmark suggests they'd matter.

---

## Phase 2 — Indexing: embeddings + vector store + lexical index ✅ DONE

**Goal:** dense and sparse indexes over each chunk set.

- Vector store: **pgvector 0.8.5** in Docker (`docker-compose.yml`, `pgvector/pgvector:pg17`
  on host port 5433). Only the stateful DB is containerized; the pipeline runs on the host.
- Dense embeddings: **BAAI/bge-small-en-v1.5** (384-dim, local, reproducible) via
  sentence-transformers, MPS-accelerated on Apple Silicon (167 texts/s). Asymmetric
  query/passage encoding, L2-normalized → exact cosine.
- **Exact search by default** (`ORDER BY embedding <=> q`); corpus is small enough that an
  ANN index would only inject approximation error into the retrieval numbers. HNSW is
  provided (`store.create_hnsw`) as the documented scale path.
- Sparse: real **BM25** (`rank-bm25`) with an identifier-preserving tokenizer
  (`CVE-2024-38545`, `1756-L8x`, `Modbus/TCP` stay single tokens) — the crux of the
  acronym/ID hybrid story. Pickled per (strategy, model).
- One `chunks` table keyed by (strategy, model, chunk_id) so both chunk sets and future
  models coexist and stay comparable.

**Result:** `fixed` 11,077 and `structure` 22,138 chunks indexed into pgvector + BM25.
Smoke query confirms dense and sparse surface *different* candidates (the reason hybrid is
worth building). 9 passing tests (incl. pgvector round-trip). Wheel risk resolved: torch
2.13.0 ships cp314 wheels, so no container needed for the Python side.

**Done when:** ✅ `python -m manifold.index.build --strategy all` populates pgvector + BM25,
reports counts/timings, and a smoke query returns sane neighbors.

---

## Phase 3 — Retrieval: dense · sparse · hybrid · rerank ✅ DONE

**Goal:** a retrieval API returning ranked, scored passages under a configurable matrix.

- One `retrieve(query, RetrievalConfig) -> [RetrievalResult]` API (`retrieve/retriever.py`)
  backing every config: `dense` (pgvector exact cosine), `bm25`, `hybrid` (**Reciprocal
  Rank Fusion**, rrf_k=60, weightable), each optionally + cross-encoder **rerank**
  (`ms-marco-MiniLM-L-6-v2`, top-`candidate_k` re-scored). A `RetrievalConfig` fully
  determines a run, so Phase 4 sweeps the matrix by enumerating configs. Resources are
  lazy + cached for reuse across a sweep.
- Side-by-side demo: `python -m manifold.retrieve.demo [query] [--strategy ...]`.

**The acronym/ID failure mode, demonstrated (Phase 4 will quantify it):**
- Query `CVE-2023-38545` → **dense misses entirely** (returns unrelated advisories ~0.77
  cosine); **BM25 surfaces the correct advisory** (ICSA-24-004-01 Rockwell FactoryTalk,
  which carries that CVE). Locked as a regression test.
- Conceptual query ("protect safety instrumented systems") → dense finds NIST 5.3.1; BM25
  finds the real Triton/Triconex SIS attack (MITRE C0030); hybrid boosts the doc in both
  lists (MITRE M0812) to #1. Complementary strengths → fusion helps.

**Done when:** ✅ every config works via one API; demo shows them side by side. 14 passing
tests (RRF logic + DB-gated retriever integration + the CVE regression).

---

## Phase 4 — The benchmark (headline deliverable) ⭐ — harness DONE, gold set drafted, verification IN PROGRESS

**Built (`src/manifold/eval/`):**
- **Metrics harness** (`metrics.py`) — recall@k, precision@k, MRR, nDCG@k, hand-written and
  unit-tested (7 tests, incl. hand-verified nDCG). **Doc-level qrels** so one gold set scores
  every chunking strategy fairly (retrieved chunks collapse to parent docs). No fragile
  native dep; qrels stay BEIR/TREC-compatible.
- **Gold-set model + I/O** (`goldset.py`) — `queries.jsonl` + TREC `qrels.tsv`; query types
  lookup / multi_hop / synthesis / **unanswerable**; `verified` flag for the human pass.
- **Auto-draft generator** (`generate.py`) — Claude (claude-opus-4-8, configurable) generates
  a question per stratified seed doc, then **judges every retrieved candidate** for relevance
  (adaptive thinking) so qrels capture all relevant docs, not just the seed; plus a batch of
  unanswerable/out-of-corpus questions. Emits a `REVIEW.md` for human verification.
- **Benchmark runner** (`run.py`) — scores the full matrix (2 chunking × 3 methods × rerank
  = 12 configs) against the gold set, prints a table sorted by nDCG@10, saves `results.json`.

**Actual state on disk (truthseeking — do not overstate this):**
- The **public gold set is drafted and judged, but barely verified.** `corpus/goldset/` holds
  `queries.jsonl` (76 queries: 30 lookup / 15 multi_hop / 16 synthesis / 15 unanswerable),
  `qrels.tsv` (**2,075 judgments** — 147 rel=2, 851 rel=1, 1,077 explicit rel=0), `pool.json`
  (depth 15, 61 answerable queries, mean 34.4 docs), `pool_runs.json` (per-system runs cached
  to depth 50), `REVIEW.md`, and `relabel_log.tsv`.
- **Human verification: 6 of 61 answerable queries accepted, 3 rejected, 52 untouched**
  (target ≥50). `eval.run` scores `verified` queries only, so there is still **no committed
  `results.json`** — the headline table does not exist. `corpus/answers/` is empty.
- **The 15 `unanswerable` queries are unreviewed and hold no qrels rows** (they are excluded
  from pooling by design). Their review is a different task with an inverted failure mode:
  the risk is a query being *accidentally answerable* from the corpus, which would silently
  break the abstention half of the benchmark.
- Verification is finding **systematic judge over-labelling**, corrected through
  `scripts/relabel_doc.py` into an append-only `relabel_log.tsv` (**93 grade edits** so far,
  e.g. 29 on one synthesis query whose pool was inflated with ~30 topically-adjacent
  advisories). Rejections are recorded, not deleted: `verified=false` + `reviewed=true` + a
  `review_note` explaining why, so "rejected" is distinguishable from "not yet reviewed".
- A **local** draft also remains (`corpus/goldset_local/`, 10 queries on a licensed-source
  corpus). It cannot be published and is not the artifact above.

**What review is actually catching (the gold set is the deliverable, so this matters):**
- **Boilerplate false positives.** A CISA advisory's only tie to a query is often generic
  recommendation text. One vendor sentence appears verbatim in **63 advisories**, making all of
  them look relevant to any removable-media query. `scripts/scan_suspect_labels.py` finds these
  automatically (marker-based truncation *plus* stripping any sentence repeated across ≥20
  advisories, since vendor boilerplate is interleaved with substantive text rather than trailing).
- **Under-graded seeds.** At least one query had its own seed doc — the only document matching
  every query discriminator — sitting at rel=1 while a generic architecture chapter held rel=2.
- **Queries that are unanswerable as posed.** 2 of 3 rejections were query defects, not label
  defects, and no relabelling can fix them: one asserted a cross-source mapping that does not
  exist in the corpus, another was so under-specified that three retrievers produced a diffuse
  30-doc pool. A diffuse pool is now read as *evidence about the query*, not noise to relabel.
- **Seed leakage.** A query generated *from* a seed doc can look precise only because it reuses
  that doc's wording. Queries are therefore judged on their own text first, with the seed
  withheld, before any label is inspected.

**Known methodological risks:**
- **Pooling bias — FIXED IN CODE AND RUN, but the pool is shallow.** The depth-_k_ pipeline is
  built and executed: `eval.generate --stage queries` drafts questions, `scripts/build_pool.py`
  unions top-_k_ doc IDs across a diverse retriever fleet into `pool.json`, and
  `eval.generate --stage judge` labels that union (storing rel=0 explicitly). Bias-robust
  reporting — **bpref**, **judged@k** pool coverage, **leave-one-out** unique contribution — is
  wired into `eval.run`, which also refuses to score over a licensed-augmented index.
  **The committed pool is depth 15 across 3 systems** (bge-base, e5-base, bm25) — not the full
  fleet, and shallower than the TREC convention of ~100. This was a deliberate cost decision:
  judging is bounded by a single self-funded API account, and depth 50 was more than that
  budget allowed.
  **Measured consequence:** `nist-800-82r3:6.2.8` ("Personnel Security") was ranked by *all
  three* systems (28/46/49) yet excluded by the depth-15 cutoff; two further answer-bearing docs
  for that query are missing even at depth 50. Deepening is now cheap to resume — per-system runs
  are cached to depth 50, `build_pool.py --from-runs --depth N` re-derives without re-encoding,
  and judge resume is **per-(query, doc)**, so only newly pooled docs cost tokens while existing
  grades and human relabels merge rather than being overwritten. Cost of moving to depth 30 is
  ~1,968 new judgments (~0.9× the existing volume). Full methodology: [POOLING.md](POOLING.md).
  The legacy single-retriever draft (`--stage all`) is retained only for the local Dragos/IEC
  eval, which it explicitly labels as biased.
- **Recall holes are documented, never hand-patched.** Adding a qrel for a document no system
  retrieved would fabricate a candidate the pool never surfaced (`relabel_doc.py` refuses it by
  design) and would break pool-coverage accounting, since bpref/judged@k assume judged ⊆ pool.
  One such patch was applied during review and then reverted for exactly this reason. Known
  holes are listed in [POOLING.md](POOLING.md) instead.
- **The draft judge was cheap AND had thinking disabled — a likely cause of the over-labelling.**
  All 2,075 drafted judgments came from **`claude-haiku-4-5`** on an explicit API key (recorded in
  `corpus/goldset/PROVENANCE.md`). Haiku was a deliberate cost choice, but `_judge()` requests
  adaptive thinking and `generate.py` silently drops it for Haiku (the model rejects the
  parameter), so the grades came from the cheapest judge *without* its judgment-sharpening
  feature. That is consistent with what review keeps finding: boilerplate FPs, under-graded seed
  docs, generic front matter at rel=2. **Re-judging must stay on `claude-haiku-4-5`** or grades
  become incomparable across queries; a stronger model belongs in a separate calibration pass
  over a verified subset, reporting judge↔human agreement. Stamping provenance *in* the artifact
  (sidecar JSON) rather than by hand is still an open work item. A run once inherited Claude Code
  session credentials instead of an explicit key; those judgments were reverted and a guard added
  ([`src/manifold/llm.py`](src/manifold/llm.py)).
- **Doc-level qrels** can't credit a chunk that pinpoints the answer over a blob that
  contains it — the very distinction the `fixed` vs `structure` comparison is meant to test.
- **Corpus imbalance** (~89% CISA advisories) can make "hybrid wins" an artifact of
  identifier-lookup dominance; report per-query-type breakdowns.
- Precision/recall/MRR binarize relevance (`rel>0`) while only nDCG uses the 0/1/2 grades.

**Remaining — three gates to a published table:**
1. **Human verification (the bottleneck).** 6 of 61 answerable verified; need ≥50. Plus the 15
   `unanswerable` queries, which need answerability confirmation rather than label review.
   `scripts/scan_suspect_labels.py` pre-triages boilerplate FPs; `scripts/verify_query.py`
   records accept/reject; `scripts/relabel_doc.py` handles corrections with an audit log.
   Re-judging needs no API key — use `scripts/cc_judge_export.py` → subagents →
   `scripts/cc_judge_ingest.py` for a $0 pass.
2. **Rebuild the index from the public corpus.** `eval.run` refuses to publish over a
   licensed-augmented index, and the pgvector `structure` set currently holds 3,139 Dragos
   chunks (25,277 total). `fixed` is already clean (11,077, no Dragos). Because the `chunks`
   table keys on `strategy`, load the augmented set under a *distinct* strategy label so the
   private delta study survives instead of being rebuilt twice.
3. **Score it.** `python -m manifold.eval.run --only-verified` → `results.json`, reported with
   pool coverage / leave-one-out / bpref alongside nDCG, and with the pool depth stated.

**Optional but recommended before (1) scales:** deepen the pool to 30. Doing it at 6 verified
costs 6 re-reviews; doing it after hitting 50 costs 50.

**Goal:** the public retrieval gold set + metrics harness that make everything above
*measured*, not asserted.

- Gold set: ≥50 queries with query→relevant-passage judgments (qrels), TREC/BEIR format.
  Mix of lookup, multi-hop, and synthesis queries, **plus unanswerable / out-of-corpus
  queries** (the credibility tell most public sets omit). Queries authored from real OT
  security questions; relevance judged against chunks with documented adjudication rules.
- Metrics: recall@k, precision@k, MRR, nDCG@k via `pytrec_eval` (BEIR-interoperable).
- Run the full matrix (chunking × retrieval config) → a results table, with honest
  negative results reported.
- Publish the qrels + harness so the benchmark is reusable and citable.

**Done when:** `manifold.eval retrieval` reproduces the published table from the committed
gold set; the table lives in the README.

**Risk:** gold-set quality is the whole artifact. Document judging criteria; consider a
second-pass review of relevance labels; keep the set small but rigorous over large but
sloppy.

---

## Phase 5 — Grounded generation ✅ DONE (built ahead; runs when a key is available)

**Goal:** answers built only from retrieved context, with enforced citations.

- **Native Citations, not prompt markers** (`generate/answer.py`) — each retrieved chunk is
  passed as a citations-enabled `document` block, so Claude can only cite the provided
  sources and the API returns the exact cited span + source index per claim. Citation-to-
  source enforcement is done by the platform, not a regex.
- **Grounding scored** — `build_answer()` (pure, unit-tested) computes cited-fraction, flags
  substantive **uncited claims**, maps each citation back to its `chunk_id`/`url`, and detects
  correct **abstention** when the corpus doesn't cover the question.
- **Model:** claude-opus-4-8 (one call per query, so cheap; visible-output quality matters);
  override with `MANIFOLD_GEN_MODEL`. Adaptive thinking on.
- **Batch runner** (`generate/run.py`) — answers every gold query → `corpus/answers/answers.jsonl`
  + aggregate report (mean cited-fraction; unanswerable-correctly-abstained count), the fixed
  answer set Phase 6 scores.

**Done when:** ✅ `python -m manifold.generate.answer "<query>"` returns a cited answer or an
explicit "not supported by corpus", with uncited claims flagged. 5 tests (31 total).
**To run:** needs `ANTHROPIC_API_KEY` + built indexes.

---

## Phase 6 — Safety-weighted, calibrated groundedness (the former `judge`, now merged here)

**Goal:** the second novel contribution — faithfulness scoring that reflects OT risk. This
was once scoped as a separate `judge` repo; it now lives in Manifold as a `manifold.judge`
(or `manifold.groundedness`) module, reusing this project's corpus, gold set, and Phase-5
answer set rather than duplicating them elsewhere.

**Current state: not built.** Today only a lightweight heuristic exists in
`generate/answer.py` — cited-fraction, uncited-claim flagging, and regex-based abstention
detection. None of the below is implemented yet.

- Decompose answers into atomic claims; verify each against retrieved context.
- OT claim taxonomy with severity weights (mis-stating an air-gap/segmentation control or
  a safety-instrumented-system action = critical; background context = minor); domain-
  tiered thresholds. (Consider grounding severity in an existing standard — e.g. MITRE
  ATT&CK for ICS Impact tactics or IEC 62443 SLs — so weights aren't hand-waved.)
- **Judge-vs-human calibration:** hand-label a subset, report judge↔human agreement
  (κ / correlation) with confidence intervals (ARES/PPI-style; validate the evaluator à la
  GroUSE). Reporting this is the tell of someone who has actually run evals.

**Scope boundary (vs. Interlock):** this metric scores *whether an answer is faithful to the
retrieved corpus*, weighted by OT consequence. It is **not** an action-boundary or
prompt-injection defense — that is [Interlock](https://github.com/gregcarriger/Interlock)'s
job, and Manifold should not grow into it. Interlock guards what an agent *does*; Phase 6
measures whether what it *says* is grounded.

**Done when:** a groundedness report runs over Phase-5 answers and publishes the judge↔human
agreement numbers, as a module inside this repository.

---

## Phase 7 — Productize, document, reproduce

**Goal:** a repo a reviewer trusts in five minutes and can run in one command.

- `scripts/pull_corpus.py` (re-pull raw sources) + `make corpus` / `make bench` targets;
  raw dumps stay out of git, everything derivable is scripted.
- CLI surface (`manifold ingest|index|retrieve|eval|answer`) and a minimal demo (small
  API or notebook) — enough to *show* retrieval + citation, not a heavy UI.
- README: two-sentence problem statement above the fold, mermaid architecture diagram,
  **results tables with real numbers**, honest limitations + what's untested, a "what I'd
  do differently at scale" section (the governance voice), cross-link to Interlock (with the
  scope boundary made explicit — Interlock guards actions, Manifold measures retrieval and
  faithfulness), MIT license, meaningful commit history.
- **Put the project under version control.** It is not currently a git repository, so the
  "meaningful commit history" and "clean checkout reproduces …" goals below are unmet at the
  most basic level. `git init` + a real history is a prerequisite for a reviewer to trust it.
- Commit a **frozen corpus manifest / content hash** so "deterministic rebuild" survives the
  live CISA/MITRE sources drifting over time.
- Test suite for the schema, loaders, chunking, fusion, and metrics. (Loaders and the
  `eval.generate` / `generate.run` orchestration are currently untested — note that honestly.)

**Done when:** a clean checkout reproduces the corpus, the index, and the benchmark table
from documented commands.

---

## Stretch goals (post-v1)

- **Local / offline judge backend.** Add a pluggable LLM backend so
  `MANIFOLD_JUDGE_MODEL=ollama:qwen2.5:32b` routes gold-set relevance judging through a
  local Ollama model (JSON-schema structured output) instead of the Anthropic API. The
  judge stage is the token-heavy part — hundreds of 0/1/2 relevance calls that get
  human-verified regardless — so it's the natural thing to run locally: a full gold-set
  draft becomes free, reproducible, and key-less. Grounded generation stays on the API
  (platform-enforced Citations have no local equivalent), which keeps the two-backend split
  principled rather than a cost hack. **Done when:** `eval.generate` produces an equivalent
  draft against a local model *and* the README reports judge↔Claude label agreement, so the
  cost/quality trade-off is measured, not assumed.

---

## Sequencing & scope

Phases run in order; the benchmark (Phase 4) is the gravitational center — Phases 1–3
exist to be measured by it, Phases 5–6 build on it. Keep scale claims honest: ~4.3K
documents rigorously measured beats a vague "enterprise scale." An honest negative result
(e.g. hybrid not beating dense on some query class) is a feature of the artifact, not a
failure.
