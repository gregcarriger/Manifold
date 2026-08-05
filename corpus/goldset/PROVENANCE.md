# Gold-set judgment provenance

What produced the grades in `qrels.tsv`. Kept because "which model judged this, under what
settings" is not recoverable from the artifact itself, and a benchmark whose gold set has
unknown provenance cannot be defended.

## Current contents of `qrels.tsv`

| | |
|---|---|
| Judgments | **2,075** (147 rel=2 · 851 rel=1 · 1,077 explicit rel=0) |
| Queries with rows | 60 of 61 answerable (q075 pending re-judge) |
| Judge model | **`claude-haiku-4-5`** |
| Adaptive thinking | **disabled** (see below) |
| Credentials | personal Anthropic API key |
| Pool | depth 15, union of `bge-base` + `e5-base` + `bm25` |
| Human corrections | 93 grade edits, audited in `relabel_log.tsv` |

**Adaptive thinking was off, silently.** `_judge()` in `src/manifold/eval/generate.py` requests
`thinking=True`, but the module computes
`_THINKING_KW = {} if "haiku" in MODEL.lower() else {"thinking": {"type": "adaptive"}}`
because Haiku 4.5 rejects the parameter. So every one of these judgments came from the cheapest
model *with its judgment-sharpening feature dropped*. Haiku was chosen deliberately — judging
is self-funded and cost is the binding constraint (see POOLING.md on pool depth for
the same trade-off) — but it is very likely part of why human review is finding systematic
over-labelling: boilerplate false positives, under-graded seed docs, and generic front matter
graded rel=2.

**Consequence for re-judging: use `claude-haiku-4-5`.** Mixing a stronger judge into the same
`qrels.tsv` would make grades incomparable across queries. If a stronger model is wanted, run it
as a *separate calibration pass* over an already-verified subset and report judge↔human agreement,
rather than overwriting rows in place.

## Reverted batches (not in `qrels.tsv`)

Both were removed rather than kept, because judgments that cannot be reproduced from a clean
checkout should not sit in a public artifact.

| Batch | Rows | Model | Why reverted |
|---|---|---|---|
| q075 | 41 | `claude-sonnet-5` | Inherited Claude Code session credentials; also the wrong model for this corpus (sonnet vs. the corpus-wide haiku), so doubly incomparable |
| q000, q001 | 79 | `claude-haiku-4-5` | Correct model, but inherited Claude Code session credentials during a depth-30 trial that was rolled back |

`src/manifold/llm.py` now refuses to call the API on inherited session credentials, so this
failure mode cannot recur silently. The `$0` alternative is
`scripts/cc_judge_export.py` → Claude Code subagents → `scripts/cc_judge_ingest.py`; note that
grades from that route are subagent-produced, not `claude-haiku-4-5`, and any batch judged that
way should be added to the table above.

## Open item

Provenance is recorded here by hand. Stamping model, thinking setting, credential source, and
pool depth per judgment batch *in the artifact* (a sidecar JSON next to `qrels.tsv`) would remove
the manual step and make mixed-provenance rows detectable automatically.
