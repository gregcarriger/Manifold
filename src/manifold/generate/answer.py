"""Grounded generation with citation-to-source enforcement.

Retrieve → hand each chunk to Claude as a citations-enabled ``document`` block → Claude
answers using only those documents, and the API attaches a citation (with the exact cited
span and source index) to every claim it draws from a source. Because citations can only
point at the provided documents, "cite your source" is enforced by the platform, not by a
regex over prompt-injected markers.

The response is scored for grounding: what fraction of the answer text is backed by a
citation, which claims are uncited (flagged), and whether the model correctly abstained
when the corpus doesn't cover the question. Heavier, safety-weighted faithfulness scoring
is Phase 6 (the `judge` repo).

CLI:  python -m manifold.generate.answer "how should safety instrumented systems be secured"
Requires ANTHROPIC_API_KEY (or an `ant auth login` profile) and built indexes (Phase 2).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import asdict, dataclass, field

from ..llm import client as llm_client
from ..retrieve.config import RetrievalConfig
from ..retrieve.retriever import Retriever

# One call per query (not per candidate), so Opus is cheap here and gives the best answer —
# this is the artifact's visible output. Override with MANIFOLD_GEN_MODEL.
MODEL = os.environ.get("MANIFOLD_GEN_MODEL", "claude-opus-4-8")

_SYSTEM = (
    "You are an OT/ICS (operational technology / industrial control systems) security "
    "assistant. Answer the question using ONLY the provided source documents.\n"
    "- Ground every factual claim in a specific source; do not use outside knowledge.\n"
    "- If the provided documents do not contain enough information to answer, say so "
    "explicitly (e.g. \"The provided sources do not cover this.\") and stop — do not guess "
    "or fabricate CVE ids, product names, controls, or technique ids.\n"
    "- Be concise and precise. Wrong OT security guidance is dangerous."
)

_ABSTAIN_RE = re.compile(
    r"\b(do(es)? not (cover|contain|include|address)|not covered|no (relevant )?information|"
    r"cannot (answer|be answered)|insufficient (information|context)|not (found|present) in "
    r"the provided|provided (sources|documents) do not)\b",
    re.IGNORECASE,
)


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    url: str
    cited_text: str = ""


@dataclass
class AnswerBlock:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    query: str
    blocks: list[AnswerBlock]
    abstained: bool
    cited_fraction: float
    uncited_claims: list[str]      # substantive text blocks with no citation
    sources: list[str]             # unique chunk_ids cited
    config: dict

    def text(self) -> str:
        return "".join(b.text for b in self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)


def _substantive(text: str) -> bool:
    """A claim worth citing — not whitespace or a bare connective/heading."""
    return len(text.strip()) >= 40


def build_answer(query: str, content, chunks: list, config: dict) -> GroundedAnswer:
    """Assemble a GroundedAnswer from the API response content and the chunk list.

    Pure and duck-typed (no SDK import) so it can be unit-tested with synthetic blocks.
    ``content`` is the response's content list; ``chunks[i]`` is the document at
    citation ``document_index == i``.
    """
    blocks: list[AnswerBlock] = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        cits: list[Citation] = []
        for cit in (getattr(block, "citations", None) or []):
            idx = getattr(cit, "document_index", None)
            if idx is None or idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            cits.append(Citation(chunk_id=ch.chunk_id, doc_id=ch.doc_id, url=ch.url or "",
                                 cited_text=getattr(cit, "cited_text", "") or ""))
        blocks.append(AnswerBlock(text=block.text, citations=cits))

    total = sum(len(b.text.strip()) for b in blocks)
    cited = sum(len(b.text.strip()) for b in blocks if b.citations)
    cited_fraction = round(cited / total, 4) if total else 0.0
    uncited = [b.text.strip() for b in blocks if not b.citations and _substantive(b.text)]
    sources = list(dict.fromkeys(c.chunk_id for b in blocks for c in b.citations))

    full = "".join(b.text for b in blocks)
    abstained = bool(_ABSTAIN_RE.search(full)) and not sources

    return GroundedAnswer(query=query, blocks=blocks, abstained=abstained,
                          cited_fraction=cited_fraction, uncited_claims=uncited,
                          sources=sources, config=config)


def generate_answer(retriever: Retriever, client, query: str,
                    cfg: RetrievalConfig | None = None, model: str = MODEL,
                    max_context: int = 8, max_tokens: int = 2048) -> GroundedAnswer:
    cfg = cfg or RetrievalConfig(method="hybrid", strategy="structure", rerank=True,
                                 k=max_context, candidate_k=40)
    chunks = retriever.retrieve(query, cfg)[:max_context]

    documents = [{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": c.text},
        "title": f"{c.chunk_id} | {c.source}: {c.title}"[:250],
        "citations": {"enabled": True},
    } for c in chunks]

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": documents + [
            {"type": "text", "text": f"Question: {query}"}]}],
    )
    return build_answer(query, resp.content, chunks,
                        {"model": model, "retrieval": cfg.label(), "n_context": len(chunks)})


def _print(ans: GroundedAnswer) -> None:
    # Number sources in order of first citation; append [n] markers to cited blocks.
    order: dict[str, int] = {}
    print("\n=== ANSWER ===")
    for b in ans.blocks:
        marks = ""
        for c in b.citations:
            if c.chunk_id not in order:
                order[c.chunk_id] = len(order) + 1
            marks += f"[{order[c.chunk_id]}]"
        print(b.text + (f" {marks}" if marks else ""), end="")
    print("\n\n=== SOURCES ===")
    id_to_cit = {c.chunk_id: c for b in ans.blocks for c in b.citations}
    for cid, n in order.items():
        c = id_to_cit[cid]
        print(f"  [{n}] {cid}  {c.url}")
    print("\n=== GROUNDING ===")
    print(f"  cited fraction: {ans.cited_fraction:.0%}")
    print(f"  distinct sources cited: {len(ans.sources)}")
    print(f"  abstained (no answer in corpus): {ans.abstained}")
    if ans.uncited_claims:
        print(f"  ⚠ {len(ans.uncited_claims)} uncited claim(s):")
        for u in ans.uncited_claims:
            print(f"    - {u[:100]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Answer a question grounded in the OT/ICS corpus.")
    ap.add_argument("query")
    ap.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    ap.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="documents given to the model")
    args = ap.parse_args()


    retriever = Retriever()
    cfg = RetrievalConfig(method=args.method, strategy=args.strategy,
                          rerank=not args.no_rerank, k=args.k, candidate_k=40)
    try:
        ans = generate_answer(retriever, llm_client(getattr(args, 'allow_session_auth', False)), args.query, cfg,
                              max_context=args.k)
    finally:
        retriever.close()
    _print(ans)


if __name__ == "__main__":
    main()
