"""Two deliberately different chunking strategies, so the benchmark can measure which
wins on this corpus.

* ``fixed`` — structure-agnostic overlapping windows of ~target tokens. The baseline every
  RAG demo ships. It flattens the document and slides a window across it, so a window can
  begin mid-topic and a single CVE's detail can straddle two chunks.

* ``structure`` — respects the boundaries preserved during normalization: CISA per-CVE and
  per-section ``##``/``###`` blocks, NIST outline sections, whole MITRE objects. Tiny
  blocks merge with neighbors; oversized blocks split on paragraph/sentence boundaries.
  Each chunk is prefixed with a compact context header (e.g. the advisory id + title, or
  the NIST section path) so an embedded passage carries where-it-came-from with it. No
  overlap — the structural boundaries are already clean.

Both share the offset-preserving splitter core, so any quality difference is attributable
to boundary choice and context headers, not to different low-level splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import Chunk, Document
from .splitter import Span, atomize, n_tok, pack

_HEADER_LINE = re.compile(r"(?m)^#{2,3} ")


@dataclass
class ChunkConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    max_tokens: int = 1024
    min_tokens: int = 48  # blocks smaller than this are merged (structure strategy)


def _doc_meta(doc: Document) -> dict:
    """Carry forward the parent metadata fields useful as retrieval filters."""
    keep = ("advisory_id", "cve_ids", "cwe_ids", "vendors", "products", "max_severity",
            "attack_id", "tactics", "section_number", "publication",
            "serial", "report_type", "threat_groups", "attack_techniques", "perspective")
    return {k: doc.metadata[k] for k in keep if k in doc.metadata}


def _context_header(doc: Document) -> str:
    if doc.source == "nist":
        path = " > ".join(doc.section_path[-2:]) if doc.section_path else doc.title
        return f"NIST SP 800-82r3 — {path}"
    if doc.source == "cisa":
        return f"CISA {doc.metadata.get('advisory_id', '')}: {doc.title}".strip()
    if doc.source == "mitre":
        aid = doc.metadata.get("attack_id")
        return f"MITRE ATT&CK for ICS {aid or ''} — {doc.title}".strip(" —")
    if doc.source == "iec62443":
        return f"{doc.metadata.get('part', '')}: {doc.title}".strip(": ")
    if doc.source == "dragos":
        return f"Dragos WorldView {doc.metadata.get('serial', '')} — {doc.title}".strip(" —")
    return doc.title


def _make_chunk(doc: Document, span: Span, idx: int, strategy: str, header: str = "") -> Chunk:
    body = doc.text[span[0] : span[1]].strip()
    text = f"{header}\n\n{body}" if header else body
    return Chunk(
        chunk_id=f"{doc.doc_id}#{idx}",
        doc_id=doc.doc_id,
        source=doc.source,
        doc_type=doc.doc_type,
        strategy=strategy,
        title=doc.title,
        text=text,
        url=doc.url,
        section_path=list(doc.section_path),
        char_start=span[0],
        char_end=span[1],
        chunk_index=idx,
        n_chunks=1,  # patched after the full list is known
        context_header=header,
        metadata=_doc_meta(doc),
    )


def _finalize(chunks: list[Chunk]) -> list[Chunk]:
    n = len(chunks)
    for c in chunks:
        c.n_chunks = n
    return chunks


def fixed(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Structure-agnostic overlapping windows over the whole document."""
    atoms = atomize(doc.text, 0, len(doc.text), cfg.max_tokens)
    spans = pack(doc.text, atoms, cfg.target_tokens, cfg.overlap_tokens, cfg.max_tokens)
    return _finalize([_make_chunk(doc, sp, i, "fixed") for i, sp in enumerate(spans)])


def _structural_blocks(doc: Document) -> list[Span]:
    """Boundaries that reflect the document's own structure."""
    text = doc.text
    if doc.source == "cisa":
        starts = [m.start() for m in _HEADER_LINE.finditer(text)]
        bounds = sorted({0, *starts, len(text)})
        blocks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        return [(s, e) for s, e in blocks if text[s:e].strip()]
    # NIST sections and MITRE objects are already coherent single units.
    return [(0, len(text))]


def _merge_small(text: str, blocks: list[Span], min_tokens: int) -> list[Span]:
    """Greedily coalesce adjacent blocks until each reaches ``min_tokens``.

    Prevents lone one-line blocks (e.g. a CISA "Countries deployed: Worldwide") from
    becoming their own chunk, while keeping genuinely distinct blocks (a per-CVE section)
    separate. The final block may fall under ``min_tokens`` if nothing follows it.
    """
    if not blocks:
        return []
    merged: list[Span] = []
    cur_s, cur_e = blocks[0]
    for s, e in blocks[1:]:
        if n_tok(text[cur_s:cur_e]) < min_tokens:
            cur_e = e  # keep accreting until the running block is big enough
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def structure(doc: Document, cfg: ChunkConfig) -> list[Chunk]:
    """Structure-aware chunks with context headers and no overlap."""
    header = _context_header(doc)
    blocks = _merge_small(doc.text, _structural_blocks(doc), cfg.min_tokens)

    spans: list[Span] = []
    for bs, be in blocks:
        if n_tok(doc.text[bs:be]) <= cfg.max_tokens:
            spans.append((bs, be))
        else:  # split an oversized block on natural boundaries, no overlap
            atoms = atomize(doc.text, bs, be, cfg.max_tokens)
            spans.extend(
                pack(doc.text, atoms, cfg.target_tokens, 0, cfg.max_tokens)
            )

    return _finalize(
        [_make_chunk(doc, sp, i, "structure", header) for i, sp in enumerate(spans)]
    )


STRATEGIES = {"fixed": fixed, "structure": structure}
