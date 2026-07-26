"""Invariants for the chunking layer. Run: pytest -q"""

from __future__ import annotations

from manifold.chunk.splitter import atomize, n_tok, pack
from manifold.chunk.strategies import ChunkConfig, fixed, structure
from manifold.schema import Document

_CFG = ChunkConfig(target_tokens=64, overlap_tokens=8, max_tokens=128, min_tokens=8)


def _doc(text: str, source: str = "nist", **meta) -> Document:
    return Document(
        doc_id=f"{source}:test",
        source=source,
        doc_type="standard_section" if source == "nist" else "advisory",
        title="Test Doc",
        text=text,
        section_path=["1. Root", "1.1 Child"],
        metadata=meta,
    )


def test_pack_respects_max_cap():
    text = "word " * 400  # ~500 tokens of small atoms
    atoms = atomize(text, 0, len(text), _CFG.max_tokens)
    spans = pack(text, atoms, _CFG.target_tokens, _CFG.overlap_tokens, _CFG.max_tokens)
    assert spans, "expected at least one window"
    for s, e in spans:
        assert n_tok(text[s:e]) <= _CFG.max_tokens


def test_offsets_reconstruct_body():
    doc = _doc("Alpha bravo charlie. " * 60)
    for c in fixed(doc, _CFG):
        body = doc.text[c.char_start : c.char_end].strip()
        assert body and body in doc.text


def test_fixed_has_overlap_multiple_chunks():
    doc = _doc("".join(f"Sentence number {i} here. " for i in range(120)))
    chunks = fixed(doc, _CFG)
    assert len(chunks) > 1
    # overlapping windows: chunk i+1 should start before chunk i ends
    assert any(chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1))


def test_structure_adds_context_header_and_no_overlap():
    doc = _doc("Body text here. " * 50)
    chunks = structure(doc, _CFG)
    assert all(c.context_header for c in chunks)
    assert all(c.context_header in c.text for c in chunks)


def test_structure_splits_cisa_per_cve():
    text = (
        "Acme PLC Advisory\n\n"
        "## Risk evaluation\n" + "Exploitation could allow remote code execution. " * 6 + "\n\n"
        "### CVE-2024-0001\n" + "Weakness: CWE-787. " * 20 + "\n\n"
        "### CVE-2024-0002\n" + "Weakness: CWE-125. " * 20
    )
    doc = _doc(text, source="cisa", advisory_id="ICSA-24-000-01", cve_ids=["CVE-2024-0001"])
    chunks = structure(doc, _CFG)
    joined = [c.text for c in chunks]
    # the two CVEs should not end up in the same chunk
    assert not any("CVE-2024-0001" in t and "CVE-2024-0002" in t for t in joined)


def test_unique_chunk_ids():
    doc = _doc("Paragraph one. " * 40 + "\n\n" + "Paragraph two. " * 40)
    ids = [c.chunk_id for c in fixed(doc, _CFG)]
    assert len(ids) == len(set(ids))
