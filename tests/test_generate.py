"""Grounded-generation tests. Exercise build_answer with synthetic (duck-typed) response
blocks — no API or DB needed."""

from __future__ import annotations

from dataclasses import dataclass, field

from manifold.generate.answer import build_answer


# --- minimal duck-typed stand-ins for SDK objects ---
@dataclass
class FakeCitation:
    document_index: int
    cited_text: str = ""


@dataclass
class FakeBlock:
    type: str
    text: str = ""
    citations: list = field(default_factory=list)


@dataclass
class FakeChunk:
    chunk_id: str
    doc_id: str
    url: str = ""
    title: str = ""


CHUNKS = [
    FakeChunk("cisa:ICSA-24-004-01#2", "cisa:ICSA-24-004-01", "https://x/a", "Rockwell"),
    FakeChunk("nist-800-82r3:5.3.1#0", "nist-800-82r3:5.3.1", "https://x/b", "Safety"),
]


def test_cited_block_maps_to_chunk():
    content = [
        FakeBlock("thinking", "internal..."),  # skipped
        FakeBlock("text", "Air-gapping is advised. ",
                  citations=[FakeCitation(document_index=1, cited_text="isolate SIS")]),
    ]
    ans = build_answer("q", content, CHUNKS, {})
    assert ans.blocks[0].citations[0].chunk_id == "nist-800-82r3:5.3.1#0"
    assert ans.sources == ["nist-800-82r3:5.3.1#0"]
    assert ans.cited_fraction == 1.0
    assert ans.uncited_claims == []
    assert ans.abstained is False


def test_uncited_substantive_claim_is_flagged():
    content = [
        FakeBlock("text", "You should also rotate all your passwords every 30 days as a "
                          "general best practice for these systems.")  # no citations
    ]
    ans = build_answer("q", content, CHUNKS, {})
    assert ans.cited_fraction == 0.0
    assert len(ans.uncited_claims) == 1


def test_abstention_detected_when_uncited_and_refusal_phrase():
    content = [FakeBlock("text", "The provided sources do not cover this topic.")]
    ans = build_answer("q", content, CHUNKS, {})
    assert ans.abstained is True
    assert ans.sources == []


def test_partial_grounding_fraction():
    content = [
        FakeBlock("text", "CVE-2023-38545 is a buffer overflow in FactoryTalk. ",
                  citations=[FakeCitation(document_index=0)]),
        FakeBlock("text", "It is definitely the worst vulnerability ever recorded anywhere."),
    ]
    ans = build_answer("q", content, CHUNKS, {})
    assert 0.0 < ans.cited_fraction < 1.0
    assert len(ans.uncited_claims) == 1


def test_out_of_range_citation_index_ignored():
    content = [FakeBlock("text", "Some claim about controls here for testing.",
                         citations=[FakeCitation(document_index=99)])]
    ans = build_answer("q", content, CHUNKS, {})
    assert ans.sources == []  # bad index dropped, not crashed
