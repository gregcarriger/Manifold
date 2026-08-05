"""Unified document schema for the Manifold OT/ICS corpus.

Every source (NIST SP 800-82r3, CISA ICS advisories, MITRE ATT&CK for ICS, and the
optional licensed IEC 62443 set) is normalized into a stream of ``Document`` objects
with a common shape. Downstream stages (chunking, embedding, retrieval, evaluation)
operate only on this schema and never touch the raw source formats again.

Design notes
------------
* A ``Document`` is a *coherent source unit*, not a chunk: one NIST section, one CISA
  advisory, one MITRE technique. Splitting into retrieval passages happens later in the
  chunking stage, which is deliberately kept separate so we can compare strategies.
* ``section_path`` preserves source hierarchy so structure-aware chunking has something
  to work with (e.g. ``["2. OT Overview", "2.3. ...", "2.3.2. SCADA Systems"]``).
* ``metadata`` carries source-specific structured fields (CVEs, CVSS, vendors, ATT&CK
  IDs, tactics, ...). These become filterable facets and feed the gold-set construction.
* ``url`` is the canonical, citable source location — the anchor for groundedness /
  citation-to-source enforcement during generation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """A normalized, source-agnostic document."""

    doc_id: str
    """Stable, globally unique id, namespaced by source. e.g. 'cisa:ICSA-24-004-01',
    'nist-800-82r3:2.3.2', 'mitre-ics:T0800'."""

    source: str
    """One of: 'nist', 'cisa', 'mitre', 'iec62443', 'dragos'. The last two are optional,
    licensed, local-only sources — never committed and excluded from published results."""

    doc_type: str
    """Fine-grained kind within a source, e.g. 'standard_section', 'advisory',
    'technique', 'mitigation', 'software', 'group', 'asset', 'campaign', 'standard',
    'report', 'indicators'."""

    title: str
    text: str
    """Normalized plain text (whitespace collapsed, PDF hyphenation repaired, running
    headers/footers stripped). This is what gets chunked and embedded."""

    url: str = ""
    """Canonical source URL for citation. Empty for local/licensed sources."""

    section_path: list[str] = field(default_factory=list)
    """Ordered hierarchy of section titles, outermost first. Empty if flat."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Source-specific structured fields (see loader docstrings)."""

    source_published: str | None = None
    """ISO date the source item was published/revised, if known."""

    text_hash: str = ""
    """sha256 of ``text`` — set automatically; used for dedup and integrity."""

    def __post_init__(self) -> None:
        if not self.text_hash and self.text:
            self.text_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def token_estimate(self) -> int:
        """Rough token count (~4 chars/token) — good enough for corpus stats."""
        return len(self.text) // 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(docs: Iterable[Document], path: str) -> int:
    """Write documents as JSON Lines. Returns the count written."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Document]:
    """Stream documents back from a JSON Lines file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Document(**json.loads(line))


@dataclass
class Chunk:
    """A retrieval passage derived from a Document by a chunking strategy.

    Chunks are what actually get embedded, indexed, and retrieved. Provenance back to the
    parent document is preserved (``doc_id``, ``char_start``/``char_end`` offsets into the
    parent ``text``, ``section_path``, citable ``url``) so citation-to-source enforcement
    works downstream. ``text`` is what is embedded and may include a synthetic
    ``context_header`` (structure strategy); the char offsets always refer to the parent
    body span, not the header.
    """

    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    strategy: str
    """Chunking strategy that produced this chunk, e.g. 'fixed', 'structure'."""

    title: str
    text: str
    url: str = ""
    section_path: list[str] = field(default_factory=list)

    char_start: int = 0
    char_end: int = 0
    chunk_index: int = 0
    n_chunks: int = 1

    context_header: str = ""
    """Synthetic prefix prepended to ``text`` for embedding (structure strategy). Empty if
    none. Present in ``text`` but not counted in char offsets."""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_chunks_jsonl(path: str) -> Iterator[Chunk]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Chunk(**json.loads(line))
