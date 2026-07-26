"""Load NIST SP 800-82r3 (PDF) into unified Documents, one per outline section.

The PDF ships with a clean 277-entry hierarchical outline (bookmarks) with page targets.
We use it to slice the document into sections: for each outline entry we take the text
from its page up to the next entry's page, then trim to the heading boundaries. This
gives structure-aware, citable sections (with a section_path) rather than arbitrary page
blobs — and preserves the hierarchy the chunking stage compares strategies against.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from pypdf import PdfReader

from ..schema import Document
from .base import clean_text, detect_repeated_lines, drop_lines, strip_bare_page_numbers

_PDF_URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf"
_SEC_NUM = re.compile(r"^([A-Z]?\d+(?:\.\d+)*)\.?\s+(.*)$")


def _flatten_outline(reader: PdfReader) -> list[tuple[int, str, int]]:
    """Return ordered (depth, title, page_index) from the PDF outline."""
    out: list[tuple[int, str, int]] = []

    def walk(items, depth=0):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                try:
                    pg = reader.get_destination_page_number(item)
                except Exception:  # noqa: BLE001 - unresolvable outline dest: treat as no page
                    pg = None
                title = (item.title or "").strip()
                if title and pg is not None:
                    out.append((depth, title, pg))

    walk(reader.outline)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _split_number(title: str) -> tuple[str | None, str]:
    """'2.3.2. SCADA Systems' -> ('2.3.2', 'SCADA Systems')."""
    m = _SEC_NUM.match(title)
    if m:
        return m.group(1), m.group(2).strip()
    return None, title


def load(raw_path: str, doc_slug: str = "nist-800-82r3") -> Iterator[Document]:
    reader = PdfReader(raw_path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    banned = detect_repeated_lines(pages, min_fraction=0.4)

    entries = _flatten_outline(reader)
    if not entries:
        return

    # Maintain an ancestor stack to build section_path from outline depth.
    stack: list[str] = []

    for i, (depth, title, page) in enumerate(entries):
        next_page = entries[i + 1][2] if i + 1 < len(entries) else len(pages) - 1
        # Slice pages spanning this section (inclusive of the next entry's page so we
        # can trim precisely at its heading), then clean.
        raw = "\n".join(pages[page : max(next_page + 1, page + 1)])
        raw = drop_lines(raw, banned)
        raw = strip_bare_page_numbers(raw)

        # Trim to heading boundaries within the slice.
        norm_raw = raw
        start = 0
        h = title
        idx = norm_raw.lower().find(_norm(h))
        if idx == -1:  # try without the leading number
            _, bare = _split_number(title)
            idx = norm_raw.lower().find(_norm(bare)) if bare else -1
        if idx != -1:
            start = idx
        body = raw[start:]

        if i + 1 < len(entries):
            nxt = entries[i + 1][1]
            end = body.lower().find(_norm(nxt), len(h))
            if end != -1:
                body = body[:end]

        text = clean_text(body)
        if len(text) < 40:  # skip empty/near-empty stubs (bare headings)
            # still update the stack so descendants get the right path
            del stack[depth:]
            stack.append(title)
            continue

        del stack[depth:]
        stack.append(title)

        sec_num, sec_title = _split_number(title)
        yield Document(
            doc_id=f"{doc_slug}:{sec_num or i}",
            source="nist",
            doc_type="standard_section",
            title=sec_title,
            text=text,
            url=(f"{_PDF_URL}#section-{sec_num}" if sec_num else _PDF_URL),
            section_path=list(stack),
            source_published="2023-09-28",
            metadata={
                "publication": "NIST SP 800-82r3",
                "section_number": sec_num,
                "depth": depth,
                "page_start": page + 1,  # human-facing 1-indexed
            },
        )
