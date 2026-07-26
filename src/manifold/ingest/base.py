"""Shared text-normalization helpers for ingestion.

The goal of normalization is a clean, consistent plain-text surface across very different
source formats (PDF text, CSAF JSON prose, STIX markdown) so that chunking and embedding
see uniform input. We normalize conservatively — we do not paraphrase or drop content.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

# Characters PDFs love to emit that we want to normalize away.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
    "•": "-", "﻿": "",
}

_WS_RUN = re.compile(r"[ \t]+")
_BLANKLINES = re.compile(r"\n{3,}")
# A word split across a line break by hyphenation: "vulnera-\nbility" -> "vulnerability".
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def clean_text(text: str) -> str:
    """Normalize unicode, repair PDF hyphenation, and collapse whitespace runs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = _HYPHEN_BREAK.sub("", text)
    # Normalize line endings, strip trailing space per line.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS_RUN.sub(" ", ln).rstrip() for ln in text.split("\n"))
    text = _BLANKLINES.sub("\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    """Remove HTML tags (some CSAF/STIX prose carries light markup) and unescape."""
    import html

    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def detect_repeated_lines(pages: Iterable[str], min_fraction: float = 0.5) -> set[str]:
    """Identify running headers/footers: short lines that recur on many PDF pages.

    Returns the set of offending lines so callers can strip them. We only flag short
    lines (headers/footers, page numbers) to avoid nuking real repeated content.
    """
    pages = list(pages)
    if not pages:
        return set()
    counts: Counter[str] = Counter()
    for pg in pages:
        for ln in {l.strip() for l in pg.split("\n") if l.strip()}:
            if len(ln) <= 90:  # headers/footers are short
                counts[ln] += 1
    threshold = max(3, int(len(pages) * min_fraction))
    return {ln for ln, c in counts.items() if c >= threshold}


def drop_lines(text: str, banned: set[str]) -> str:
    """Remove lines matching the banned set (post-strip comparison)."""
    if not banned:
        return text
    keep = [ln for ln in text.split("\n") if ln.strip() not in banned]
    return "\n".join(keep)


_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$")


def strip_bare_page_numbers(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not _PAGE_NUM.match(ln))
