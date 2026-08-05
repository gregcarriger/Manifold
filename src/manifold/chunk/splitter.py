"""Offset-preserving text segmentation and windowing.

The core primitive both chunking strategies share. It works in terms of *character spans*
``(start, end)`` into the parent document text, so every resulting chunk can point back to
exactly where it came from — provenance survives chunking, which citation enforcement
depends on later.

Token counts use the same ~4-chars/token estimate as the rest of the pipeline. That is a
deliberate, documented approximation: chunking strategy (where boundaries fall) dominates
retrieval quality far more than exact tokenizer choice, and a dependency-free estimator
keeps the corpus build reproducible. A real tokenizer can be swapped into ``n_tok`` later
without changing any boundary logic.
"""

from __future__ import annotations

import re

Span = tuple[int, int]

_SEP_PARA = re.compile(r"\n\s*\n")
_SEP_SENT = re.compile(r"(?<=[.!?])\s+")


def n_tok(text: str) -> int:
    """Estimated token count for a string (~4 chars/token)."""
    return max(1, len(text) // 4)


def _spans(text: str, lo: int, hi: int, sep: re.Pattern) -> list[Span]:
    """Split text[lo:hi] on ``sep``, returning absolute non-empty spans."""
    out: list[Span] = []
    pos = lo
    for m in sep.finditer(text, lo, hi):
        if text[pos : m.start()].strip():
            out.append((pos, m.start()))
        pos = m.end()
    if text[pos:hi].strip():
        out.append((pos, hi))
    return out


def atomize(text: str, lo: int, hi: int, max_tokens: int) -> list[Span]:
    """Break text[lo:hi] into atomic spans no larger than ``max_tokens``.

    Prefers paragraph boundaries, then sentence boundaries, then a hard character cut as a
    last resort for pathological single sentences.
    """
    atoms: list[Span] = []
    paras = _spans(text, lo, hi, _SEP_PARA) or [(lo, hi)]
    for ps, pe in paras:
        if n_tok(text[ps:pe]) <= max_tokens:
            atoms.append((ps, pe))
            continue
        for ss, se in _spans(text, ps, pe, _SEP_SENT) or [(ps, pe)]:
            if n_tok(text[ss:se]) <= max_tokens:
                atoms.append((ss, se))
            else:
                step = max_tokens * 4
                c = ss
                while c < se:
                    atoms.append((c, min(se, c + step)))
                    c += step
    return atoms


def pack(
    text: str,
    atoms: list[Span],
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int | None = None,
) -> list[Span]:
    """Pack atomic spans into windows of ~``target_tokens`` with sliding overlap.

    Returns one merged span per window (first atom's start .. last atom's end). A window
    grows until it reaches ``target_tokens``; if ``max_tokens`` is set it is a hard cap —
    an atom that would push the window past it starts a new window instead (unless the
    window is still empty, since a lone oversized atom must go somewhere). Overlap is
    achieved by rewinding over trailing atoms until ~``overlap_tokens`` are re-included.
    Guarantees forward progress even with oversized atoms.
    """
    if not atoms:
        return []
    windows: list[Span] = []
    i, n = 0, len(atoms)
    while i < n:
        cur: list[Span] = []
        tok = 0
        j = i
        while j < n and (tok < target_tokens or not cur):
            atok = n_tok(text[atoms[j][0] : atoms[j][1]])
            if cur and max_tokens is not None and tok + atok > max_tokens:
                break  # respect the hard cap
            cur.append(atoms[j])
            tok += atok
            j += 1
        windows.append((cur[0][0], cur[-1][1]))
        if j >= n:
            break
        if overlap_tokens <= 0:
            i = j
            continue
        back, k = 0, j - 1
        while k > i and back < overlap_tokens:
            back += n_tok(text[atoms[k][0] : atoms[k][1]])
            k -= 1
        i = max(k + 1, i + 1)  # never stall
    return windows
