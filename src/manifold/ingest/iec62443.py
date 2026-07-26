"""Optional loader for licensed IEC 62443 PDFs.

IEC 62443 is copyrighted and paywalled, so nothing here ships in the repo. This loader
is a no-op unless a user who holds a license has placed PDFs in
``corpus/raw/iec62443/`` and created a ``manifest.json`` there (see the README and
manifest.template.json in that directory). When active it produces the same Document
schema as every other source, so IEC content flows through the identical pipeline.

Public benchmark results are always computed on the public corpus only; any IEC-
augmented run is a private, local extension reported separately.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from ..schema import Document
from .base import clean_text


def is_available(raw_dir: str) -> bool:
    return os.path.isfile(os.path.join(raw_dir, "manifest.json"))


def load(raw_dir: str) -> Iterator[Document]:
    manifest_path = os.path.join(raw_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return  # not licensed / not present — skip silently

    from pypdf import PdfReader

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest.get("documents", []):
        fname = entry.get("filename")
        if not fname:
            continue
        path = os.path.join(raw_dir, fname)
        if not os.path.isfile(path):
            print(f"  [iec62443] listed but missing on disk, skipping: {fname}")
            continue

        part = entry.get("part", fname)
        reader = PdfReader(path)
        text = clean_text("\n".join((p.extract_text() or "") for p in reader.pages))
        if not text:
            continue

        yield Document(
            doc_id=f"iec62443:{part.replace(' ', '')}",
            source="iec62443",
            doc_type="standard",
            title=f"{part} — {entry.get('title', '')}".strip(" —"),
            text=text,
            url="",  # licensed; no public URL
            section_path=[part],
            source_published=(str(entry["year"]) if entry.get("year") else None),
            metadata={
                "part": part,
                "edition": entry.get("edition"),
                "year": entry.get("year"),
                "license": "proprietary-local-only",
            },
        )
