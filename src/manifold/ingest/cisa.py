"""Load CISA ICS advisories (CSAF 2.0 JSON) into unified Documents.

Each advisory becomes one Document. We keep the advisory-specific content — summary,
per-CVE detail, remediations, affected products — and deliberately drop the boilerplate
legal disclaimers and generic "Recommended Practices" notes, which are byte-identical
across thousands of advisories and would otherwise flood retrieval with duplicate text.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator
from typing import Any

from ..schema import Document
from .base import clean_text, strip_html

# Notes that repeat verbatim across the whole advisory set — not advisory-specific.
_BLOCK_NOTE_TITLES = {
    "legal notice",
    "cisa disclaimer",
    "recommended practices",
    "general recommendations",
    "general security recommendations",
}
_BLOCK_NOTE_CATEGORIES = {"legal_disclaimer"}


def _collect_products(branches: list[dict[str, Any]], vendors: set, products: set) -> None:
    """Recurse the CSAF product_tree, collecting vendor and product names."""
    for b in branches or []:
        cat, name = b.get("category"), b.get("name")
        if cat == "vendor" and name:
            vendors.add(name.strip())
        elif cat == "product_name" and name:
            products.add(name.strip())
        _collect_products(b.get("branches", []), vendors, products)


def _keep_note(note: dict[str, Any]) -> bool:
    if note.get("category") in _BLOCK_NOTE_CATEGORIES:
        return False
    title = (note.get("title") or "").strip().lower()
    return title not in _BLOCK_NOTE_TITLES


def _best_cvss(scores: list[dict[str, Any]]) -> tuple[float | None, str | None, str | None]:
    """Return (base_score, base_severity, vector) for the highest-scoring CVSS entry."""
    best = (None, None, None)
    best_val = -1.0
    for s in scores or []:
        for key in ("cvss_v4", "cvss_v3", "cvss_v2"):
            c = s.get(key)
            if not c:
                continue
            val = c.get("baseScore")
            if isinstance(val, (int, float)) and val > best_val:
                best_val = float(val)
                best = (float(val), c.get("baseSeverity"), c.get("vectorString"))
    return best


def _advisory_url(adv_id: str) -> str:
    return f"https://www.cisa.gov/news-events/ics-advisories/{adv_id.lower()}"


def load_advisory(path: str) -> Document | None:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    doc = d.get("document", {})
    adv_id = doc.get("tracking", {}).get("id")
    if not adv_id:
        return None
    title = doc.get("title", "").strip()

    parts: list[str] = [title, ""]

    # Advisory-level narrative notes (summary, risk evaluation, ...).
    for n in doc.get("notes", []):
        if _keep_note(n) and n.get("text"):
            t = (n.get("title") or n.get("category") or "Note").strip()
            parts.append(f"## {t}\n{strip_html(n['text']).strip()}")

    # Affected products.
    vendors: set[str] = set()
    products: set[str] = set()
    _collect_products(d.get("product_tree", {}).get("branches", []), vendors, products)
    if products:
        parts.append("## Affected Products\n" + "; ".join(sorted(products)))

    # Per-vulnerability detail.
    cve_ids: list[str] = []
    cwe_ids: list[str] = []
    max_score, max_sev = None, None
    for v in d.get("vulnerabilities", []):
        cve = v.get("cve")
        if cve:
            cve_ids.append(cve)
        cwe = v.get("cwe") or {}
        if cwe.get("id"):
            cwe_ids.append(cwe["id"])
        score, sev, vector = _best_cvss(v.get("scores", []))
        if score is not None and (max_score is None or score > max_score):
            max_score, max_sev = score, sev

        vblock = [f"### {cve or 'Vulnerability'}"]
        if cwe.get("id"):
            vblock.append(f"Weakness: {cwe['id']} {cwe.get('name', '')}".strip())
        if score is not None:
            vblock.append(f"CVSS: {score} ({sev}) {vector or ''}".strip())
        for n in v.get("notes", []):
            if n.get("text"):
                vblock.append(strip_html(n["text"]).strip())
        for r in v.get("remediations", []):
            if r.get("details"):
                cat = r.get("category", "remediation")
                vblock.append(f"Remediation ({cat}): {strip_html(r['details']).strip()}")
        parts.append("\n".join(vblock))

    text = clean_text("\n\n".join(p for p in parts if p and p.strip()))
    if not text:
        return None

    tracking = doc.get("tracking", {})
    published = tracking.get("current_release_date") or tracking.get("initial_release_date")

    return Document(
        doc_id=f"cisa:{adv_id}",
        source="cisa",
        doc_type="advisory",
        title=title,
        text=text,
        url=_advisory_url(adv_id),
        section_path=[],
        source_published=(published[:10] if published else None),
        metadata={
            "advisory_id": adv_id,
            "cve_ids": cve_ids,
            "cwe_ids": sorted(set(cwe_ids)),
            "vendors": sorted(vendors),
            "products": sorted(products),
            "max_cvss": max_score,
            "max_severity": max_sev,
            "num_vulnerabilities": len(d.get("vulnerabilities", [])),
        },
    )


def load(raw_dir: str) -> Iterator[Document]:
    """Yield a Document for every advisory JSON under ``raw_dir/advisories``."""
    pattern = os.path.join(raw_dir, "advisories", "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            doc = load_advisory(path)
        except Exception as e:  # noqa: BLE001 - stay resilient to a few malformed files
            print(f"  [cisa] skipped {os.path.basename(path)}: {e}")
            continue
        if doc:
            yield doc
