"""Optional loader for licensed Dragos WorldView intelligence.

Dragos WorldView is a proprietary, subscription threat-intelligence product, so nothing
here ships in the repo. This loader is a no-op unless a user who holds a subscription has
downloaded WorldView products (from https://portal.dragos.com/#/products, typically scoped
to a *perspective* such as Manufacturing) into ``corpus/raw/dragos/``. When active it
produces the same Document schema as every other source, so WorldView content flows through
the identical pipeline. Public benchmark results are always computed on the public corpus
only; any WorldView-augmented run is a private, local extension reported separately.

Report bundles live under ``reports/``. Each WorldView report exports as up to three files
sharing a serial prefix (``AIR-2026-14``):

* ``<serial> <title>.pdf``          — narrative analysis (the primary retrieval content).
* ``<serial>-<ts>.stix2.json``      — STIX 2.x bundle used to enrich report metadata
                                      (threat-actors, ATT&CK patterns, indicator count).
* ``<serial>-IOCs-<ts>.csv``        — indicator list, emitted as a separate, exact-match
                                      retrievable ``indicators`` document.

PDF-only products are supported; the STIX/CSV files are optional per bundle.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

from ..schema import Document
from .base import (
    clean_text,
    detect_repeated_lines,
    drop_lines,
    strip_bare_page_numbers,
    strip_html,
)

# A WorldView serial: 2-5 letter product code, 4-digit year, sequence number. The export
# timestamp suffix on STIX/CSV filenames (``-1784836687``) is excluded by stopping at the
# first non-serial hyphen group.
# Canonical (auth-gated) portal reference for a report, by serial.
_PORTAL_URL = "https://portal.dragos.com/products/{serial}/report"

_SERIAL = re.compile(r"^([A-Za-z]{2,5}-\d{4}-\d+)")
_CVE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_TLP = re.compile(r"TLP:\s*(RED|AMBER\+STRICT|AMBER|GREEN|WHITE|CLEAR)", re.IGNORECASE)

# WorldView product families confirmed from real exports. Unknown prefixes (WVW, TR, AA,
# ...) fall back to a label derived from the report's own title, which is authoritative —
# we do not guess expansions we haven't verified.
_REPORT_TYPES = {
    "AIR": "Adversary Infrastructure Report",
    "RAN": "Industrial Ransomware Report",
}


def is_available(raw_dir: str) -> bool:
    """Active if any report bundle files are present (a manifest is optional)."""
    reports = os.path.join(raw_dir, "reports")
    if not os.path.isdir(reports):
        return False
    return any(
        glob.glob(os.path.join(reports, ext)) for ext in ("*.pdf", "*.stix2.json", "*.csv")
    )


def _perspective(raw_dir: str) -> str:
    path = os.path.join(raw_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("perspective") or "unspecified"
    return "unspecified"


def _serial(filename: str) -> str | None:
    m = _SERIAL.match(os.path.basename(filename))
    return m.group(1) if m else None


def _report_type(serial: str, title: str) -> str:
    prefix = serial.split("-", 1)[0].upper()
    if prefix in _REPORT_TYPES:
        return _REPORT_TYPES[prefix]
    # Derive from the title minus any trailing date range.
    return re.sub(r"\s+\d.*$", "", title).strip() or "WorldView Report"


def _title_from_pdf(pdf_path: str, serial: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    # Drop the leading serial and any separators; keep the human title + date range.
    title = re.sub(rf"^{re.escape(serial)}[\s_-]*", "", stem).strip()
    return title or serial


def _pdf_text(path: str) -> str:
    """Extract PDF text, dropping running headers/footers and bare page numbers."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    banned = detect_repeated_lines(pages)
    body = "\n".join(pages)
    body = drop_lines(body, banned)
    body = strip_bare_page_numbers(body)
    return clean_text(body)


def _stix_enrichment(path: str) -> dict[str, Any]:
    """Pull threat-actors, ATT&CK patterns and indicator count out of a STIX 2.x bundle."""
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)
    objects = bundle.get("objects", [])
    threat_groups = sorted({o["name"] for o in objects if o.get("type") == "threat-actor"})
    techniques = sorted({o["name"] for o in objects if o.get("type") == "attack-pattern"})
    indicator_count = sum(1 for o in objects if o.get("type") == "indicator")
    modified = [o["modified"][:10] for o in objects if o.get("modified")]
    return {
        "threat_groups": threat_groups,
        "attack_techniques": techniques,
        "indicator_count": indicator_count,
        "stix_last_modified": max(modified) if modified else None,
    }


def _indicators_doc(
    csv_path: str, serial: str, report_title: str, base_meta: dict[str, Any]
) -> Document | None:
    """One retrievable document listing a report's IOCs, values kept verbatim for exact
    match (the same lexical/identifier story as CVE ids elsewhere in the corpus)."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    type_counts: Counter[str] = Counter()
    groups: set[str] = set()
    techniques: set[str] = set()
    lines: list[str] = []
    for r in rows:
        value = (r.get("Indicator Value") or "").strip()
        if not value:
            continue
        ioc_type = (r.get("Type") or "").strip()
        type_counts[ioc_type] += 1
        conf = (r.get("Confidence") or "").strip()
        first, last = (r.get("First Seen") or "").strip(), (r.get("Last Seen") or "").strip()
        tg = (r.get("Threat Groups") or "").strip()
        tech = (r.get("ATT&CK Techniques") or "").strip()
        if tg:
            groups.update(g.strip() for g in tg.split(",") if g.strip())
        if tech:
            techniques.update(t.strip() for t in tech.split(",") if t.strip())
        parts = [f"{value} ({ioc_type})"]
        if conf:
            parts.append(f"confidence {conf}")
        if first or last:
            parts.append(f"seen {first} to {last}".strip())
        if tech:
            parts.append(f"technique {tech}")
        if tg:
            parts.append(f"attributed to {tg}")
        lines.append("- " + "; ".join(parts))

    if not lines:
        return None

    summary = ", ".join(f"{n} {t or 'unknown'}" for t, n in type_counts.most_common())
    text = clean_text(
        f"Indicators of compromise from Dragos WorldView report {serial} "
        f"({report_title}). {len(lines)} indicators: {summary}.\n\n" + "\n".join(lines)
    )
    meta = dict(base_meta)
    meta.update(
        {
            "serial": serial,
            "ioc_count": len(lines),
            "ioc_types": dict(type_counts),
            "threat_groups": sorted(groups) or base_meta.get("threat_groups", []),
            "attack_techniques": sorted(techniques) or base_meta.get("attack_techniques", []),
        }
    )
    return Document(
        doc_id=f"dragos:indicators:{serial}",
        source="dragos",
        doc_type="indicators",
        title=f"{serial} — Indicators of Compromise",
        text=text,
        url=_PORTAL_URL.format(serial=serial),  # auth-gated portal reference
        section_path=[serial, "Indicators of Compromise"],
        metadata=meta,
        source_published=base_meta.get("source_published"),
    )


def _serial_for(path: str, known: list[str]) -> str | None:
    """Serial a report file belongs to: the longest known (index) serial its basename
    starts with — handles compound serials like 'DOM-2022-47 & 48' — else the regex."""
    base = os.path.basename(path)
    for s in known:  # `known` is pre-sorted longest-first
        if base.startswith(s):
            return s
    return _serial(path)


def _group_bundles(reports_dir: str, known_serials: list[str]) -> dict[str, dict[str, Any]]:
    """Group report files by serial into {serial: {pdfs: [...], stix, csv}}.

    A product may ship more than one PDF (a ``/report`` and a ``/slides`` deck), so PDFs
    accumulate into a list; there is at most one STIX bundle and one IOC CSV per serial.
    """
    known = sorted(known_serials, key=len, reverse=True)
    bundles: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(reports_dir, "*"))):
        if os.path.isdir(path):
            continue
        serial = _serial_for(path, known)
        if not serial:
            continue
        b = bundles.setdefault(serial, {"pdfs": []})
        low = path.lower()
        if low.endswith(".csv"):
            b["csv"] = path
        elif low.endswith(".stix2.json"):
            b["stix"] = path
        elif low.endswith(".pdf"):
            b["pdfs"].append(path)
    return bundles


def _primary_pdf(pdfs: list[str]) -> str | None:
    """Prefer the narrative report over a slides deck when both are present."""
    if not pdfs:
        return None
    narrative = [p for p in pdfs if "slide" not in os.path.basename(p).lower()]
    return (narrative or pdfs)[0]


def _read_json(path: str) -> Any:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _ptags(product: dict, tag_type: str) -> list[str]:
    """Text values of a product's tags of a given tag_type (API products_index)."""
    return sorted(
        {
            t["text"]
            for t in (product.get("tags") or [])
            if t.get("tag_type") == tag_type and t.get("text")
        }
    )


def _report_meta(
    serial: str, perspective: str, product: dict | None, enrichment: dict, title: str
) -> dict[str, Any]:
    """Build report metadata, preferring API product fields, falling back to STIX/filename."""
    meta: dict[str, Any] = {
        "serial": serial,
        "perspective": perspective,
        "license": "proprietary-local-only",
    }
    if product:
        meta["report_type"] = product.get("type") or _report_type(serial, title)
        if product.get("tlp_level"):
            meta["tlp"] = str(product["tlp_level"]).upper()
        if product.get("threat_level") is not None:
            meta["threat_level"] = product["threat_level"]
        techniques = _ptags(product, "ATT&CK Technique") + _ptags(product, "ICS ATT&CK Technique")
        malware = _ptags(product, "Malware") + _ptags(product, "ICS Malware") + _ptags(product, "Ransomware")
        meta["threat_groups"] = _ptags(product, "ThreatGroup") or enrichment.get("threat_groups", [])
        meta["attack_techniques"] = techniques or enrichment.get("attack_techniques", [])
        cves = _ptags(product, "CVE")
        if cves:
            meta["cve_ids"] = cves  # cross-links to CISA advisories / MITRE
        if malware:
            meta["malware"] = malware
        for key, tt in (("vendors", "Vendor"), ("external_names", "ExternalName"),
                        ("industries", "Industry"), ("naics", "NAICS")):
            vals = _ptags(product, tt)
            if vals:
                meta[key] = vals
        meta["indicator_count"] = product.get("ioc_count") or enrichment.get("indicator_count", 0)
    else:
        meta["report_type"] = _report_type(serial, title)
        meta["threat_groups"] = enrichment.get("threat_groups", [])
        meta["attack_techniques"] = enrichment.get("attack_techniques", [])
        meta["indicator_count"] = enrichment.get("indicator_count", 0)
    return meta


def _threat_group_docs(raw_dir: str, perspective: str) -> Iterator[Document]:
    groups = _read_json(os.path.join(raw_dir, "threat_groups.json")) or {}
    for name, serials in groups.items():
        serials = sorted({s for s in serials if s})
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
        text = clean_text(
            f"{name} is an OT/ICS threat group tracked by Dragos WorldView. Referenced in "
            f"{len(serials)} {perspective} WorldView reports: {', '.join(serials)}."
        )
        yield Document(
            doc_id=f"dragos:group:{slug}",
            source="dragos",
            doc_type="threat_group",
            title=f"{name} — Dragos threat group",
            text=text,
            url="",
            section_path=["Threat Groups", name],
            metadata={
                "threat_group": name,
                "report_serials": serials,
                "perspective": perspective,
                "license": "proprietary-local-only",
            },
        )


def _vulnerability_docs(raw_dir: str) -> Iterator[Document]:
    """Dragos-scored CVEs from an optional /tags?tag_type=CVE pull (vulnerabilities.json)."""
    records = _read_json(os.path.join(raw_dir, "vulnerabilities.json")) or []
    for rec in records:
        cve = (rec.get("text") or "").upper()
        if not cve.startswith("CVE-"):
            continue
        st = rec.get("special_tag") or {}
        parts = [f"Dragos vulnerability assessment for {cve}."]
        for label, score, vector in (
            ("Dragos", st.get("dragos_cvss_score"), st.get("dragos_cvss_string")),
            ("NVD", st.get("nvd_cvss_score"), st.get("nvd_cvss_string")),
            ("ICS-CERT", st.get("icsa_cvss_score"), st.get("icsa_cvss_string")),
        ):
            if score is not None:
                parts.append(f"{label} CVSS {score}" + (f" ({vector})." if vector else "."))
        yield Document(
            doc_id=f"dragos:vuln:{cve}",
            source="dragos",
            doc_type="vulnerability",
            title=f"{cve} — Dragos vulnerability assessment",
            text=clean_text(" ".join(parts)),
            url="",
            section_path=["Vulnerabilities", cve],
            metadata={
                "cve_ids": [cve],
                "dragos_cvss_score": st.get("dragos_cvss_score"),
                "nvd_cvss_score": st.get("nvd_cvss_score"),
                "icsa_cvss_score": st.get("icsa_cvss_score"),
                "license": "proprietary-local-only",
            },
        )


def load(raw_dir: str) -> Iterator[Document]:
    if not is_available(raw_dir):
        return  # not subscribed / not present — skip silently

    perspective = _perspective(raw_dir)
    reports_dir = os.path.join(raw_dir, "reports")

    index_list = _read_json(os.path.join(raw_dir, "products_index.json")) or []
    index = {p["serial"]: p for p in index_list if p.get("serial")}
    bundles = _group_bundles(reports_dir, list(index))

    for serial in sorted(set(index) | set(bundles)):
        product = index.get(serial)
        files = bundles.get(serial, {})
        pdf = _primary_pdf(files.get("pdfs", []))
        stix, csv_path = files.get("stix"), files.get("csv")

        enrichment = _stix_enrichment(stix) if stix else {}
        title = (product or {}).get("title") or (_title_from_pdf(pdf, serial) if pdf else serial)
        published = (product or {}).get("release_date", "")[:10] or enrichment.get("stix_last_modified")
        report_type = (product or {}).get("type") or _report_type(serial, title)

        base_meta = _report_meta(serial, perspective, product, enrichment, title)
        if published:
            base_meta["source_published"] = published

        # Report body: full PDF text if downloaded, else the API executive summary.
        text, body_source = "", ""
        if pdf:
            text, body_source = _pdf_text(pdf), "pdf"
        if not text and product and product.get("executive_summary"):
            text, body_source = clean_text(strip_html(product["executive_summary"])), "executive_summary"

        if text:
            meta = dict(base_meta)
            meta["body_source"] = body_source
            body_cves = {m.group(0).upper() for m in _CVE.finditer(text)}
            if body_cves:
                meta["cve_ids"] = sorted(set(meta.get("cve_ids", [])) | body_cves)
            if "tlp" not in meta:
                tlp = _TLP.search(text)
                if tlp:
                    meta["tlp"] = tlp.group(1).upper()
            yield Document(
                doc_id=f"dragos:report:{serial}",
                source="dragos",
                doc_type="report",
                title=f"{report_type}: {title}" if report_type.lower() not in title.lower() else title,
                text=text,
                url=_PORTAL_URL.format(serial=serial),
                section_path=[report_type, serial],
                metadata=meta,
                source_published=published or None,
            )

        if csv_path:
            doc = _indicators_doc(csv_path, serial, title, base_meta)
            if doc:
                yield doc

    yield from _threat_group_docs(raw_dir, perspective)
    yield from _vulnerability_docs(raw_dir)
