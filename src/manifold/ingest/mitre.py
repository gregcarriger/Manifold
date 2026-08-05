"""Load MITRE ATT&CK for ICS (STIX 2.1 JSON) into unified Documents.

We emit one Document per meaningful ATT&CK object — techniques, mitigations, software
(malware/tools), groups (intrusion sets), assets, and campaigns — and enrich each with
relationships resolved from the STIX bundle (e.g. which mitigations address a technique,
which assets it targets). Deprecated/revoked objects are skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from ..schema import Document
from .base import clean_text

# STIX object type -> (our doc_type)
_TYPE_MAP = {
    "attack-pattern": "technique",
    "course-of-action": "mitigation",
    "malware": "software",
    "tool": "software",
    "intrusion-set": "group",
    "x-mitre-asset": "asset",
    "campaign": "campaign",
}


def _attack_id(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (attack_id, url) from the mitre-attack external reference."""
    for e in obj.get("external_references", []):
        if e.get("source_name") == "mitre-attack" and e.get("external_id"):
            return e["external_id"], e.get("url", "")
    return None, None


def load(raw_path: str) -> Iterator[Document]:
    """Yield Documents from the ICS ATT&CK STIX bundle at ``raw_path``."""
    with open(raw_path, encoding="utf-8") as f:
        bundle = json.load(f)
    objects = bundle["objects"]

    by_ref: dict[str, dict] = {o["id"]: o for o in objects}

    # Resolve relationships: target_ref -> [(rel_type, source_obj)]
    incoming: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    outgoing: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for o in objects:
        if o.get("type") == "relationship":
            src, tgt = by_ref.get(o["source_ref"]), by_ref.get(o["target_ref"])
            if src and tgt:
                incoming[o["target_ref"]].append((o["relationship_type"], src))
                outgoing[o["source_ref"]].append((o["relationship_type"], tgt))

    # Tactic shortname -> display name (for kill_chain_phases).
    tactic_name = {
        t.get("x_mitre_shortname"): t.get("name")
        for t in objects
        if t.get("type") == "x-mitre-tactic"
    }

    def _name_of(obj: dict) -> str:
        aid, _ = _attack_id(obj)
        return f"{obj.get('name', '?')} ({aid})" if aid else obj.get("name", "?")

    for o in objects:
        stix_type = o.get("type")
        doc_type = _TYPE_MAP.get(stix_type)
        if not doc_type:
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue

        attack_id, url = _attack_id(o)
        name = o.get("name", "").strip()
        if not name:
            continue

        parts = [name, "", o.get("description", "").strip()]
        meta: dict[str, Any] = {"attack_id": attack_id, "stix_type": stix_type}

        # Tactics (for techniques).
        tactics = [
            tactic_name.get(kc.get("phase_name"), kc.get("phase_name"))
            for kc in o.get("kill_chain_phases", [])
        ]
        if tactics:
            meta["tactics"] = tactics
            parts.append("Tactics: " + ", ".join(tactics))

        platforms = [p for p in o.get("x_mitre_platforms", []) if p and p != "None"]
        if platforms:
            meta["platforms"] = platforms

        # Relationship-derived context.
        if doc_type == "technique":
            mitigations = [_name_of(s) for rt, s in incoming.get(o["id"], []) if rt == "mitigates"]
            assets = [_name_of(t) for rt, t in outgoing.get(o["id"], []) if rt == "targets"]
            if mitigations:
                meta["mitigations"] = mitigations
                parts.append("Mitigations: " + "; ".join(mitigations))
            if assets:
                meta["targeted_assets"] = assets
                parts.append("Targeted assets: " + "; ".join(assets))
        elif doc_type == "mitigation":
            addressed = [_name_of(t) for rt, t in outgoing.get(o["id"], []) if rt == "mitigates"]
            if addressed:
                meta["addresses_techniques"] = addressed
                parts.append("Addresses techniques: " + "; ".join(addressed))

        text = clean_text("\n\n".join(p for p in parts if p and p.strip()))
        if not text:
            continue

        yield Document(
            doc_id=f"mitre-ics:{attack_id or o['id']}",
            source="mitre",
            doc_type=doc_type,
            title=name,
            text=text,
            url=url or "",
            section_path=[],
            source_published=(o.get("modified") or o.get("created", ""))[:10] or None,
            metadata=meta,
        )
