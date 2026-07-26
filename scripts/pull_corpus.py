#!/usr/bin/env python3
"""Reproducibly pull the public raw corpus into corpus/raw/.

The raw dumps are intentionally kept out of git (large, redistribution nuance); this
script regenerates them from authoritative sources so the pipeline is reproducible from a
clean checkout. IEC 62443 is licensed and never pulled here.

Usage:
    python scripts/pull_corpus.py                              # everything, default location
    python scripts/pull_corpus.py --source mitre               # one source
    python scripts/pull_corpus.py --cisa-years 2024 2025 2026  # recent CISA advisories only
    python scripts/pull_corpus.py --out /tmp/manifold-corpus   # custom output dir

With no --cisa-years, all ~3,800 CISA OT advisories in the feed are pulled.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import urllib.request

DEFAULT_RAW = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "corpus", "raw")
)

NIST_URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf"
MITRE_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/ics-attack/ics-attack.json"
CISA_FEED_URL = "https://raw.githubusercontent.com/cisagov/CSAF/develop/csaf_files/OT/white/cisa-csaf-ot-feed-tlp-white.json"

# Some CDNs reject Python's default urllib User-Agent; use a plain browser-ish one.
_HEADERS = {"User-Agent": "manifold-corpus-puller/0.1"}


def _fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())


def pull_nist(raw: str) -> None:
    print("[nist] NIST SP 800-82r3 PDF ...")
    _fetch(NIST_URL, os.path.join(raw, "nist", "NIST.SP.800-82r3.pdf"))
    print("[nist] done")


def pull_mitre(raw: str) -> None:
    print("[mitre] ATT&CK for ICS STIX ...")
    _fetch(MITRE_URL, os.path.join(raw, "mitre", "ics-attack.json"))
    print("[mitre] done")


def pull_cisa(raw: str, years: list[str] | None) -> None:
    print("[cisa] fetching CSAF OT feed manifest ...")
    req = urllib.request.Request(CISA_FEED_URL, headers=_HEADERS)
    with urllib.request.urlopen(req) as r:
        feed = json.load(r)
    entries = feed["feed"]["entry"]
    if years:
        yy = {y[-2:] for y in years}
        entries = [e for e in entries if e["id"].split("-")[1] in yy]
    adv_dir = os.path.join(raw, "cisa", "advisories")
    os.makedirs(adv_dir, exist_ok=True)
    urls = [e["content"]["src"] for e in entries]
    print(f"[cisa] downloading {len(urls)} advisories ...")

    failures: list[str] = []

    def one(u: str) -> None:
        try:
            _fetch(u, os.path.join(adv_dir, os.path.basename(u)))
        except Exception as e:  # noqa: BLE001 - keep going; report at the end
            failures.append(f"{os.path.basename(u)}: {e}")

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, urls))
    print(f"[cisa] done: {len(urls) - len(failures)}/{len(urls)} files")
    if failures:
        print(f"[cisa] {len(failures)} failed, e.g. {failures[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull the public raw corpus for Manifold.")
    ap.add_argument("--source", choices=["nist", "cisa", "mitre", "all"], default="all")
    ap.add_argument("--cisa-years", nargs="*", default=None,
                    help="restrict CISA pull to these years (e.g. 2024 2025 2026)")
    ap.add_argument("--out", default=DEFAULT_RAW,
                    help="output directory for corpus/raw (default: repo corpus/raw)")
    args = ap.parse_args()
    raw = os.path.abspath(args.out)
    print(f"corpus/raw -> {raw}")
    if args.source in ("nist", "all"):
        pull_nist(raw)
    if args.source in ("mitre", "all"):
        pull_mitre(raw)
    if args.source in ("cisa", "all"):
        pull_cisa(raw, args.cisa_years)
    print("done.")


if __name__ == "__main__":
    main()
