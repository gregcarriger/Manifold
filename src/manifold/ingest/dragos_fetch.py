"""Fetch licensed Dragos WorldView data via the WorldView API into ``corpus/raw/dragos/``.

This is the *acquisition* step for the local-only ``dragos`` source; ``dragos.py`` then
normalizes whatever lands on disk. Nothing fetched here is ever committed (the whole
directory is git-ignored). Requires a WorldView subscription and API credentials.

The API (https://portal.dragos.com/api/v1) serves product **metadata**, per-product
**IOCs** (``/products/{id}/csv`` and ``/stix2``), the global **/indicators** feed, and
**/tags** (where threat groups and vulnerabilities live). It does *not* serve the report
narrative PDFs — those are downloaded from the portal UI and dropped into ``reports/``
alongside what this tool fetches (both share a serial prefix, so the loader groups them).

Credentials (put in ``.env`` — never committed), one of:
  * ``DRAGOS_API_TOKEN`` + ``DRAGOS_API_SECRET``  (API-Token / API-Secret headers), or
  * ``DRAGOS_NPE_TOKEN``                          (Authorization header).
Optional ``DRAGOS_PERSPECTIVE`` (e.g. ``Manufacturing``) restricts the pull client-side.

Usage:
    python -m manifold.ingest.dragos_fetch --probe   # dump sample responses (schema discovery)
    python -m manifold.ingest.dragos_fetch           # full pull into corpus/raw/dragos/
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

BASE = "https://portal.dragos.com/api/v1"

# Org-wide API budget: 60 requests/min and 1000/week, shared across all members. We count
# and throttle every request so a pull can't silently blow the weekly budget.
_REQUEST_COUNT = 0
_MIN_INTERVAL = 1.1  # seconds between requests -> <=60/min
_LAST_CALL = 0.0


def _throttle() -> None:
    global _LAST_CALL
    wait = _MIN_INTERVAL - (time.monotonic() - _LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL = time.monotonic()


class ApiError(Exception):
    """A non-transient HTTP error from the WorldView API (carries the status code)."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
RAW = os.path.join(_ROOT, "corpus", "raw", "dragos")


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): populate os.environ for keys not already set."""
    path = os.path.join(_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _auth_headers() -> dict[str, str]:
    token, secret = os.environ.get("DRAGOS_API_TOKEN"), os.environ.get("DRAGOS_API_SECRET")
    npe = os.environ.get("DRAGOS_NPE_TOKEN")
    if token and secret:
        return {"API-Token": token, "API-Secret": secret}
    if npe:
        return {"Authorization": npe}
    raise SystemExit(
        "No Dragos credentials found. Set DRAGOS_API_TOKEN + DRAGOS_API_SECRET "
        "(or DRAGOS_NPE_TOKEN) in .env — see .env.example."
    )


def _request(path: str, params: dict[str, Any] | None = None) -> tuple[bytes, str]:
    """GET a path (relative to BASE). Returns (body_bytes, content_type)."""
    qs = ""
    if params:
        # doseq handles list params like serials[]/serial[]/tags[].
        qs = "?" + urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE}{path}{qs}"
    req = urllib.request.Request(url, headers={**_auth_headers(), "Accept": "*/*"})
    global _REQUEST_COUNT
    for attempt in range(4):
        try:
            _throttle()
            with urllib.request.urlopen(req, timeout=60) as resp:
                _REQUEST_COUNT += 1
                return resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:  # rate limited / transient
                time.sleep(2 * (attempt + 1))
                continue
            body = e.read().decode("utf-8", "replace")[:300]
            raise ApiError(e.code, f"{url}\n{body}") from e
    raise ApiError(0, f"Repeated failures for {url}")


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    body, _ = _request(path, params)
    return json.loads(body)


def _ppath(serial: str, sub: str) -> str:
    """Build a /products/<serial>/<sub> path, URL-encoding the serial (some carry spaces
    or '&', e.g. 'DOM-2022-47 & 48')."""
    return f"/products/{urllib.parse.quote(str(serial), safe='')}/{sub}"


def _items(payload: Any) -> list[dict]:
    """Tolerant list extraction: a bare list, or the first list-valued field of an object."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "products", "indicators", "tags", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        for v in payload.values():
            if isinstance(v, list):
                return v
    return []


def _pick(item: dict, *names: str) -> Any:
    for n in names:
        if item.get(n) not in (None, ""):
            return item[n]
    return None


def _paginate(path: str, params: dict[str, Any], page_size: int = 500) -> Iterator[dict]:
    page = 1
    while True:
        payload = _get_json(path, {**params, "page": str(page), "page_size": str(page_size)})
        batch = _items(payload)
        if not batch:
            return
        yield from batch
        if len(batch) < page_size:
            return
        page += 1  # throttling handled in _request


# ---------------------------------------------------------------------------- probe

def probe() -> None:
    """Dump small samples of each endpoint for schema discovery (into RAW/_probe)."""
    out = os.path.join(RAW, "_probe")
    os.makedirs(out, exist_ok=True)

    def dump(name: str, obj: Any) -> None:
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    products = _get_json("/products", {"page": "1", "page_size": "3"})
    dump("products_page1.json", products)
    plist = _items(products)
    print(f"/products: top-level={_shape(products)}  n_items={len(plist)}")
    if plist:
        print(f"  first product keys: {sorted(plist[0].keys())}")
        print(f"  sample types: {sorted({str(_pick(p, 'type')) for p in plist})}")
        print(f"  sample report_link: {plist[0].get('report_link')!r}")
        # Prefer a product that actually has IOCs so the csv/stix2 sample is non-empty.
        with_iocs = [p for p in plist if (p.get("ioc_count") or 0) > 0]
        target = (with_iocs or plist)[0]
        pid = _pick(target, "id", "serial", "product_id")
        if pid is not None:
            meta = _get_json(f"/products/{pid}")
            dump("product_detail.json", meta)
            print(f"  /products/{pid} keys: {sorted(meta.keys())}")
            for kind in ("csv", "stix2"):
                try:
                    body, ct = _request(f"/products/{pid}/{kind}")
                    ext = "csv" if kind == "csv" else "stix2.json"
                    with open(os.path.join(out, f"product_iocs.{ext}"), "wb") as f:
                        f.write(body)
                    print(f"  /products/{pid}/{kind}: {ct}, {len(body)} bytes")
                except ApiError as e:
                    print(f"  /products/{pid}/{kind}: skipped ({e})")

    indicators = _get_json("/indicators", {"page": "1", "page_size": "5"})
    dump("indicators_page1.json", indicators)
    ilist = _items(indicators)
    print(f"/indicators: top-level={_shape(indicators)}  n_items={len(ilist)}")
    if ilist:
        print(f"  first indicator keys: {sorted(ilist[0].keys())}")

    tags = _get_json("/tags", {"page": "1", "page_size": "50"})
    dump("tags_page1.json", tags)
    tlist = _items(tags)
    print(f"/tags: top-level={_shape(tags)}  n_items={len(tlist)}")
    if tlist:
        print(f"  first tag keys: {sorted(tlist[0].keys())}")
        types = sorted({str(_pick(t, 'tag_type', 'type')) for t in tlist})
        print(f"  observed tag_type values: {types}")
    print(f"\nProbe written to {out} — inspect these to finalize field mapping.")


def _shape(obj: Any) -> str:
    if isinstance(obj, dict):
        return f"object(keys={sorted(obj.keys())})"
    if isinstance(obj, list):
        return f"list(len={len(obj)})"
    return type(obj).__name__


# ---------------------------------------------------------------------------- pull

def _tags_of_type(product: dict, tag_type: str) -> list[str]:
    return [
        t.get("text")
        for t in (product.get("tags") or [])
        if t.get("tag_type") == tag_type and t.get("text")
    ]


def _matches_perspective(product: dict, perspective: str | None) -> bool:
    """A product is in-perspective if it carries the sector as an Industry tag."""
    if not perspective:
        return True
    return any(perspective.lower() == i.lower() for i in _tags_of_type(product, "Industry"))


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def pull(
    perspective: str | None,
    with_iocs: bool = False,
    with_pdfs: bool = False,
    cve_pages: int = 0,
    max_products: int = 0,
    all_pdf_types: bool = False,
) -> None:
    reports_dir = os.path.join(RAW, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # --- Tier 0 (cheap): product list + Manufacturing filter + metadata/exec-summary. ---
    print(f"Fetching product catalog (filter perspective={perspective or 'ALL'}) ...")
    all_products = list(_paginate("/products", {"sort_by": "release_date"}, page_size=500))
    scoped = [p for p in all_products if _matches_perspective(p, perspective)]
    with_ioc_count = sum(1 for p in scoped if (p.get("ioc_count") or 0) > 0)
    _write(os.path.join(RAW, "products_index.json"), json.dumps(scoped, indent=2).encode())
    print(f"  {len(all_products)} products total; {len(scoped)} in perspective "
          f"({with_ioc_count} have IOCs).  -> products_index.json")

    # Threat groups: aggregate ThreatGroup tags across the scoped products (no extra calls).
    groups: dict[str, list[str]] = {}
    for p in scoped:
        for g in _tags_of_type(p, "ThreatGroup"):
            groups.setdefault(g, []).append(p.get("serial"))
    _write(os.path.join(RAW, "threat_groups.json"), json.dumps(groups, indent=2).encode())
    print(f"  {len(groups)} threat groups referenced -> threat_groups.json")

    # --- Tier 1 (opt-in, bounded): per-product IOC CSVs for scoped products with IOCs. ---
    if with_iocs:
        todo = [p for p in scoped if (p.get("ioc_count") or 0) > 0]
        print(f"IOCs: fetching CSVs for {len(todo)} products (skipping any already on disk) ...")
        for p in todo:
            serial = p.get("serial")
            dest = os.path.join(reports_dir, f"{serial}-IOCs-api.csv")
            if os.path.exists(dest):
                continue
            try:
                body, _ = _request(_ppath(serial, "csv"))
                if body.strip():
                    _write(dest, body)
            except ApiError as e:
                if e.code != 404:
                    print(f"  [warn] {serial} csv: {e}")
            except Exception as e:  # noqa: BLE001 - one bad product must not abort the pull
                print(f"  [warn] {serial} csv: {e}")

    # --- Tier 2 (opt-in, expensive): report PDFs (1 request each). ---
    if with_pdfs:
        candidates = scoped
        if not all_pdf_types:
            # Suspect Domain Reports are auto-generated IOC dumps with no narrative; their
            # IOCs come via the IOC tier, so skip their PDFs by default.
            candidates = [p for p in scoped if "suspect domain" not in str(p.get("type", "")).lower()]
            print(f"  (skipping {len(scoped) - len(candidates)} Suspect Domain reports; "
                  f"use --all-pdf-types to include)")
        todo = candidates[:max_products] if max_products else candidates
        print(f"PDFs: fetching report PDFs for up to {len(todo)} products "
              f"(skipping any already on disk) ...")
        for p in todo:
            serial = p.get("serial")
            dest = os.path.join(reports_dir, f"{serial}-report.pdf")
            if os.path.exists(dest):
                continue
            try:
                body, ct = _request(_ppath(serial, "report"))
                if b"%PDF" in body[:8]:
                    _write(dest, body)
                else:
                    print(f"  [warn] {serial} report not a PDF (ct={ct})")
            except ApiError as e:
                if e.code != 404:
                    print(f"  [warn] {serial} report: {e}")
            except Exception as e:  # noqa: BLE001 - one bad product must not abort the pull
                print(f"  [warn] {serial} report: {e}")

    # --- Tier 3 (opt-in, expensive): Dragos-scored CVEs (vulnerabilities). ---
    if cve_pages > 0:
        print(f"Vulnerabilities: fetching up to {cve_pages} pages of CVE tags ...")
        cves: list[dict] = []
        for page in range(1, cve_pages + 1):
            batch = _items(_get_json(
                "/tags", {"tag_type": "CVE", "page": str(page), "page_size": "500"}
            ))
            if not batch:
                break
            cves.extend(batch)
        _write(os.path.join(RAW, "vulnerabilities.json"), json.dumps(cves, indent=2).encode())
        print(f"  {len(cves)} CVE records -> vulnerabilities.json")

    print(f"\nDone. API requests this run: {_REQUEST_COUNT} "
          f"(org budget: 60/min, 1000/week, shared).")
    print("Next: python -m manifold.ingest.run --source dragos")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Dragos WorldView data (licensed, local-only).")
    ap.add_argument("--probe", action="store_true", help="dump sample responses for schema discovery")
    ap.add_argument("--perspective", default=None, help="override DRAGOS_PERSPECTIVE (e.g. Manufacturing)")
    ap.add_argument("--with-iocs", action="store_true", help="also fetch per-product IOC CSVs")
    ap.add_argument("--with-pdfs", action="store_true", help="also fetch report PDFs (1 request each)")
    ap.add_argument("--max-products", type=int, default=0, help="cap PDF fetches (0 = all scoped)")
    ap.add_argument("--cve-pages", type=int, default=0, help="pages of Dragos CVE scoring to pull (500/page)")
    ap.add_argument("--all-pdf-types", action="store_true", help="include Suspect Domain report PDFs")
    args = ap.parse_args()

    _load_dotenv()
    if args.probe:
        probe()
        return
    perspective = args.perspective or os.environ.get("DRAGOS_PERSPECTIVE")
    pull(
        perspective,
        with_iocs=args.with_iocs,
        with_pdfs=args.with_pdfs,
        cve_pages=args.cve_pages,
        max_products=args.max_products,
        all_pdf_types=args.all_pdf_types,
    )


if __name__ == "__main__":
    main()
