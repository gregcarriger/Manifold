# Dragos WorldView — Optional Licensed Intelligence (not included)

[Dragos WorldView](https://www.dragos.com/threat-intelligence/) is a commercial OT/ICS
threat-intelligence product: intelligence **Reports**, **Vulnerabilities** (Dragos-scored,
"now/next/never"), **Indicators** (IOCs), and **Threat Groups** (e.g. KAMACITE, ELECTRUM,
VOLTZITE). It is a high-value, OT-native complement to the public Manifold corpus.

**It is not included in this repository, and never will be.** WorldView is proprietary,
licensed content. Redistributing it publicly is not permitted by the subscription. The
`.gitignore` in this directory blocks every file except this README and the manifest
template, so a licensed copy dropped here locally can never be committed by accident.

## For someone who holds a WorldView subscription

Data is downloaded from the portal (<https://portal.dragos.com/#/products>) with a
**perspective** selected (e.g. *Manufacturing*) so the export is scoped to the sectors you
care about. The loader auto-activates when files are present and is skipped otherwise, so
the pipeline still runs on the public corpus alone.

### Report bundles → `reports/`

Each WorldView report exports as up to three files sharing a serial prefix
(`<PREFIX>-<YEAR>-<N>`, e.g. `AIR-2026-14`). Drop them into `reports/`:

- `<serial> <title>.pdf` — the narrative analysis (the primary retrieval content).
- `<serial>-<ts>.stix2.json` — STIX 2.x bundle (indicators, threat-actors, ATT&CK
  patterns, relationships). Used to enrich the report's metadata.
- `<serial>-IOCs-<ts>.csv` — the indicator list. Emitted as a separately retrievable
  indicators document (so an IOC value like a domain or hash is findable by exact match).

PDF-only products (e.g. weekly ransomware summaries) are fine — the STIX/CSV are optional.

### Provenance (optional but recommended)

Copy `manifest.template.json` to `manifest.json` and record the `perspective` and
`pulled_at` date. It is git-ignored. The perspective is stamped onto every document's
metadata so any Dragos-augmented result is clearly labeled.

Then re-run ingestion:

```bash
python -m manifold.ingest.run --source dragos   # Dragos only
python -m manifold.ingest.run                    # all sources (Dragos included if present)
```

See `src/manifold/ingest/dragos.py`.

## Provenance note for the Manifold benchmark

The public gold set and all **published** evaluation results are computed on the **public
corpus only** (NIST SP 800-82r3, CISA ICS advisories, MITRE ATT&CK for ICS), so the
numbers are reproducible by anyone. Any WorldView-augmented run is a **private, local
extension** — reported separately, with the *delta* shared but never the underlying Dragos
data. Every Dragos document carries `metadata.license = "proprietary-local-only"`.
