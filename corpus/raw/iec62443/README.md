# IEC 62443 — Optional Licensed Corpus (not included)

The IEC 62443 / ISA-62443 series (*Security for Industrial Automation and Control
Systems*) is the central international standard for OT/ICS security. It is a natural,
high-value addition to the Manifold corpus.

**It is not included in this repository, and never will be.** IEC 62443 is a
copyrighted, paywalled standard published by the IEC and ISA. Redistributing its text
publicly is not permitted. The `.gitignore` in this directory blocks every file except
this README and the manifest template, so a licensed copy dropped here locally can never
be committed by accident.

## For someone who holds a license

If you have legitimate access to the IEC 62443 documents (e.g. an organizational
subscription such as an ISA membership or an enterprise standards license), you can
extend the corpus locally without changing any pipeline code:

1. Place the PDFs in this directory. Recommended parts (most relevant to OT security):
   - **IEC 62443-3-3** — System security requirements and security levels
   - **IEC 62443-3-2** — Security risk assessment for system design
   - **IEC 62443-2-1** — Security program requirements for IACS asset owners
   - **IEC 62443-4-2** — Technical security requirements for IACS components
   - **IEC 62443-4-1** — Secure product development lifecycle requirements
2. Copy `manifest.template.json` to `manifest.json` and fill in one entry per file
   (title, part number, edition/year, local filename).
3. Re-run ingestion. The IEC loader auto-activates when `manifest.json` is present and
   the referenced files exist; otherwise it is skipped and the pipeline runs on the
   public corpus only. See `src/manifold/ingest/iec62443.py`.

## What ships publicly instead

The public Manifold corpus stands on its own using copyright-clean sources that cover
much of the same control ground: **NIST SP 800-82r3** (OT security controls, public
domain), **CISA ICS advisories** (TLP:WHITE), and **MITRE ATT&CK for ICS**. IEC 62443 is
purely an optional enrichment for users who already hold the rights.

> Provenance note for the Manifold benchmark: the public gold-set and all published
> evaluation results are computed on the **public corpus only**, so the numbers are
> reproducible by anyone. Any IEC-62443-augmented run is a private, local extension and
> is reported separately, never mixed into the public benchmark.
