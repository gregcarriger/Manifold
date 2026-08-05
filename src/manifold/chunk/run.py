"""Chunking orchestrator: turn the normalized corpus into retrieval passages.

Usage:
    python -m manifold.chunk.run                       # run all strategies
    python -m manifold.chunk.run --strategy structure  # one strategy
    python -m manifold.chunk.run --compare             # print a side-by-side table

Reads ``corpus/normalized/documents.jsonl`` and writes ``corpus/chunks/<strategy>.jsonl``
plus a per-strategy stats block and (with --compare) a comparison table.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from collections import Counter

from ..schema import Chunk, read_jsonl, write_chunks_jsonl
from .strategies import STRATEGIES, ChunkConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
NORMALIZED = os.path.join(_ROOT, "corpus", "normalized", "documents.jsonl")
OUT_DIR = os.path.join(_ROOT, "corpus", "chunks")


def _stats(chunks: list[Chunk], cfg: ChunkConfig) -> dict:
    tok = [c.token_estimate for c in chunks]
    per_doc = Counter(c.doc_id for c in chunks)
    return {
        "chunks": len(chunks),
        "per_source": dict(Counter(c.source for c in chunks)),
        "tokens_mean": round(st.mean(tok), 1) if tok else 0,
        "tokens_median": int(st.median(tok)) if tok else 0,
        "tokens_p95": int(sorted(tok)[int(len(tok) * 0.95)]) if tok else 0,
        "tokens_max": max(tok) if tok else 0,
        "chunks_per_doc_mean": round(len(chunks) / len(per_doc), 2) if per_doc else 0,
        "oversized_gt_max": sum(1 for t in tok if t > cfg.max_tokens),
        "tiny_lt_min": sum(1 for t in tok if t < cfg.min_tokens),
        "with_context_header": sum(1 for c in chunks if c.context_header),
    }


def run_strategy(name: str, docs: list, cfg: ChunkConfig) -> tuple[list[Chunk], dict]:
    fn = STRATEGIES[name]
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(fn(d, cfg))
    return chunks, _stats(chunks, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk the normalized corpus.")
    ap.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    ap.add_argument("--target-tokens", type=int, default=512)
    ap.add_argument("--overlap-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--compare", action="store_true", help="print side-by-side stats")
    ap.add_argument("--normalized", default=NORMALIZED, help="input corpus (default: public)")
    ap.add_argument("--out-dir", default=OUT_DIR, help="chunk output dir (default: corpus/chunks)")
    args = ap.parse_args()

    cfg = ChunkConfig(
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
        max_tokens=args.max_tokens,
    )
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    docs = list(read_jsonl(args.normalized))
    print(f"loaded {len(docs)} documents from {args.normalized}")

    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    all_stats: dict[str, dict] = {}
    for name in names:
        chunks, stats = run_strategy(name, docs, cfg)
        out = os.path.join(out_dir, f"{name}.jsonl")
        write_chunks_jsonl(chunks, out)
        all_stats[name] = stats
        print(f"[{name}] {len(chunks)} chunks -> {out}")

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "strategies": all_stats}, f, indent=2)

    if args.compare and len(all_stats) > 1:
        keys = ["chunks", "chunks_per_doc_mean", "tokens_mean", "tokens_median",
                "tokens_p95", "tokens_max", "oversized_gt_max", "tiny_lt_min",
                "with_context_header"]
        names_sorted = list(all_stats)
        w = max(len(k) for k in keys) + 2
        print("\n" + "metric".ljust(w) + "".join(n.rjust(14) for n in names_sorted))
        print("-" * (w + 14 * len(names_sorted)))
        for k in keys:
            row = k.ljust(w) + "".join(str(all_stats[n][k]).rjust(14) for n in names_sorted)
            print(row)
    else:
        print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    main()
