"""Show the same query across every retrieval config, side by side.

Usage:
    python -m manifold.retrieve.demo "modbus command injection on a PLC"
    python -m manifold.retrieve.demo --strategy fixed "CVE-2024-38434"
    python -m manifold.retrieve.demo            # runs a few illustrative built-in queries

The built-in queries include an exact CVE id and a model number — cases where lexical BM25
should beat dense, illustrating why hybrid exists.
"""

from __future__ import annotations

import argparse

from .config import RetrievalConfig
from .retriever import Retriever

_BUILTIN = [
    "how should safety instrumented systems be protected from cyber attack",  # semantic
    "CVE-2023-38545",                                                          # exact id
    "buffer overflow in Rockwell FactoryTalk",                                # mixed
]


def _print_config(r: Retriever, query: str, cfg: RetrievalConfig, k: int = 5) -> None:
    results = r.retrieve(query, cfg)
    print(f"\n  ── {cfg.label()} ──")
    if not results:
        print("     (no results)")
        return
    for res in results[:k]:
        comp = " ".join(f"{name}={v:.3f}" for name, v in res.scores.items()
                         if isinstance(v, (int, float)))
        print(f"     {res.rank}. [{res.source}] {res.doc_id:22} {res.title[:38]:38} | {comp}")


def run(query: str, strategy: str, r: Retriever) -> None:
    print("=" * 100)
    print(f"QUERY: {query!r}   (chunk set: {strategy})")
    configs = [
        RetrievalConfig(method="dense", strategy=strategy),
        RetrievalConfig(method="bm25", strategy=strategy),
        RetrievalConfig(method="hybrid", strategy=strategy),
        RetrievalConfig(method="hybrid", strategy=strategy, rerank=True),
    ]
    for cfg in configs:
        _print_config(r, query, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare retrieval configs on a query.")
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--strategy", choices=["fixed", "structure"], default="structure")
    args = ap.parse_args()

    r = Retriever()
    queries = [args.query] if args.query else _BUILTIN
    for q in queries:
        run(q, args.strategy, r)
    r.close()


if __name__ == "__main__":
    main()
