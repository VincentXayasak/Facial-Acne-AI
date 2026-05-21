#!/usr/bin/env python3
"""
Ingest Marker markdown papers into ChromaDB for RAG.

Pipeline: clean → markdown header split → recursive token chunks → LM Studio embeddings.

Prerequisites:
  1. LM Studio: load an *embedding* model (e.g. nomic-embed-text), start the local server.
  2. pip install -r requirements-ingest.txt
  3. cp .env.example .env   # optional; defaults work for stock LM Studio

Examples:
  python ingest_papers.py --dry-run
  python ingest_papers.py --reset
  python ingest_papers.py --max-docs 3 --reset
  python ingest_papers.py --ping
  python ingest_papers.py --query "benzoyl peroxide inflammatory acne"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.embeddings import LMStudioEmbeddingFunction
from rag.ingest import ingest_manifest, load_manifest_records, query_collection
from rag.chunk import chunk_all

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest acne research markdown into ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "markdown" / "manifest.json",
        help="manifest.json from convert_pdfs.py",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=SCRIPT_DIR / "chroma_db",
        help="Persistent Chroma directory",
    )
    parser.add_argument(
        "--collection",
        default="acne_research",
        help="Chroma collection name",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before ingest",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Only ingest the first N papers (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk only; do not embed or write to Chroma",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size in tokens",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Token overlap between chunks",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Test LM Studio embedding endpoint and exit",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a test retrieval query against an existing collection",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return for --query",
    )
    return parser.parse_args()


def print_query_results(results: dict) -> None:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    print(f"\nTop {len(docs)} results:\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        title = meta.get("title", "?")
        section = meta.get("section", "?")
        base = meta.get("base_name", "?")
        preview = doc.replace("\n", " ")[:240]
        print(f"[{i}] {title}")
        print(f"    paper={base}  section={section}  distance={dist:.4f}")
        print(f"    {preview}...\n")


def main() -> int:
    load_dotenv(SCRIPT_DIR / ".env")
    args = parse_args()

    embed_fn = LMStudioEmbeddingFunction()

    if args.ping:
        print("Pinging LM Studio embeddings...")
        try:
            dim = embed_fn.ping()
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print(
                "\nCheck: LM Studio server is running, an embedding model is loaded, "
                "and LM_STUDIO_EMBED_MODEL matches the model id.",
                file=sys.stderr,
            )
            if "No embedding data" in str(exc):
                print(
                    "Tip: LM_STUDIO_BASE_URL can be http://HOST:1234 or http://HOST:1234/v1 "
                    "(both work after the URL fix).",
                    file=sys.stderr,
                )
            return 1
        print(
            f"OK — model={embed_fn.model}  base_url={embed_fn.base_url}  dim={dim}"
        )
        return 0

    if args.query:
        print(f"Query: {args.query!r}")
        try:
            results = query_collection(
                chroma_path=args.chroma_path,
                collection_name=args.collection,
                embedding_fn=embed_fn,
                query_text=args.query,
                n_results=args.top_k,
            )
        except Exception as exc:
            print(f"Query failed: {exc}", file=sys.stderr)
            return 1
        print_query_results(results)
        return 0

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    print(f"Manifest: {args.manifest}")
    print(f"Chroma:   {args.chroma_path}  collection={args.collection}")
    if args.dry_run:
        print("Mode:     dry-run (no Chroma write)")

    try:
        stats = ingest_manifest(
            manifest_path=args.manifest,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            embedding_fn=embed_fn,
            reset=args.reset,
            max_docs=args.max_docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        if not args.dry_run:
            print(
                "\nTip: run  python ingest_papers.py --ping  "
                "after starting LM Studio with an embedding model.",
                file=sys.stderr,
            )
        return 1

    print(json.dumps(stats, indent=2))
    if args.dry_run:
        records = load_manifest_records(args.manifest, max_docs=args.max_docs)
        sample = chunk_all(records[:1], chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if sample:
            print("\nSample chunk (first paper, first chunk):")
            print("-" * 60)
            print(sample[0].page_content[:800])
            if len(sample[0].page_content) > 800:
                print("...")
    else:
        print("\nDone. Test retrieval:")
        print(
            '  python ingest_papers.py --query "omega-3 fatty acids acne inflammation"'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
