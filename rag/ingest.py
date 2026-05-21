"""Load manifest, chunk papers, write to ChromaDB."""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata
from langchain_core.documents import Document

from rag.chunk import PaperRecord, chunk_all
from rag.embeddings import LMStudioEmbeddingFunction

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def _sanitize_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    return safe[:200]


def load_manifest_records(
    manifest_path: Path,
    *,
    max_docs: int | None = None,
) -> list[PaperRecord]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[PaperRecord] = []

    for doc in manifest.get("documents", []):
        md_path = doc.get("markdown_path")
        if not md_path:
            continue
        path = Path(md_path)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue

        records.append(
            PaperRecord(
                base_name=doc.get("base_name") or path.stem,
                title=(doc.get("title") or "").strip(),
                pdf_path=doc.get("pdf") or "",
                markdown_path=str(path),
                text=text,
            )
        )
        if max_docs and len(records) >= max_docs:
            break

    return records


def _chunk_id(base_name: str, chunk_index: int) -> str:
    return _sanitize_id(f"{base_name}::{chunk_index}")


def _chroma_metadata(doc: Document) -> Metadata:
    meta: Metadata = {}
    for key, value in doc.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = str(value)
    return meta


def ingest_manifest(
    *,
    manifest_path: Path,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction,
    reset: bool = False,
    max_docs: int | None = None,
    batch_size: int = 100,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dry_run: bool = False,
) -> dict:
    records = load_manifest_records(manifest_path, max_docs=max_docs)
    if not records:
        raise RuntimeError(f"No markdown documents found via {manifest_path}")

    all_chunks: list[Document] = chunk_all(
        records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    stats = {
        "papers": len(records),
        "chunks": len(all_chunks),
        "dry_run": dry_run,
        "ingested": 0,
    }

    if dry_run:
        return stats

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        collection.add(
            ids=[_chunk_id(d.metadata["base_name"], d.metadata["chunk_index"]) for d in batch],
            documents=[d.page_content for d in batch],
            metadatas=[_chroma_metadata(d) for d in batch],
        )
        stats["ingested"] += len(batch)

    return stats


def query_collection(
    *,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction,
    query_text: str,
    n_results: int = 5,
) -> dict:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
