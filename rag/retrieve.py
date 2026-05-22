"""Retrieve and format Chroma chunks for the Llama prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.ingest import query_collection


@dataclass(frozen=True)
class RetrievedChunk:
    index: int
    text: str
    title: str
    section: str
    base_name: str
    distance: float


def _dedupe_key(meta: dict) -> str:
    return f"{meta.get('base_name', '')}::{meta.get('section', '')}"


def retrieve_chunks(
    query_text: str,
    *,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction,
    top_k: int = 5,
    fetch_k: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k unique (paper, section) chunks."""
    fetch_k = fetch_k or max(top_k * 3, top_k)
    raw = query_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        query_text=query_text,
        n_results=fetch_k,
    )

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    seen: set[str] = set()
    chunks: list[RetrievedChunk] = []

    for doc, meta, dist in zip(docs, metas, dists):
        key = _dedupe_key(meta)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            RetrievedChunk(
                index=len(chunks) + 1,
                text=doc,
                title=str(meta.get("title", "Unknown")),
                section=str(meta.get("section", "")),
                base_name=str(meta.get("base_name", "")),
                distance=float(dist),
            )
        )
        if len(chunks) >= top_k:
            break

    return chunks


def format_context_block(chunk: RetrievedChunk) -> str:
    header = f"[{chunk.index}] {chunk.title}"
    if chunk.section:
        header += f" — {chunk.section}"
    if chunk.base_name:
        header += f" (id: {chunk.base_name})"
    return f"{header}\n{chunk.text}"


def build_retrieval_query(question: str, observation: dict | None) -> str:
    """Combine user question with Gemini vision fields for embedding search."""
    if not observation:
        return question

    parts = [question]
    field_map = [
        ("acne_types", "acne types"),
        ("severity_estimate", "severity"),
        ("affected_areas", "areas"),
        ("skin_tone_description", "skin tone"),
        ("apparent_sex_presentation", "presentation"),
        ("other_observations", "notes"),
    ]
    for key, label in field_map:
        value = observation.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{label}: {value}")
    return " | ".join(parts)


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    observation: dict | None = None,
) -> str:
    context = "\n\n".join(format_context_block(c) for c in chunks)
    obs_block = ""
    if observation:
        obs_block = (
            "Vision model observation (from user photo; not medical diagnosis):\n"
            f"{json.dumps(observation, indent=2)}\n\n"
        )

    return (
        f"{obs_block}"
        f"User question: {question}\n\n"
        f"Research context:\n{context}\n\n"
        "Write a clear, helpful answer grounded in the research context. "
        "Use inline citations [1], [2], etc."
    )
