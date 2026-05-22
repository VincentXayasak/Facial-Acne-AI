"""Retrieve and format Chroma chunks for the Llama prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.ingest import query_collection
from rag.topics import (
    CORPUS_TOPICS,
    TOPIC_SEARCH_TERMS,
    infer_topics_from_observation,
)


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


def _merge_chunk(
    best: dict[str, RetrievedChunk],
    doc: str,
    meta: dict,
    dist: float,
) -> None:
    key = _dedupe_key(meta)
    chunk = RetrievedChunk(
        index=0,
        text=doc,
        title=str(meta.get("title", "Unknown")),
        section=str(meta.get("section", "")),
        base_name=str(meta.get("base_name", "")),
        distance=float(dist),
    )
    if key not in best or dist < best[key].distance:
        best[key] = chunk


def retrieve_chunks(
    query_text: str,
    *,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction,
    top_k: int = 5,
    fetch_k: int | None = None,
    extra_queries: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k unique (paper, section) chunks; optional multi-query merge."""
    queries = [query_text]
    if extra_queries:
        for q in extra_queries:
            if q and q not in queries:
                queries.append(q)

    per_query_k = fetch_k or max(top_k * 2, 8)
    best: dict[str, RetrievedChunk] = {}

    for q in queries:
        raw = query_collection(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_fn=embedding_fn,
            query_text=q,
            n_results=per_query_k,
        )
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            _merge_chunk(best, doc, meta, dist)

    ranked = sorted(best.values(), key=lambda c: c.distance)[:top_k]
    return [
        RetrievedChunk(
            index=i + 1,
            text=c.text,
            title=c.title,
            section=c.section,
            base_name=c.base_name,
            distance=c.distance,
        )
        for i, c in enumerate(ranked)
    ]


def format_context_block(chunk: RetrievedChunk) -> str:
    header = f"[{chunk.index}] {chunk.title}"
    if chunk.section:
        header += f" — {chunk.section}"
    if chunk.base_name:
        header += f" (id: {chunk.base_name})"
    return f"{header}\n{chunk.text}"


def _format_observation_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_retrieval_query(question: str, observation: dict | None) -> str:
    """Primary embedding query: question + vision + topic vocabulary."""
    if not observation:
        return question

    parts = [question]
    field_map = [
        ("acne_types", "lesions"),
        ("severity_estimate", "severity"),
        ("inflammation_level", "inflammation"),
        ("affected_areas", "areas"),
        ("scarring_or_post_inflammatory_marks", "marks"),
        ("apparent_age_group", "age group"),
        ("other_observations", "clinical notes"),
    ]
    for key, label in field_map:
        value = observation.get(key)
        if value and str(value).lower() not in ("unclear", "none", "none visible", ""):
            parts.append(f"{label}: {_format_observation_value(value)}")

    keywords = observation.get("retrieval_keywords") or []
    if keywords:
        parts.append("keywords: " + ", ".join(str(k) for k in keywords[:15]))

    topics = observation.get("relevant_research_topics") or infer_topics_from_observation(
        observation
    )
    for topic_id in topics:
        label = CORPUS_TOPICS.get(topic_id, topic_id)
        parts.append(label)

    return " | ".join(parts)


def build_extra_retrieval_queries(
    question: str, observation: dict | None
) -> list[str]:
    """Topic-focused queries so retrieval spans hormones, bacteria, treatments, diet, etc."""
    if not observation:
        return []

    extra: list[str] = []
    topics = observation.get("relevant_research_topics") or infer_topics_from_observation(
        observation
    )
    keywords = observation.get("retrieval_keywords") or []

    for topic_id in topics[:4]:
        terms = TOPIC_SEARCH_TERMS.get(topic_id, [])
        if terms:
            extra.append(f"{question} {' '.join(terms[:10])}")

    if keywords:
        extra.append(f"{question} {' '.join(str(k) for k in keywords[:12])}")

    return extra


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    observation: dict | None = None,
) -> str:
    context = "\n\n".join(format_context_block(c) for c in chunks)
    obs_block = ""
    topic_block = ""
    if observation:
        obs_block = (
            "Vision model observation (from user photo; not a medical diagnosis):\n"
            f"{json.dumps(observation, indent=2)}\n\n"
        )
        topics = observation.get("relevant_research_topics") or []
        if topics:
            labels = [CORPUS_TOPICS.get(t, t) for t in topics]
            topic_block = (
                "Likely relevant research areas for this case: "
                + "; ".join(labels)
                + "\n\n"
            )

    return (
        f"{obs_block}{topic_block}"
        f"User question: {question}\n\n"
        f"Research context ({len(chunks)} paper excerpts — for your facts only, do not copy their wording):\n"
        f"{context}\n\n"
        "Write a clear, easy-to-read answer using ONLY the facts in the excerpts above.\n"
        "Structure example:\n"
        "1) Brief intro (what you see in the photo if provided + what the question is about)\n"
        "2) **Main causes** — numbered list in plain English, with [1], [2] citations\n"
        "3) **What research suggests for care** — numbered practical steps or options, with citations\n"
        "4) **Extra tips** — short bullets if sources support them\n"
        "5) Short closing reminder (patience, see a dermatologist if needed)\n"
        "6) **Sources** — list paper titles\n\n"
        "Remember: simplify the science. No words like pathogenesis, phylotypes, dysbiosis, or "
        "follicular hyper-keratinization unless you immediately explain them in simple terms."
    )


def build_base_prompt(question: str, observation: dict | None = None) -> str:
    """User prompt for Llama without RAG (Gemini observation optional)."""
    obs_block = ""
    if observation:
        obs_block = (
            "Vision model observation (from user photo; not medical diagnosis):\n"
            f"{json.dumps(observation, indent=2)}\n\n"
        )

    return (
        f"{obs_block}"
        f"User question: {question}\n\n"
        "Answer using your general knowledge. Do not cite research papers."
    )
