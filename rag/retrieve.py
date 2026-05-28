"""Retrieve and format Chroma chunks for the Llama prompt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.ingest import query_collection
from rag.relevance import (
    chunk_passes_relevance,
    is_causes_query,
    is_treatment_query,
)
from rag.clean import sanitize_inline_citations
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
    base = meta.get("base_name", "")
    idx = meta.get("chunk_index", "")
    return f"{base}::{idx}"


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


def _one_best_chunk_per_paper(
    candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    """Keep the closest chunk per paper so sources are diverse."""
    by_paper: dict[str, RetrievedChunk] = {}
    for chunk in sorted(candidates, key=lambda c: c.distance):
        if chunk.base_name not in by_paper:
            by_paper[chunk.base_name] = chunk
    ranked = sorted(by_paper.values(), key=lambda c: c.distance)[:top_k]
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


def retrieve_chunks(
    query_text: str,
    *,
    question: str | None = None,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction,
    top_k: int = 5,
    fetch_k: int | None = None,
    extra_queries: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks from distinct papers after relevance filtering."""
    question = question or query_text
    queries = [query_text]
    if extra_queries:
        for q in extra_queries:
            if q and q not in queries:
                queries.append(q)

    per_query_k = fetch_k or max(top_k * 5, 20)
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

    filtered = [
        c
        for c in best.values()
        if chunk_passes_relevance(c.text, c.title, c.distance, question=question)
    ]

    # If filters are too strict, fall back to best distance matches that at least mention acne.
    if len(filtered) < top_k:
        fallback = [
            c for c in sorted(best.values(), key=lambda x: x.distance)
            if "acne" in (c.title + c.text).lower()
        ]
        seen = {c.base_name for c in filtered}
        for c in fallback:
            if c.base_name not in seen:
                filtered.append(c)
                seen.add(c.base_name)
            if len(filtered) >= top_k:
                break

    return _one_best_chunk_per_paper(filtered, top_k)


def format_context_block(chunk: RetrievedChunk) -> str:
    header = f"[{chunk.index}] {chunk.title}"
    if chunk.section:
        header += f" — {chunk.section}"
    if chunk.base_name:
        header += f" (id: {chunk.base_name})"
    body = sanitize_inline_citations(chunk.text)
    return f"{header}\n{body}"


def _format_observation_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_retrieval_query(question: str, observation: dict | None) -> str:
    """Primary embedding query: question + vision + topic vocabulary."""
    parts = [question]

    if is_treatment_query(question):
        parts.append(
            "acne vulgaris treatment benzoyl peroxide retinoid topical therapy clinical"
        )
    elif is_causes_query(question):
        parts.append("acne vulgaris causes hormones sebum Cutibacterium acnes inflammation")

    if not observation:
        return " | ".join(parts)

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
    question: str, observation: dict | None = None
) -> list[str]:
    """Extra searches tuned to what the user actually asked."""
    extra: list[str] = []

    if is_treatment_query(question):
        med = " ".join(TOPIC_SEARCH_TERMS["medicine_skincare"][:10])
        extra.append(f"{question} {med}")
        extra.append(
            f"{question} acne vulgaris randomized topical benzoyl peroxide retinoid treatment efficacy"
        )
    elif is_causes_query(question):
        extra.append(
            f"{question} acne vulgaris pathogenesis sebaceous hormones Cutibacterium acnes"
        )
    else:
        extra.append(f"{question} acne vulgaris facial acne research")

    if observation:
        topics = observation.get("relevant_research_topics") or infer_topics_from_observation(
            observation
        )
        keywords = observation.get("retrieval_keywords") or []
        for topic_id in topics[:2]:
            terms = TOPIC_SEARCH_TERMS.get(topic_id, [])
            if terms:
                extra.append(f"{question} {' '.join(terms[:8])}")
        if keywords:
            extra.append(f"{question} {' '.join(str(k) for k in keywords[:10])}")

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

    treatment_note = ""
    if is_treatment_query(question):
        treatment_note = (
            "The user asked about treatment or a product routine. "
            "If the excerpts do NOT describe specific acne treatments, products, or regimens, "
            "say clearly: 'The papers we retrieved don't lay out a step-by-step product routine.' "
            "Do NOT repurpose eczema, contact dermatitis, vinegar, or general moisturizer studies as acne advice.\n\n"
        )

    return (
        f"{obs_block}{topic_block}"
        f"User question: {question}\n\n"
        f"{treatment_note}"
        f"Research context ({len(chunks)} excerpts from {len(chunks)} different papers — "
        f"these are your ONLY allowed facts):\n"
        f"{context}\n\n"
        f"CITATIONS: Use ONLY excerpt numbers [1] through [{len(chunks)}] from the headers above. "
        "Never use numbers copied from inside an excerpt (those were removed when possible). "
        "Each fact must cite the excerpt it came from.\n\n"
        "Write a detailed, consumer-friendly answer using ONLY information from the excerpts above.\n"
        "LENGTH: Aim for a full, helpful reply — usually at least 4–6 paragraphs or equivalent depth. "
        "Do not give a thin bullet list of one-liners.\n"
        "- For each idea, write 2–4 sentences that explain what the research found and what it means "
        "for someone reading this, with citation(s) [n] inline.\n"
        "- Combine related facts from different [n] blocks into the same paragraph when they fit.\n"
        "- Translate study language into plain English; do not copy academic wording.\n"
        "- Do NOT invent lifestyle tips, hygiene routines, or warnings that are not in the excerpts.\n"
        "- If a topic is missing from the excerpts, say the papers do not address it.\n"
        "Suggested flow:\n"
        "1) Friendly intro (2–3 sentences): photo + what you will cover\n"
        "2) **What might be going on** — developed paragraphs with [n] citations (skip if user only asked about treatment)\n"
        "3) **What studies suggest** — developed paragraphs on treatments/findings with [n] citations\n"
        "4) Optional: 1–2 sentences suggesting a dermatologist for personal medical advice\n"
        "5) **Sources** — paper titles for each [n] you used"
    )


def build_base_prompt(question: str, observation: dict | None = None) -> str:
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
