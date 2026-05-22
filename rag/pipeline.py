"""End-to-end: optional vision → optional RAG → Llama answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.llm import BASE_SYSTEM, LMStudioChat, RAG_SYSTEM
from rag.retrieve import (
    build_base_prompt,
    build_extra_retrieval_queries,
    build_retrieval_query,
    build_user_prompt,
    retrieve_chunks,
)
from rag.vision import analyze_face_gemini


@dataclass
class RAGResult:
    answer: str
    observation: dict | None
    retrieval_query: str | None
    chunks: list
    use_rag: bool


def run_rag(
    question: str,
    *,
    image_path: Path | None = None,
    observation: dict | None = None,
    chroma_path: Path | None = None,
    collection_name: str = "acne_research",
    embedding_fn: LMStudioEmbeddingFunction | None = None,
    chat: LMStudioChat | None = None,
    top_k: int = 5,
    skip_vision: bool = False,
    use_rag: bool = True,
) -> RAGResult:
    chat = chat or LMStudioChat()
    obs = observation

    if image_path and not skip_vision and obs is None:
        obs = analyze_face_gemini(image_path)

    chunks: list = []
    retrieval_query: str | None = None

    if use_rag:
        if chroma_path is None:
            raise ValueError("chroma_path is required when use_rag=True")
        embedding_fn = embedding_fn or LMStudioEmbeddingFunction()
        retrieval_query = build_retrieval_query(question, obs)
        extra_queries = build_extra_retrieval_queries(question, obs)
        chunks = retrieve_chunks(
            retrieval_query,
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_fn=embedding_fn,
            top_k=top_k,
            extra_queries=extra_queries,
        )
        if not chunks:
            raise RuntimeError(
                "No chunks retrieved from Chroma. Run ingest_papers.py --reset first."
            )
        user_prompt = build_user_prompt(question, chunks, obs)
        answer = chat.complete(user_prompt, system=RAG_SYSTEM, max_tokens=2048)
    else:
        user_prompt = build_base_prompt(question, obs)
        answer = chat.complete(user_prompt, system=BASE_SYSTEM)

    return RAGResult(
        answer=answer,
        observation=obs,
        retrieval_query=retrieval_query,
        chunks=chunks,
        use_rag=use_rag,
    )


def result_to_dict(result: RAGResult) -> dict:
    out = {
        "answer": result.answer,
        "observation": result.observation,
        "use_rag": result.use_rag,
        "retrieval_query": result.retrieval_query,
        "sources": [
            {
                "cite": c.index,
                "title": c.title,
                "section": c.section,
                "paper_id": c.base_name,
                "distance": c.distance,
            }
            for c in result.chunks
        ],
    }
    return out


def print_result(result: RAGResult) -> None:
    mode = "RAG (research-grounded)" if result.use_rag else "Base model (no RAG)"
    print(f"Mode: {mode}\n")

    if result.observation:
        print("--- Vision observation (Gemini) ---")
        print(json.dumps(result.observation, indent=2))
        print()

    if result.use_rag:
        print(f"Retrieval query: {result.retrieval_query}")
        if result.observation and result.observation.get("relevant_research_topics"):
            print(
                "Research topics: "
                + ", ".join(result.observation["relevant_research_topics"])
            )
        print()
        print("--- Retrieved sources ---")
        for c in result.chunks:
            print(f"[{c.index}] {c.title} ({c.base_name}) — {c.section}  d={c.distance:.4f}")
        print()
        print("--- Answer (Llama + RAG) ---")
    else:
        print("--- Answer (Llama base model) ---")

    print(result.answer)
