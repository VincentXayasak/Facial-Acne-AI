"""End-to-end RAG: optional vision → retrieve → Llama answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.llm import LMStudioChat
from rag.retrieve import (
    build_retrieval_query,
    build_user_prompt,
    retrieve_chunks,
)
from rag.vision import analyze_face_gemini


@dataclass
class RAGResult:
    answer: str
    observation: dict | None
    retrieval_query: str
    chunks: list


def run_rag(
    question: str,
    *,
    image_path: Path | None = None,
    observation: dict | None = None,
    chroma_path: Path,
    collection_name: str,
    embedding_fn: LMStudioEmbeddingFunction | None = None,
    chat: LMStudioChat | None = None,
    top_k: int = 5,
    skip_vision: bool = False,
) -> RAGResult:
    embedding_fn = embedding_fn or LMStudioEmbeddingFunction()
    chat = chat or LMStudioChat()

    obs = observation

    if image_path and not skip_vision and obs is None:
        obs = analyze_face_gemini(image_path)

    retrieval_query = build_retrieval_query(question, obs)
    chunks = retrieve_chunks(
        retrieval_query,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        top_k=top_k,
    )

    if not chunks:
        raise RuntimeError("No chunks retrieved from Chroma. Run ingest_papers.py --reset first.")

    user_prompt = build_user_prompt(question, chunks, obs)
    answer = chat.complete(user_prompt)

    return RAGResult(
        answer=answer,
        observation=obs,
        retrieval_query=retrieval_query,
        chunks=chunks,
    )


def result_to_dict(result: RAGResult) -> dict:
    return {
        "answer": result.answer,
        "observation": result.observation,
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


def print_result(result: RAGResult) -> None:
    if result.observation:
        print("--- Vision observation (Gemini) ---")
        print(json.dumps(result.observation, indent=2))
        print()

    print(f"Retrieval query: {result.retrieval_query}\n")
    print("--- Retrieved sources ---")
    for c in result.chunks:
        print(f"[{c.index}] {c.title} ({c.base_name}) — {c.section}  d={c.distance:.4f}")
    print()

    print("--- Answer (Llama + RAG) ---")
    print(result.answer)
