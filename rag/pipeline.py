"""End-to-end: optional vision → optional RAG → Llama answer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rag.embeddings import LMStudioEmbeddingFunction
from rag.llm import BASE_SYSTEM, DEFAULT_MAX_TOKENS, LMStudioChat, RAG_SYSTEM
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


def chunks_to_sources(chunks: list) -> list[dict]:
    return [
        {
            "cite": c.index,
            "title": c.title,
            "section": c.section,
            "paper_id": c.base_name,
            "distance": c.distance,
        }
        for c in chunks
    ]


def parse_history(raw: str | list | None) -> list[dict[str, str]]:
    """Normalize chat history from the web UI."""
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _prepare_prompts(
    question: str,
    *,
    image_path: Path | None = None,
    observation: dict | None = None,
    chroma_path: Path | None = None,
    collection_name: str = "acne_research",
    embedding_fn: LMStudioEmbeddingFunction | None = None,
    top_k: int = 5,
    skip_vision: bool = False,
    use_rag: bool = True,
) -> tuple[dict | None, list, str | None, str, str, float]:
    """Returns obs, chunks, retrieval_query, user_prompt, system, temperature."""
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
            question=question,
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
        return obs, chunks, retrieval_query, user_prompt, RAG_SYSTEM, 0.35

    user_prompt = build_base_prompt(question, obs)
    return obs, chunks, retrieval_query, user_prompt, BASE_SYSTEM, 0.35


def stream_rag_events(
    question: str,
    *,
    image_path: Path | None = None,
    observation: dict | None = None,
    history: list[dict[str, str]] | None = None,
    chroma_path: Path | None = None,
    collection_name: str = "acne_research",
    embedding_fn: LMStudioEmbeddingFunction | None = None,
    chat: LMStudioChat | None = None,
    top_k: int = 5,
    skip_vision: bool = False,
    use_rag: bool = True,
) -> Iterator[dict]:
    """Yield status/meta/token/done events for SSE streaming."""
    chat = chat or LMStudioChat()

    if image_path and not skip_vision and observation is None:
        yield {"type": "status", "message": "Analyzing photo with Gemini…"}
    if use_rag:
        yield {"type": "status", "message": "Searching research papers…"}

    obs, chunks, retrieval_query, user_prompt, system, temperature = _prepare_prompts(
        question,
        image_path=image_path,
        observation=observation,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        top_k=top_k,
        skip_vision=skip_vision,
        use_rag=use_rag,
    )

    yield {
        "type": "meta",
        "use_rag": use_rag,
        "observation": obs,
        "retrieval_query": retrieval_query,
        "sources": chunks_to_sources(chunks),
        "question": question,
    }
    yield {"type": "status", "message": "Writing answer…"}

    for token in chat.stream_tokens(
        user_prompt,
        system=system,
        history=history,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
    ):
        yield {"type": "token", "text": token}

    yield {"type": "done"}


def run_rag(
    question: str,
    *,
    image_path: Path | None = None,
    observation: dict | None = None,
    history: list[dict[str, str]] | None = None,
    chroma_path: Path | None = None,
    collection_name: str = "acne_research",
    embedding_fn: LMStudioEmbeddingFunction | None = None,
    chat: LMStudioChat | None = None,
    top_k: int = 5,
    skip_vision: bool = False,
    use_rag: bool = True,
) -> RAGResult:
    chat = chat or LMStudioChat()
    obs, chunks, retrieval_query, user_prompt, system, temperature = _prepare_prompts(
        question,
        image_path=image_path,
        observation=observation,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        top_k=top_k,
        skip_vision=skip_vision,
        use_rag=use_rag,
    )
    answer = chat.complete(
        user_prompt,
        system=system,
        history=history,
        temperature=temperature,
        max_tokens=DEFAULT_MAX_TOKENS,
    )

    return RAGResult(
        answer=answer,
        observation=obs,
        retrieval_query=retrieval_query,
        chunks=chunks,
        use_rag=use_rag,
    )


def result_to_dict(result: RAGResult) -> dict:
    return {
        "answer": result.answer,
        "observation": result.observation,
        "use_rag": result.use_rag,
        "retrieval_query": result.retrieval_query,
        "sources": chunks_to_sources(result.chunks),
    }


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
