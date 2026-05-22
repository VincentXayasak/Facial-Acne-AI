"""LM Studio chat completions for grounded answers."""

from __future__ import annotations

import os
from collections.abc import Iterator

from openai import OpenAI

from rag.embeddings import normalize_lm_studio_base_url

RAG_SYSTEM = """You help everyday users understand facial acne using plain language.
You have short excerpts from peer-reviewed papers (hormones, bacteria, treatments, diet, stress, etc.).

STYLE (very important):
- Write like the no-nonsense, friendly tone of a health blog — NOT like a journal article.
- Use simple words. If you must use a medical term, explain it in parentheses right after.
  Example: "Cutibacterium acnes (a common skin bacteria)" not "gram-positive anaerobic bacterium."
- Use clear section headers and numbered lists for causes, treatments, and tips.
- Keep paragraphs short (2–4 sentences). No walls of dense science text.
- Do NOT copy academic phrasing from the sources. Translate study findings into everyday English.
- Still ground every main point in the provided context and cite as [1], [2], etc.
- When photo observations are given, briefly acknowledge what was seen, then answer the question.

CONTENT:
- Focus on what the research suggests: likely causes, what treatments studies mention, practical takeaways.
- If the sources do not cover something, say "The papers we have don't really cover that" — don't pad with jargon.
- Do not diagnose or prescribe. Encourage seeing a dermatologist for persistent or severe acne.
- End with a short "Sources" section listing paper titles for each [n] you used."""

BASE_SYSTEM = """You are a helpful assistant discussing facial acne.
Answer from your general knowledge only — you do NOT have access to a research paper database in this mode.
- Do not invent citations, paper titles, or study results.
- If unsure, say so clearly.
- Do not present yourself as a clinician; this is informational only."""

# Backward-compatible alias
DEFAULT_SYSTEM = RAG_SYSTEM


class LMStudioChat:
    """Chat via LM Studio OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        raw_base = base_url or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
        self.base_url = normalize_lm_studio_base_url(raw_base)
        self.api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        self.model = model or os.environ.get(
            "LM_STUDIO_CHAT_MODEL",
            os.environ.get("LM_STUDIO_MODEL", "llama-3.2-8b-instruct"),
        )
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(
        self,
        user_prompt: str,
        *,
        system: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0].message.content
        return (choice or "").strip()

    def stream_tokens(
        self,
        user_prompt: str,
        *,
        system: str = DEFAULT_SYSTEM,
        temperature: float = 0.2,
        max_tokens: int = 1536,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def ping(self) -> str:
        text = self.complete("Reply with exactly: ok", max_tokens=16)
        if not text:
            raise RuntimeError("Chat model returned an empty response.")
        return text
