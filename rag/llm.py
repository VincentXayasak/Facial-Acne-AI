"""LM Studio chat completions for grounded answers."""

from __future__ import annotations

import os

from openai import OpenAI

from rag.embeddings import normalize_lm_studio_base_url

DEFAULT_SYSTEM = """You are a research assistant specializing in facial acne.
Answer the user's question using ONLY the research context provided below.
- Cite evidence inline as [1], [2], etc., matching the numbered context blocks.
- If the context does not support an answer, say you cannot find support in the corpus.
- Do not present yourself as a clinician; summarize what the cited research reports.
- End with a "Sources" section listing the paper title for each citation you used."""


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
        max_tokens: int = 1024,
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

    def ping(self) -> str:
        text = self.complete("Reply with exactly: ok", max_tokens=16)
        if not text:
            raise RuntimeError("Chat model returned an empty response.")
        return text
