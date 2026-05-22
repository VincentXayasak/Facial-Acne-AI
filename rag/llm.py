"""LM Studio chat completions for grounded answers."""

from __future__ import annotations

import os

from openai import OpenAI

from rag.embeddings import normalize_lm_studio_base_url

RAG_SYSTEM = """You are a research assistant specializing in facial acne vulgaris.
You have excerpts from a curated corpus on: hormones & genetics, bacteria & inflammation,
environment & stress, medicine & skincare (retinoids, benzoyl peroxide, azelaic acid, etc.),
and diet & habits.

Answer using ONLY the numbered research context provided.
- Write a substantive answer (typically 4–8 paragraphs when sources allow), not a brief bullet list.
- Synthesize findings across multiple sources; compare or combine results when relevant.
- Cite every major claim inline as [1], [2], etc., matching the context block numbers.
- Organize by theme when it helps (causes, mechanisms, treatments, lifestyle factors).
- If the context does not cover part of the question, state that the corpus does not address it.
- Do not present yourself as a clinician; report what the cited studies describe.
- End with a "Sources" section listing the full paper title for each [n] you cited."""

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

    def ping(self) -> str:
        text = self.complete("Reply with exactly: ok", max_tokens=16)
        if not text:
            raise RuntimeError("Chat model returned an empty response.")
        return text
