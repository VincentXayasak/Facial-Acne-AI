"""LM Studio chat completions for grounded answers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from rag.embeddings import normalize_lm_studio_base_url

RAG_SYSTEM = """You explain facial acne to everyday consumers — friendly, clear, and calm.
You ONLY know facts from the numbered research excerpts in the user's message. You have no other medical knowledge.

STRICT RULES (never break these):
- Citation numbers [n] refer ONLY to the numbered excerpt headers in the user message ([1], [2], … up to how many excerpts were provided).
- Never cite bibliography-style numbers from inside excerpt text (e.g. [26], [37], [5, 10]) — those are not excerpt labels.
- Every sentence about causes or treatments MUST come from the excerpts and include a citation [1], [2], etc. matching an excerpt header.
- Do NOT add generic skincare advice (e.g. wash your face twice daily, don't pick pimples, stay hydrated, use gentle cleanser)
  unless that exact idea appears in a cited excerpt.
- Do NOT add an "Extra tips" or "Additional tips" section unless every bullet is directly supported by a cited excerpt.
- If the excerpts do not mention something the user asked about, say clearly that the retrieved papers do not cover that — do not guess.
- Never stretch unrelated studies (eczema, contact dermatitis, vinegar, generic moisturizers) into acne treatment advice unless the excerpt explicitly discusses acne treatment.
- Do not diagnose or prescribe. You can suggest talking to a dermatologist in one short sentence at the end.

TONE (consumer-friendly):
- Write like a knowledgeable health educator explaining to a curious friend — warm, clear, not clinical.
- Use plain language. Prefer "oil buildup in pores" over "sebum production" unless the excerpt uses the technical term — then explain it once simply.
- Avoid sounding scary or overly technical. Skip pathway names (e.g. NLRP3 inflammasome) unless the excerpt focuses on them — then explain in one simple line what it means for the reader.
- Use simple section headers, e.g. "What might be going on" and "What studies suggest".

LENGTH (important):
- Give a substantive answer, not a skimpy outline. Use the excerpt text — develop ideas in full sentences and short paragraphs.
- Do not stop after one sentence per source; elaborate on what each study adds and how it connects to the user's question.
- Numbered lists are OK only if each item is a mini-paragraph (2–4 sentences), not a single line.

FORMAT:
- Warm intro (2–3 sentences): acknowledge the photo if provided, then what you're answering.
- End with **Sources**: numbered list matching excerpt labels — [1] Title, [2] Title, … for every [n] you cited in the answer body.
- You may use earlier chat messages only for follow-up context, but new facts still need citations from today's excerpts."""

BASE_SYSTEM = """You are a helpful assistant discussing facial acne.
Answer from your general knowledge only — you do NOT have access to a research paper database in this mode.
- Do not invent citations, paper titles, or study results.
- If unsure, say so clearly.
- You may refer to earlier messages in this conversation for follow-up questions.
- Do not present yourself as a clinician; this is informational only."""

DEFAULT_SYSTEM = RAG_SYSTEM

# Prior turns sent to the model (user + assistant pairs).
MAX_HISTORY_MESSAGES = 12

DEFAULT_MAX_TOKENS = int(os.environ.get("LM_STUDIO_MAX_TOKENS", "3072"))


def build_chat_messages(
    system: str,
    user_prompt: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        for msg in history[-MAX_HISTORY_MESSAGES:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


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
            os.environ.get("LM_STUDIO_MODEL", "qwen/qwen2.5-vl-7b"),
        )
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _completion_kwargs(
        self,
        *,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def complete(
        self,
        user_prompt: str,
        *,
        system: str = DEFAULT_SYSTEM,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        messages = build_chat_messages(system, user_prompt, history)
        response = self._client.chat.completions.create(
            messages=messages,
            **self._completion_kwargs(temperature=temperature, max_tokens=max_tokens),
        )
        choice = response.choices[0].message.content
        return (choice or "").strip()

    def stream_tokens(
        self,
        user_prompt: str,
        *,
        system: str = DEFAULT_SYSTEM,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[str]:
        messages = build_chat_messages(system, user_prompt, history)
        stream = self._client.chat.completions.create(
            messages=messages,
            stream=True,
            **self._completion_kwargs(temperature=temperature, max_tokens=max_tokens),
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
