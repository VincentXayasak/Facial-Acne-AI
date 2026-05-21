"""LM Studio OpenAI-compatible embedding endpoint."""

from __future__ import annotations

import os
from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI


def normalize_lm_studio_base_url(base_url: str) -> str:
    """Ensure OpenAI client targets .../v1 (required for /embeddings)."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


class LMStudioEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embed text via LM Studio's /v1/embeddings API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        raw_base = base_url or os.environ.get(
            "LM_STUDIO_BASE_URL", "http://localhost:1234"
        )
        self.base_url = normalize_lm_studio_base_url(raw_base)
        self.api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
        self.model = model or os.environ.get(
            "LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
        )
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def name(self) -> str:
        return f"lmstudio:{self.model}"

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        # LM Studio accepts batched inputs; keep batches modest for RAM.
        batch_size = 32
        all_embeddings: list[list[float]] = []
        for start in range(0, len(input), batch_size):
            batch = input[start : start + batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=list(batch),
            )
            ordered = sorted(response.data, key=lambda row: row.index)
            for row in ordered:
                vec = row.embedding
                all_embeddings.append(
                    list(vec) if not isinstance(vec, list) else cast(list[float], vec)
                )
        return all_embeddings

    def ping(self) -> int:
        """Raise if the embedding server or model is unavailable. Returns vector dim."""
        vectors = self(["ping"])
        if not vectors:
            raise RuntimeError("Embedding server returned no vectors.")
        dim = len(vectors[0])
        if dim == 0:
            raise RuntimeError("Embedding server returned an empty vector.")
        return dim
