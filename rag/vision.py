"""LM Studio vision (Qwen2.5-VL): structured facial acne observations only."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from openai import OpenAI

from rag.embeddings import normalize_lm_studio_base_url
from rag.topics import CORPUS_TOPICS_PROMPT_BLOCK

DEFAULT_VISION_MODEL = "qwen/qwen2.5-vl-7b"

VISION_PROMPT = f"""Analyze this facial photo for dermatology research purposes only.
Do NOT give treatment advice or a medical diagnosis.

Our research library covers these topics (use these exact topic ids):
{CORPUS_TOPICS_PROMPT_BLOCK}

Return ONLY valid JSON with these keys:
{{
  "acne_types": ["papules, pustules, comedones, nodules, cysts — only what you clearly see"],
  "severity_estimate": "mild | moderate | severe | unclear",
  "inflammation_level": "none | mild | moderate | severe | unclear",
  "affected_areas": ["forehead, cheeks, chin, jawline, nose, etc."],
  "scarring_or_post_inflammatory_marks": "brief description or none visible",
  "skin_tone_description": "brief Fitzpatrick-style description if visible, else unclear",
  "apparent_sex_presentation": "female | male | androgynous | unclear",
  "apparent_age_group": "adolescent | young adult | adult | unclear",
  "relevant_research_topics": ["1-3 topic ids from the list above that best match what you see"],
  "retrieval_keywords": ["8-15 concrete search terms for finding papers: lesion names, treatments that may apply, mechanisms like sebum, C. acnes, retinoid, dairy, stress, etc."],
  "other_observations": "distribution pattern, oiliness, dryness, other visible features"
}}

Pick relevant_research_topics that fit the visible presentation (e.g. inflamed papules → bacteria_inflammation + medicine_skincare).
retrieval_keywords should help search a scientific corpus — use specific medical terms, not generic words like "acne face".
Use "unclear" or empty lists when not visible. Be factual and conservative."""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _image_mime(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def _vision_client() -> tuple[OpenAI, str]:
    raw_base = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
    base_url = normalize_lm_studio_base_url(raw_base)
    api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    model = (
        os.environ.get("LM_STUDIO_VISION_MODEL")
        or os.environ.get("LM_STUDIO_CHAT_MODEL")
        or os.environ.get("LM_STUDIO_MODEL")
        or DEFAULT_VISION_MODEL
    )
    return OpenAI(base_url=base_url, api_key=api_key), model


def analyze_face(
    image_path: Path,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Return structured JSON observation from a face photo via LM Studio vision."""
    if base_url or api_key or model:
        raw_base = base_url or os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
        client = OpenAI(
            base_url=normalize_lm_studio_base_url(raw_base),
            api_key=api_key or os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
        )
        model = model or os.environ.get("LM_STUDIO_VISION_MODEL") or os.environ.get(
            "LM_STUDIO_CHAT_MODEL", DEFAULT_VISION_MODEL
        )
    else:
        client, model = _vision_client()

    mime = _image_mime(image_path)
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(
            f"Vision model ({model}) returned an empty response. "
            "Load qwen/qwen2.5-vl-7b in LM Studio and start the server."
        )

    try:
        return _parse_json_response(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Vision model response was not valid JSON: {text[:500]}"
        ) from exc
