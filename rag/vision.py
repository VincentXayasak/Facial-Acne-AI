"""Gemini vision: structured facial acne observations only."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from rag.topics import CORPUS_TOPICS_PROMPT_BLOCK

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


def analyze_face_gemini(
    image_path: Path,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    from google import genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env for Gemini vision."
        )

    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    client = genai.Client(api_key=api_key)

    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    image_bytes = image_path.read_bytes()
    response = client.models.generate_content(
        model=model,
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime),
            VISION_PROMPT,
        ],
    )

    text = getattr(response, "text", None) or ""
    if not text and response.candidates:
        parts = response.candidates[0].content.parts
        text = "".join(getattr(p, "text", "") or "" for p in parts)

    if not text.strip():
        raise RuntimeError("Gemini returned an empty vision response.")

    try:
        return _parse_json_response(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini response was not valid JSON: {text[:500]}") from exc
