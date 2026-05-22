"""Gemini vision: structured facial acne observations only."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

VISION_PROMPT = """Analyze this facial photo for dermatology research purposes only.
Do NOT give treatment advice or a medical diagnosis.

Return ONLY valid JSON with these keys:
{
  "acne_types": ["list of lesion types you see, e.g. papules, pustules, comedones, nodules"],
  "severity_estimate": "mild | moderate | severe | unclear",
  "affected_areas": ["e.g. forehead, cheeks, chin, jawline"],
  "skin_tone_description": "brief Fitzpatrick-style description if visible, else unclear",
  "apparent_sex_presentation": "female | male | androgynous | unclear",
  "other_observations": "brief notes on inflammation, scarring, distribution patterns"
}

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
