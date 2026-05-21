"""Normalize Marker markdown before chunking."""

from __future__ import annotations

import re

# Drop figure/table image lines and bare image-only blocks.
_IMAGE_LINE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$", re.MULTILINE)

# HTML-ish artifacts from PDF conversion.
_HTML_TAG = re.compile(r"<[^>]+>")
_SPAN_ID = re.compile(r'<span id="[^"]*"></span>\s*', re.IGNORECASE)

# Section headers that usually start the bibliography (case-insensitive line match).
_REFERENCES_HEADERS = re.compile(
    r"^#{1,4}\s*\*{0,2}\s*(references|bibliography|literature cited)\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Boilerplate lines often repeated per page.
_PAGE_HEADER = re.compile(
    r"^(Life \d{4},|Copyright:|Citation:|Received:|Revised:|Accepted:|Published:).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _truncate_at_references(text: str) -> str:
    match = _REFERENCES_HEADERS.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def clean_markdown(text: str) -> str:
    """Remove noise that hurts retrieval (images, refs, HTML artifacts)."""
    text = _IMAGE_LINE.sub("", text)
    text = _SPAN_ID.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _PAGE_HEADER.sub("", text)
    text = _truncate_at_references(text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_substantive(text: str, min_chars: int = 80) -> bool:
    """Skip empty or tiny fragments after cleaning."""
    stripped = re.sub(r"\s+", "", text)
    return len(stripped) >= min_chars
