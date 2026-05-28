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


# In-paper bibliography (not RAG excerpt labels). Applied when building the LLM prompt.
_BRACKET_NUM_CITE = re.compile(r"\[(?:\s*\d+\s*(?:,\s*\d+\s*)*)\]")
_PAGE_ANCHOR = re.compile(r"\]\(#page-\d+-\d+\)")


def _strip_page_anchor_refs(text: str) -> str:
    """Remove Marker/PDF links like [26](#page-9-0) or [[42\\]](#page-10-0)."""
    while True:
        match = _PAGE_ANCHOR.search(text)
        if not match:
            break
        bracket_start = text.rfind("[", 0, match.start())
        if bracket_start < 0:
            break
        text = text[:bracket_start] + text[match.end() :]
    return text


def sanitize_inline_citations(text: str) -> str:
    """Strip the source paper's in-text reference numbers from excerpt bodies.

    RAG uses [1], [2], … only as excerpt header labels. Without this, models
    often copy bibliography markers like [26, 37] from the chunk text.
    """
    text = _strip_page_anchor_refs(text)
    text = _BRACKET_NUM_CITE.sub("", text)
    text = re.sub(r"\\\]", "", text)
    text = re.sub(r"\[\s*,\s*", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\[\s+and\s+", " and ", text)
    text = re.sub(r"\s+\]\s*", " ", text)
    text = re.sub(r"\s+\[\s+", " ", text)
    text = re.sub(r"\s+\]\.", ".", text)
    text = re.sub(r"\s+\[\.", ".", text)
    text = re.sub(r"\[\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()
