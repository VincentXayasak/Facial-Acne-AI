"""Header-aware then recursive token chunking."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.clean import clean_markdown, is_substantive

HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


@dataclass(frozen=True)
class PaperRecord:
    base_name: str
    title: str
    pdf_path: str
    markdown_path: str
    text: str


def _section_label(header_meta: dict) -> str:
    parts = []
    for key in ("h1", "h2", "h3", "h4"):
        value = header_meta.get(key)
        if value and str(value).strip():
            parts.append(str(value).strip())
    return " > ".join(parts) if parts else "body"


def _prepend_context(title: str, section: str, body: str) -> str:
    title = title.strip() or "Unknown"
    section = section.strip() or "body"
    return f"Paper: {title}\nSection: {section}\n---\n{body.strip()}"


def chunk_document(
    record: PaperRecord,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split one paper: clean → headers → recursive token chunks."""
    cleaned = clean_markdown(record.text)
    if not is_substantive(cleaned):
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    sections = header_splitter.split_text(cleaned)

    recursive = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    base_meta = {
        "base_name": record.base_name,
        "title": (record.title or record.base_name)[:500],
        "pdf_path": record.pdf_path,
        "markdown_path": record.markdown_path,
    }

    out: list[Document] = []
    chunk_index = 0

    for section_doc in sections:
        section = _section_label(section_doc.metadata)
        splits = recursive.split_text(section_doc.page_content)
        for piece in splits:
            if not is_substantive(piece, min_chars=60):
                continue
            page_content = _prepend_context(record.title, section, piece)
            meta = {
                **base_meta,
                "section": section[:300],
                "chunk_index": chunk_index,
            }
            out.append(Document(page_content=page_content, metadata=meta))
            chunk_index += 1

    return out


def chunk_all(
    records: list[PaperRecord],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    chunks: list[Document] = []
    for record in records:
        chunks.extend(
            chunk_document(
                record,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
