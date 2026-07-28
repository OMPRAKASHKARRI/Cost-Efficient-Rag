

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup
from markdown import markdown as render_markdown
from pypdf import PdfReader

from src.logger import get_logger

logger = get_logger()

FileType = Literal["pdf", "html", "md"]

_DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

_EXTENSION_MAP: dict[str, FileType] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
}


@dataclass(frozen=True)
class ChunkRecord:
    """A single chunk of a source document, ready for embedding + storage.

    Attributes:
        id: Deterministic SHA-256 hex digest — the chunk's primary key.
        text: The chunk's text content.
        source: Path (or logical name) of the originating document.
        chunk_index: 0-based position of this chunk within its source document.
        file_type: One of "pdf", "html", "md".
        char_count: ``len(text)`` — stored redundantly for cheap analytics
            without re-scanning chunk text.
        metadata: Free-form filter tags (e.g. ``{"category": "policy",
            "date_created": "2026-01-01"}``) merged into the vector store
            record at insert time.
    """

    id: str
    text: str
    source: str
    chunk_index: int
    file_type: FileType
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension has no registered loader."""


def detect_file_type(path: str | Path) -> FileType:
    """Map a file's extension to a supported :data:`FileType`.

    Raises:
        UnsupportedFileTypeError: if the extension isn't pdf/html/md.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in _EXTENSION_MAP:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{suffix}' for {path}. "
            f"Supported: {sorted(_EXTENSION_MAP)}"
        )
    return _EXTENSION_MAP[suffix]


def load_pdf(path: str | Path) -> str:
    """Extract concatenated text from every page of a PDF."""
    reader = PdfReader(str(path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages_text)
    logger.debug(f"Loaded PDF {path}: {len(reader.pages)} pages, {len(text)} chars")
    return text


def load_html(path: str | Path) -> str:
    """Extract visible text from an HTML file, dropping script/style tags."""
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _html_to_text(raw)


def load_markdown(path: str | Path) -> str:
    """Render Markdown to HTML, then extract visible text.

    Rendering to HTML first (rather than treating markdown as plain text)
    strips syntax noise (``#``, ``**``, link markup) so embeddings are
    computed on the same kind of clean prose as the HTML/PDF paths.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    html = render_markdown(raw, extensions=["tables", "fenced_code"])
    return _html_to_text(html)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # get_text with a separator keeps block-level elements from smashing
    # together into one run-on word (e.g. "</h1><p>" -> "Header Paragraph"
    # instead of "HeaderParagraph").
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


_LOADERS = {
    "pdf": load_pdf,
    "html": load_html,
    "md": load_markdown,
}


def load_document(path: str | Path) -> tuple[str, FileType]:
    """Dispatch to the correct loader based on file extension.

    Returns:
        A tuple of ``(extracted_text, file_type)``.
    """
    file_type = detect_file_type(path)
    text = _LOADERS[file_type](path)
    return text, file_type


def generate_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    """Deterministic content-addressed chunk ID: SHA256(source + index + text).

    Identical (source, index, text) always yields the same ID — this is
    what makes re-ingestion idempotent. Changing the chunk's text (e.g. the
    source document was edited) intentionally produces a *different* ID
    rather than reusing the old one, since the old ID no longer represents
    that content.
    """
    raw_identifier = f"{source_path}_{chunk_index}_{text}"
    return hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()


def recursive_character_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split ``text`` into chunks of at most ~``chunk_size`` characters.

    Tries separators in priority order (paragraph, line, sentence, word,
    character) so splits prefer natural boundaries and only fall back to a
    hard character cut when a single unit (e.g. one giant unbroken line)
    doesn't fit on its own. Adjacent chunks overlap by ``chunk_overlap``
    characters so context isn't lost at chunk boundaries.

    Args:
        text: Input text to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        separators: Override the default separator priority list.

    Returns:
        List of non-empty, whitespace-trimmed text chunks, in document order.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
    text = text.strip()
    if not text:
        return []
    seps = separators if separators is not None else _DEFAULT_SEPARATORS
    pieces = _split_recursive(text, seps, chunk_size)
    merged = _merge_pieces(pieces, chunk_size, chunk_overlap)
    return [m.strip() for m in merged if m.strip()]


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively break ``text`` into atomic pieces each <= chunk_size where possible."""
    if len(text) <= chunk_size:
        return [text] if text else []

    separator = separators[-1]
    remaining_separators: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining_separators = []
            break
        if sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    raw_pieces = list(text) if separator == "" else text.split(separator)

    result: list[str] = []
    for piece in raw_pieces:
        if not piece:
            continue
        if len(piece) > chunk_size and separator != "":
            result.extend(_split_recursive(piece, remaining_separators or [""], chunk_size))
        else:
            result.append(piece)
    return result


def _merge_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack small atomic pieces into ~chunk_size windows with overlap."""
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _current_text() -> str:
        return " ".join(current)

    for piece in pieces:
        piece_len = len(piece) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + piece_len > chunk_size:
            chunks.append(_current_text())
            # Build overlap: keep trailing pieces whose combined length <= chunk_overlap.
            overlap_pieces: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                add_len = len(prev) + (1 if overlap_pieces else 0)
                if overlap_len + add_len > chunk_overlap:
                    break
                overlap_pieces.insert(0, prev)
                overlap_len += add_len
            current = overlap_pieces
            current_len = overlap_len
            piece_len = len(piece) + (1 if current else 0)

        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append(_current_text())

    return chunks


def build_chunk_records(
    text: str,
    source: str,
    file_type: FileType,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    """Split text and wrap each piece into a hashed :class:`ChunkRecord`."""
    raw_chunks = recursive_character_split(text, chunk_size, chunk_overlap)
    records = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = generate_chunk_id(source, idx, chunk_text)
        records.append(
            ChunkRecord(
                id=chunk_id,
                text=chunk_text,
                source=source,
                chunk_index=idx,
                file_type=file_type,
                char_count=len(chunk_text),
                metadata=dict(metadata or {}),
            )
        )
    return records


def deduplicate_chunks(
    records: list[ChunkRecord],
    existing_ids: set[str],
) -> list[ChunkRecord]:
    """Filter out chunks whose ID is already present in the vector store.

    This is what makes ``ingest_document``/``ingest_documents`` idempotent:
    re-running ingestion on an unchanged file produces the same IDs, all of
    which are filtered out here, yielding zero new inserts.
    """
    new_records = [r for r in records if r.id not in existing_ids]
    skipped = len(records) - len(new_records)
    if skipped:
        logger.info(f"Skipped {skipped} already-ingested chunk(s) (idempotent re-ingestion)")
    return new_records


def ingest_document(
    path: str | Path,
    existing_ids: set[str],
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    """Load, chunk, hash, and deduplicate a single document.

    Args:
        path: Path to a .pdf, .html, .htm, .md, or .markdown file.
        existing_ids: Chunk IDs already present in the vector store (for
            idempotent re-ingestion). Pass an empty set for a fresh table.
        chunk_size: Target max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        metadata: Extra filter-tag metadata (e.g. category, date_created)
            attached to every chunk from this document.

    Returns:
        New (not-yet-stored) :class:`ChunkRecord` objects ready to embed
        and upsert. May be empty if every chunk was already ingested.
    """
    source = str(path)
    text, file_type = load_document(path)
    records = build_chunk_records(
        text=text,
        source=source,
        file_type=file_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )
    new_records = deduplicate_chunks(records, existing_ids)
    logger.info(
        f"Ingested {path}: {len(records)} chunk(s) total, "
        f"{len(new_records)} new, {len(records) - len(new_records)} deduplicated"
    )
    return new_records


def ingest_documents(
    paths: list[str | Path],
    existing_ids: set[str],
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    """Ingest multiple documents, accumulating ``existing_ids`` across files.

    Accumulation matters within a single batch: if two files in the same
    call happened to produce an identical chunk (same source path can't
    collide, but this guards against calling it twice with a stale
    ``existing_ids`` snapshot) the second occurrence is still deduplicated.
    """
    all_new_records: list[ChunkRecord] = []
    seen_ids = set(existing_ids)
    for path in paths:
        new_records = ingest_document(
            path, seen_ids, chunk_size, chunk_overlap, metadata
        )
        seen_ids.update(r.id for r in new_records)
        all_new_records.extend(new_records)
    return all_new_records
