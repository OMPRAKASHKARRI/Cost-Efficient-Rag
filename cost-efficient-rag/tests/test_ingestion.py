"""Tests for src/ingestion.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import (
    ChunkRecord,
    UnsupportedFileTypeError,
    build_chunk_records,
    deduplicate_chunks,
    detect_file_type,
    generate_chunk_id,
    ingest_document,
    ingest_documents,
    load_html,
    load_markdown,
    load_pdf,
    recursive_character_split,
)


# ── Chunk ID hashing ──────────────────────────────────────────────────────


def test_hash_is_deterministic():
    assert generate_chunk_id("doc.pdf", 0, "hello") == generate_chunk_id("doc.pdf", 0, "hello")


def test_hash_changes_with_chunk_index():
    assert generate_chunk_id("doc.pdf", 0, "hello") != generate_chunk_id("doc.pdf", 1, "hello")


def test_hash_changes_with_text():
    assert generate_chunk_id("doc.pdf", 0, "hello") != generate_chunk_id("doc.pdf", 0, "world")


def test_hash_changes_with_source():
    assert generate_chunk_id("a.pdf", 0, "hello") != generate_chunk_id("b.pdf", 0, "hello")


def test_hash_is_64_char_hex():
    h = generate_chunk_id("doc.pdf", 0, "hello")
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex


# ── Recursive splitter ────────────────────────────────────────────────────


def test_splitter_respects_chunk_size_on_long_text():
    text = ("Paragraph one has some content in it. " * 3 + "\n\n") * 5
    chunks = recursive_character_split(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # allow small slack: merge never splits mid-word, so a chunk may exceed
    # chunk_size by up to the length of the one word that pushed it over
    assert all(len(c) <= 250 for c in chunks)


def test_splitter_returns_single_chunk_for_short_text():
    text = "short text"
    chunks = recursive_character_split(text, chunk_size=500, chunk_overlap=50)
    assert chunks == ["short text"]


def test_splitter_empty_text_returns_empty_list():
    assert recursive_character_split("", chunk_size=100, chunk_overlap=10) == []
    assert recursive_character_split("   ", chunk_size=100, chunk_overlap=10) == []


def test_splitter_overlap_ge_chunk_size_raises():
    with pytest.raises(ValueError, match="must be <"):
        recursive_character_split("some text", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        recursive_character_split("some text", chunk_size=10, chunk_overlap=20)


def test_splitter_produces_overlap_between_consecutive_chunks():
    text = ("word " * 200).strip()
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=30)
    assert len(chunks) > 1
    overlap_found = any(
        chunks[i].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )
    assert overlap_found


def test_splitter_handles_single_word_longer_than_chunk_size():
    text = "a" * 500
    chunks = recursive_character_split(text, chunk_size=100, chunk_overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).replace("", "") != ""  # no content silently dropped
    assert sum(len(c) for c in chunks) >= 500  # overlap means >=, not ==


# ── Loaders ────────────────────────────────────────────────────────────────


def test_detect_file_type_supported():
    assert detect_file_type("a.pdf") == "pdf"
    assert detect_file_type("a.html") == "html"
    assert detect_file_type("a.htm") == "html"
    assert detect_file_type("a.md") == "md"
    assert detect_file_type("a.markdown") == "md"


def test_detect_file_type_unsupported_raises():
    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type("a.docx")


def test_load_html_strips_script_and_style(tmp_path):
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><h1>Title</h1><script>alert('x')</script>"
        "<p>Hello <b>world</b></p></body></html>"
    )
    f = tmp_path / "page.html"
    f.write_text(html)
    text = load_html(f)
    assert "alert" not in text
    assert "color:red" not in text
    assert "Title" in text
    assert "Hello" in text and "world" in text


def test_load_markdown_strips_syntax_keeps_content(tmp_path):
    md = "# Heading\n\nSome **bold** text and a [link](http://example.com).\n"
    f = tmp_path / "doc.md"
    f.write_text(md)
    text = load_markdown(f)
    assert "Heading" in text
    assert "bold" in text
    assert "link" in text
    assert "http://example.com" not in text  # URL itself stripped, anchor text kept
    assert "**" not in text and "#" not in text


def test_load_pdf_uses_pypdf_reader(tmp_path):
    """Mock pypdf.PdfReader so this test doesn't require a real binary PDF fixture."""
    fake_page1 = MagicMock()
    fake_page1.extract_text.return_value = "Page one text."
    fake_page2 = MagicMock()
    fake_page2.extract_text.return_value = "Page two text."

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake bytes")  # content irrelevant, reader is mocked

    with patch("src.ingestion.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value.pages = [fake_page1, fake_page2]
        text = load_pdf(f)

    assert "Page one text." in text
    assert "Page two text." in text


def test_load_pdf_handles_none_extract_text(tmp_path):
    """Some PDF pages (e.g. scanned images) return None from extract_text()."""
    fake_page = MagicMock()
    fake_page.extract_text.return_value = None

    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    with patch("src.ingestion.PdfReader") as mock_reader_cls:
        mock_reader_cls.return_value.pages = [fake_page]
        text = load_pdf(f)  # must not raise on None

    assert text == ""


# ── ChunkRecord construction ────────────────────────────────────────────────


def test_build_chunk_records_produces_sequential_indices():
    text = "Sentence one. " * 40
    records = build_chunk_records(text, "doc.md", "md", chunk_size=100, chunk_overlap=20)
    assert [r.chunk_index for r in records] == list(range(len(records)))
    assert all(isinstance(r, ChunkRecord) for r in records)
    assert all(r.source == "doc.md" for r in records)
    assert all(r.file_type == "md" for r in records)
    assert all(r.char_count == len(r.text) for r in records)


def test_build_chunk_records_attaches_metadata():
    records = build_chunk_records(
        "some text here", "doc.md", "md", chunk_size=100, chunk_overlap=10,
        metadata={"category": "policy", "date_created": "2026-01-01"},
    )
    assert all(r.metadata == {"category": "policy", "date_created": "2026-01-01"} for r in records)


# ── Deduplication / idempotency ──────────────────────────────────────────────


def test_deduplicate_chunks_filters_existing_ids():
    text = "Sentence one. " * 40
    records = build_chunk_records(text, "doc.md", "md", chunk_size=100, chunk_overlap=20)
    all_ids = {r.id for r in records}

    fresh = deduplicate_chunks(records, existing_ids=set())
    assert len(fresh) == len(records)

    none_new = deduplicate_chunks(records, existing_ids=all_ids)
    assert len(none_new) == 0

    half_ids = set(list(all_ids)[: len(all_ids) // 2])
    partial = deduplicate_chunks(records, existing_ids=half_ids)
    assert len(partial) == len(records) - len(half_ids)


def test_ingest_document_is_idempotent(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n" + "Some repeated content sentence. " * 30)

    first_pass = ingest_document(f, existing_ids=set(), chunk_size=100, chunk_overlap=20)
    assert len(first_pass) > 0

    ids_after_first = {r.id for r in first_pass}
    second_pass = ingest_document(f, existing_ids=ids_after_first, chunk_size=100, chunk_overlap=20)
    assert len(second_pass) == 0  # re-ingesting unchanged content yields nothing new


def test_ingest_documents_accumulates_across_files(tmp_path):
    f1 = tmp_path / "a.md"
    f1.write_text("Content A. " * 30)
    f2 = tmp_path / "b.md"
    f2.write_text("Content B. " * 30)

    new_records = ingest_documents([f1, f2], existing_ids=set(), chunk_size=100, chunk_overlap=20)
    sources = {r.source for r in new_records}
    assert str(f1) in sources
    assert str(f2) in sources

    # Second call with accumulated IDs from the first should yield nothing new.
    all_ids = {r.id for r in new_records}
    repeat = ingest_documents([f1, f2], existing_ids=all_ids, chunk_size=100, chunk_overlap=20)
    assert len(repeat) == 0
