"""Tests for eval/local_retriever.py."""

from __future__ import annotations

from src.ingestion import ChunkRecord
from eval.local_retriever import OfflineTfidfRetriever


def _chunk(id_, text, source="doc.md", file_type="md", metadata=None):
    return ChunkRecord(
        id=id_, text=text, source=source, chunk_index=0, file_type=file_type,
        char_count=len(text), metadata=metadata or {},
    )


def test_search_ranks_most_relevant_chunk_first():
    chunks = [
        _chunk("c1", "The Pro plan costs 49 dollars per month per seat."),
        _chunk("c2", "Our office is located in downtown Seattle near the water."),
        _chunk("c3", "Support hours are Monday through Friday nine to five."),
    ]
    retriever = OfflineTfidfRetriever(chunks)
    results = retriever.search("How much does the Pro plan cost per month?", top_k=3)
    assert results[0]["id"] == "c1"


def test_search_respects_top_k():
    chunks = [_chunk(f"c{i}", f"chunk number {i} about pricing plans") for i in range(10)]
    retriever = OfflineTfidfRetriever(chunks)
    results = retriever.search("pricing plans", top_k=3)
    assert len(results) <= 3


def test_search_applies_similarity_threshold():
    chunks = [
        _chunk("c1", "The Pro plan costs 49 dollars per month."),
        _chunk("c2", "Completely unrelated content about gardening and soil."),
    ]
    retriever = OfflineTfidfRetriever(chunks)
    all_results = retriever.search("Pro plan cost", top_k=2, similarity_threshold=None)
    filtered = retriever.search("Pro plan cost", top_k=2, similarity_threshold=0.5)
    assert len(filtered) <= len(all_results)
    assert all(r["similarity"] >= 0.5 for r in filtered)


def test_search_applies_metadata_filter():
    chunks = [
        _chunk("c1", "pricing info here", file_type="md"),
        _chunk("c2", "pricing info here too", file_type="html"),
    ]
    retriever = OfflineTfidfRetriever(chunks)
    results = retriever.search("pricing", top_k=5, metadata_filter={"file_type": "html"})
    assert all(r["file_type"] == "html" for r in results)
    assert len(results) == 1


def test_search_empty_corpus_returns_empty_list():
    retriever = OfflineTfidfRetriever([])
    assert retriever.search("anything", top_k=5) == []


def test_search_result_shape_matches_vector_store_search():
    """Results must carry the same keys src.vector_store.VectorStoreManager.search()
    returns, since downstream eval code treats both backends interchangeably."""
    chunks = [_chunk("c1", "some pricing content", metadata={"category": "policy"})]
    retriever = OfflineTfidfRetriever(chunks)
    results = retriever.search("pricing", top_k=1)
    result = results[0]
    for key in ("id", "text", "source", "chunk_index", "file_type", "metadata", "similarity"):
        assert key in result
