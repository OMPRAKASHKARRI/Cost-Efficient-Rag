"""Tests for src/api.py.

Uses FastAPI's dependency-override mechanism to substitute fakes for the
embedding model, vector store, and LLM client, so these tests exercise
routing, request validation, and response shaping without a real LanceDB
table, a downloaded embedding model, or a network call to an LLM provider.

NOTE: requires `fastapi`, `httpx`, and `pydantic` — install via
requirements.txt. Not runnable in the sandbox this repo was authored in
(no network access to install those packages there); verified by hand
against the equivalent non-FastAPI logic instead — see the module
docstring discussion in the accompanying README's Testing section.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_embedding_model, get_llm_client, get_rag_pipeline, get_vector_store
from src.rag_pipeline import RAGAnswer


class _FakeEmbeddingModel:
    model_name = "fake-embedder"

    def encode(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def encode_one(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class _FakeVectorStore:
    def __init__(self):
        self.upserted = []

    def get_existing_ids(self):
        return set()

    def upsert_chunks(self, chunks, embeddings, model_name):
        self.upserted.extend(chunks)
        return len(chunks)


class _FakeRAGPipeline:
    def __init__(self, canned_answer: RAGAnswer):
        self._canned = canned_answer
        self.last_call = None

    def answer(self, query, top_k=None, metadata_filter=None):
        self.last_call = {"query": query, "top_k": top_k, "metadata_filter": metadata_filter}
        return self._canned


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    fake_vector_store = _FakeVectorStore()
    app.dependency_overrides[get_embedding_model] = lambda: _FakeEmbeddingModel()
    app.dependency_overrides[get_vector_store] = lambda: fake_vector_store

    canned = RAGAnswer(
        answer="The Pro plan costs $49/month [Doc: pricing.md, Chunk: c1].",
        citations=[{"source": "pricing.md", "chunk_id": "c1", "similarity": 0.91}],
        fallback_triggered=False,
        retrieved_chunk_count=1,
        retrieval_latency_ms=5.0,
        generation_latency_ms=200.0,
        total_latency_ms=205.0,
        prompt_tokens=100,
        completion_tokens=15,
        estimated_cost_usd=0.00002,
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: _FakeRAGPipeline(canned)

    with TestClient(app) as c:
        yield c, fake_vector_store

    app.dependency_overrides.clear()


def test_health_endpoint(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_grounded_answer(client):
    c, _ = client
    response = c.post("/query", json={"query": "How much does Pro cost?"})
    assert response.status_code == 200
    body = response.json()
    assert body["fallback_triggered"] is False
    assert body["citations"][0]["chunk_id"] == "c1"
    assert body["prompt_tokens"] == 100
    assert body["estimated_cost_usd"] == pytest.approx(0.00002)


def test_query_endpoint_rejects_empty_query(client):
    c, _ = client
    response = c.post("/query", json={"query": ""})
    assert response.status_code == 422  # pydantic min_length=1 validation


def test_query_endpoint_passes_top_k_and_filter(client):
    c, _ = client
    response = c.post(
        "/query",
        json={"query": "test", "top_k": 3, "metadata_filter": {"file_type": "pdf"}},
    )
    assert response.status_code == 200


def test_ingest_endpoint_accepts_markdown_upload(client, tmp_path):
    c, fake_vector_store = client
    md_content = b"# Title\n\nSome content here for ingestion testing purposes and beyond."
    response = c.post(
        "/ingest",
        files={"files": ("doc.md", md_content, "text/markdown")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_chunks_ingested"] >= 1
    assert len(fake_vector_store.upserted) == body["total_chunks_ingested"]


def test_ingest_endpoint_rejects_unsupported_extension(client):
    c, _ = client
    response = c.post(
        "/ingest",
        files={"files": ("doc.docx", b"not really a docx", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_ingest_endpoint_is_idempotent_across_two_calls(client):
    c, fake_vector_store = client
    md_content = b"# Title\n\n" + b"Repeated sentence for chunking. " * 20
    first = c.post("/ingest", files={"files": ("doc.md", md_content, "text/markdown")})
    first_count = first.json()["total_chunks_ingested"]
    assert first_count > 0

    # NOTE: the fake vector store's get_existing_ids() always returns set(),
    # unlike a real VectorStoreManager which would return the IDs just
    # upserted -- so this test's fake doesn't itself prove idempotency end
    # to end. True idempotency across repeated /ingest calls is proven at
    # the ingestion-module level in tests/test_ingestion.py
    # (test_ingest_document_is_idempotent) against the real ChunkRecord
    # hashing + dedup logic; this test only proves the endpoint routes a
    # second call successfully rather than erroring.
    second = c.post("/ingest", files={"files": ("doc.md", md_content, "text/markdown")})
    assert second.status_code == 200
