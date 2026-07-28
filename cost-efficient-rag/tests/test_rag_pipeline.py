"""Tests for src/rag_pipeline.py.

Uses hand-written fakes for the vector store, embedding model, and LLM
client (all accessed through narrow interfaces in rag_pipeline.py) so the
orchestration logic — fallback-before-LLM-call, fallback-from-LLM-output,
citation building, cost accounting — is tested without a real vector DB,
embedding model, or network/API key, none of which are available offline.
"""

from __future__ import annotations

import pytest

from src.rag_pipeline import (
    NO_CONTEXT_FALLBACK_MESSAGE,
    RAGPipeline,
    SimpleLLMResponse,
    build_system_prompt,
    estimate_cost_usd,
    format_context_block,
)


# ── Pure prompt / cost helpers ────────────────────────────────────────────


def test_format_context_block_includes_source_and_chunk_id():
    chunks = [
        {"source": "pricing.md", "id": "abc123", "text": "Pro plan is $49/month."},
        {"source": "faq.html", "id": "def456", "text": "Support hours are 9-5 ET."},
    ]
    block = format_context_block(chunks)
    assert "[Doc: pricing.md, Chunk: abc123]" in block
    assert "Pro plan is $49/month." in block
    assert "[Doc: faq.html, Chunk: def456]" in block


def test_format_context_block_empty_list():
    assert format_context_block([]) == ""


def test_build_system_prompt_contains_fallback_instruction_and_context():
    chunks = [{"source": "doc.md", "id": "x1", "text": "Some fact."}]
    prompt = build_system_prompt(chunks)
    assert NO_CONTEXT_FALLBACK_MESSAGE in prompt
    assert "Some fact." in prompt
    assert "ONLY the context provided" in prompt


def test_estimate_cost_usd_arithmetic():
    # 1000 prompt tokens @ $0.15/1M + 500 completion tokens @ $0.60/1M
    cost = estimate_cost_usd(1000, 500, input_cost_per_1m=0.15, output_cost_per_1m=0.60)
    expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    assert cost == pytest.approx(expected)


def test_estimate_cost_usd_zero_tokens_is_zero_cost():
    assert estimate_cost_usd(0, 0, 0.15, 0.60) == 0.0


# ── Fakes for RAGPipeline collaborators ──────────────────────────────────


class _FakeSettings:
    default_top_k = 5
    similarity_threshold = 0.35
    llm_model_name = "fake-model"
    llm_input_cost_per_1m_tokens = 0.15
    llm_output_cost_per_1m_tokens = 0.60


class _FakeEmbeddingModel:
    def encode_one(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class _FakeVectorStore:
    def __init__(self, chunks_to_return):
        self._chunks = chunks_to_return
        self.search_calls = []

    def search(self, query_vector, top_k, metadata_filter=None, similarity_threshold=None):
        self.search_calls.append(
            {"top_k": top_k, "metadata_filter": metadata_filter, "threshold": similarity_threshold}
        )
        return self._chunks


class _FakeLLMClient:
    def __init__(self, response: SimpleLLMResponse):
        self._response = response
        self.call_count = 0
        self.last_call = None

    def generate(self, system_prompt, user_query, model):
        self.call_count += 1
        self.last_call = {"system_prompt": system_prompt, "user_query": user_query, "model": model}
        return self._response


@pytest.fixture(autouse=True)
def _patch_metrics_logging(monkeypatch):
    """Replace log_query_metrics with a no-op recorder so tests don't need the
    real (pydantic/loguru-backed) logging stack, which is unavailable offline."""
    calls = []

    def fake_log_query_metrics(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr("src.rag_pipeline.log_query_metrics", fake_log_query_metrics)
    return calls


# ── RAGPipeline.answer() ──────────────────────────────────────────────────


def test_answer_normal_flow_returns_citations_and_cost(_patch_metrics_logging):
    chunks = [
        {"id": "c1", "source": "pricing.md", "text": "Pro plan is $49/month.", "similarity": 0.9},
    ]
    llm_response = SimpleLLMResponse(
        text="The Pro plan costs $49/month [Doc: pricing.md, Chunk: c1].",
        prompt_tokens=120,
        completion_tokens=20,
    )
    pipeline = RAGPipeline(
        vector_store=_FakeVectorStore(chunks),
        embedding_model=_FakeEmbeddingModel(),
        llm_client=_FakeLLMClient(llm_response),
        settings=_FakeSettings(),
    )

    result = pipeline.answer("How much does Pro cost?")

    assert result.fallback_triggered is False
    assert result.answer == llm_response.text
    assert result.retrieved_chunk_count == 1
    assert result.citations == [{"source": "pricing.md", "chunk_id": "c1", "similarity": 0.9}]
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 20
    assert result.estimated_cost_usd > 0
    assert len(_patch_metrics_logging) == 1  # metrics were logged exactly once


def test_answer_empty_retrieval_never_calls_llm(_patch_metrics_logging):
    llm = _FakeLLMClient(SimpleLLMResponse(text="should never be returned", prompt_tokens=0, completion_tokens=0))
    pipeline = RAGPipeline(
        vector_store=_FakeVectorStore([]),  # nothing clears the similarity floor
        embedding_model=_FakeEmbeddingModel(),
        llm_client=llm,
        settings=_FakeSettings(),
    )

    result = pipeline.answer("What is the meaning of life?")

    assert llm.call_count == 0  # no wasted / potentially hallucinated API call
    assert result.fallback_triggered is True
    assert result.answer == NO_CONTEXT_FALLBACK_MESSAGE
    assert result.citations == []
    assert result.estimated_cost_usd == 0.0
    assert result.retrieved_chunk_count == 0


def test_answer_llm_triggered_fallback_has_no_citations(_patch_metrics_logging):
    """Chunks WERE retrieved (clear the similarity floor) but the LLM itself
    judged them insufficient and returned the exact fallback string — this
    must still surface as fallback_triggered=True with no citations, since
    citing chunks the model explicitly said were insufficient would be
    misleading."""
    chunks = [{"id": "c1", "source": "unrelated.md", "text": "unrelated content", "similarity": 0.4}]
    llm_response = SimpleLLMResponse(
        text=NO_CONTEXT_FALLBACK_MESSAGE, prompt_tokens=90, completion_tokens=15
    )
    pipeline = RAGPipeline(
        vector_store=_FakeVectorStore(chunks),
        embedding_model=_FakeEmbeddingModel(),
        llm_client=_FakeLLMClient(llm_response),
        settings=_FakeSettings(),
    )

    result = pipeline.answer("Some off-topic question")

    assert result.fallback_triggered is True
    assert result.citations == []
    assert result.answer == NO_CONTEXT_FALLBACK_MESSAGE
    # tokens/cost still tracked even on a fallback answer -- the call did happen
    assert result.prompt_tokens == 90


def test_answer_passes_top_k_and_metadata_filter_through_to_search(_patch_metrics_logging):
    vector_store = _FakeVectorStore([])
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_model=_FakeEmbeddingModel(),
        llm_client=_FakeLLMClient(SimpleLLMResponse("x", 0, 0)),
        settings=_FakeSettings(),
    )

    pipeline.answer("query", top_k=7, metadata_filter={"file_type": "pdf"})

    assert vector_store.search_calls[0]["top_k"] == 7
    assert vector_store.search_calls[0]["metadata_filter"] == {"file_type": "pdf"}
    assert vector_store.search_calls[0]["threshold"] == 0.35


def test_answer_defaults_top_k_from_settings(_patch_metrics_logging):
    vector_store = _FakeVectorStore([])
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_model=_FakeEmbeddingModel(),
        llm_client=_FakeLLMClient(SimpleLLMResponse("x", 0, 0)),
        settings=_FakeSettings(),
    )

    pipeline.answer("query")  # no top_k passed

    assert vector_store.search_calls[0]["top_k"] == _FakeSettings.default_top_k
