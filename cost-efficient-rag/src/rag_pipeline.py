

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.config import Settings, get_settings
from src.logger import get_logger, log_query_metrics, timer
from src.vector_store import EmbeddingModel, VectorStoreManager

logger = get_logger()

NO_CONTEXT_FALLBACK_MESSAGE = (
    "I do not have sufficient information in the provided context to answer this question."
)

SYSTEM_PROMPT_TEMPLATE = """You are a precise QA assistant. Answer the user question using ONLY the context provided below.
For every factual claim, cite the source using the chunk ID in brackets [Doc: <source>, Chunk: <id>].
Do not use any outside knowledge. If the provided context does not contain enough information to \
answer the question, respond with exactly this text and nothing else:
"{fallback_message}"

Context:
{context_block}
"""


class LLMResponse(Protocol):
    """Shape expected back from an LLMClient.generate() call."""

    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class SimpleLLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """Minimal interface rag_pipeline.py depends on — swap in a fake for tests."""

    def generate(self, system_prompt: str, user_query: str, model: str) -> SimpleLLMResponse: ...

class GroqChatClient:
    """Thin wrapper over the Groq chat completions API implementing LLMClient."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_query: str,
        model: str,
    ) -> SimpleLLMResponse:

        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
        )

        usage = response.usage

        return SimpleLLMResponse(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


def format_context_block(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into the ``[Doc: source, Chunk: id]\\ntext`` blocks the prompt expects."""
    parts = []
    for chunk in retrieved_chunks:
        parts.append(f"[Doc: {chunk['source']}, Chunk: {chunk['id']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def build_system_prompt(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Build the grounded-QA system prompt from retrieved chunks."""
    context_block = format_context_block(retrieved_chunks)
    return SYSTEM_PROMPT_TEMPLATE.format(
        fallback_message=NO_CONTEXT_FALLBACK_MESSAGE, context_block=context_block
    )


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> float:
    """Estimate USD cost of one LLM call from token counts and per-1M-token pricing."""
    return (prompt_tokens / 1_000_000) * input_cost_per_1m + (
        completion_tokens / 1_000_000
    ) * output_cost_per_1m


@dataclass
class RAGAnswer:
    """Full result of a /query request — everything the API response and metrics logging need."""

    answer: str
    citations: list[dict[str, Any]]
    fallback_triggered: bool
    retrieved_chunk_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class RAGPipeline:
    """Orchestrates retrieval + grounded generation for one query."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Embed the query and run top-k similarity search. Returns (chunks, latency_ms)."""
        k = top_k or self.settings.default_top_k
        with timer() as elapsed:
            query_vector = self.embedding_model.encode_one(query)
            chunks = self.vector_store.search(
                query_vector,
                top_k=k,
                metadata_filter=metadata_filter,
                similarity_threshold=self.settings.similarity_threshold,
            )
        return chunks, elapsed()

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGAnswer:
        """Run the full retrieve -> ground -> generate pipeline for one query.

        Fallback logic (no hallucination on missing context) triggers in
        two places: before ever calling the LLM if retrieval returns zero
        chunks (saves a wasted, potentially hallucinated API call
        entirely), and via the prompt's own instruction if the LLM itself
        judges the retrieved chunks insufficient despite some existing
        above the similarity floor.
        """
        top_k = top_k or self.settings.default_top_k
        overall_start_elapsed = None
        with timer() as total_elapsed:
            retrieved_chunks, retrieval_latency_ms = self.retrieve(query, top_k, metadata_filter)

            if not retrieved_chunks:
                logger.info(f"No chunks cleared the similarity threshold for query: {query!r}")
                result = RAGAnswer(
                    answer=NO_CONTEXT_FALLBACK_MESSAGE,
                    citations=[],
                    fallback_triggered=True,
                    retrieved_chunk_count=0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=0.0,
                    total_latency_ms=total_elapsed(),
                    prompt_tokens=0,
                    completion_tokens=0,
                    estimated_cost_usd=0.0,
                )
                self._log(query, result, top_k)
                return result

            system_prompt = build_system_prompt(retrieved_chunks)
            with timer() as gen_elapsed:
                llm_response = self.llm_client.generate(
                    system_prompt=system_prompt,
                    user_query=query,
                    model=self.settings.llm_model_name,
                )
            generation_latency_ms = gen_elapsed()

            fallback_triggered = llm_response.text.strip() == NO_CONTEXT_FALLBACK_MESSAGE
            citations = [] if fallback_triggered else [
                {"source": c["source"], "chunk_id": c["id"], "similarity": c["similarity"]}
                for c in retrieved_chunks
            ]
            cost = estimate_cost_usd(
                llm_response.prompt_tokens,
                llm_response.completion_tokens,
                self.settings.llm_input_cost_per_1m_tokens,
                self.settings.llm_output_cost_per_1m_tokens,
            )

            result = RAGAnswer(
                answer=llm_response.text,
                citations=citations,
                fallback_triggered=fallback_triggered,
                retrieved_chunk_count=len(retrieved_chunks),
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_elapsed(),
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                estimated_cost_usd=cost,
            )
        self._log(query, result, top_k)
        return result

    def _log(self, query: str, result: RAGAnswer, top_k: int) -> None:
        log_query_metrics(
            query=query,
            retrieval_latency_ms=result.retrieval_latency_ms,
            generation_latency_ms=result.generation_latency_ms,
            total_latency_ms=result.total_latency_ms,
            retrieved_chunk_count=result.retrieved_chunk_count,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            fallback_triggered=result.fallback_triggered,
            top_k=top_k,
        )
