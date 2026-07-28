

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.ingestion import (
    UnsupportedFileTypeError,
    build_chunk_records,
    deduplicate_chunks,
    load_document,
)
from src.logger import get_logger
from src.rag_pipeline import GroqChatClient, RAGPipeline
from src.vector_store import EmbeddingModel, VectorStoreManager

logger = get_logger()

app = FastAPI(
    title="Cost-Efficient RAG Application",
    description="Retrieval-Augmented Generation over PDF/HTML/Markdown, backed by an embedded LanceDB store.",
    version="1.0.0",
)


# ── Dependency singletons ─────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    return EmbeddingModel(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreManager:
    settings = get_settings()
    embedding_model = get_embedding_model()
    return VectorStoreManager(
        db_path=settings.vector_store_dir,
        table_name=settings.vector_table_name,
        embedding_dim=embedding_model.dimension,
    )


@lru_cache(maxsize=1)
def get_llm_client() -> GroqChatClient:
    settings = get_settings()
    return GroqChatClient(api_key=settings.require_llm_key())


def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> RAGPipeline:
    # LLM client is resolved lazily inside here (not as its own Depends)
    # so that /ingest — which never needs an LLM — keeps working even
    # when OPENAI_API_KEY isn't set.
    return RAGPipeline(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_client=get_llm_client(),
        settings=settings,
    )


# ── Request / response schemas ───────────────────────────────────────────


class IngestFileResult(BaseModel):
    source: str
    chunks_ingested: int
    chunks_skipped_duplicate: int


class IngestResponse(BaseModel):
    total_chunks_ingested: int
    files: list[IngestFileResult]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(default=None, gt=0, description="Number of chunks to retrieve.")
    metadata_filter: dict[str, Any] | None = Field(
        default=None, description="Equality filters, e.g. {\"file_type\": \"pdf\"}."
    )


class Citation(BaseModel):
    source: str
    chunk_id: str
    similarity: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    fallback_triggered: bool
    retrieved_chunk_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
) -> IngestResponse:
    """Ingest one or more PDF/HTML/Markdown files: chunk, embed, and idempotently upsert."""
    upload_dir = Path(settings.log_dir).parent / "data" / "raw_documents"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for upload in files:
        dest = upload_dir / upload.filename
        contents = await upload.read()
        dest.write_bytes(contents)
        saved_paths.append(dest)

    existing_ids = vector_store.get_existing_ids()
    file_results: list[IngestFileResult] = []
    all_new_chunks = []

    for path in saved_paths:
        try:
            text, file_type = load_document(path)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        all_chunks_for_file = build_chunk_records(
            text=text,
            source=str(path),
            file_type=file_type,
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
        )
        new_chunks = deduplicate_chunks(all_chunks_for_file, existing_ids)
        existing_ids.update(c.id for c in new_chunks)
        all_new_chunks.extend(new_chunks)

        file_results.append(
            IngestFileResult(
                source=str(path),
                chunks_ingested=len(new_chunks),
                chunks_skipped_duplicate=len(all_chunks_for_file) - len(new_chunks),
            )
        )

    if all_new_chunks:
        texts = [c.text for c in all_new_chunks]
        embeddings = embedding_model.encode(texts)
        vector_store.upsert_chunks(all_new_chunks, embeddings, embedding_model.model_name)

    return IngestResponse(
        total_chunks_ingested=len(all_new_chunks),
        files=file_results,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Answer a question using top-k retrieval + grounded, cited generation."""
    try:
        result = pipeline.answer(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )
    except RuntimeError as e:
        # e.g. missing OPENAI_API_KEY, surfaced from Settings.require_llm_key()
        raise HTTPException(status_code=503, detail=str(e)) from e

    return QueryResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        fallback_triggered=result.fallback_triggered,
        retrieved_chunk_count=result.retrieved_chunk_count,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )
