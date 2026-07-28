"""
Central application configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========= LLM =========

    llm_provider: str = Field(default="groq")

    groq_api_key: str = Field(default="")

    llm_model_name: str = Field(default="openai/gpt-oss-120b")

    llm_judge_model_name: str = Field(default="openai/gpt-oss-120b")

    # ========= Vector Store =========

    vector_store_path: str = Field(default="./data/lancedb_store")

    vector_table_name: str = Field(default="rag_chunks")

    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")

    embedding_dimension: int = Field(default=384, gt=0)

    # ========= Chunking =========

    default_chunk_size: int = Field(default=500, gt=0)

    default_chunk_overlap: int = Field(default=50, ge=0)

    # ========= Retrieval =========

    default_top_k: int = Field(default=5, gt=0)

    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # ========= Logging =========

    log_level: str = Field(default="INFO")

    log_dir: str = Field(default="./logs")

    environment: str = Field(default="development")

    # ========= Cost =========

    llm_input_cost_per_1m_tokens: float = Field(default=0.15)

    llm_output_cost_per_1m_tokens: float = Field(default=0.60)

    monthly_query_volume: int = Field(default=50000)

    @field_validator("default_chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        chunk = info.data.get("default_chunk_size")

        if chunk is not None and v >= chunk:
            raise ValueError(
                "default_chunk_overlap must be smaller than default_chunk_size"
            )

        return v

    @property
    def vector_store_dir(self):
        path = Path(self.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir_path(self):
        path = Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_llm_key(self):

        if self.llm_provider.lower() == "groq":

            if not self.groq_api_key:

                raise RuntimeError(
                    "GROQ_API_KEY is missing in your .env"
                )

            return self.groq_api_key

        raise RuntimeError(
            f"Unsupported LLM provider: {self.llm_provider}"
        )


@lru_cache(maxsize=1)
def get_settings():

    return Settings()