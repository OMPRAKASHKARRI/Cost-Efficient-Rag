"""Tests for src/config.py."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure get_settings()'s lru_cache doesn't leak state between tests."""
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_load_without_env_file(monkeypatch, tmp_path):
    """With no .env and no relevant env vars, sane defaults are used."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.lower().startswith(("openai_", "vector_", "embedding_", "default_", "log_", "llm_")):
            monkeypatch.delenv(key, raising=False)

    from src.config import Settings

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.default_chunk_size == 500
    assert settings.default_chunk_overlap == 50
    assert settings.embedding_dimension == 384
    assert settings.openai_api_key == ""


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("DEFAULT_CHUNK_SIZE", "800")
    monkeypatch.setenv("DEFAULT_TOP_K", "10")

    from src.config import Settings

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.default_chunk_size == 800
    assert settings.default_top_k == 10


def test_overlap_must_be_smaller_than_chunk_size():
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, default_chunk_size=100, default_chunk_overlap=100)  # type: ignore[call-arg]


def test_overlap_equal_to_zero_is_valid():
    from src.config import Settings

    settings = Settings(_env_file=None, default_chunk_size=100, default_chunk_overlap=0)  # type: ignore[call-arg]
    assert settings.default_chunk_overlap == 0


def test_require_llm_key_raises_when_missing():
    from src.config import Settings

    settings = Settings(_env_file=None, openai_api_key="")  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        settings.require_llm_key()


def test_require_llm_key_returns_value_when_present():
    from src.config import Settings

    settings = Settings(_env_file=None, openai_api_key="sk-test-123")  # type: ignore[call-arg]
    assert settings.require_llm_key() == "sk-test-123"


def test_vector_store_dir_creates_directory(tmp_path):
    from src.config import Settings

    target = tmp_path / "nested" / "lancedb_store"
    settings = Settings(_env_file=None, vector_store_path=str(target))  # type: ignore[call-arg]
    resolved = settings.vector_store_dir
    assert resolved.exists()
    assert resolved.is_dir()


def test_get_settings_is_cached():
    from src.config import get_settings

    a = get_settings()
    b = get_settings()
    assert a is b


def test_invalid_similarity_threshold_rejected():
    from pydantic import ValidationError

    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_threshold=1.5)  # type: ignore[call-arg]
