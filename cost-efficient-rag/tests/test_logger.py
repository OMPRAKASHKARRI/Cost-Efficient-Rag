"""Tests for src/logger.py."""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolated_log_dir(monkeypatch, tmp_path):
    """Point LOG_DIR at a tmp dir and reset the module-level configured flag."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    from src.config import get_settings
    import src.logger as logger_module

    get_settings.cache_clear()
    logger_module._CONFIGURED = False
    yield tmp_path
    get_settings.cache_clear()
    logger_module._CONFIGURED = False


def test_log_query_metrics_writes_jsonl(_isolated_log_dir):
    from src.logger import log_query_metrics

    record = log_query_metrics(
        query="what is the refund policy?",
        retrieval_latency_ms=12.345,
        generation_latency_ms=430.1,
        total_latency_ms=442.445,
        retrieved_chunk_count=3,
        prompt_tokens=512,
        completion_tokens=64,
        estimated_cost_usd=0.0001234,
        fallback_triggered=False,
        top_k=5,
    )

    assert record["retrieved_chunk_count"] == 3
    assert record["total_tokens"] == 576
    assert record["fallback_triggered"] is False

    metrics_file = _isolated_log_dir / "logs" / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["query"] == "what is the refund policy?"
    assert parsed["prompt_tokens"] == 512


def test_multiple_metrics_calls_append(_isolated_log_dir):
    from src.logger import log_query_metrics

    for i in range(3):
        log_query_metrics(
            query=f"q{i}",
            retrieval_latency_ms=1.0,
            generation_latency_ms=1.0,
            total_latency_ms=2.0,
            retrieved_chunk_count=1,
            prompt_tokens=10,
            completion_tokens=10,
            estimated_cost_usd=0.0,
            fallback_triggered=False,
            top_k=3,
        )

    metrics_file = _isolated_log_dir / "logs" / "metrics.jsonl"
    lines = metrics_file.read_text().strip().splitlines()
    assert len(lines) == 3


def test_timer_reports_positive_elapsed_ms():
    from src.logger import timer

    with timer() as elapsed:
        time.sleep(0.01)
        mid = elapsed()
    end = elapsed()

    assert mid >= 10.0
    assert end >= mid
