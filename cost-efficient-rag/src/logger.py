

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from loguru import logger as _logger

from src.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotently configure loguru's console + metrics sinks."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    _logger.remove()  # drop loguru's default stderr sink so we control format/level

    _logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    metrics_path = settings.log_dir_path / "metrics.jsonl"
    _logger.add(
        metrics_path,
        level="INFO",
        format="{message}",
        filter=lambda record: record["extra"].get("metrics_record", False),
        rotation="10 MB",
        serialize=False,
    )

    _CONFIGURED = True


def get_logger():
    """Return the configured loguru logger for general-purpose app logging."""
    configure_logging()
    return _logger


def log_query_metrics(
    *,
    query: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    retrieved_chunk_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    estimated_cost_usd: float,
    fallback_triggered: bool,
    top_k: int,
) -> dict[str, Any]:
    """Emit one structured metrics record for a single /query request.

    Returns the record dict as well (in addition to logging it) so API
    handlers can include the same numbers in the HTTP response without
    recomputing anything.
    """
    configure_logging()
    record = {
        "timestamp": time.time(),
        "query": query,
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "generation_latency_ms": round(generation_latency_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "retrieved_chunk_count": retrieved_chunk_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "fallback_triggered": fallback_triggered,
        "top_k": top_k,
    }
    _logger.bind(metrics_record=True).info(json.dumps(record))
    return record


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Context manager yielding a callable that returns elapsed ms so far.

    Usage::

        with timer() as elapsed:
            do_work()
        ms = elapsed()
    """
    start = time.perf_counter()

    def _elapsed() -> float:
        return (time.perf_counter() - start) * 1000.0

    yield _elapsed
