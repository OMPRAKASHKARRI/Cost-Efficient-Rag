"""
Cost & latency analysis.

Two independent pieces:

1. :func:`estimate_monthly_cost` — a pure arithmetic model of embedded
   (self-hosted disk) vs. managed vector DB cost at a given vector count.
   No embeddings, LLM, or network needed to run this — it's just storage
   math plus stated pricing assumptions, so it's fully reproducible and
   testable anywhere.

2. :func:`compute_latency_percentiles` — reads the JSONL metrics log
   written by ``src/logger.log_query_metrics`` and computes p50/p95
   latency, matching what the reference guide asks the README to report.

Assumptions (all overridable via arguments, defaults documented inline):
    * Embedding dimension: 384 (all-MiniLM-L6-v2, float32 = 4 bytes/dim)
    * Metadata overhead: ~0.5 KB/vector (id, source, chunk_index,
      file_type, category, date_created, metadata_json, embedding_model —
      see the schema in src/vector_store.py)
    * Disk cost: $0.08/GB/month (AWS EBS gp3 list price ballpark, Jan 2026)
    * Managed vector DB: tiered flat "always-on pod" pricing bands
      loosely modeled on Pinecone's public pricing shape (base pod cost
      rising in steps as vector count crosses common tier boundaries) —
      NOT pulled from a live price list; treat as an order-of-magnitude
      comparison, not a quote. See README's Cost Analysis section for the
      full assumption list and citation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BYTES_PER_FLOAT32 = 4
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_METADATA_BYTES_PER_VECTOR = 512  # ~0.5 KB: id/source/chunk_index/file_type/category/date/model strings
DEFAULT_DISK_COST_PER_GB_MONTH = 0.08

# Managed vector DB cost bands (always-on pod pricing), approximate.
_MANAGED_COST_BANDS: list[tuple[int, float]] = [
    (100_000, 70.00),
    (1_000_000, 280.00),
    (10_000_000, 1200.00),
]


def _managed_db_cost(vector_count: int) -> float:
    """Piecewise "always-on pod" cost model — flat within a tier, step up at tier boundaries."""
    for threshold, cost in _MANAGED_COST_BANDS:
        if vector_count <= threshold:
            return cost
    return _MANAGED_COST_BANDS[-1][1]  # beyond the largest modeled tier: use its cost as a floor


def estimate_monthly_cost(
    vector_count: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    metadata_bytes_per_vector: int = DEFAULT_METADATA_BYTES_PER_VECTOR,
    disk_cost_per_gb_month: float = DEFAULT_DISK_COST_PER_GB_MONTH,
) -> dict[str, Any]:
    """Estimate monthly storage cost for an embedded vector store vs. a managed one.

    Returns:
        Dict with vector_count, storage size, embedded cost, managed cost,
        and the percentage savings of embedded vs. managed.
    """
    raw_vector_bytes = vector_count * embedding_dim * BYTES_PER_FLOAT32
    metadata_bytes = vector_count * metadata_bytes_per_vector
    total_bytes = raw_vector_bytes + metadata_bytes
    total_gb = total_bytes / (1024**3)

    embedded_cost = total_gb * disk_cost_per_gb_month
    managed_cost = _managed_db_cost(vector_count)
    savings_pct = (1 - (embedded_cost / managed_cost)) * 100 if managed_cost > 0 else 0.0

    return {
        "vector_count": vector_count,
        "embedding_dim": embedding_dim,
        "storage_size_gb": round(total_gb, 4),
        "embedded_db_monthly_cost_usd": round(embedded_cost, 4),
        "managed_db_monthly_cost_usd": round(managed_cost, 2),
        "savings_percentage": round(savings_pct, 2),
    }


def estimate_llm_query_cost(
    monthly_query_volume: int,
    avg_prompt_tokens: int,
    avg_completion_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> dict[str, Any]:
    """Estimate monthly LLM generation spend, separate from vector storage cost.

    This is broken out from :func:`estimate_monthly_cost` because it scales
    with *query volume*, not vector count — a corpus can grow 100x with no
    change in LLM spend if query traffic stays flat, which is worth stating
    explicitly rather than folding into one blended number.
    """
    cost_per_query = (avg_prompt_tokens / 1_000_000) * input_cost_per_1m + (
        avg_completion_tokens / 1_000_000
    ) * output_cost_per_1m
    monthly_cost = cost_per_query * monthly_query_volume
    return {
        "monthly_query_volume": monthly_query_volume,
        "avg_cost_per_query_usd": round(cost_per_query, 8),
        "estimated_monthly_llm_cost_usd": round(monthly_cost, 2),
    }


def compute_latency_percentiles(metrics_jsonl_path: str | Path) -> dict[str, Any]:
    """Compute p50/p95 for retrieval, generation, and total latency from a metrics log.

    Reads the JSON-Lines file written by ``src/logger.log_query_metrics``.
    Returns zeros with ``sample_count: 0`` if the file doesn't exist or is
    empty, rather than raising, so this can run safely before any queries
    have been logged.
    """
    path = Path(metrics_jsonl_path)
    if not path.exists():
        return _empty_percentile_result()

    retrieval_ms, generation_ms, total_ms = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            retrieval_ms.append(record["retrieval_latency_ms"])
            generation_ms.append(record["generation_latency_ms"])
            total_ms.append(record["total_latency_ms"])

    if not total_ms:
        return _empty_percentile_result()

    return {
        "sample_count": len(total_ms),
        "retrieval_latency_ms": {"p50": _percentile(retrieval_ms, 50), "p95": _percentile(retrieval_ms, 95)},
        "generation_latency_ms": {"p50": _percentile(generation_ms, 50), "p95": _percentile(generation_ms, 95)},
        "total_latency_ms": {"p50": _percentile(total_ms, 50), "p95": _percentile(total_ms, 95)},
    }


def _empty_percentile_result() -> dict[str, Any]:
    zero = {"p50": 0.0, "p95": 0.0}
    return {
        "sample_count": 0,
        "retrieval_latency_ms": dict(zero),
        "generation_latency_ms": dict(zero),
        "total_latency_ms": dict(zero),
    }


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation) — simple, deterministic, dependency-free."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = max(0, min(len(sorted_vals) - 1, int(round(pct / 100 * (len(sorted_vals) - 1)))))
    return round(sorted_vals[index], 3)


def build_cost_benchmark_table(scales: list[int] | None = None) -> list[dict[str, Any]]:
    """Convenience wrapper: run estimate_monthly_cost across the standard 100K/1M/10M scales."""
    scales = scales or [100_000, 1_000_000, 10_000_000]
    return [estimate_monthly_cost(n) for n in scales]
