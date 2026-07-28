"""Tests for eval/cost_analysis.py."""

from __future__ import annotations

import json

import pytest

from eval.cost_analysis import (
    build_cost_benchmark_table,
    compute_latency_percentiles,
    estimate_llm_query_cost,
    estimate_monthly_cost,
)


def test_estimate_monthly_cost_matches_manual_calculation():
    result = estimate_monthly_cost(100_000)
    raw_bytes = 100_000 * 384 * 4
    metadata_bytes = 100_000 * 512
    expected_gb = (raw_bytes + metadata_bytes) / (1024**3)
    assert result["storage_size_gb"] == round(expected_gb, 4)
    assert result["embedded_db_monthly_cost_usd"] == round(expected_gb * 0.08, 4)


def test_estimate_monthly_cost_scales_up_with_vector_count():
    small = estimate_monthly_cost(100_000)
    large = estimate_monthly_cost(10_000_000)
    assert large["storage_size_gb"] > small["storage_size_gb"]
    assert large["embedded_db_monthly_cost_usd"] > small["embedded_db_monthly_cost_usd"]
    assert large["managed_db_monthly_cost_usd"] >= small["managed_db_monthly_cost_usd"]


def test_estimate_monthly_cost_embedded_much_cheaper_than_managed():
    for n in (100_000, 1_000_000, 10_000_000):
        result = estimate_monthly_cost(n)
        assert result["embedded_db_monthly_cost_usd"] < result["managed_db_monthly_cost_usd"]
        assert result["savings_percentage"] > 90


def test_estimate_monthly_cost_custom_assumptions():
    result = estimate_monthly_cost(
        100_000, embedding_dim=1536, metadata_bytes_per_vector=1000, disk_cost_per_gb_month=0.10
    )
    assert result["embedding_dim"] == 1536
    # larger embedding dim -> more storage than the 384-dim default at the same vector count
    default_result = estimate_monthly_cost(100_000)
    assert result["storage_size_gb"] > default_result["storage_size_gb"]


def test_estimate_llm_query_cost_arithmetic():
    result = estimate_llm_query_cost(
        monthly_query_volume=1000, avg_prompt_tokens=500, avg_completion_tokens=100,
        input_cost_per_1m=0.15, output_cost_per_1m=0.60,
    )
    expected_per_query = (500 / 1_000_000) * 0.15 + (100 / 1_000_000) * 0.60
    assert result["avg_cost_per_query_usd"] == pytest.approx(expected_per_query)
    assert result["estimated_monthly_llm_cost_usd"] == pytest.approx(expected_per_query * 1000, abs=0.01)


def test_compute_latency_percentiles_missing_file_returns_zeros(tmp_path):
    result = compute_latency_percentiles(tmp_path / "does_not_exist.jsonl")
    assert result["sample_count"] == 0
    assert result["total_latency_ms"]["p50"] == 0.0


def test_compute_latency_percentiles_empty_file_returns_zeros(tmp_path):
    f = tmp_path / "metrics.jsonl"
    f.write_text("")
    result = compute_latency_percentiles(f)
    assert result["sample_count"] == 0


def test_compute_latency_percentiles_real_records(tmp_path):
    f = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"retrieval_latency_ms": i, "generation_latency_ms": i * 10, "total_latency_ms": i * 11})
        for i in range(1, 11)
    ]
    f.write_text("\n".join(lines))

    result = compute_latency_percentiles(f)
    assert result["sample_count"] == 10
    assert result["total_latency_ms"]["p95"] >= result["total_latency_ms"]["p50"]
    assert result["retrieval_latency_ms"]["p50"] > 0


def test_build_cost_benchmark_table_default_scales():
    table = build_cost_benchmark_table()
    assert [row["vector_count"] for row in table] == [100_000, 1_000_000, 10_000_000]
    assert len(table) == 3


def test_build_cost_benchmark_table_custom_scales():
    table = build_cost_benchmark_table([500, 5000])
    assert [row["vector_count"] for row in table] == [500, 5000]
