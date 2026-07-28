"""Tests for eval/evaluate_retrieval.py."""

from __future__ import annotations

import math

from eval.evaluate_retrieval import (
    compute_context_precision_at_k,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retrieval_dataset,
)


def test_recall_at_k_partial_match():
    assert compute_recall_at_k(["a", "b", "c", "d", "e"], ["c", "z"], k=5) == 0.5


def test_recall_at_k_truncated_before_hit():
    assert compute_recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_at_k_empty_gold_is_zero():
    assert compute_recall_at_k(["a"], [], k=5) == 0.0


def test_hit_rate_at_k():
    assert compute_hit_rate_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert compute_hit_rate_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert compute_hit_rate_at_k(["a"], [], k=5) == 0.0


def test_mrr_uses_full_ranked_list_not_truncated():
    # matches the reference guide's compute_mrr contract: uses full list, no k truncation
    assert compute_mrr(["a", "b", "c"], ["c"]) == 1 / 3
    assert compute_mrr(["c", "a", "b"], ["c"]) == 1.0
    assert compute_mrr(["a", "b"], ["z"]) == 0.0
    assert compute_mrr(["a"], []) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert compute_ndcg_at_k(["x", "y"], ["x", "y"], k=2) == 1.0


def test_ndcg_partial_credit_for_lower_rank():
    ndcg = compute_ndcg_at_k(["a", "b", "c"], ["c"], k=3)
    expected = (1 / math.log2(4)) / (1 / math.log2(2))
    assert abs(ndcg - expected) < 1e-9


def test_ndcg_no_gold_is_zero():
    assert compute_ndcg_at_k(["a"], [], k=5) == 0.0


def test_context_precision_at_k():
    assert compute_context_precision_at_k(["a", "b", "c", "d", "e"], ["c"], k=5) == 1 / 5
    assert compute_context_precision_at_k(["x", "y"], ["x", "y"], k=2) == 1.0
    assert compute_context_precision_at_k([], ["x"], k=0) == 0.0


def test_evaluate_retrieval_dataset_excludes_no_gold_queries_from_mean():
    per_query_results = [
        {"question_id": "q1", "retrieved_ids": ["x", "y", "z"], "gold_ids": ["x"]},
        {"question_id": "q2", "retrieved_ids": ["a", "b", "c"], "gold_ids": ["c"]},
        {"question_id": "q3", "retrieved_ids": ["m", "n"], "gold_ids": []},
    ]
    agg = evaluate_retrieval_dataset(per_query_results, k=3)
    assert agg["num_queries_evaluated"] == 2
    assert agg["num_queries_excluded_no_gold"] == 1
    assert agg["mean_recall_at_k"] == 1.0
    assert abs(agg["mean_mrr"] - (1.0 + 1 / 3) / 2) < 1e-9
    assert len(agg["per_query"]) == 3  # per-query breakdown includes ALL queries, incl. excluded


def test_evaluate_retrieval_dataset_all_no_gold_returns_zero_means_not_error():
    per_query_results = [{"question_id": "q1", "retrieved_ids": ["a"], "gold_ids": []}]
    agg = evaluate_retrieval_dataset(per_query_results, k=1)
    assert agg["num_queries_evaluated"] == 0
    assert agg["mean_recall_at_k"] == 0.0  # no ZeroDivisionError
