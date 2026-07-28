"""
Retrieval evaluation metrics: Recall@k, Hit Rate, MRR, nDCG@k, Context Precision.

Every metric function here is pure — it takes a ranked list of retrieved
chunk IDs and a set of gold-relevant chunk IDs and returns a float. This
makes them trivially unit-testable without a vector store, embedding
model, or corpus at all (see ``tests/test_evaluate_retrieval.py``), and it
means the same functions work whether the ranked list came from
:class:`src.vector_store.VectorStoreManager` or any other retriever.

Metric definitions used here (binary relevance: a chunk is either in the
gold set or not):

* Recall@k    = |retrieved[:k] ∩ gold| / |gold|
* Hit Rate@k  = 1.0 if retrieved[:k] contains ANY gold chunk, else 0.0
* MRR         = 1 / (rank of first relevant chunk), 0.0 if none found
                (uses the full ranked list, not truncated to k, matching
                the reference guide's definition)
* nDCG@k      = DCG@k / IDCG@k, with binary relevance grades
* Context Precision@k = |retrieved[:k] ∩ gold| / k

Aggregate (dataset-level) scores are the mean of the per-query scores,
computed by :func:`evaluate_retrieval_dataset`.
"""

from __future__ import annotations

import math
from typing import Any


def compute_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of gold-relevant chunks that appear in the top-k retrieved results."""
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold = set(gold_ids)
    return len(top_k & gold) / len(gold)


def compute_hit_rate_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1.0 if at least one gold-relevant chunk appears in the top-k, else 0.0."""
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(gold_ids) else 0.0


def compute_mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant chunk in the full ranked list."""
    gold = set(gold_ids)
    for index, item_id in enumerate(retrieved_ids):
        if item_id in gold:
            return 1.0 / (index + 1)
    return 0.0


def compute_ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, with binary relevance grades.

    DCG@k  = sum_{i=1}^{k} rel_i / log2(i + 1)   (i is 1-indexed rank)
    IDCG@k = DCG@k of the ideal ranking (all relevant chunks ranked first)
    nDCG@k = DCG@k / IDCG@k  (0.0 if IDCG@k is 0, i.e. no gold chunks)
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k], start=1):
        relevance = 1.0 if item_id in gold else 0.0
        dcg += relevance / math.log2(i + 1)

    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_context_precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved chunks that are actually gold-relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    gold = set(gold_ids)
    hits = sum(1 for item_id in top_k if item_id in gold)
    return hits / len(top_k)


def evaluate_single_query(
    retrieved_ids: list[str], gold_ids: list[str], k: int
) -> dict[str, float]:
    """Compute every retrieval metric for one query."""
    return {
        "recall_at_k": compute_recall_at_k(retrieved_ids, gold_ids, k),
        "hit_rate_at_k": compute_hit_rate_at_k(retrieved_ids, gold_ids, k),
        "mrr": compute_mrr(retrieved_ids, gold_ids),
        "ndcg_at_k": compute_ndcg_at_k(retrieved_ids, gold_ids, k),
        "context_precision_at_k": compute_context_precision_at_k(retrieved_ids, gold_ids, k),
    }


def evaluate_retrieval_dataset(
    per_query_results: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    """Aggregate per-query retrieval metrics across an entire eval dataset.

    Args:
        per_query_results: list of dicts, each with ``question_id``,
            ``retrieved_ids`` (ranked list), and ``gold_ids``.
        k: the k used for @k metrics (must match how ``retrieved_ids`` was truncated upstream).

    Returns:
        Dict with per-query breakdown and dataset-level means. Queries with
        an empty gold set (e.g. designed to trigger the no-context
        fallback) are excluded from the retrieval-metric averages, since
        recall/precision/nDCG are undefined for "nothing is relevant" —
        they're evaluated separately as fallback-correctness cases instead.
    """
    per_query = []
    scored_queries = []
    for item in per_query_results:
        metrics = evaluate_single_query(item["retrieved_ids"], item["gold_ids"], k)
        row = {"question_id": item["question_id"], **metrics}
        per_query.append(row)
        if item["gold_ids"]:
            scored_queries.append(metrics)

    def _mean(key: str) -> float:
        if not scored_queries:
            return 0.0
        return sum(m[key] for m in scored_queries) / len(scored_queries)

    return {
        "k": k,
        "num_queries_evaluated": len(scored_queries),
        "num_queries_excluded_no_gold": len(per_query_results) - len(scored_queries),
        "mean_recall_at_k": _mean("recall_at_k"),
        "mean_hit_rate_at_k": _mean("hit_rate_at_k"),
        "mean_mrr": _mean("mrr"),
        "mean_ndcg_at_k": _mean("ndcg_at_k"),
        "mean_context_precision_at_k": _mean("context_precision_at_k"),
        "per_query": per_query,
    }
