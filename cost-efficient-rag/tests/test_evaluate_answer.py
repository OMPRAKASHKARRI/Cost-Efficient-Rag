"""Tests for eval/evaluate_answer.py."""

from __future__ import annotations

import pytest

from eval.evaluate_answer import (
    HeuristicJudge,
    check_fallback_correctness,
    evaluate_answer_dataset,
)


def test_fallback_correctness_all_combinations():
    assert check_fallback_correctness(True, True) == 1.0
    assert check_fallback_correctness(False, False) == 1.0
    assert check_fallback_correctness(True, False) == 0.0
    assert check_fallback_correctness(False, True) == 0.0


def test_heuristic_judge_grounded_answer_scores_higher_than_fabricated():
    judge = HeuristicJudge()
    context = "The Pro plan costs $49 per month per seat and includes 100000 API calls."
    gold = "The Pro plan costs $49 per month per seat."

    grounded = judge.score("How much is Pro?", "The Pro plan costs 49 dollars per month per seat.", context, gold)
    fabricated = judge.score("How much is Pro?", "The Pro plan includes unlimited storage and a free t-shirt.", context, gold)

    assert grounded["faithfulness"] > fabricated["faithfulness"]
    assert grounded["answer_relevance"] > fabricated["answer_relevance"]


def test_heuristic_judge_empty_answer_scores_zero():
    judge = HeuristicJudge()
    scores = judge.score("q", "", "some context", "some gold answer")
    assert scores["faithfulness"] == 0.0
    assert scores["answer_relevance"] == 0.0


def test_heuristic_judge_empty_context_faithfulness_is_zero():
    judge = HeuristicJudge()
    scores = judge.score("q", "some answer text here", "", "gold")
    assert scores["faithfulness"] == 0.0


def test_evaluate_answer_dataset_excludes_fallback_from_faithfulness_mean():
    judge = HeuristicJudge()
    results = [
        {
            "question_id": "q1", "question": "x", "answer": "grounded answer here",
            "context_text": "grounded answer here in context", "ground_truth_answer": "grounded answer here",
            "fallback_triggered": False, "expect_fallback": False,
        },
        {
            "question_id": "q2", "question": "x", "answer": "fallback text",
            "context_text": "", "ground_truth_answer": "NO_ANSWER_IN_CONTEXT",
            "fallback_triggered": True, "expect_fallback": True,
        },
    ]
    agg = evaluate_answer_dataset(results, judge)
    assert agg["num_queries"] == 2
    assert agg["num_fallback_responses"] == 1
    assert agg["num_answers_scored"] == 1
    assert agg["fallback_accuracy"] == 1.0  # both q1 (correctly didn't fallback) and q2 (correctly did) match
    assert len(agg["per_query"]) == 2
    # fallback row shouldn't carry faithfulness/answer_relevance keys
    fallback_row = next(r for r in agg["per_query"] if r["question_id"] == "q2")
    assert "faithfulness" not in fallback_row


def test_evaluate_answer_dataset_detects_missed_fallback():
    judge = HeuristicJudge()
    results = [
        {
            "question_id": "q1", "question": "x", "answer": "made up answer",
            "context_text": "totally unrelated content", "ground_truth_answer": "NO_ANSWER_IN_CONTEXT",
            "fallback_triggered": False, "expect_fallback": True,  # should have fallen back but didn't
        },
    ]
    agg = evaluate_answer_dataset(results, judge)
    assert agg["fallback_accuracy"] == 0.0


def test_evaluate_answer_dataset_all_fallback_gives_zero_mean_not_error():
    judge = HeuristicJudge()
    results = [
        {
            "question_id": "q1", "question": "x", "answer": "fallback",
            "context_text": "", "ground_truth_answer": "NO_ANSWER_IN_CONTEXT",
            "fallback_triggered": True, "expect_fallback": True,
        },
    ]
    agg = evaluate_answer_dataset(results, judge)
    assert agg["num_answers_scored"] == 0
    assert agg["mean_faithfulness"] == 0.0  # no ZeroDivisionError
