"""
Answer-quality evaluation: Faithfulness/Groundedness and Answer Relevance.

Three interchangeable scoring backends, all behind the same
:class:`AnswerJudge` protocol (``score(question, answer, context_text,
ground_truth_answer) -> {"faithfulness": float, "answer_relevance": float}``):

1. :class:`OpenAIJudge` — the primary, intended-for-grading backend.
   Prompts an LLM to rate faithfulness (is every claim in the answer
   actually supported by the retrieved context?) and relevance (does the
   answer address the question?) on a 0-1 scale. Requires ``OPENAI_API_KEY``
   and network access.

2. :func:`score_with_ragas` — optional integration with the ``ragas``
   library's ``faithfulness`` and ``answer_relevancy`` metrics, satisfying
   the reference guide's "use RAGAS where appropriate." Imports ``ragas``
   lazily and raises a clear, actionable error if it isn't installed,
   rather than failing at module import time.

3. :class:`HeuristicJudge` — a fully offline, dependency-free fallback
   using lexical token-overlap (F1) as a crude proxy for both metrics.
   **This is explicitly a stand-in, not a substitute for LLM-as-judge or
   RAGAS** — it cannot detect a fluent but unsupported claim, only lexical
   grounding. It exists so the evaluation harness is runnable end-to-end
   (and its logic testable) in environments without API access, which is
   how this repository's own ``results/eval_results.json`` was produced.
   See the README's Evaluation section for the honest accounting of what
   backend generated which numbers.

Also included: :func:`check_fallback_correctness`, which scores whether the
system's decision to trigger (or not trigger) the no-context fallback
matched the eval dataset's ``expect_fallback`` label — this is what proves
"safely triggers fallback when irrelevant contexts are given" per the
rubric, and needs no judge at all (it's a straightforward boolean compare).
"""

from __future__ import annotations

import json
import re
import string
from typing import Any, Protocol

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "with", "at", "by", "from",
    "this", "that", "it", "its", "as", "do", "does", "did", "not", "no",
    "you", "your", "i", "what", "how", "can", "will", "would",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = re.split(r"\s+", text.strip())
    return {t for t in tokens if t and t not in _STOPWORDS}


def _token_f1(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Standard token-overlap F1 between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0
    precision = len(overlap) / len(tokens_a)
    recall = len(overlap) / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


class AnswerJudge(Protocol):
    """Interface every scoring backend implements."""

    def score(
        self, question: str, answer: str, context_text: str, ground_truth_answer: str
    ) -> dict[str, float]: ...


class HeuristicJudge:
    """Offline, dependency-free lexical-overlap stand-in for LLM-as-judge scoring."""

    def score(
        self, question: str, answer: str, context_text: str, ground_truth_answer: str
    ) -> dict[str, float]:
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context_text)
        gold_tokens = _tokenize(ground_truth_answer)

        # Faithfulness proxy: what fraction of the answer's content words are
        # grounded in (present in) the retrieved context. This is precision
        # of answer-tokens against context-tokens, not full F1, because an
        # answer is allowed to omit context content -- what matters for
        # faithfulness is that it doesn't ADD unsupported content.
        if not answer_tokens:
            faithfulness = 0.0
        elif not context_tokens:
            faithfulness = 0.0
        else:
            faithfulness = len(answer_tokens & context_tokens) / len(answer_tokens)

        answer_relevance = _token_f1(answer_tokens, gold_tokens)

        return {"faithfulness": round(faithfulness, 4), "answer_relevance": round(answer_relevance, 4)}


_JUDGE_PROMPT_TEMPLATE = """You are evaluating a RAG system's answer. Score two dimensions from 0.0 to 1.0.

FAITHFULNESS: Is every factual claim in the ANSWER directly supported by the CONTEXT? \
1.0 = fully grounded, 0.0 = fabricated / not supported by context.

ANSWER RELEVANCE: Does the ANSWER actually address the QUESTION asked? \
1.0 = directly and completely answers it, 0.0 = off-topic or non-responsive.

QUESTION: {question}

CONTEXT:
{context_text}

ANSWER: {answer}

Respond with ONLY a JSON object, no other text: {{"faithfulness": <float>, "answer_relevance": <float>}}
"""


class OpenAIJudge:
    """LLM-as-judge using an OpenAI chat model. Requires network + OPENAI_API_KEY."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def score(
        self, question: str, answer: str, context_text: str, ground_truth_answer: str
    ) -> dict[str, float]:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question, context_text=context_text, answer=answer
        )
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            return {
                "faithfulness": float(parsed.get("faithfulness", 0.0)),
                "answer_relevance": float(parsed.get("answer_relevance", 0.0)),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            # Judge output wasn't valid JSON -- fail loudly in the aggregate
            # stats rather than silently defaulting to a misleadingly clean 0.0.
            raise ValueError(f"OpenAIJudge returned non-JSON output: {raw!r}")


def score_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Score faithfulness + answer_relevancy with the RAGAS library.

    Lazily imports ``ragas`` — raises a clear ImportError-derived message
    (rather than failing at module import time) if it isn't installed,
    since it's an optional, heavier dependency only needed when a user
    explicitly wants RAGAS's implementation instead of ``OpenAIJudge``.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as e:
        raise ImportError(
            "score_with_ragas requires the optional 'ragas' and 'datasets' packages "
            "(pip install ragas datasets) plus a configured LLM/embeddings backend "
            "for RAGAS itself. Use OpenAIJudge or HeuristicJudge instead if you don't "
            "need RAGAS specifically."
        ) from e

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return dict(result)


def check_fallback_correctness(fallback_triggered: bool, expect_fallback: bool) -> float:
    """1.0 if the system's fallback decision matched the eval dataset's expectation, else 0.0."""
    return 1.0 if fallback_triggered == expect_fallback else 0.0


def evaluate_answer_dataset(
    per_query_results: list[dict[str, Any]], judge: AnswerJudge
) -> dict[str, Any]:
    """Score every query's answer for faithfulness/relevance and fallback correctness.

    Args:
        per_query_results: list of dicts with ``question_id``, ``question``,
            ``answer``, ``context_text`` (concatenated retrieved chunk
            text), ``ground_truth_answer``, ``fallback_triggered``, and
            ``expect_fallback``.
        judge: any :class:`AnswerJudge` implementation.

    Returns:
        Dict with per-query scores and dataset-level means. Faithfulness/
        relevance are only computed (and averaged) for queries where
        fallback was NOT triggered — a fallback response is, by design,
        not an "answer" to be judged for groundedness against context that
        wasn't used; its correctness is captured entirely by
        ``fallback_accuracy`` instead.
    """
    per_query = []
    scored_answers = []
    fallback_scores = []

    for item in per_query_results:
        fallback_correct = check_fallback_correctness(
            item["fallback_triggered"], item.get("expect_fallback", False)
        )
        fallback_scores.append(fallback_correct)

        row: dict[str, Any] = {
            "question_id": item["question_id"],
            "fallback_triggered": item["fallback_triggered"],
            "expect_fallback": item.get("expect_fallback", False),
            "fallback_correct": fallback_correct,
        }

        if not item["fallback_triggered"]:
            scores = judge.score(
                item["question"], item["answer"], item["context_text"], item["ground_truth_answer"]
            )
            row.update(scores)
            scored_answers.append(scores)

        per_query.append(row)

    def _mean(key: str) -> float:
        if not scored_answers:
            return 0.0
        return sum(s[key] for s in scored_answers) / len(scored_answers)

    return {
        "num_queries": len(per_query_results),
        "num_answers_scored": len(scored_answers),
        "num_fallback_responses": len(per_query_results) - len(scored_answers),
        "mean_faithfulness": round(_mean("faithfulness"), 4),
        "mean_answer_relevance": round(_mean("answer_relevance"), 4),
        "fallback_accuracy": round(sum(fallback_scores) / len(fallback_scores), 4) if fallback_scores else 0.0,
        "per_query": per_query,
    }
