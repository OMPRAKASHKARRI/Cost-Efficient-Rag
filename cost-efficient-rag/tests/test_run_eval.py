"""Tests for scripts/run_eval.py's extractive_answer() -- the offline
generation stand-in used only because this environment has no LLM access
(see the module docstring in scripts/run_eval.py for the full rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_SPEC)
sys.modules["run_eval"] = run_eval
_SPEC.loader.exec_module(run_eval)

extractive_answer = run_eval.extractive_answer
NO_CONTEXT_FALLBACK_MESSAGE = run_eval.NO_CONTEXT_FALLBACK_MESSAGE


def test_extractive_answer_falls_back_on_empty_retrieval():
    answer, fallback, citations = extractive_answer("Any question?", [])
    assert fallback is True
    assert answer == NO_CONTEXT_FALLBACK_MESSAGE
    assert citations == []


def test_extractive_answer_picks_highest_overlap_sentence():
    chunks = [
        {
            "id": "c1", "source": "pricing.md", "similarity": 0.9,
            "text": "The Free plan is limited. The Pro plan costs 49 dollars per month per seat. Cancellation takes effect at period end.",
        }
    ]
    answer, fallback, citations = extractive_answer("How much does the Pro plan cost?", chunks)
    assert fallback is False
    assert "49 dollars" in answer
    assert "[Doc: pricing.md, Chunk: c1]" in answer
    assert citations[0]["chunk_id"] == "c1"


def test_extractive_answer_falls_back_when_no_sentence_overlaps_question():
    chunks = [
        {
            "id": "c1", "source": "unrelated.md", "similarity": 0.09,
            "text": "Completely unrelated gardening advice about soil pH levels.",
        }
    ]
    answer, fallback, _ = extractive_answer("What is the refund policy?", chunks)
    assert fallback is True
    assert answer == NO_CONTEXT_FALLBACK_MESSAGE


def test_extractive_answer_selects_best_chunk_among_multiple():
    chunks = [
        {"id": "c1", "source": "a.md", "similarity": 0.3, "text": "Some other topic entirely, nothing relevant here."},
        {"id": "c2", "source": "b.md", "similarity": 0.5, "text": "Support hours are Monday through Friday nine to five Eastern."},
    ]
    answer, fallback, _ = extractive_answer("What are the support hours?", chunks)
    assert fallback is False
    assert "[Doc: b.md, Chunk: c2]" in answer
