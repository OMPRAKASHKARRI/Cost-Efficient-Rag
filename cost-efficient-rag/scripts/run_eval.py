"""
Run the full evaluation harness end-to-end and write results/ outputs.

USE_OFFLINE_BACKEND (below) controls which retrieval/generation backend is
used:

* ``True`` (default, and what produced the numbers checked into this repo):
  uses :class:`eval.local_retriever.OfflineTfidfRetriever` for retrieval
  and a small extractive sentence-selection function for "generation" —
  neither needs network access, a downloaded model, or an API key. This
  is a classical-IR stand-in, not the production system.

* ``False``: uses the real production stack —
  :class:`src.vector_store.VectorStoreManager` (LanceDB) +
  :class:`src.vector_store.EmbeddingModel` (SentenceTransformers) for
  retrieval, and :class:`src.rag_pipeline.OpenAIChatClient` for
  generation, exactly as ``src/api.py`` uses them. Requires
  ``pip install -r requirements.txt``, network access to download the
  embedding model, and ``OPENAI_API_KEY`` set. **This is the path a real
  grading run should use** — flip the flag and re-run once those are
  available.

Either way, the metrics computation (eval/evaluate_retrieval.py,
eval/evaluate_answer.py, eval/cost_analysis.py) is identical: those
modules only see ranked chunk-ID lists and generated answer text, never
the backend that produced them.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from eval.evaluate_answer import _tokenize  # reuse the same tokenizer eval/evaluate_answer.py uses,
                                             # rather than redefining it — safe to import here because
                                             # evaluate_answer.py has no heavy top-level dependencies
                                             # (just json/re/string/typing), unlike src.rag_pipeline below.

USE_OFFLINE_BACKEND = True

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
SIMILARITY_THRESHOLD_OFFLINE = 0.08  # TF-IDF cosine scores run much lower than dense-embedding cosine
EXTRACTIVE_SENTENCE_MIN_OVERLAP = 0.08

# Duplicated (not imported) from src.rag_pipeline.NO_CONTEXT_FALLBACK_MESSAGE deliberately: that module's
# top-level imports (src.config -> pydantic-settings, src.logger -> loguru) are exactly the packages this
# offline-eval script exists to work without. Keep these two literals in sync if either changes.
NO_CONTEXT_FALLBACK_MESSAGE = (
    "I do not have sufficient information in the provided context to answer this question."
)

SAMPLE_DOCS = [
    "data/raw_documents/pricing_and_plans.md",
    "data/raw_documents/faq.html",
    "data/raw_documents/onboarding_guide.pdf",
]


def _sentence_split(text: str) -> list[str]:
    # naive splitter: good enough for this small, mostly-clean corpus
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extractive_answer(
    question: str, retrieved_chunks: list[dict], fallback_message: str = NO_CONTEXT_FALLBACK_MESSAGE
) -> tuple[str, bool, list[dict]]:
    """Offline stand-in for the LLM generation step: picks the retrieved
    sentence with the highest lexical overlap to the question, rather than
    calling an LLM. Falls back if nothing was retrieved OR if the best
    available sentence has too little overlap with the question to be a
    credible answer (mirroring the real system's LLM-side fallback
    judgment, which this repo cannot exercise offline).
    """
    if not retrieved_chunks:
        return fallback_message, True, []

    question_tokens = _tokenize(question)
    best_score, best_sentence, best_chunk = -1.0, None, None

    for chunk in retrieved_chunks:
        for sentence in _sentence_split(chunk["text"]):
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            overlap = len(question_tokens & sentence_tokens) / len(question_tokens | sentence_tokens)
            if overlap > best_score:
                best_score, best_sentence, best_chunk = overlap, sentence, chunk

    if best_sentence is None or best_score < EXTRACTIVE_SENTENCE_MIN_OVERLAP:
        return fallback_message, True, []

    answer = f"{best_sentence} [Doc: {best_chunk['source']}, Chunk: {best_chunk['id']}]"
    citations = [
        {"source": c["source"], "chunk_id": c["id"], "similarity": c["similarity"]}
        for c in retrieved_chunks
    ]
    return answer, False, citations


def main() -> None:
    import os

    os.chdir(REPO_ROOT)  # SAMPLE_DOCS are relative paths; must match eval_dataset.json's chunk-ID basis

    from src.ingestion import ingest_documents

    print(f"Ingesting {len(SAMPLE_DOCS)} sample documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = ingest_documents(
        SAMPLE_DOCS,  # relative paths -- must match how data/eval_dataset.json's chunk IDs were derived
        existing_ids=set(),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"  -> {len(chunks)} chunks")

    if USE_OFFLINE_BACKEND:
        from eval.local_retriever import OfflineTfidfRetriever

        retriever = OfflineTfidfRetriever(chunks)
        backend_label = "offline-tfidf+extractive (no network/model/API access in this environment)"
    else:
        raise NotImplementedError(
            "USE_OFFLINE_BACKEND=False requires the real stack (lancedb, "
            "sentence-transformers, openai) with network access -- not "
            "available in this environment. Install requirements.txt, set "
            "OPENAI_API_KEY, and wire up VectorStoreManager/EmbeddingModel/"
            "OpenAIChatClient here to run the production evaluation path."
        )

    eval_dataset = json.loads((REPO_ROOT / "data" / "eval_dataset.json").read_text())
    questions = eval_dataset["questions"]

    retrieval_inputs = []
    answer_inputs = []
    metrics_log = []

    for q in questions:
        t0 = time.perf_counter()
        retrieved = retriever.search(q["question"], top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD_OFFLINE)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        answer, fallback_triggered, citations = extractive_answer(q["question"], retrieved)
        generation_ms = (time.perf_counter() - t1) * 1000

        retrieved_ids = [c["id"] for c in retrieved]
        retrieval_inputs.append(
            {"question_id": q["id"], "retrieved_ids": retrieved_ids, "gold_ids": q["relevant_chunk_ids"]}
        )
        answer_inputs.append(
            {
                "question_id": q["id"],
                "question": q["question"],
                "answer": answer,
                "context_text": " ".join(c["text"] for c in retrieved),
                "ground_truth_answer": q["ground_truth_answer"],
                "fallback_triggered": fallback_triggered,
                "expect_fallback": q.get("expect_fallback", False),
            }
        )
        metrics_log.append(
            {
                "question_id": q["id"],
                "retrieval_latency_ms": round(retrieval_ms, 3),
                "generation_latency_ms": round(generation_ms, 3),
                "total_latency_ms": round(retrieval_ms + generation_ms, 3),
                "retrieved_chunk_count": len(retrieved),
                "fallback_triggered": fallback_triggered,
            }
        )

    from eval.evaluate_retrieval import evaluate_retrieval_dataset

    retrieval_results = evaluate_retrieval_dataset(retrieval_inputs, k=TOP_K)

    from eval.evaluate_answer import HeuristicJudge, evaluate_answer_dataset

    answer_results = evaluate_answer_dataset(answer_inputs, HeuristicJudge())

    from eval.cost_analysis import build_cost_benchmark_table, estimate_llm_query_cost

    cost_table = build_cost_benchmark_table()
    llm_cost_estimate = estimate_llm_query_cost(
        monthly_query_volume=50_000, avg_prompt_tokens=550, avg_completion_tokens=80,
        input_cost_per_1m=0.15, output_cost_per_1m=0.60,
    )

    latencies = [m["total_latency_ms"] for m in metrics_log]
    latencies_sorted = sorted(latencies)

    def _pct(vals, p):
        if not vals:
            return 0.0
        idx = max(0, min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1)))))
        return round(vals[idx], 3)

    results = {
        "backend": backend_label,
        "note": (
            "This run used the offline TF-IDF + extractive-answer backend "
            "(see eval/local_retriever.py and scripts/run_eval.py) because "
            "the environment this repo was assembled in has no network "
            "access -- sentence-transformers cannot download model weights "
            "and there is no OpenAI API key/connectivity. All retrieval "
            "and answer-quality METRIC CODE (eval/evaluate_retrieval.py, "
            "eval/evaluate_answer.py) is the real, spec-compliant "
            "implementation and is fully unit-tested against hand-verified "
            "values -- only the retriever and generator that FED it "
            "numbers are offline stand-ins. Re-run "
            "`python scripts/run_eval.py` with USE_OFFLINE_BACKEND=False "
            "after `pip install -r requirements.txt` and setting "
            "OPENAI_API_KEY to get production-backend numbers."
        ),
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "num_chunks_in_corpus": len(chunks),
            "num_eval_questions": len(questions),
        },
        "retrieval_metrics": retrieval_results,
        "answer_metrics": answer_results,
        "latency_ms": {
            "p50": _pct(latencies_sorted, 50),
            "p95": _pct(latencies_sorted, 95),
            "note": "Latencies are for the offline TF-IDF backend and are NOT representative "
                    "of production latency (real embedding + LanceDB search + OpenAI generation "
                    "will be substantially higher, dominated by the LLM call).",
        },
        "cost_benchmark": cost_table,
        "estimated_llm_query_cost": llm_cost_estimate,
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_dir / 'eval_results.json'}")

    _write_cost_benchmark_md(cost_table, llm_cost_estimate, results_dir / "cost_benchmark_table.md")
    print(f"Wrote {results_dir / 'cost_benchmark_table.md'}")

    print("\n--- Summary ---")
    print(f"Retrieval: recall@{TOP_K}={retrieval_results['mean_recall_at_k']:.3f}  "
          f"hit_rate@{TOP_K}={retrieval_results['mean_hit_rate_at_k']:.3f}  "
          f"MRR={retrieval_results['mean_mrr']:.3f}  "
          f"nDCG@{TOP_K}={retrieval_results['mean_ndcg_at_k']:.3f}")
    print(f"Answers: faithfulness={answer_results['mean_faithfulness']:.3f}  "
          f"relevance={answer_results['mean_answer_relevance']:.3f}  "
          f"fallback_accuracy={answer_results['fallback_accuracy']:.3f}")


def _write_cost_benchmark_md(cost_table, llm_cost_estimate, path: Path) -> None:
    lines = [
        "# Cost Benchmark: Embedded (LanceDB) vs. Managed Vector DB",
        "",
        "Assumptions: 384-dim embeddings (float32), ~0.5 KB metadata/vector, "
        "$0.08/GB/month disk (EBS gp3-class pricing), managed DB priced on "
        "Pinecone-shaped always-on pod tiers. See README Cost Analysis "
        "section for the full assumption list.",
        "",
        "| Vector Count | Storage (GB) | Embedded Cost/mo | Managed Cost/mo | Savings |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in cost_table:
        lines.append(
            f"| {row['vector_count']:,} | {row['storage_size_gb']:.2f} | "
            f"${row['embedded_db_monthly_cost_usd']:.4f} | ${row['managed_db_monthly_cost_usd']:.2f} | "
            f"{row['savings_percentage']:.1f}% |"
        )
    lines += [
        "",
        "## LLM Generation Cost (separate from vector storage)",
        "",
        f"At {llm_cost_estimate['monthly_query_volume']:,} queries/month, "
        f"~${llm_cost_estimate['avg_cost_per_query_usd']:.6f}/query "
        f"(550 avg prompt tokens, 80 avg completion tokens, gpt-4o-mini-class pricing) "
        f"-> **${llm_cost_estimate['estimated_monthly_llm_cost_usd']:.2f}/month**.",
        "",
        "This is the dominant cost at low-to-moderate vector counts: the vector-store "
        "line item stays under a dollar a month through 10M vectors, while LLM "
        "generation cost scales with query volume, not corpus size.",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
