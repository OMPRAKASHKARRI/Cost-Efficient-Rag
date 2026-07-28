"""
Offline TF-IDF retriever.

**This is explicitly NOT the production retriever.** ``src/vector_store.py``
(SentenceTransformers embeddings + LanceDB, per the assignment spec) is the
real implementation and is what ``src/api.py`` uses. This module exists
for exactly one reason: producing this repository's own
``results/eval_results.json`` required *running* the evaluation harness,
and the environment it was authored in has no network access — so
``sentence-transformers`` (needs to download model weights) and ``lancedb``
(installable, but pointless to spin up without real embeddings behind it)
could not be exercised end-to-end there.

TF-IDF + cosine similarity, computed entirely locally with scikit-learn, is
a reasonable *classical* retrieval baseline that needs no model download
and no network call, letting the harness — chunking, hashing, retrieval
metrics, answer metrics, fallback logic, cost math — run for real and
produce real (if not embedding-quality) numbers instead of a fabricated
JSON file. Swapping this for :class:`src.vector_store.VectorStoreManager`
is a one-line change in ``scripts/run_eval.py`` (see the ``USE_OFFLINE_BACKEND``
flag there) once real network/API access is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion import ChunkRecord


@dataclass
class _IndexedChunk:
    chunk: ChunkRecord


class OfflineTfidfRetriever:
    """In-memory TF-IDF index over a fixed set of chunks, offering the same
    ``search(query, top_k, similarity_threshold) -> list[dict]`` shape as
    :meth:`src.vector_store.VectorStoreManager.search`, so downstream eval
    code doesn't need to know which backend produced the results.
    """

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english")
        texts = [c.text for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts) if texts else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._chunks or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]

        candidates = list(zip(self._chunks, similarities))
        if metadata_filter:
            candidates = [
                (c, s) for c, s in candidates
                if all(getattr(c, k, c.metadata.get(k)) == v for k, v in metadata_filter.items())
            ]

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        results = []
        for chunk, similarity in candidates[:top_k]:
            similarity = float(similarity)
            if similarity_threshold is not None and similarity < similarity_threshold:
                continue
            results.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "file_type": chunk.file_type,
                    "metadata": chunk.metadata,
                    "similarity": similarity,
                }
            )
        return results
