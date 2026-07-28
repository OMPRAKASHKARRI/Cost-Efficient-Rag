"""
Vector store: embedding generation + LanceDB table management.

Two responsibilities live here, kept as separate classes so each is
independently testable and swappable:

* :class:`EmbeddingModel` — wraps ``sentence-transformers`` and records the
  model name + output dimensionality every embedding batch was produced
  with (the reference guide requires recording both in metadata).
* :class:`VectorStoreManager` — owns the LanceDB connection/table, builds
  the on-disk schema, performs idempotent upserts, and runs top-k
  similarity search with optional metadata filtering and a similarity
  floor.

Both ``lancedb``/``pyarrow`` and ``sentence-transformers`` are imported
lazily, inside methods rather than at module load time. This is not just a
sandbox workaround: it means importing this module (e.g. for a health
check, or for unit-testing the pure SQL-filter-building logic below) never
pays the cost of loading a transformer model or a Rust extension, and it
means a machine without those packages installed can still run the parts
of the test suite that don't need them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ingestion import ChunkRecord
from src.logger import get_logger

logger = get_logger()


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when an embedding's dimension doesn't match the table's schema."""


class EmbeddingModel:
    """Lazy-loaded wrapper around a SentenceTransformer embedding model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # loaded on first use, not at construction

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Output vector dimensionality of this model (e.g. 384 for MiniLM-L6-v2)."""
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]):
        """Embed a batch of texts. Returns a numpy array of shape (len(texts), dim)."""
        if not texts:
            import numpy as np

            return np.empty((0, self.dimension), dtype="float32")
        return self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
        )

    def encode_one(self, text: str):
        """Embed a single text. Returns a 1-D numpy array of length ``dimension``."""
        return self.encode([text])[0]


def _escape_sql_literal(value: str) -> str:
    """Escape single quotes for safe inclusion in a LanceDB SQL ``where`` clause."""
    return value.replace("'", "''")


def build_where_clause(metadata_filter: dict[str, Any]) -> str:
    """Translate a metadata filter dict into a LanceDB/DataFusion SQL ``where`` clause.

    Supports string, bool, int, and float values. ``None`` values are
    skipped (treated as "no constraint on this field") rather than
    generating ``field = NULL``, which would never match under standard
    SQL null semantics.

    Args:
        metadata_filter: e.g. ``{"file_type": "pdf", "category": "policy"}``.

    Returns:
        A SQL boolean expression, e.g. ``"file_type = 'pdf' AND category = 'policy'"``.

    Raises:
        ValueError: if every value was ``None`` (nothing to filter on).
        TypeError: if a value is a type we don't know how to serialize safely.
    """
    clauses: list[str] = []
    for key, value in metadata_filter.items():
        if value is None:
            continue
        if isinstance(value, bool):
            clauses.append(f"{key} = {str(value).upper()}")
        elif isinstance(value, str):
            clauses.append(f"{key} = '{_escape_sql_literal(value)}'")
        elif isinstance(value, (int, float)):
            clauses.append(f"{key} = {value}")
        else:
            raise TypeError(
                f"Unsupported metadata_filter value type for key '{key}': {type(value)}"
            )
    if not clauses:
        raise ValueError("metadata_filter contained no usable (non-None) values")
    return " AND ".join(clauses)


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert LanceDB's cosine *distance* to a cosine *similarity* in [0, 2].

    LanceDB's cosine metric returns ``distance = 1 - cosine_similarity``,
    so ``similarity = 1 - distance``. Kept as a standalone function (rather
    than inlined) so the conversion has one definition and one test.
    """
    return 1.0 - distance


_METADATA_COLUMNS = ("category", "date_created")  # promoted to real columns for fast filtering


class VectorStoreManager:
    """Owns a LanceDB table: schema, idempotent upsert, and filtered top-k search."""

    def __init__(self, db_path: str | Path, table_name: str, embedding_dim: int) -> None:
        self.db_path = str(db_path)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._db = None

    @property
    def db(self):
        if self._db is None:
            import lancedb

            self._db = lancedb.connect(self.db_path)
        return self._db

    def _table_exists(self) -> bool:
        return self.table_name in self.db.table_names()

    def _open_table(self):
        if not self._table_exists():
            return None
        return self.db.open_table(self.table_name)

    def _build_schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("file_type", pa.string()),
                pa.field("char_count", pa.int32()),
                pa.field("category", pa.string()),
                pa.field("date_created", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("embedding_model", pa.string()),
                pa.field("embedding_dim", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            ]
        )

    def _rows_from_chunks(
        self, chunks: list[ChunkRecord], embeddings, embedding_model_name: str
    ) -> list[dict[str, Any]]:
        rows = []
        for chunk, vector in zip(chunks, embeddings):
            vec_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            if len(vec_list) != self.embedding_dim:
                raise EmbeddingDimensionMismatchError(
                    f"Embedding for chunk {chunk.id} has dimension {len(vec_list)}, "
                    f"table expects {self.embedding_dim}"
                )
            rows.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "file_type": chunk.file_type,
                    "char_count": chunk.char_count,
                    "category": chunk.metadata.get("category"),
                    "date_created": chunk.metadata.get("date_created"),
                    "metadata_json": json.dumps(chunk.metadata),
                    "embedding_model": embedding_model_name,
                    "embedding_dim": self.embedding_dim,
                    "vector": vec_list,
                }
            )
        return rows

    def get_existing_ids(self) -> set[str]:
        """Return every chunk ID currently stored — used upstream for idempotent re-ingestion.

        Note on scale: this does a full-column scan. That's the right
        trade-off at eval-harness / take-home scale (thousands of chunks),
        but at 10M+ vectors this is exactly the kind of operation that
        argues for a sidecar ID index (a small SQLite table, or a bloom
        filter checkpointed alongside the LanceDB directory) rather than
        scanning the vector table itself — see the cost-analysis writeup.
        """
        table = self._open_table()
        if table is None:
            return set()
        df = table.to_pandas()
        if "id" not in df.columns or df.empty:
            return set()
        return set(df["id"].tolist())

    def count(self) -> int:
        table = self._open_table()
        return 0 if table is None else table.count_rows()

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        embeddings,
        embedding_model_name: str,
    ) -> int:
        """Insert chunk rows, creating the table on first use.

        Chunks are expected to already be deduplicated by the caller
        (``src/ingestion.py`` filters against :meth:`get_existing_ids`
        before this is called) — this method additionally uses LanceDB's
        ``merge_insert`` upsert-on-id where available as defense in depth,
        so calling it twice with overlapping IDs is still safe rather than
        producing duplicate rows.

        Returns:
            Number of rows written (post-dedup within this call).
        """
        if not chunks:
            return 0

        rows = self._rows_from_chunks(chunks, embeddings, embedding_model_name)

        import pyarrow as pa

        table_data = pa.Table.from_pylist(rows, schema=self._build_schema())

        if not self._table_exists():
            self.db.create_table(self.table_name, data=table_data)
            logger.info(f"Created table '{self.table_name}' with {len(rows)} row(s)")
            return len(rows)

        table = self.db.open_table(self.table_name)
        try:
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(table_data)
            )
        except AttributeError:
            # Older lancedb without merge_insert: fall back to plain add.
            # Safe here because chunks are already pre-deduplicated by the caller.
            table.add(table_data)
        logger.info(f"Upserted {len(rows)} row(s) into '{self.table_name}'")
        return len(rows)

    def search(
        self,
        query_vector,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k cosine similarity search with optional metadata filter + score floor.

        Args:
            query_vector: Embedding of the query (same dim as the table).
            top_k: Max number of results to return.
            metadata_filter: Optional equality filters, e.g. ``{"file_type": "pdf"}``.
            similarity_threshold: Drop results below this cosine similarity.
                ``None`` disables the floor (returns whatever LanceDB ranks top-k).

        Returns:
            List of result dicts (id, text, source, chunk_index, file_type,
            category, date_created, metadata, embedding_model, similarity),
            ordered by descending similarity. Empty list if the table
            doesn't exist yet or nothing clears the threshold.
        """
        table = self._open_table()
        if table is None:
            logger.warning(f"Table '{self.table_name}' does not exist yet — returning no results")
            return []

        query = table.search(query_vector).metric("cosine").limit(top_k)
        if metadata_filter:
            where = build_where_clause(metadata_filter)
            query = query.where(where)

        df = query.to_pandas()
        results = []
        for _, row in df.iterrows():
            distance = float(row["_distance"]) if "_distance" in row else None
            similarity = cosine_distance_to_similarity(distance) if distance is not None else None
            if (
                similarity_threshold is not None
                and similarity is not None
                and similarity < similarity_threshold
            ):
                continue
            metadata = {}
            if row.get("metadata_json"):
                try:
                    metadata = json.loads(row["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            results.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "source": row["source"],
                    "chunk_index": int(row["chunk_index"]),
                    "file_type": row["file_type"],
                    "category": row.get("category"),
                    "date_created": row.get("date_created"),
                    "metadata": metadata,
                    "embedding_model": row.get("embedding_model"),
                    "similarity": similarity,
                }
            )
        return results
