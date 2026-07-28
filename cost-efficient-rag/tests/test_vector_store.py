"""Tests for src/vector_store.py.

Pure helpers (build_where_clause, cosine_distance_to_similarity,
_rows_from_chunks) are tested directly. upsert_chunks()/search() are
tested against lightweight in-memory fakes for lancedb/pyarrow (patched
into sys.modules) so their control flow — create-vs-merge-insert branching,
threshold filtering, metadata JSON round-trip — is exercised even in
environments where the real packages aren't installed. Anywhere the real
lancedb/pyarrow ARE installed, these fakes are simply irrelevant — the
public API surface used (search/where/limit/metric/to_pandas,
Table.from_pylist) matches the real libraries.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from src.ingestion import ChunkRecord
from src.vector_store import (
    EmbeddingDimensionMismatchError,
    VectorStoreManager,
    build_where_clause,
    cosine_distance_to_similarity,
)


# ── Pure helpers ─────────────────────────────────────────────────────────


def test_where_clause_single_field():
    assert build_where_clause({"file_type": "pdf"}) == "file_type = 'pdf'"


def test_where_clause_multiple_fields_joined_with_and():
    clause = build_where_clause({"file_type": "pdf", "category": "policy"})
    assert clause == "file_type = 'pdf' AND category = 'policy'"


def test_where_clause_skips_none_values():
    clause = build_where_clause({"file_type": "html", "category": None})
    assert clause == "file_type = 'html'"


def test_where_clause_escapes_quotes():
    clause = build_where_clause({"category": "o'brien"})
    assert clause == "category = 'o''brien'"


def test_where_clause_numeric_and_bool_values():
    assert build_where_clause({"chunk_index": 3}) == "chunk_index = 3"
    assert build_where_clause({"is_active": True}) == "is_active = TRUE"


def test_where_clause_raises_on_all_none():
    with pytest.raises(ValueError):
        build_where_clause({"category": None})


def test_where_clause_raises_on_unsupported_type():
    with pytest.raises(TypeError):
        build_where_clause({"tags": ["a", "b"]})


def test_cosine_distance_to_similarity_boundaries():
    assert cosine_distance_to_similarity(0.0) == 1.0
    assert cosine_distance_to_similarity(1.0) == 0.0
    assert cosine_distance_to_similarity(2.0) == -1.0


# ── _rows_from_chunks (no pyarrow/lancedb needed) ───────────────────────


def _make_chunk(id_="abc123", metadata=None):
    return ChunkRecord(
        id=id_, text="hello", source="doc.md", chunk_index=0,
        file_type="md", char_count=5, metadata=metadata or {},
    )


def test_rows_from_chunks_maps_fields_correctly():
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    chunk = _make_chunk(metadata={"category": "policy", "date_created": "2026-01-01"})
    rows = vs._rows_from_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "all-MiniLM-L6-v2")
    row = rows[0]
    assert row["id"] == "abc123"
    assert row["category"] == "policy"
    assert row["date_created"] == "2026-01-01"
    assert row["embedding_model"] == "all-MiniLM-L6-v2"
    assert row["embedding_dim"] == 4
    assert row["vector"] == [0.1, 0.2, 0.3, 0.4]


def test_rows_from_chunks_missing_metadata_defaults_to_none():
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    chunk = _make_chunk(metadata={})
    rows = vs._rows_from_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "model-x")
    assert rows[0]["category"] is None
    assert rows[0]["date_created"] is None


def test_rows_from_chunks_dimension_mismatch_raises():
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    chunk = _make_chunk()
    with pytest.raises(EmbeddingDimensionMismatchError):
        vs._rows_from_chunks([chunk], [[0.1, 0.2]], "model-x")


# ── Fakes for lancedb / pyarrow ──────────────────────────────────────────


class _FakeQuery:
    """Fakes LanceDB's fluent table.search(vec).metric(...).limit(...).where(...) chain."""

    def __init__(self, rows: list[dict], query_vector):
        self._rows = rows
        self._query_vector = query_vector
        self._limit = len(rows)
        self._where = None

    def metric(self, name):
        return self

    def limit(self, k):
        self._limit = k
        return self

    def where(self, clause):
        self._where = clause
        return self

    def to_pandas(self):
        rows = self._rows
        if self._where:
            # Extremely small subset-of-SQL evaluator sufficient for these tests:
            # only handles "col = 'val'" / "col = num" clauses joined by AND.
            for clause in self._where.split(" AND "):
                col, _, val = clause.partition(" = ")
                val = val.strip("'")
                rows = [r for r in rows if str(r.get(col)) == val]
        # naive "distance" = negative dot product rank order (fixed order in these tests)
        rows = rows[: self._limit]
        return pd.DataFrame(rows)


class _FakeMergeInsertBuilder:
    def __init__(self, table, key):
        self._table = table
        self._key = key

    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    def execute(self, new_rows):
        existing_ids = {r["id"] for r in self._table._rows}
        for row in new_rows:
            if row["id"] in existing_ids:
                self._table._rows = [r if r["id"] != row["id"] else row for r in self._table._rows]
            else:
                self._table._rows.append(row)


class _FakeTable:
    def __init__(self, rows: list[dict]):
        self._rows = list(rows)

    def to_pandas(self):
        return pd.DataFrame(self._rows)

    def count_rows(self):
        return len(self._rows)

    def search(self, query_vector):
        # attach a synthetic _distance column, closest-first, deterministic by insertion order
        rows_with_distance = [
            {**r, "_distance": i * 0.1} for i, r in enumerate(self._rows)
        ]
        return _FakeQuery(rows_with_distance, query_vector)

    def merge_insert(self, key):
        return _FakeMergeInsertBuilder(self, key)

    def add(self, new_rows):
        self._rows.extend(new_rows)


class _FakeDB:
    def __init__(self):
        self.tables: dict[str, _FakeTable] = {}

    def table_names(self):
        return list(self.tables.keys())

    def open_table(self, name):
        return self.tables[name]

    def create_table(self, name, data):
        self.tables[name] = _FakeTable(list(data))


@pytest.fixture
def fake_lancedb_and_pyarrow(monkeypatch):
    """Patch sys.modules so `import lancedb` / `import pyarrow as pa` inside
    vector_store.py resolve to lightweight fakes, and hand back the fake DB.
    """
    fake_db = _FakeDB()

    fake_lancedb = types.ModuleType("lancedb")
    fake_lancedb.connect = lambda path: fake_db

    fake_pa = types.ModuleType("pyarrow")

    class _FakeArrowTable(list):
        @staticmethod
        def from_pylist(rows, schema=None):
            return _FakeArrowTable(rows)

    fake_pa.Table = _FakeArrowTable
    fake_pa.schema = lambda fields: fields
    fake_pa.field = lambda name, dtype: (name, dtype)
    fake_pa.string = lambda: "string"
    fake_pa.int32 = lambda: "int32"
    fake_pa.list_ = lambda inner, size: f"list<{inner}, {size}>"
    fake_pa.float32 = lambda: "float32"

    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)
    return fake_db


# ── upsert_chunks ─────────────────────────────────────────────────────────


def test_upsert_creates_table_on_first_call(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    chunk = _make_chunk(id_="c1")
    written = vs.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "model-x")
    assert written == 1
    assert vs.count() == 1


def test_upsert_is_idempotent_via_merge_insert(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    chunk = _make_chunk(id_="c1", metadata={"category": "a"})
    vs.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]], "model-x")
    # Re-upsert the SAME id: row count must not grow.
    vs.upsert_chunks([chunk], [[0.9, 0.9, 0.9, 0.9]], "model-x")
    assert vs.count() == 1


def test_upsert_empty_list_is_noop(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    assert vs.upsert_chunks([], [], "model-x") == 0
    assert vs.count() == 0


def test_get_existing_ids_empty_when_no_table(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    assert vs.get_existing_ids() == set()


def test_get_existing_ids_after_upsert(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    vs.upsert_chunks([_make_chunk(id_="c1"), _make_chunk(id_="c2")], [[0.1] * 4, [0.2] * 4], "model-x")
    assert vs.get_existing_ids() == {"c1", "c2"}


# ── search ──────────────────────────────────────────────────────────────


def test_search_returns_empty_when_table_missing(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    assert vs.search([0.1, 0.2, 0.3, 0.4], top_k=5) == []


def test_search_applies_metadata_filter(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    vs.upsert_chunks(
        [_make_chunk(id_="c1", metadata={"category": "policy"}),
         _make_chunk(id_="c2", metadata={"category": "faq"})],
        [[0.1] * 4, [0.2] * 4],
        "model-x",
    )
    results = vs.search([0.1] * 4, top_k=5, metadata_filter={"category": "faq"})
    assert len(results) == 1
    assert results[0]["id"] == "c2"
    assert results[0]["metadata"] == {"category": "faq"}


def test_search_applies_similarity_threshold(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    # 3 chunks; fake distances will be 0.0, 0.1, 0.2 -> similarities 1.0, 0.9, 0.8
    vs.upsert_chunks(
        [_make_chunk(id_=f"c{i}") for i in range(3)],
        [[0.1] * 4] * 3,
        "model-x",
    )
    results = vs.search([0.1] * 4, top_k=3, similarity_threshold=0.85)
    # only similarity 1.0 and 0.9 clear the 0.85 floor
    assert len(results) == 2
    assert all(r["similarity"] >= 0.85 for r in results)


def test_search_results_include_embedding_model_and_similarity(fake_lancedb_and_pyarrow):
    vs = VectorStoreManager(db_path="/tmp/vs_test", table_name="rag_chunks", embedding_dim=4)
    vs.upsert_chunks([_make_chunk(id_="c1")], [[0.1] * 4], "all-MiniLM-L6-v2")
    results = vs.search([0.1] * 4, top_k=1)
    assert results[0]["embedding_model"] == "all-MiniLM-L6-v2"
    assert results[0]["similarity"] == 1.0  # first result, fake distance 0.0
