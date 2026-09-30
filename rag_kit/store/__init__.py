"""Storage sub-package — LanceDB vector store operations.

Provides ``VectorStore`` for persisting text chunks + embeddings in a local
LanceDB database with full-text-search and vector similarity search.

LanceDB is chosen over ChromaDB because:
  - Pure-local persistence (no server process)
  - Arrow/Parquet columnar storage (compact, fast)
  - Built-in vector search with L2 distance
  - SQL-like filters for metadata queries
"""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
from typing import Any

import numpy as np
import pyarrow as pa

# --------------------------------------------------------------------------- #
# Schema definition
# --------------------------------------------------------------------------- #

# Default embedding dimension for paraphrase-multilingual-MiniLM-L12-v2
# (used only when no vectors have been written yet; the real dim is taken
# from the first ``add_chunks`` call so any embedding model works).
_DEFAULT_DIM = 384

_TABLE_NAME = "documents"


def _make_schema(dim: int = _DEFAULT_DIM) -> pa.Schema:
    """Return the PyArrow schema for the documents table."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_idx", pa.int32()),
            pa.field("vlm_generated", pa.bool_()),
            pa.field("source_type", pa.string()),
        ]
    )


# --------------------------------------------------------------------------- #
# VectorStore
# --------------------------------------------------------------------------- #


class VectorStore:
    """Local vector database backed by LanceDB.

    The store persists chunks (text + metadata) and their embeddings to a
    local directory.  All data survives across restarts.

    Args:
        db_path:     Directory for the LanceDB database.
        table_name:  Name of the table to use (default: "documents").
        dim:         Embedding dimension (default: 384).
    """

    def __init__(
        self,
        db_path: str = "",
        table_name: str = _TABLE_NAME,
        dim: int = _DEFAULT_DIM,
    ) -> None:
        self.db_path = str(db_path) if db_path else str(Path.home() / "lancedb")
        self.table_name = table_name
        self.dim = dim
        self._db: Any = None
        self._table: Any = None

    # -- internal ----------------------------------------------------------- #

    def _connect(self) -> Any:
        """Connect to (or create) the LanceDB database."""
        if self._db is not None:
            return self._db

        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        import lancedb

        # Watchers and CLI processes share this database. Refresh cached
        # table versions before reads so long-lived readers see new commits.
        self._db = lancedb.connect(self.db_path, read_consistency_interval=timedelta(0))
        return self._db

    def _ensure_table(self) -> Any:
        """Open or create the documents table.

        Missing metadata columns are added with defaults, preserving rows.
        """
        db = self._connect()

        if self.table_name in db.table_names():
            existing = db.open_table(self.table_name)
            existing_cols = set(existing.schema.names)
            defaults = {"vlm_generated": "false", "source_type": "'text'"}
            missing = {k: v for k, v in defaults.items() if k not in existing_cols}
            if missing:
                existing.add_columns(missing)
            self._table = existing
            self.dim = existing.schema.field("vector").type.list_size
            return self._table

        schema = _make_schema(self.dim)
        self._table = db.create_table(
            self.table_name,
            schema=schema,
            mode="create",
        )
        return self._table

    def _get_table(self) -> Any:
        """Return the table, raising a clear error if it doesn't exist."""
        db = self._connect()
        if self.table_name not in db.table_names():
            raise RuntimeError(
                f"Table '{self.table_name}' does not exist in {self.db_path}. "
                f"Add chunks first via add_chunks()."
            )
        if self._table is None:
            try:
                self._table = db.open_table(self.table_name)
            except ValueError as exc:
                # A concurrent first ingest can create the table directory
                # before its initial manifest is committed.
                if "was not found" in str(exc):
                    raise RuntimeError(f"Table '{self.table_name}' is not ready yet") from exc
                raise
        return self._table

    # -- public API: write -------------------------------------------------- #

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: np.ndarray,
    ) -> int:
        """Store chunks with their embeddings.

        Args:
            chunks:  List of chunk dicts (from ``Chunk.to_dict()``).
                     Each must have: id, text, source, page, chunk_idx.
            vectors: ``np.ndarray`` of shape ``(len(chunks), dim)``.

        Returns:
            Number of rows added.
        """
        if not chunks:
            return 0

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Expected vectors shape ({len(chunks)}, dim), "
                f"got {vectors.shape}"
            )

        # Update dimension if this is the first insert and dim differs.
        if vectors.shape[1] != self.dim:
            self.dim = vectors.shape[1]

        table = self._ensure_table()

        rows: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "id": str(chunk.get("id", chunk.get("chunk_id", ""))),
                    "text": chunk["text"],
                    "vector": vectors[i].tolist(),
                    "source": chunk["source"],
                    "page": int(chunk.get("page", 0)),
                    "chunk_idx": int(chunk.get("chunk_idx", 0)),
                    "vlm_generated": bool(chunk.get("vlm_generated", False)),
                    "source_type": str(chunk.get("source_type", "text")),
                }
            )

        table.add(rows)
        self._fts_ready = False
        return len(rows)

    def replace_source(self, source: str, chunks: list[dict[str, Any]],
                       vectors: np.ndarray) -> int:
        """Atomically replace one source after extraction and embedding succeed.

        Merge-insert commits insertion and deletion together. An empty valid
        document is a single source-scoped delete. Failed validation never
        removes the previous document.
        """
        if any(c.get("source") != source for c in chunks):
            raise ValueError("All replacement chunks must belong to the source")
        vectors = np.asarray(vectors, dtype=np.float32)
        if chunks and (vectors.ndim != 2 or vectors.shape[0] != len(chunks)):
            raise ValueError("Replacement vectors must match the chunk count")
        if not chunks:
            if self.count_rows():
                self.delete_by_source(source)
            return 0
        self.dim = vectors.shape[1]
        table = self._ensure_table()
        if table.schema.field("vector").type.list_size != vectors.shape[1]:
            raise ValueError("Embedding dimension differs from the existing index; use a new database")
        rows = [{
            "id": str(c.get("id", c.get("chunk_id", ""))),
            "text": c["text"], "vector": vectors[i].tolist(), "source": source,
            "page": int(c.get("page", 0)), "chunk_idx": int(c.get("chunk_idx", i)),
            "vlm_generated": bool(c.get("vlm_generated", False)),
            "source_type": str(c.get("source_type", "text")),
        } for i, c in enumerate(chunks)]
        safe_source = source.replace("'", "''")
        (table.merge_insert("id").when_matched_update_all()
         .when_not_matched_insert_all()
         .when_not_matched_by_source_delete(f"source = '{safe_source}'")
         .execute(pa.Table.from_pylist(rows, schema=table.schema)))
        self._fts_ready = False
        return len(rows)

    # -- public API: search ------------------------------------------------- #

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_sql: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            query_vector:  1-D ``np.ndarray`` of shape ``(dim,)``.
            top_k:         Number of results to return.
            filter_sql:    Optional SQL filter (e.g. ``source = 'doc.pdf'``).

        Returns:
            List of result dicts with keys:
            ``id, text, source, page, chunk_idx, score``.
            ``score`` is a similarity score in [0, 1] (1 = identical).
        """
        table = self._get_table()

        query_vector = np.asarray(query_vector, dtype=np.float32).tolist()

        query = table.search(query_vector).limit(top_k)
        if filter_sql:
            query = query.where(filter_sql)

        raw = query.to_list()

        # Convert _distance (L2 squared) to similarity score [0, 1].
        # For normalised vectors, L2_sq = 2 - 2*cos_sim, so
        # cos_sim = 1 - L2_sq / 2.  Clamp to [0, 1].
        results: list[dict[str, Any]] = []
        for row in raw:
            distance = row.pop("_distance", 0.0)
            score = max(0.0, 1.0 - distance / 2.0)
            row["score"] = round(score, 4)
            results.append(row)

        return results

    def search_by_source(
        self,
        query_vector: np.ndarray,
        source: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search within chunks from a specific source file."""
        # Escape single quotes in the source path for SQL.
        safe_source = source.replace("'", "''")
        return self.search(
            query_vector,
            top_k=top_k,
            filter_sql=f"source = '{safe_source}'",
        )

    _fts_ready = False
    _fts_rows = -1

    def _ensure_fts(self) -> None:
        """Create (or refresh) the full-text-search index on the text column.

        The index is rebuilt whenever new rows are added, otherwise newly
        ingested documents would be missing from keyword search.
        """
        count = self.count_rows()
        if count == 0:
            self._fts_ready = False
            return
        if self._fts_ready and self._fts_rows == count:
            return
        table = self._get_table()
        try:
            try:
                from lancedb.index import FTS
                table.create_index("text", config=FTS(), replace=True)
            except Exception:
                table.create_fts_index("text", replace=True)
            self._fts_ready = True
            self._fts_rows = count
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Could not create FTS index: %s", exc)

    def search_hybrid(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_sql: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + keyword search with Reciprocal Rank Fusion.

        Combines dense vector search (semantic) with full-text keyword
        search so exact terms that dense embeddings miss are still found.
        ``alpha`` is the weight on the semantic side (1.0 = vector-only,
        0.0 = keyword-only); default 0.5 blends both.

        Returns the same result shape as :meth:`search`.
        """
        alpha = min(max(alpha, 0.0), 1.0)

        # vector-only shortcut
        if alpha >= 1.0:
            return self.search(query_vector, top_k=top_k, filter_sql=filter_sql)
        self._ensure_fts()
        table = self._get_table()
        candidate_count = max(top_k * 4, 20)
        try:
            q = table.search(query_text, query_type="fts").limit(candidate_count)
            if filter_sql:
                q = q.where(filter_sql)
            keyword = q.to_list()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Keyword search failed (%s).", exc
            )
            if alpha <= 0.0:
                raise RuntimeError(f"Keyword search unavailable: {exc}") from exc
            return self.search(query_vector, top_k=top_k, filter_sql=filter_sql)
        if alpha <= 0.0:
            for row in keyword:
                row["score"] = round(float(row.get("_score", row.get("score", 0))), 4)
            return keyword[:top_k]
        semantic = self.search(query_vector, top_k=candidate_count, filter_sql=filter_sql)
        fused: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for weight, rows in ((alpha, semantic), (1 - alpha, keyword)):
            for rank, row in enumerate(rows, 1):
                key = row["id"]
                fused.setdefault(key, dict(row))
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
        ordered = sorted(scores, key=lambda k: (-scores[k], k))[:top_k]
        return [{**fused[k], "score": round(scores[k], 8)} for k in ordered]

    # -- public API: delete ------------------------------------------------- #

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks belonging to *source*.

        Returns:
            Number of rows deleted (best-effort count).
        """
        table = self._get_table()
        before = table.count_rows()
        # Escape single quotes for SQL filter.
        safe_source = source.replace("'", "''")
        table.delete(f"source = '{safe_source}'")
        self._fts_ready = False
        after = table.count_rows()
        return before - after

    def clear(self) -> None:
        """Delete all data in the table."""
        db = self._connect()
        if self.table_name in db.table_names():
            db.drop_table(self.table_name)
        self._table = None

    # -- public API: introspection ----------------------------------------- #

    def count_rows(self) -> int:
        """Return the total number of chunks in the store."""
        try:
            table = self._get_table()
            return table.count_rows()
        except RuntimeError:
            return 0

    def list_sources(self) -> list[str]:
        """Return a list of unique source file paths in the store."""
        try:
            table = self._get_table()
        except RuntimeError:
            return []

        # Use Arrow (no pandas dependency).
        arrow_table = table.to_arrow()
        sources = arrow_table.column("source").to_pylist()
        # Deduplicate while preserving order.
        seen: set[str] = set()
        result: list[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the store."""
        try:
            table = self._get_table()
            row_count = table.count_rows()
            sources = self.list_sources()
        except RuntimeError:
            row_count = 0
            sources = []

        return {
            "db_path": self.db_path,
            "table_name": self.table_name,
            "row_count": row_count,
            "source_count": len(sources),
            "sources": sources,
            "embedding_dim": self.dim,
        }

    # -- context manager support ------------------------------------------- #

    def close(self) -> None:
        """Release resources."""
        self._table = None
        self._db = None

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #


def create_store_from_config(config: Any) -> VectorStore:
    """Create a :class:`VectorStore` from a rag-kit ``Config`` object.

    Args:
        config: A :class:`rag_kit.config.Config` instance.

    Returns:
        Configured (but not yet connected) :class:`VectorStore`.
    """
    return VectorStore(
        db_path=config.db_path,
        table_name=_TABLE_NAME,
        dim=384,  # Matches paraphrase-multilingual-MiniLM-L12-v2
    )
