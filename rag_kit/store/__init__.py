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

        self._db = lancedb.connect(self.db_path)
        return self._db

    def _ensure_table(self) -> Any:
        """Open or create the documents table.

        If the table exists but lacks the expected columns (vlm_generated,
        source_type), it is dropped and recreated — those were added after
        the initial scaffold.
        """
        if self._table is not None:
            return self._table

        db = self._connect()

        if self.table_name in db.table_names():
            existing = db.open_table(self.table_name)
            existing_cols = set(existing.schema.names)
            required = {"vlm_generated", "source_type"}
            if not required.issubset(existing_cols):
                import logging
                _logger = logging.getLogger(__name__)
                _logger.info(
                    "Table %r is missing columns %s; dropping and recreating.",
                    self.table_name,
                    required - existing_cols,
                )
                db.drop_table(self.table_name)
            else:
                self._table = existing
                return self._table

        schema = _make_schema(self.dim)
        self._table = db.create_table(
            self.table_name,
            schema=schema,
            mode="overwrite",
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
            self._table = db.open_table(self.table_name)
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
                table.create_index(column="text", config=FTS(), replace=True)
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
        self._ensure_fts()
        table = self._get_table()

        query_vec = np.asarray(query_vector, dtype=np.float32).tolist()
        alpha = min(max(alpha, 0.0), 1.0)

        # vector-only shortcut
        if alpha >= 1.0:
            return self.search(query_vector, top_k=top_k, filter_sql=filter_sql)
        # keyword-only shortcut (FTS raw = relevance as score)
        if alpha <= 0.0:
            q = table.search(query_text, query_type="fts").limit(top_k)
            if filter_sql:
                q = q.where(filter_sql)
            rows = q.to_list()
            for row in rows:
                row["score"] = round(row.get("score", row.get("_score", 0.0)), 4)
            return rows

        try:
            from lancedb.rerankers import RRFReranker
        except Exception:
            return self.search(query_vector, top_k=top_k, filter_sql=filter_sql)

        try:
            q = (
                table.search(query_type="hybrid")
                .vector(query_vec)
                .text(query_text)
                .rerank(reranker=RRFReranker())
                .limit(top_k)
            )
            if filter_sql:
                q = q.where(filter_sql)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Hybrid search failed (%s); falling back to vector search.", exc
            )
            return self.search(query_vector, top_k=top_k, filter_sql=filter_sql)

        rows = q.to_list()
        for row in rows:
            score = row.get("_relevance_score", row.get("score", 0.0))
            row["score"] = round(float(score), 4)
            row.pop("_relevance_score", None)
        return rows

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
