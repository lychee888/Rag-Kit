"""Embedding sub-package — sentence-transformers model management.

Provides ``EmbeddingEngine`` for generating vector embeddings from text
and a ``chunk_text`` utility for splitting documents into overlapping chunks.

Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ~120 MB, 50+ languages).
Well within the 2 GB total memory budget.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Chunk dataclass
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """A text chunk extracted from a document.

    Attributes:
        text:       The chunk text content.
        source:     Absolute or relative path of the source file.
        page:       Page number (0 for non-paginated documents).
        chunk_idx:  Index of this chunk within the source file.
        chunk_id:   Unique identifier (auto-generated if not provided).
    """

    text: str
    source: str
    page: int = 0
    chunk_idx: int = 0
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_idx": self.chunk_idx,
        }


# --------------------------------------------------------------------------- #
# Text chunking
# --------------------------------------------------------------------------- #


def chunk_text(
    text: str,
    source: str = "",
    page: int = 0,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split *text* into overlapping chunks of approximately *chunk_size* tokens.

    Uses a sentence-aware splitter that tries to break on sentence boundaries
    (handles both Western ``.!?`` and CJK ``。！？`` punctuation) while
    respecting the chunk_size and chunk_overlap parameters measured in
    characters (a reasonable proxy for tokens on multilingual text).

    Args:
        text:           The full text to split.
        source:         Source file path (attached to each chunk).
        page:           Page number (attached to each chunk).
        chunk_size:     Target chunk length in characters.
        chunk_overlap:  Overlap between consecutive chunks in characters.

    Returns:
        List of :class:`Chunk` objects with sequential ``chunk_idx``.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace but preserve CJK characters.
    text = text.strip()

    # Sentence boundary pattern: Western .!? and CJK 。！？ followed by
    # optional quotes/brackets then whitespace or end-of-string.
    sentence_end = re.compile(r"([.!?。！？][\"'\)\]]*)\s+")

    # Split into sentences, keeping the delimiter attached.
    parts = sentence_end.split(text)
    sentences: list[str] = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i + 1]  # re-attach delimiter
            i += 2
        else:
            i += 1
        if s.strip():
            sentences.append(s.strip())

    # If no sentence boundaries found, treat the whole text as one sentence.
    if not sentences:
        sentences = [text]

    # Greedily pack sentences into chunks of ~chunk_size chars.
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for sentence in sentences:
        # If a single sentence exceeds chunk_size, hard-split it.
        if len(sentence) > chunk_size:
            # Flush current buffer first.
            if current:
                chunks.append(
                    Chunk(
                        text=current,
                        source=source,
                        page=page,
                        chunk_idx=idx,
                    )
                )
                idx += 1
                current = ""

            # Hard-split the long sentence.
            for start in range(0, len(sentence), chunk_size - chunk_overlap):
                piece = sentence[start : start + chunk_size]
                if len(piece) < 50 and chunks:
                    # Merge tiny tail into previous chunk.
                    chunks[-1].text += " " + piece
                else:
                    chunks.append(
                        Chunk(
                            text=piece,
                            source=source,
                            page=page,
                            chunk_idx=idx,
                        )
                    )
                    idx += 1
            continue

        # Normal packing.
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(
                    Chunk(
                        text=current,
                        source=source,
                        page=page,
                        chunk_idx=idx,
                    )
                )
                idx += 1
                # Overlap: carry the tail of the previous chunk.
                if chunk_overlap > 0 and len(current) > chunk_overlap:
                    overlap = current[-chunk_overlap:]
                    current = (overlap + " " + sentence).strip()
                else:
                    current = sentence
            else:
                current = sentence

    # Flush remaining buffer.
    if current:
        chunks.append(
            Chunk(
                text=current,
                source=source,
                page=page,
                chunk_idx=idx,
            )
        )
        idx += 1

    return chunks


# --------------------------------------------------------------------------- #
# Embedding engine
# --------------------------------------------------------------------------- #


def _model_is_cached(model_name: str, cache_dir: str) -> bool:
    """Return True if *model_name* is already downloaded into *cache_dir*.

    Checks both the flat folder (``cache_dir/<name>``) and the standard
    HuggingFace hub layout (``cache_dir/models--org--name/snapshots/...``).
    This decides whether we may enable offline mode.
    """
    from pathlib import Path

    base = Path(cache_dir)
    # 1. Flat directory: cache_dir/model_name with config.json (or safetensors)
    flat = base / model_name
    if (flat / "model.safetensors").exists() or (flat / "pytorch_model.bin").exists() \
            or (flat / "config.json").exists():
        return True
    # 2. HF hub layout: cache_dir/models--org--name/snapshots/<sha>/config.json
    org, _, name = model_name.partition("/")
    hub = base / f"models--{org}--{name}" / "snapshots"
    if hub.exists():
        for snapshot in hub.iterdir():
            if (snapshot / "config.json").exists():  # coarse; weights may still be partial
                return True
    return False


class EmbeddingEngine:
    """Lazy-loaded sentence-transformers embedding model.

    The model is loaded on first call to :meth:`embed_texts` or
    :meth:`embed_query`, not at construction time.  This allows the
    ``EmbeddingEngine`` to be instantiated cheaply (e.g. for config
    inspection) without downloading or loading the model weights.

    Model: ``paraphrase-multilingual-MiniLM-L12-v2``
      - 384-dimensional vectors
      - ~470 MB on disk (safetensors, 117M params, float32)
      - ~500 MB RAM at runtime
      - Supports 50+ languages (Chinese, English, etc.)
      - Fits within the 2 GB memory budget
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        model_dir: str = "",
        hf_endpoint: str = "",
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.hf_endpoint = hf_endpoint
        self._model: Any = None  # Lazy-loaded SentenceTransformer

    # -- internal ----------------------------------------------------------- #

    def _load_model(self) -> Any:
        """Load the sentence-transformers model (lazy, once)."""
        if self._model is not None:
            return self._model

        # Set HF endpoint for China mirror if configured.
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint

        # Set cache directory if configured.
        cache_kwargs: dict[str, Any] = {}
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
            cache_kwargs["cache_folder"] = self.model_dir
            # Go offline ONLY if the model is already present in the cache dir.
            # Otherwise we stay online so the model can download on first run.
            if _model_is_cached(self.model_name, self.model_dir):
                os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers could not be imported — this usually means a "
                "dependency version conflict (most commonly an incompatible "
                "tokenizers build). Underlying error:\n"
                f"  {exc}\n"
                "Fix: reinstall the project dependencies (a clean venv works best):\n"
                '  pip install -e ".[ocr]"'
            ) from exc

        self._model = SentenceTransformer(self.model_name, **cache_kwargs)
        return self._model

    # -- public API --------------------------------------------------------- #

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), dim)`` with ``float32``
            dtype.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            text: Query text.

        Returns:
            1-D ``np.ndarray`` of shape ``(dim,)`` with ``float32`` dtype.
        """
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        model = self._load_model()
        # get_embedding_dimension is the new name (st>=5.0);
        # fall back to the old name for older versions.
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded into memory."""
        return self._model is not None

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata without loading the model."""
        return {
            "model_name": self.model_name,
            "dimension": 384,  # Known for paraphrase-multilingual-MiniLM-L12-v2
            "approx_size_mb": 470,  # safetensors float32, 117M params
            "loaded": self._model is not None,
            "model_dir": self.model_dir,
            "hf_endpoint": self.hf_endpoint or "(default)",
        }


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #


def create_engine_from_config(config: Any) -> EmbeddingEngine:
    """Create an :class:`EmbeddingEngine` from a rag-kit ``Config`` object.

    Args:
        config: A :class:`rag_kit.config.Config` instance.

    Returns:
        Configured (but not yet loaded) :class:`EmbeddingEngine`.
    """
    return EmbeddingEngine(
        model_name=config.embedding_model,
        model_dir=config.model_dir,
        hf_endpoint=config.hf_endpoint,
    )
