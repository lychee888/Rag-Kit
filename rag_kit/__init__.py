"""
rag-kit — Agentic RAG system for Hermes Agent.

Watch a folder, auto-ingest documents (PDF, DOCX, XLSX, PPTX, TXT, MD,
and images via OCR), embed with multilingual sentence-transformers, store
in LanceDB, and query via a CLI designed for LLM agent consumption.
"""

import os
import site
import sys
from pathlib import Path

__version__ = "0.1.1"
__all__ = ["__version__"]


def _sanitize_sys_path() -> None:
    """Drop site-packages entries that don't belong to this interpreter.

    A wrapping shell with ``PYTHONPATH`` set (common when an agent runtime
    like Hermes exports its own venv) injects those packages *ahead* of this
    venv's site-packages, so ``rag`` can silently import wrong versions
    (e.g. a tokenizers build incompatible with transformers). A venv should
    behave like a venv regardless of the shell it is launched from, so we
    keep only [the current interpreter, the base interpreter, the user site]
    and drop any other site-packages entry.
    """
    owned = {
        str(Path(sys.prefix).resolve()).lower(),
        str(Path(sys.base_prefix).resolve()).lower(),
    }
    try:
        owned.add(str(Path(site.getusersitepackages()).resolve()).lower())
    except Exception:
        pass
    owned = {o for o in owned if o}  # drop empty entries
    to_drop: set[str] = set()
    for entry in sys.path:
        if not entry or "site-packages" not in entry.lower():
            continue
        try:
            under_owned = str(Path(entry).resolve()).lower().startswith(
                tuple(owned)
            )
        except Exception:
            under_owned = True  # unreadable entry — keep it, be conservative
        if not under_owned:
            to_drop.add(entry)
    if to_drop:
        sys.path[:] = [e for e in sys.path if e not in to_drop]

    # Keep child processes (pip, model downloaders) rooted in this venv too.
    os.environ.pop("PYTHONPATH", None)


_sanitize_sys_path()
