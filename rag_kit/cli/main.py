"""rag-kit CLI — Typer-based command-line interface.

Commands:
  ingest    One-shot ingestion of files/folders into the vector DB
  watch     Folder watcher using watchdog for real-time auto-ingestion
  query     Search the vector DB with source attribution
  status    Show DB stats (doc count, chunk count, DB size, model info)
  config    Print or update configuration
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import typer

from rag_kit import __version__
from rag_kit.config import Config, create_default_config, get_config, get_config_path, load_config
from rag_kit.embed import EmbeddingEngine, create_engine_from_config
from rag_kit.ingest import SUPPORTED_EXTENSIONS, ingest_file, ingest_folder
from rag_kit.store import VectorStore, create_store_from_config
from rag_kit.autostart import is_autostart_installed, remove_autostart, setup_autostart

# --------------------------------------------------------------------------- #
# Logger setup
# --------------------------------------------------------------------------- #

# We configure a root 'rag_kit' logger for the CLI so all sub-packages
# (ingest, embed, store, vlm) share the same handler and level.
_CLI_LOGGER = logging.getLogger("rag_kit")
# Default: quiet.  --verbose raises to INFO, --debug to DEBUG.
_console_handler: logging.Handler | None = None


def _setup_logging(level: int = logging.WARNING) -> None:
    """Configure the rag_kit logger for CLI output."""
    global _console_handler
    if _console_handler is not None:
        _CLI_LOGGER.removeHandler(_console_handler)
    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    _console_handler.setLevel(level)
    _CLI_LOGGER.addHandler(_console_handler)
    _CLI_LOGGER.setLevel(level)


# --------------------------------------------------------------------------- #
# Typer app
# --------------------------------------------------------------------------- #

app = typer.Typer(
    name="rag-kit",
    help=(
        "rag-kit — Agentic RAG system for Hermes Agent.\n\n"
        "Watch a folder, ingest documents (PDF/DOCX/TXT/MD), embed with "
        "multilingual sentence-transformers, store in LanceDB, and query "
        "via CLI. Supports Chinese + English, 2 GB memory budget.\n\n"
        "Commands: ingest, query, list-files, delete, status, config, "
        "watch, setup-autostart\n"
        "Use --help on any command for details. Use --json for agent output."
    ),
)

# --------------------------------------------------------------------------- #
# Shared callbacks / helpers
# --------------------------------------------------------------------------- #


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V",
                                  help="Show version and exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                  help="Enable INFO-level logging."),
    debug: bool = typer.Option(False, "--debug",
                               help="Enable DEBUG-level logging."),
) -> None:
    """rag-kit — Agentic RAG system for Hermes Agent."""
    if version:
        typer.echo(f"rag-kit {__version__}")
        raise typer.Exit()

    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    _setup_logging(level)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _ensure_config() -> Config:
    """Load and validate config, exiting on failure."""
    try:
        return load_config()
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


def _db_size_mb(db_path: str) -> int:
    """Return the total size of the LanceDB directory in megabytes."""
    p = Path(db_path)
    if not p.exists():
        return 0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total // (1024 * 1024)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or folder to ingest."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Enable/disable OCR for scanned PDFs."),
    vlm: Optional[bool] = typer.Option(
        None, "--vlm/--no-vlm", help="Enable/disable VLM for visual content (default: from config)."
    ),
) -> None:
    """Ingest a file or folder into the vector DB.

    Supported formats: PDF, DOCX, TXT, MD.

    For a single file, that file is parsed, chunked, embedded, and stored.
    For a folder, all supported files inside are ingested recursively.
    """
    config = _ensure_config()
    if vlm is None:
        vlm = config.vlm_enabled
    source = Path(path).expanduser().resolve()
    start = time.monotonic()
    store = create_store_from_config(config)
    engine = create_engine_from_config(config)
    errors = []
    incomplete = False
    added = 0
    if not source.exists():
        files = []
        errors.append({"file": str(source), "error": "Path does not exist"})
    elif source.is_file():
        files = [source]
    else:
        files = sorted(p for p in source.rglob("*") if p.is_file()
                       and p.suffix.lower() in config.supported_extensions
                       and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for file in files:
        typer.echo(f"Ingesting file: {file}", err=True)
        try:
            chunks = ingest_file(file, chunk_size=config.chunk_size,
                                 chunk_overlap=config.chunk_overlap,
                                 languages=config.languages, use_ocr=ocr, use_vlm=vlm)
            vectors = engine.embed_texts([c["text"] for c in chunks]) if chunks else []
            added += store.replace_source(str(file), chunks, vectors)
        except Exception as exc:
            from rag_kit.ingest.pipeline import IncompleteExtractionError
            incomplete = incomplete or isinstance(exc, IncompleteExtractionError)
            errors.append({"file": str(file), "error": str(exc)})
    result = {
        "status": ("partial" if incomplete else "error") if errors else "ok", "path": str(source),
        "chunks_found": added, "chunks_stored": added, "errors": errors,
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False))
    else:
        typer.echo(f"Ingested {added} chunks from {source}")
        for error in errors:
            typer.echo(f"Error ingesting {error['file']}: {error['error']}", err=True)
    if errors:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

@app.command(name="query")
def query_cmd(
    text: str = typer.Argument(..., help="Search query text."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return."),
    source: str = typer.Option("", "--source", "-s", help="Filter results to this source file."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
) -> None:
    """Search the vector DB.

    Returns top-k results with similarity scores and source attribution.
    """
    config = _ensure_config()
    store = create_store_from_config(config)

    if store.count_rows() == 0:
        msg = "Database is empty. Ingest documents first with: rag-kit ingest <folder>"
        if json_output:
            typer.echo(json.dumps({"status": "empty", "message": msg, "results": []},
                                  ensure_ascii=False))
        else:
            typer.echo(msg)
        raise typer.Exit(code=0)

    query_vec = (create_engine_from_config(config).embed_query(text)
                 if float(config.search_alpha) > 0 else None)

    filter_sql = None
    if source:
        source = source if source in store.list_sources() else str(Path(source).expanduser().resolve())
        safe_source = source.replace("'", "''")
        filter_sql = f"source = '{safe_source}'"
    # Use the configured blend even when filtering to one document.
    results = store.search_hybrid(text, query_vec, top_k=top_k,
                                  alpha=float(config.search_alpha), filter_sql=filter_sql)

    if json_output:
        output = {
            "status": "ok",
            "query": text,
            "top_k": top_k,
            "result_count": len(results),
            "results": [
                {
                    "score": r["score"],
                    "source": r["source"],
                    "page": r.get("page", 0),
                    "chunk_idx": r.get("chunk_idx", 0),
                    "text": r["text"],
                }
                for r in results
            ],
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"\nQuery: \"{text}\"\n")
        typer.echo(f"{'─' * 70}")
        for i, r in enumerate(results, 1):
            typer.echo(
                f"[{i}] score={r['score']:.4f}  "
                f"source={r.get('source', '?')}  "
                f"page={r.get('page', 0)}"
            )
            # Show a snippet (first 200 chars).
            snippet = r.get("text", "")[:200].replace("\n", " ")
            typer.echo(f"    {snippet}{'…' if len(r.get('text', '')) > 200 else ''}")
            typer.echo(f"{'─' * 70}")
        typer.echo(f"\n{len(results)} result(s) shown (top-{top_k})")


# --------------------------------------------------------------------------- #
# list-files
# --------------------------------------------------------------------------- #

@app.command(name="list-files")
def list_files(
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
) -> None:
    """List all ingested files in the vector DB."""
    config = _ensure_config()
    store = create_store_from_config(config)
    sources = store.list_sources()

    if not sources:
        msg = "No files ingested yet. Use: rag-kit ingest <folder>"
        if json_output:
            typer.echo(json.dumps({"status": "empty", "message": msg, "files": []},
                                  ensure_ascii=False))
        else:
            typer.echo(msg)
    elif json_output:
        typer.echo(json.dumps({"status": "ok", "file_count": len(sources), "files": sources},
                              indent=2, ensure_ascii=False))
    else:
        typer.echo(f"{len(sources)} file(s) ingested:")
        for s in sources:
            typer.echo(f"  {s}")


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #

@app.command()
def delete(
    file_path: str = typer.Argument(..., help="Path (full or filename) of the file to remove from DB."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
) -> None:
    """Remove a file's chunks from the vector DB.

    Accepts either a full path as stored in the DB or a bare filename
    (in which case all sources with that basename are removed).
    """
    _ensure_config()
    store = create_store_from_config(get_config())

    # 1. Try exact match on resolved path, then basename, then any stored
    #    source whose file name matches (handles full paths and bare names).
    targets = [str(Path(file_path).expanduser().resolve()), file_path]
    # Older CLI imports stored relative paths. Match only identities whose
    # resolution equals this exact file; never guess a VLM basename owner.
    targets += [s for s in store.list_sources() if not Path(s).is_absolute()
                and str(Path(s).resolve()) == targets[0]]
    if Path(file_path).name == file_path:
        # Bare filename: match against stored sources by basename.
        try:
            targets += [s for s in store.list_sources()
                        if Path(s).name == file_path]
        except RuntimeError:
            pass

    removed = 0
    for target in dict.fromkeys(targets):
        try:
            removed += store.delete_by_source(target)
        except RuntimeError:
            # Table doesn't exist yet — nothing to delete.
            pass

    if json_output:
        typer.echo(json.dumps({"status": "ok", "file": file_path, "removed_chunks": removed},
                              ensure_ascii=False))
    else:
        if removed > 0:
            typer.echo(f"Removed {removed} chunk(s) for: {file_path}")
        else:
            typer.echo(f"No chunks found for: {file_path}")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

@app.command()
def status(
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
) -> None:
    """Show DB stats: document count, chunk count, DB size, model info."""
    config = _ensure_config()
    store = create_store_from_config(config)
    engine = create_engine_from_config(config)

    stats = store.get_stats()
    model_info = engine.get_model_info()
    db_size = _db_size_mb(config.db_path)

    if json_output:
        typer.echo(json.dumps({
            "version": __version__,
            "db": {
                "path": stats["db_path"],
                "table": stats["table_name"],
                "chunk_count": stats["row_count"],
                "source_count": stats["source_count"],
                "sources": stats["sources"],
                "size_mb": db_size,
            },
            "embedding": model_info,
            "config": {
                "watch_folder": config.watch_folder,
                "vlm_enabled": config.vlm_enabled,
                "vlm_model": config.vlm_model,
                "languages": config.languages,
                "chunk_size": config.chunk_size,
                "max_memory_mb": config.max_memory_mb,
            },
        }, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"rag-kit v{__version__}")
        typer.echo()
        typer.echo("── Database ──")
        typer.echo(f"  Path:         {stats['db_path']}")
        typer.echo(f"  Table:        {stats['table_name']}")
        typer.echo(f"  Chunks:       {stats['row_count']}")
        typer.echo(f"  Documents:    {stats['source_count']}")
        typer.echo(f"  Size:         {db_size} MB")
        typer.echo()
        typer.echo("── Embedding Model ──")
        typer.echo(f"  Model:        {model_info['model_name']}")
        typer.echo(f"  Dimension:    {model_info['dimension']}")
        typer.echo(f"  Approx size:  {model_info['approx_size_mb']} MB")
        typer.echo(f"  Loaded:       {'yes' if model_info['loaded'] else 'no (lazy)'}")
        typer.echo(f"  Cache dir:    {model_info['model_dir'] or '(default)'}")
        typer.echo(f"  HF endpoint:  {model_info['hf_endpoint']}")
        typer.echo()
        typer.echo("── Configuration ──")
        typer.echo(f"  Watch folder: {config.watch_folder}")
        typer.echo(f"  VLM:          {config.vlm_model} ({'enabled' if config.vlm_enabled else 'disabled'})")
        typer.echo(f"  Languages:    {', '.join(config.languages)}")
        typer.echo(f"  Chunk size:   {config.chunk_size} (overlap: {config.chunk_overlap})")
        typer.echo(f"  Max memory:   {config.max_memory_mb} MB")
        typer.echo(f"  Extensions:   {', '.join(config.supported_extensions)}")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

config_app = typer.Typer(
    name="config",
    help="Print or update configuration.",
)
app.add_typer(config_app)


@config_app.callback(invoke_without_command=True)
def _config_default(ctx: typer.Context) -> None:
    """Print current configuration."""
    if ctx.invoked_subcommand is not None:
        return
    config = _ensure_config()
    typer.echo(f"Config file:  {get_config_path()}")
    typer.echo(f"Config exists: {'yes' if get_config_path().exists() else 'no'}")
    typer.echo()
    for key, val in config.to_dict().items():
        typer.echo(f"  {key}: {val}")


@config_app.command(name="init")
def config_init(
    path: str = typer.Option("", "--path", "-p", help="Path to write config file."),
) -> None:
    """Create a default configuration file."""
    p = create_default_config(path or None)
    typer.echo(f"Created default config at: {p}")


@config_app.command(name="path")
def config_path() -> None:
    """Print the config file path."""
    typer.echo(str(get_config_path()))


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(..., help="Config key to set (e.g. watch_folder)."),
    value: str = typer.Argument(..., help="Value to set."),
) -> None:
    """Update a configuration value and write it to disk."""
    import yaml

    target = get_config_path()

    # Load existing config file (or empty dict).
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Coerce the value type.
    from rag_kit.config import _BOOL_KEYS, _FLOAT_KEYS, _INT_KEYS, _LIST_KEYS, _coerce
    typed_val = _coerce(key, value)

    data[key] = typed_val

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    typer.echo(f"Set {key} = {typed_val} in {target}")
    # Warn if value doesn't look right after coercion.
    if key in _LIST_KEYS and isinstance(typed_val, str):
        typer.echo(f"  Note: {key} expects comma-separated list; current value: {typed_val!r}")


# --------------------------------------------------------------------------- #
# watch
# --------------------------------------------------------------------------- #

# Watchdog is imported lazily so the module loads even if watchdog is missing,
# but the watch command will fail with a clear message.

class _WatchHandler:
    """watchdog event handler that debounces file events before ingestion.

    When a file is created or modified, its path is recorded with a timestamp.
    A background thread processes files that have been stable (no new events)
    for ``debounce_seconds`` seconds.
    """

    def __init__(
        self,
        config: Config,
        debounce_seconds: float = 2.0,
        json_output: bool = False,
    ) -> None:
        from watchdog.events import FileSystemEventHandler

        self._handler = _DebouncedEventHandler(self._on_event)
        self._config = config
        self._root = Path(config.watch_folder).expanduser().resolve()
        self._debounce = debounce_seconds
        self._json = json_output
        self._pending_events: dict[str, float] = {}  # path -> last event time
        self._pending_trees: set[str] = set()
        self._pending_deletes: set[str] = set()      # resolved abs paths to remove
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._worker: threading.Thread | None = None

        _CLI_LOGGER.info("Watch handler initialised (debounce=%.1fs)", debounce_seconds)

    @property
    def handler(self) -> Any:
        return self._handler

    def _on_event(self, event_type: str, file_path: str) -> None:
        if event_type in ("tree_created", "created_or_modified"):
            if not Path(file_path).resolve().is_relative_to(self._root):
                return
        if event_type == "tree_created":
            folder = Path(file_path)
            if folder.is_dir():
                for path in folder.rglob("*"):
                    if path.is_file():
                        self._on_event("created_or_modified", str(path))
            return
        with self._lock:
            if event_type == "tree_deleted":
                self._pending_trees.add(str(Path(file_path).resolve()))
            elif event_type == "deleted":
                self._pending_deletes.add(str(Path(file_path).resolve()))
                self._pending_events.pop(file_path, None)
            elif Path(file_path).suffix.lower() not in self._config.supported_extensions:
                return
            else:
                self._pending_events[file_path] = time.monotonic()
        self._resume.set()

    def reconcile(self, folder: Path) -> None:
        """Backfill changes missed during downtime without touching other roots."""
        self._root = folder.resolve()
        files = [p for p in folder.rglob("*") if p.is_file()]
        store = create_store_from_config(self._config)
        for source in store.list_sources():
            path = Path(source)
            if (path.is_absolute() and path.is_relative_to(folder)
                    and (not path.exists() or path.suffix.lower() not in self._config.supported_extensions)):
                self._on_event("deleted", source)
        for file in files:
            self._on_event("created_or_modified", str(file))

    def _drain_deletes(self) -> None:
        with self._lock:
            deletes = list(self._pending_deletes)
            trees = list(self._pending_trees)
            self._pending_deletes.clear()
            self._pending_trees.clear()
        store = create_store_from_config(self._config)
        for tree in trees:
            for source in store.list_sources():
                if Path(source).is_absolute() and Path(source).is_relative_to(Path(tree)):
                    deletes.append(source)
        for source in set(deletes):
            self._delete_one(source)

    def start_worker(self) -> None:
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def _process_loop(self) -> None:
        """Background thread: process deletes immediately, ingest stable files."""
        _CLI_LOGGER.info("Watcher worker started")
        while not self._stop.wait(timeout=0.5):
            self._drain_deletes()

            # 2. Ingest files that have been stable for the debounce period.
            to_process: list[str] = []
            now = time.monotonic()
            with self._lock:
                for path, ts in list(self._pending_events.items()):
                    if now - ts >= self._debounce:
                        to_process.append(path)
                        del self._pending_events[path]

            for file_path in to_process:
                self._ingest_one(file_path)

        _CLI_LOGGER.info("Watcher worker stopped")

    def _delete_one(self, abs_path: str) -> None:
        """Remove all stored chunks for a deleted file and report to the user."""
        _CLI_LOGGER.info("Deleting vectors for removed file: %s", abs_path)
        try:
            store = create_store_from_config(self._config)
            deleted = store.delete_by_source(abs_path)
            _CLI_LOGGER.info("Removed %d chunk(s) for %s", deleted, abs_path)
        except RuntimeError:
            pass  # table doesn't exist yet — nothing to delete
        except Exception as exc:
            _CLI_LOGGER.error("Failed to remove vectors for %s: %s", abs_path, exc)

    def _ingest_one(self, file_path: str) -> None:
        """Ingest a single file that has stabilised."""
        start = time.monotonic()
        _CLI_LOGGER.info("Processing: %s", file_path)

        # Resolve once so the stored ``source`` matches the value used for
        # delete-by-source replacement (and later file-removal deletes).
        abs_path = str(Path(file_path).resolve())

        try:
            chunks = ingest_file(
                abs_path,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
                languages=self._config.languages,
                use_ocr=True,
                use_vlm=self._config.vlm_enabled,
            )
        except Exception as exc:
            _CLI_LOGGER.error("Failed to ingest %s: %s", file_path, exc)
            from rag_kit.ingest.pipeline import IncompleteExtractionError
            if self._json:
                typer.echo(json.dumps({
                    "event": "ingest_failed", "file": file_path,
                    "status": "partial" if isinstance(exc, IncompleteExtractionError) else "error",
                    "errors": [str(exc)], "source_replaced": False,
                }, ensure_ascii=False))
            return


        # Convert to store format.
        store_chunks: list[dict[str, Any]] = []
        for i, c in enumerate(chunks):
            store_chunks.append({
                "id": c.get("id", c.get("chunk_id", "")),
                "text": c["text"],
                "source": c.get("source", c.get("source_file", "")),
                "page": int(c.get("page", 0)),
                "chunk_idx": int(c.get("chunk_idx", i)),
                "vlm_generated": bool(c.get("vlm_generated", False)),
                "source_type": str(c.get("source_type", "text")),
            })

        try:
            store = create_store_from_config(self._config)
            engine = create_engine_from_config(self._config)
            texts = [c["text"] for c in store_chunks]
            vectors = engine.embed_texts(texts) if texts else []
            added = store.replace_source(abs_path, store_chunks, vectors)
            elapsed = time.monotonic() - start

            msg = f"Ingested {file_path}: {len(chunks)} chunks ({added} stored) in {elapsed:.1f}s"
            if self._json:
                typer.echo(json.dumps({
                    "event": "ingested",
                    "file": file_path,
                    "chunks": len(chunks),
                    "stored": added,
                    "elapsed": round(elapsed, 2),
                }, ensure_ascii=False))
            else:
                typer.echo(msg)
        except Exception as exc:
            _CLI_LOGGER.error("Failed to store %s: %s", file_path, exc)

    def flush_and_stop(self) -> None:
        """Process any remaining pending files, then stop the worker."""
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join()

        self._drain_deletes()

        # Process anything still pending.
        with self._lock:
            remaining = list(self._pending_events.keys())
            self._pending_events.clear()

        for file_path in remaining:
            _CLI_LOGGER.info("Flushing pending: %s", file_path)
            self._ingest_one(file_path)


class _DebouncedEventHandler:
    """Minimal watchdog event handler that calls a callback on file changes."""

    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def dispatch(self, event: Any) -> None:
        # watchdog's dispatch method — handle create/modify/move/delete.
        directory = bool(event.is_directory)
        if event.event_type in ("created", "modified"):
            if not directory or event.event_type == "created":
                self._callback("tree_created" if directory else "created_or_modified", event.src_path)
        elif event.event_type == "moved":
            self._callback("tree_deleted" if directory else "deleted", event.src_path)
            self._callback("tree_created" if directory else "created_or_modified", event.dest_path)
        elif event.event_type == "deleted":
            self._callback("tree_deleted" if directory else "deleted", event.src_path)


@app.command()
def watch(
    folder: str | None = typer.Argument(None, help="Folder to watch for new/modified documents (default: config watch_folder)."),
    debounce: float = typer.Option(2.0, "--debounce", "-d",
                                    help="Seconds to wait after last file event before ingesting."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
    poll: bool = typer.Option(False, "--poll",
                               help="Use polling instead of native OS events."),
) -> None:
    """Watch a folder and auto-ingest new/modified documents in real-time.

    Uses watchdog for native filesystem event monitoring (inotify on Linux,
    FSEvents on macOS, ReadDirectoryChangesW on Windows).  Files are ingested
    after they stop changing for --debounce seconds (default 2s).

    Press Ctrl+C to stop gracefully.
    """
    config = _ensure_config()
    folder_path = Path(folder or config.watch_folder).expanduser().resolve()
    if not folder_path.is_dir():
        if json_output:
            typer.echo(json.dumps({"event": "error", "error": f"Folder does not exist: {folder_path}"}))
        else:
            typer.echo(f"Error: folder does not exist: {folder_path}", err=True)
        raise typer.Exit(code=1)
    config.watch_folder = str(folder_path)
    from rag_kit import watchlock
    lock = watchlock.acquire(folder_path)
    if lock is None:
        if json_output:
            typer.echo(json.dumps({"status": "already_running", "folder": str(folder_path)}))
        else:
            typer.echo(f"Another watcher is watching: {folder_path}", err=True)
        return
    observer = None
    watch_handler = None
    originals = {}
    try:
        from watchdog.observers import Observer
        from watchdog.observers.polling import PollingObserver
        watch_handler = _WatchHandler(config, debounce_seconds=debounce, json_output=json_output)
        observer = (PollingObserver if poll else Observer)()
        observer.schedule(watch_handler.handler, str(folder_path), recursive=True)
        shutdown = threading.Event()
        def on_shutdown(signum, frame):
            shutdown.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            originals[sig] = signal.signal(sig, on_shutdown)
        # Observe before scanning, then start the worker after backfill queues.
        observer.start()
        watch_handler.reconcile(folder_path)
        watch_handler.start_worker()
        if json_output:
            typer.echo(json.dumps({"event": "watching", "folder": str(folder_path)}))
        else:
            typer.echo(f"Watching: {folder_path}. Press Ctrl+C to stop.", err=True)
        shutdown.wait()
    finally:
        if observer is not None:
            observer.stop()
            if observer.is_alive():
                observer.join()
        try:
            if watch_handler is not None:
                watch_handler.flush_and_stop()
        finally:
            for sig, handler in originals.items():
                signal.signal(sig, handler)
            watchlock.release(lock)
    if json_output:
        typer.echo(json.dumps({"event": "stopped"}))
    else:
        typer.echo("Done.", err=True)


# --------------------------------------------------------------------------- #
# setup-autostart
# --------------------------------------------------------------------------- #

@app.command(name="setup-autostart")
def autostart_cmd(
    install: bool = typer.Option(True, "--install/--remove", help="Install or remove autostart."),
    interval: int = typer.Option(0, "--interval", "-i",
                                  help="Polling interval in seconds (default: from config)."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
) -> None:
    """Configure the rag-kit watcher to start automatically on boot.

    Windows: Creates a scheduled task.
    Linux:   Creates a systemd user service with linger.
    """
    _ensure_config()
    config = get_config()

    if interval <= 0:
        interval = config.watch_interval

    if install:
        success, message = setup_autostart(interval=interval)
        if not success:
            if json_output:
                typer.echo(json.dumps({"status": "error", "message": message}, ensure_ascii=False))
            else:
                typer.echo(f"Failed to install autostart: {message}", err=True)
            raise typer.Exit(code=1)

        installed = is_autostart_installed()
        if json_output:
            typer.echo(json.dumps({
                "status": "ok",
                "action": "install",
                "message": message,
                "installed": installed,
                "interval": interval,
            }, ensure_ascii=False))
        else:
            typer.echo(message)
            typer.echo(f"  Status: {'Installed' if installed else 'NOT installed'}")
    else:
        success, message = remove_autostart()
        if json_output:
            typer.echo(json.dumps({
                "status": "ok" if success else "error",
                "action": "remove",
                "message": message,
            }, ensure_ascii=False))
        else:
            typer.echo(message)
        if not success:
            raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_cli() -> None:
    # Configure before Typer parses --help, which bypasses its callback.
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    run_cli()
