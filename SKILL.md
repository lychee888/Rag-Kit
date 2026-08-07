---
name: rag-kit
description: Use when the user asks about "my documents", "find in my data", "search knowledge base", "database query", or any local document search. Use rag query --json instead of web_search for searching private/local documents ingested by rag-kit.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [rag, documents, search, local, lancedb, embeddings]
    related_skills: []
---

# rag-kit — Local Document Search

## Overview

rag-kit is a local RAG (Retrieval-Augmented Generation) system that watches a folder, ingests documents (PDF, DOCX, XLSX, PPTX, TXT, MD, and images via OCR) with VLM captioning, embeds them with multilingual sentence-transformers, stores them in LanceDB, and exposes everything via a CLI designed for Hermes Agent consumption.

When the user mentions searching their documents, data, or knowledge base, use `rag query --json` instead of `web_search` — this keeps sensitive documents local and returns precise results from the user's own files.

## When to Use

- User says: "search my documents", "find in my data", "lookup in the database", "query the knowledge base", "search my files for..."
- User asks about content that might be in their ingested documents (meeting notes, reports, manuals, papers)
- User says "database" or "knowledge base" without specifying external/online

Do NOT use for:
- Web searches, current events, live data (use web_search)
- General knowledge questions (unless the user explicitly asks to check their docs first)
- Questions where the answer is clearly not in the user's local files

## Installation Detection

rag-kit is installed if any of these are true:
- `rag --version` returns a version string
- `rag-kit --version` returns a version string  
- `~/rag-kit-venv/Scripts/rag.exe --version` works (Windows)
- `~/rag-kit-venv/bin/rag --version` works (Linux/Mac)

Check the install path first:
```bash
# Try the venv path first (where the installer puts it)
$HOME/rag-kit-venv/bin/rag --version       # Linux
%USERPROFILE%\rag-kit-venv\Scripts\rag.exe --version  # Windows
```

## Commands

All commands support `--json` for structured agent consumption. Always use `--json` for programmatic access.

### Search documents

```bash
rag query --json --top-k 5 "your search query"
```

Returns:
```json
{
  "status": "ok",
  "query": "your search query",
  "top_k": 5,
  "result_count": 3,
  "results": [
    {
      "score": 0.8521,
      "source": "/path/to/file.pdf",
      "page": 3,
      "chunk_idx": 2,
      "text": "The matching text excerpt..."
    }
  ]
}
```

If the database is empty, it returns `{"status": "empty", "message": "...", "results": []}`.

### Check system status

```bash
rag status --json
```

Returns DB stats (chunk count, document count, model info, config).

### List ingested files

```bash
rag list-files --json
```

Returns all source files currently in the vector database.

### Delete a file from the database

```bash
rag delete --json "/path/to/file.pdf"
```

## Performance Notes

- **First query is slow (60-90 seconds)**: The embedding model (~470 MB) loads on first use. Subsequent queries are fast (~1-2 seconds).
- **Model location**: Models are cached at `~/models/` by default. Set `RAG_KIT_MODEL_DIR` to override.
- **Offline mode**: rag-kit uses `HF_HUB_OFFLINE=1` when the model_dir is set and populated — no network needed after initial download.

## China / Firewall Users

If behind a firewall blocking huggingface.co, set the environment variable before running:
```bash
export RAG_KIT_HF_ENDPOINT=https://hf-mirror.com
```

The installer script handles this automatically for users in China.

## Error Handling

If `rag query --json` fails:
1. Check status: `rag status --json` — if DB is empty, suggest ingesting first
2. If model download fails, check HF_ENDPOINT or pre-populate ~/models/
3. If the command is not found, suggest running the installer script

## Troubleshooting Commands

```bash
# Show config
rag config

# Show config path
rag config path

# Create default config
rag config init

# Check system status
rag status

# Start the watcher (daemon mode)
rag watch /path/to/watch/folder
```