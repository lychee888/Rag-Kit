"""Configuration system for rag-kit.

Loads settings from a YAML config file, with environment variable overrides
for every key. Validates paths, model names, and the 2 GB memory budget guard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Resolve paths relative to user home for cross-platform consistency.
_HOME = Path.home()

DEFAULT_CONFIG = {
    "watch_folder": str(_HOME / "Documents" / "rag-ingest"),
    "db_path": str(_HOME / "lancedb"),
    "model_dir": str(_HOME / "models"),
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "vlm_model": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "vlm_enabled": True,
    "supported_extensions": [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".tiff", ".webp"],
    "languages": ["zh", "en"],
    "chunk_size": 512,
    "chunk_overlap": 64,
    "watch_interval": 30,
    "search_alpha": 0.5,
    "max_memory_mb": 2048,  # 2 GB budget guard
    "hf_endpoint": "",  # empty = default HF endpoint; set to https://hf-mirror.com for China
}

# Environment variable mappings: RAG_KIT_<KEY> overrides any config value.
ENV_PREFIX = "RAG_KIT_"

# Type coercion for non-string config values.
_BOOL_KEYS = {"vlm_enabled"}
_INT_KEYS = {"chunk_size", "chunk_overlap", "watch_interval", "max_memory_mb"}
_FLOAT_KEYS = {"search_alpha"}
_LIST_KEYS = {"supported_extensions", "languages"}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Validated configuration for rag-kit."""

    watch_folder: str = ""
    db_path: str = ""
    model_dir: str = ""
    embedding_model: str = ""
    vlm_model: str = ""
    vlm_enabled: bool = True
    supported_extensions: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["zh", "en"])
    chunk_size: int = 512
    chunk_overlap: int = 64
    watch_interval: int = 30
    search_alpha: float = 0.5
    max_memory_mb: int = 2048
    hf_endpoint: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build Config from a dict, applying env var overrides."""
        merged = dict(DEFAULT_CONFIG)
        merged.update(data)
        # Apply environment variable overrides.
        for key in merged:
            env_key = f"{ENV_PREFIX}{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                merged[key] = _coerce(key, env_val)
        return cls(**{k: merged[k] for k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """Validate config values. Returns list of error messages (empty = OK)."""
        errors: list[str] = []

        if not self.watch_folder:
            errors.append("watch_folder must not be empty")
        if not self.db_path:
            errors.append("db_path must not be empty")
        if not self.embedding_model:
            errors.append("embedding_model must not be empty")
        if self.chunk_size <= 0:
            errors.append(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            errors.append(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size "
                f"({self.chunk_size})"
            )
        if not self.supported_extensions:
            errors.append("supported_extensions must not be empty")
        if not self.languages:
            errors.append("languages must not be empty")
        if self.search_alpha < 0 or self.search_alpha > 1:
            errors.append(f"search_alpha must be in [0, 1], got {self.search_alpha}")
        if self.max_memory_mb < 256:
            errors.append(
                f"max_memory_mb must be >= 256 (MB), got {self.max_memory_mb}"
            )
        if self.watch_interval <= 0:
            errors.append(f"watch_interval must be > 0, got {self.watch_interval}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize config back to a plain dict (for JSON output)."""
        return {
            "watch_folder": self.watch_folder,
            "db_path": self.db_path,
            "model_dir": self.model_dir,
            "embedding_model": self.embedding_model,
            "vlm_model": self.vlm_model,
            "vlm_enabled": self.vlm_enabled,
            "supported_extensions": self.supported_extensions,
            "languages": self.languages,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "watch_interval": self.watch_interval,
            "search_alpha": self.search_alpha,
            "max_memory_mb": self.max_memory_mb,
            "hf_endpoint": self.hf_endpoint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(key: str, val: str) -> Any:
    """Coerce a string env var to the correct Python type."""
    if key in _BOOL_KEYS:
        return val.lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        return int(val)
    if key in _FLOAT_KEYS:
        return float(val)
    if key in _LIST_KEYS:
        return [item.strip() for item in val.split(",") if item.strip()]
    return val


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay onto base (overlay wins)."""
    result = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CONFIG: Config | None = None
_CONFIG_PATH: Path | None = None


def get_config_path() -> Path:
    """Return the config file path to load, checking several locations."""
    candidates = [
        # 1. Explicit env var
        Path(os.environ.get("RAG_KIT_CONFIG", "")) if os.environ.get("RAG_KIT_CONFIG") else None,
        # 2. CWD
        Path.cwd() / "rag-kit.yaml",
        # 3. User home
        _HOME / ".rag-kit.yaml",
    ]
    for p in candidates:
        if p and p.exists():
            return p
    # Fall back to the default location (even if it doesn't exist yet).
    return _HOME / ".rag-kit.yaml"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load config from YAML file, applying env var overrides.

    If no path is given, searches standard locations (env var, CWD, home).
    If no file is found, returns defaults with env overrides applied.

    Raises ValueError if validation fails.
    """
    global _CONFIG, _CONFIG_PATH

    if config_path is not None:
        path = Path(config_path)
    else:
        path = get_config_path()

    _CONFIG_PATH = path

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    merged = _deep_merge(DEFAULT_CONFIG, raw)
    config = Config.from_dict(merged)

    errors = config.validate()
    if errors:
        raise ValueError(
            f"Config validation failed ({len(errors)} errors):\n  - "
            + "\n  - ".join(errors)
        )

    _CONFIG = config
    return config


def get_config() -> Config:
    """Return the cached config, loading it if necessary."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def create_default_config(path: str | Path | None = None) -> Path:
    """Write a default config file to disk and return its path."""
    if path is not None:
        out = Path(path)
    else:
        out = _HOME / ".rag-kit.yaml"

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=True, allow_unicode=True)
    return out
