"""VLM (Vision-Language Model) captioner for visual content extraction.

Uses SmolVLM-256M-Instruct (or configurable model) to generate text
descriptions of images found in documents. These descriptions are then
fed back into the chunking + embedding pipeline as additional chunks.

Model budget: SmolVLM-256M-Instruct is ~250-500 MB (fits within 2 GB
combined budget alongside embedding model + OCR engine).

Key design:
- Lazy model loading (first caption triggers download + load, cached after)
- Language-aware prompting (zh or en based on surrounding document text)
- Graceful failure (logs error, returns empty caption if model unavailable)
- Supports PyTorch and ONNX runtime backends
- Returns chunks with metadata: vlm_generated=True, source_type="image"
"""

from __future__ import annotations

import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_kit.vlm.language import detect_language, get_vlm_prompt

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _resolve_model_path(model_name: str, cache_dir: str) -> str | None:
    """Resolve the local model directory path.

    Checks several possible cache locations:
    1. cache_dir/model_name (flat directory)
    2. cache_dir/models--org--name/ (HuggingFace hub cache)
    3. ~/lancedb-models/org/name (prior rag-kit cache)

    Returns the local path if found, None to use remote download.
    """
    from pathlib import Path

    # 1. Flat directory: cache_dir/org/name
    flat = Path(cache_dir) / model_name
    if flat.exists() and (flat / "config.json").exists():
        return str(flat)

    # 2. HuggingFace hub cache: cache_dir/models--org--name/snapshots/xxx/
    org, _, name = model_name.partition("/")
    hub_cache = Path(cache_dir) / f"models--{org}--{name}"
    if hub_cache.exists():
        snapshots = hub_cache / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if (snapshot / "config.json").exists():
                    return str(snapshot)

    # 3. Prior rag-kit cache: ~/lancedb-models/org/name
    home = Path.home()
    alt = home / "lancedb-models" / org / name
    if alt.exists() and (alt / "config.json").exists():
        return str(alt)

    # 4. HuggingFace default cache: ~/.cache/huggingface/hub/models--org--name/
    hf_cache = home / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}"
    if hf_cache.exists():
        snapshots = hf_cache / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if (snapshot / "config.json").exists():
                    return str(snapshot)

    return None


@lru_cache(maxsize=1)
def _load_vlm_pytorch(
    model_name: str,
    cache_dir: str,
) -> tuple[Any, Any]:
    """Lazy-load the VLM model using HuggingFace transformers (PyTorch backend).

    Downloads ~250-500 MB on first use. Subsequent calls are instant via lru_cache.
    Uses float32 on CPU by default. GPU auto-detected if available.

    Returns:
        Tuple of (model, processor).
    """
    import torch
    from transformers import AutoProcessor

    # Try local path first, fall back to model name + cache_dir
    local_path = _resolve_model_path(model_name, cache_dir)
    load_path = local_path or model_name

    logger.info("Loading VLM model (PyTorch): %s (from %s)", model_name, load_path)

    processor = AutoProcessor.from_pretrained(load_path)

    # SmolVLM uses AutoModelForImageTextToText (transformers >= 4.46 / 5.x).
    # transformers 5.x uses ``dtype`` (``torch_dtype`` is deprecated);
    # AutoModelForVision2Seq was REMOVED in 5.x, so only use the former.
    model = None
    last_err: Exception | None = None
    from transformers import AutoModelForImageTextToText
    for kwargs in (
        {"dtype": torch.float32},
        {"torch_dtype": torch.float32},
        {"dtype": torch.float32, "trust_remote_code": True},
    ):
        try:
            model = AutoModelForImageTextToText.from_pretrained(load_path, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - try next fallback
            last_err = exc
            logger.debug("AutoModelForImageTextToText attempt failed: %s", exc)

    if model is None:
        raise RuntimeError(
            f"Could not load VLM model {model_name} via AutoModelForImageTextToText. "
            f"Last error: {last_err}. Ensure transformers>=4.46 and the model is "
            "available (pip install -e '.[ocr]')."
        )
    model = model.to("cpu")

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("VLM moved to GPU (cuda)")

    logger.info("VLM model loaded: %s", model_name)
    return model, processor


@lru_cache(maxsize=1)
def _load_vlm_onnx(
    model_name: str,
    cache_dir: str,
) -> tuple[Any, Any]:
    """Lazy-load the VLM model using ONNX Runtime backend.

    This is used as a fallback when PyTorch weights are not available
    but ONNX weights are. SmolVLM-256M has quantized ONNX variants
    (q4) that are ~264 MB total — very memory efficient.

    Returns:
        Tuple of (model, processor) where model is an ONNX session wrapper.
    """
    import torch
    from transformers import AutoProcessor

    # Try local path first
    local_path = _resolve_model_path(model_name, cache_dir)
    load_path = local_path or model_name

    logger.info("Loading VLM model (ONNX): %s (from %s)", model_name, load_path)

    processor = AutoProcessor.from_pretrained(load_path)

    # Try to load ONNX model — via Optimum's ORTModel (the transformer-native
    # AutoModelForImageTextToText class takes no ``provider`` kwarg, so it is
    # only usable for PyTorch; ONNX requires optimum).
    try:
        from optimum.onnxruntime import ORTModelForImageTextToText
        model = ORTModelForImageTextToText.from_pretrained(
            load_path,
            provider="CPUExecutionProvider",
        )
        logger.info("VLM loaded via Optimum ORT")
        return model, processor
    except Exception as e:
        logger.debug("Optimum ORT loading failed: %s", e)

    raise RuntimeError(
        f"Could not load VLM ONNX model {model_name}. "
        f"Install optimum[onnxruntime] for ONNX support."
    )


class VLMCaptioner:
    """Vision-language model captioner for document images.

    The model is loaded lazily on the first call — the first caption may
    take 30-60 seconds for model download + loading. Subsequent captions
    are faster (~2-5s per image on CPU, <1s on GPU).

    Attributes:
        model_name: HuggingFace model ID (default: from config).
        cache_dir: Model cache directory.
        backend: "pytorch" or "onnx" (auto-detected).
    """

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        backend: str = "auto",
    ):
        # Read from config if not specified
        from rag_kit.config import get_config
        config = get_config()
        self.model_name = model_name or config.vlm_model
        self.cache_dir = cache_dir or config.model_dir
        self.backend = backend
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load the VLM model if not already loaded."""
        if self._loaded:
            return

        # Set HF_ENDPOINT if configured (for users behind firewalls)
        from rag_kit.config import get_config
        config = get_config()
        if config.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", config.hf_endpoint)

        # Try PyTorch first (unless ONNX explicitly requested)
        if self.backend in ("auto", "pytorch"):
            try:
                self._model, self._processor = _load_vlm_pytorch(
                    self.model_name, self.cache_dir,
                )
                self.backend = "pytorch"
                self._loaded = True
                return
            except Exception as e:
                logger.warning(
                    "PyTorch VLM loading failed (%s), trying ONNX fallback", e,
                )

        # Fall back to ONNX
        if self.backend in ("auto", "onnx"):
            try:
                self._model, self._processor = _load_vlm_onnx(
                    self.model_name, self.cache_dir,
                )
                self.backend = "onnx"
                self._loaded = True
                return
            except Exception as e:
                logger.error("ONNX VLM loading also failed: %s", e)

        raise RuntimeError(
            f"Could not load VLM model {self.model_name} via any backend. "
            f"Check that the model is downloaded and transformers is installed."
        )

    def caption_image(
        self,
        image_bytes: bytes,
        page_num: int = 0,
        language: str = "en",
    ) -> str:
        """Generate a text caption for a PNG image.

        Args:
            image_bytes: PNG image bytes (from page render or embedded image).
            page_num: Page number for logging (0-indexed).
            language: "zh" or "en" — determines caption prompt language.

        Returns:
            Caption text string, or empty string on failure.
        """
        try:
            from PIL import Image
            import torch

            self._ensure_loaded()

            image = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")

            prompt = get_vlm_prompt(language)

            # Build chat messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template — use tokenizer if processor doesn't have one
            # (SmolVLM/Idefics3: chat_template lives on tokenizer, not processor)
            if hasattr(self._processor, "apply_chat_template"):
                try:
                    text = self._processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except (ValueError, RuntimeError):
                    text = self._processor.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
            elif hasattr(self._processor, "tokenizer"):
                text = self._processor.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                raise RuntimeError("No chat template available on processor or tokenizer")
            inputs = self._processor(
                text=[text], images=[image], return_tensors="pt",
            )

            # Move inputs to model device
            if hasattr(self._model, "device"):
                inputs = {
                    k: v.to(self._model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=1.0,
                )

            # Decode only the new tokens (skip the input prompt)
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated = output[0][input_len:]
            caption = self._processor.decode(
                generated, skip_special_tokens=True,
            ).strip()

            logger.info(
                "VLM caption for page %d (%s): %s",
                page_num + 1, language, caption[:120],
            )
            return caption

        except Exception as e:
            logger.error(
                "VLM captioning failed for page %d: %s", page_num + 1, e,
            )
            return ""

    def caption_regions(
        self,
        regions: list,  # list[ImageRegion]
        language: str = "en",
    ) -> list[dict[str, Any]]:
        """Caption multiple image regions and return chunk dicts.

        This is the main entry point for the ingestion pipeline.
        Each caption becomes a chunk with vlm_generated and source_type metadata.

        Args:
            regions: List of ImageRegion objects from extract_images_from_pdf/docx.
            language: "zh" or "en" for caption language.

        Returns:
            List of chunk dicts:
            {
                "text": "[Visual content on page 3: bar chart showing...]",
                "page": 2,
                "vlm_generated": True,
                "source_type": "image",
                "source_file": "...",  # set by caller
                "extra_metadata": {...},  # from ImageRegion
            }
        """
        if not regions:
            return []

        chunks: list[dict[str, Any]] = []

        for region in regions:
            caption = self.caption_image(
                region.image_bytes,
                page_num=region.page_num,
                language=language,
            )

            if caption:
                # Format the caption as a searchable chunk
                page_display = region.page_num + 1
                type_label = {
                    "page_render": "page",
                    "embedded_image": "image",
                    "table": "table",
                    "chart": "chart",
                }.get(region.source_type, "visual content")

                chunk_text = (
                    f"[Visual content — {type_label} on page {page_display}: "
                    f"{caption}]"
                )

                chunk = {
                    "text": chunk_text,
                    "page": region.page_num,
                    "vlm_generated": True,
                    "source_type": "image",
                    "extra_metadata": {
                        "region_type": region.source_type,
                        "region_width": region.width,
                        "region_height": region.height,
                        **region.metadata,
                    },
                }
                chunks.append(chunk)
            else:
                logger.warning(
                    "Empty VLM caption for page %d (region type: %s)",
                    region.page_num + 1, region.source_type,
                )

        logger.info(
            "VLM captioned %d/%d regions (%s)",
            len(chunks), len(regions), language,
        )
        return chunks

    def is_available(self) -> bool:
        """Check if the VLM model can be loaded (without actually loading it).

        Returns True if either PyTorch or ONNX backend is available
        and the model files exist in the cache directory.
        """
        try:
            import transformers  # noqa: F401
            import PIL  # noqa: F401
        except ImportError:
            return False

        # Check if model directory exists
        from pathlib import Path
        model_path = Path(self.cache_dir) / self.model_name
        if model_path.exists():
            return True

        # Check HuggingFace cache structure
        # Models are cached as: cache_dir/models--<org>--<name>/
        org, _, name = self.model_name.partition("/")
        hf_cache = Path(self.cache_dir) / f"models--{org}--{name}"
        if hf_cache.exists():
            return True

        # Also check lancedb-models cache (from prior rag-kit installation)
        home = Path.home()
        alt_cache = home / "lancedb-models" / org / name
        if alt_cache.exists():
            return True

        return False


def caption_document(
    file_path: str,
    language: str | None = None,
    vlm_model: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """High-level: extract and caption all visual content from a document.

    This is the main entry point for the ingestion pipeline.
    Given a file path, extracts images, runs VLM captioning, and returns
    chunk dicts with vlm_generated=True and source_type="image" metadata.

    Args:
        file_path: Path to a PDF or DOCX file.
        language: Override language detection ("zh" or "en"). If None,
            auto-detects from surrounding text in the document.
        vlm_model: Override VLM model name.
        cache_dir: Override model cache directory.

    Returns:
        List of chunk dicts ready for embedding and storage.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _caption_pdf(file_path, language, vlm_model, cache_dir)
    elif ext == ".docx":
        return _caption_docx(file_path, language, vlm_model, cache_dir)
    else:
        logger.debug("VLM captioning not supported for %s files", ext)
        return []


def _caption_pdf(
    file_path: str,
    language: str | None,
    vlm_model: str | None,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    """Extract and caption visual content from a PDF."""
    import fitz

    from rag_kit.vlm.extractor import extract_images_from_pdf

    doc = fitz.open(file_path)

    # Auto-detect language from document text if not specified
    if language is None:
        sample_text = ""
        for page in doc:
            text = page.get_text()
            sample_text += text
            if len(sample_text) > 2000:
                break
        language = detect_language(sample_text)

    regions = extract_images_from_pdf(doc)
    doc.close()

    if not regions:
        logger.info("No visual regions found in %s", file_path)
        return []

    logger.info(
        "Found %d visual regions in %s (language: %s)",
        len(regions), Path(file_path).name, language,
    )

    captioner = VLMCaptioner(model_name=vlm_model, cache_dir=cache_dir)
    return captioner.caption_regions(regions, language=language)


def _caption_docx(
    file_path: str,
    language: str | None,
    vlm_model: str | None,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    """Extract and caption visual content from a DOCX file."""
    from rag_kit.vlm.extractor import extract_images_from_docx

    # Auto-detect language from document text
    if language is None:
        try:
            from docx import Document
            doc = Document(file_path)
            sample_text = " ".join(p.text for p in doc.paragraphs[:50])
            language = detect_language(sample_text)
        except Exception:
            language = "en"

    regions = extract_images_from_docx(file_path)
    if not regions:
        logger.info("No embedded images found in %s", file_path)
        return []

    logger.info(
        "Found %d embedded images in %s (language: %s)",
        len(regions), Path(file_path).name, language,
    )

    captioner = VLMCaptioner(model_name=vlm_model, cache_dir=cache_dir)
    return captioner.caption_regions(regions, language=language)
