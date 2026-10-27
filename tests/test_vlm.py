"""Tests for the VLM (Vision-Language Model) sub-package.

Tests cover:
- Language detection (zh vs en)
- Image extraction from PDFs (vector graphics + raster images)
- Cache detection consistency (audit F8: is_available agrees with loader)
- Real-VLM captioning: opt-in integration lane (marker `vlm_integration` or
  RAG_KIT_VLM_INTEGRATION=1); RAG_KIT_VLM_REQUIRED=1 turns a missing
  prerequisite into a loud failure instead of a skip
- Chunk metadata (vlm_generated, source_type)
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _vlm_cache_dir() -> str:
    """Model cache dir honored by the lane (mirrors rag-kit defaults)."""
    return os.environ.get("RAG_KIT_MODEL_DIR", str(Path.home() / "models"))


def _vlm_caption_prereq(request) -> str:
    """Gate for the opt-in real-captioning lane; honors REQUIRED mode.

    Returns the cache dir when the lane may run. Skips when the lane was not
    opted into or (non-required mode) the model is absent; FAILS (not skips)
    when RAG_KIT_VLM_REQUIRED=1 and prerequisites are missing. Called from
    the test body so a missing prereq surfaces as a FAILED test, not an
    ERROR from fixture setup.
    """
    markexpr = getattr(request.config.option, "markexpr", "") or ""
    opted_in = (
        os.environ.get("RAG_KIT_VLM_INTEGRATION") == "1"
        or "vlm_integration" in markexpr
    )
    if not opted_in:
        pytest.skip(
            "real-VLM integration lane is opt-in: run with "
            "`-m vlm_integration` or RAG_KIT_VLM_INTEGRATION=1"
        )

    from rag_kit.vlm import VLMCaptioner

    cache_dir = _vlm_cache_dir()
    captioner = VLMCaptioner(model_name=VLM_MODEL, cache_dir=cache_dir)
    if not captioner.is_available():
        detail = (
            f"SmolVLM weights not found under cache dir {cache_dir!r} "
            f"(expected models--HuggingFaceTB--SmolVLM-256M-Instruct/"
            f"snapshots/*/model.safetensors); pass RAG_KIT_MODEL_DIR to "
            f"point at a populated cache"
        )
        if os.environ.get("RAG_KIT_VLM_REQUIRED") == "1":
            pytest.fail(f"REQUIRED real-VLM integration lane: {detail}")
        pytest.skip(detail)
    return cache_dir


class TestLanguageDetection:
    """Tests for language detection module."""

    def test_english_detection(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("The quick brown fox jumps over the lazy dog") == "en"

    def test_chinese_detection(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("这是一份关于第三季度销售收入的报告") == "zh"

    def test_mixed_en_dominant(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("The revenue is 50M in Q3 第三季度") == "en"

    def test_mixed_zh_dominant(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("第三季度收入为50M，这是我们的季度报告") == "zh"

    def test_empty_string(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("") == "en"

    def test_numbers_only(self):
        from rag_kit.vlm.language import detect_language
        assert detect_language("1234567890") == "en"

    def test_get_vlm_prompt_en(self):
        from rag_kit.vlm.language import get_vlm_prompt
        prompt = get_vlm_prompt("en")
        assert "describe" in prompt.lower()
        assert "charts" in prompt.lower()

    def test_get_vlm_prompt_zh(self):
        from rag_kit.vlm.language import get_vlm_prompt
        prompt = get_vlm_prompt("zh")
        assert "描述" in prompt
        assert "图表" in prompt


class TestImageExtraction:
    """Tests for image extraction from PDFs."""

    @pytest.fixture
    def test_pdf(self):
        """Create a test PDF with vector graphics (bar chart + diagram)."""
        from tests.create_test_pdf import create_test_pdf
        pdf_path = PROJECT_ROOT / "tests" / "test_visual.pdf"
        if not pdf_path.exists():
            create_test_pdf(str(pdf_path))
        return str(pdf_path)

    def test_detect_image_pages(self, test_pdf):
        """detect_image_pages should find pages with visual content."""
        import fitz
        from rag_kit.vlm import detect_image_pages

        doc = fitz.open(test_pdf)
        pages = detect_image_pages(doc)
        doc.close()

        assert len(pages) >= 2, f"Expected ≥2 image pages, got {pages}"
        # Pages are 0-indexed
        assert 0 in pages, "Page 1 (bar chart) should be detected"
        assert 1 in pages, "Page 2 (flow diagram) should be detected"

    def test_extract_images_from_pdf(self, test_pdf):
        """extract_images_from_pdf should return ImageRegion objects."""
        import fitz
        from rag_kit.vlm import extract_images_from_pdf, ImageRegion

        doc = fitz.open(test_pdf)
        regions = extract_images_from_pdf(doc)
        doc.close()

        assert len(regions) >= 2, f"Expected ≥2 regions, got {len(regions)}"
        for region in regions:
            assert isinstance(region, ImageRegion)
            assert len(region.image_bytes) > 0, "Image bytes should not be empty"
            assert region.page_num >= 0
            assert region.source_type in ("page_render", "embedded_image")
            assert region.width > 0
            assert region.height > 0

    def test_pymupdf_128_compatibility(self, test_pdf):
        """get_image_rects should work with pymupdf 1.28+ (xref parameter)."""
        import fitz
        from rag_kit.vlm.extractor import _MIN_VECTOR_DRAWINGS

        doc = fitz.open(test_pdf)
        page = doc[0]  # First page (bar chart)

        # Check that get_drawings works (for vector graphics detection)
        drawings = page.get_drawings()
        assert len(drawings) > 0, "Should have vector drawings on bar chart page"

        # Check that get_image_rects accepts xref (pymupdf 1.28+ API)
        images = page.get_images(full=True)
        if images:
            xref = images[0][0]
            try:
                rects = page.get_image_rects(xref)
                # Should not raise TypeError
            except TypeError as e:
                if "required positional argument" in str(e):
                    pytest.fail(f"get_image_rects requires xref but code didn't pass it: {e}")

        doc.close()


class TestVLMCaptioning:
    """Real-captioning integration lane (requires the SmolVLM model).

    Opt-in two ways (either enables the lane):
      - ``pytest -m vlm_integration``
      - ``RAG_KIT_VLM_INTEGRATION=1 pytest``

    With ``RAG_KIT_VLM_REQUIRED=1`` a missing prerequisite FAILS loudly
    instead of skipping, so broken visual ingestion can never hide behind
    permanent skips again (audit F8). The default suite stays light: with
    no opt-in the lane skips deterministically.
    """

    @pytest.fixture
    def test_pdf(self):
        from tests.create_test_pdf import create_test_pdf
        pdf_path = PROJECT_ROOT / "tests" / "test_visual.pdf"
        if not pdf_path.exists():
            create_test_pdf(str(pdf_path))
        return str(pdf_path)

    def test_captioner_availability(self):
        """is_available() must agree with where the loader actually looks."""
        from rag_kit.vlm import VLMCaptioner
        from rag_kit.vlm.captioner import _resolve_model_path

        captioner = VLMCaptioner(model_name=VLM_MODEL, cache_dir=_vlm_cache_dir())
        result = captioner.is_available()
        assert isinstance(result, bool)
        # F8: detection and load-path resolution are the same computation.
        assert result == (
            _resolve_model_path(VLM_MODEL, str(captioner.cache_dir)) is not None
        )

    def test_captioner_unavailable_on_empty_cache(self, tmp_path):
        """An empty cache dir must report unavailable to BOTH detector and
        resolver, and a config.json-only snapshot (no weights) must not
        count as available (audit F8: no half-populated cache false-positives).
        """
        from rag_kit.vlm import VLMCaptioner
        from rag_kit.vlm.captioner import _resolve_model_path

        empty = tmp_path / "models-empty"
        captioner = VLMCaptioner(model_name=VLM_MODEL, cache_dir=str(empty))
        assert captioner.is_available() is False
        assert _resolve_model_path(VLM_MODEL, str(empty)) is None

        # Snapshot dir with config.json but no weights -> still unavailable.
        snap = empty / f"models--HuggingFaceTB--SmolVLM-256M-Instruct" / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        assert captioner.is_available() is False
        assert _resolve_model_path(VLM_MODEL, str(empty)) is None

    @pytest.mark.vlm_integration
    def test_caption_document(self, test_pdf, request):
        """Full pipeline: extract + caption all visuals from test PDF."""
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from rag_kit.vlm import caption_document

        chunks = caption_document(test_pdf, cache_dir=_vlm_caption_prereq(request))

        assert len(chunks) >= 2, f"Expected ≥2 VLM chunks, got {len(chunks)}"

        for chunk in chunks:
            # Verify required metadata
            assert chunk["vlm_generated"] is True, "vlm_generated must be True"
            assert chunk["source_type"] == "image", "source_type must be 'image'"
            assert "text" in chunk and len(chunk["text"]) > 0, "Chunk must have text"
            assert "page" in chunk and chunk["page"] >= 0, "Chunk must have page number"
            assert "[Visual content" in chunk["text"], "Text should be formatted as visual content"

    @pytest.mark.vlm_integration
    def test_caption_coherence(self, test_pdf, request):
        """VLM captions should be coherent (contain relevant keywords)."""
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from rag_kit.vlm import caption_document

        chunks = caption_document(test_pdf, cache_dir=_vlm_caption_prereq(request))
        assert len(chunks) >= 2

        # Page 1 (bar chart) caption should mention chart/revenue/bar
        page1_text = next((c["text"] for c in chunks if c["page"] == 0), "")
        page1_lower = page1_text.lower()
        assert any(kw in page1_lower for kw in ["chart", "bar", "revenue", "quarter"]), \
            f"Page 1 caption should mention chart/revenue/bar: {page1_text[:200]}"

        # Page 2 (flow diagram) caption should mention diagram/flow/system
        page2_text = next((c["text"] for c in chunks if c["page"] == 1), "")
        page2_lower = page2_text.lower()
        assert any(kw in page2_lower for kw in ["diagram", "flow", "system", "architecture"]), \
            f"Page 2 caption should mention diagram/flow/system: {page2_text[:200]}"


class TestImportStructure:
    """Test that all public API exports are importable."""

    def test_imports(self):
        from rag_kit.vlm import (
            VLMCaptioner,
            caption_document,
            extract_images_from_pdf,
            extract_images_from_docx,
            detect_image_pages,
            detect_language,
            get_vlm_prompt,
            ImageRegion,
        )
        # All should be callable/non-None
        assert callable(caption_document)
        assert callable(extract_images_from_pdf)
        assert callable(extract_images_from_docx)
        assert callable(detect_image_pages)
        assert callable(detect_language)
        assert callable(get_vlm_prompt)
        assert VLMCaptioner is not None
        assert ImageRegion is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
