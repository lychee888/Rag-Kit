"""VLM sub-package — vision-language model for visual content extraction.

When the ingestion pipeline encounters images, charts, diagrams, or tables
embedded in PDFs/DOCX files, this module extracts those visual regions and
passes them to a lightweight VLM (HuggingFaceTB/SmolVLM-256M-Instruct) to generate
textual descriptions. These descriptions are then fed back into the
chunking + embedding pipeline as additional chunks with metadata flagging
them as vlm_generated=True and source_type="image".

Public API:
    VLMCaptioner       — lazy-loaded VLM model, generates captions for images
    caption_document   — high-level: extract + caption all visuals from a file
    extract_images_from_pdf  — extract visual regions from a PDF
    extract_images_from_docx — extract embedded images from a DOCX
    detect_image_pages       — find pages with significant image content
    detect_language          — detect zh vs en from surrounding text
"""

from rag_kit.vlm.captioner import (
    VLMCaptioner,
    caption_document,
)
from rag_kit.vlm.extractor import (
    ImageRegion,
    detect_image_pages,
    extract_images_from_docx,
    extract_images_from_pdf,
)
from rag_kit.vlm.language import detect_language, get_vlm_prompt

__all__ = [
    "VLMCaptioner",
    "caption_document",
    "extract_images_from_pdf",
    "extract_images_from_docx",
    "detect_image_pages",
    "detect_language",
    "get_vlm_prompt",
    "ImageRegion",
]
