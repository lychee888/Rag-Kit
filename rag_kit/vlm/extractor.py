"""Image extraction from PDFs and DOCX files.

Extracts visual regions (charts, diagrams, tables, embedded images)
from documents and returns them as PIL Images ready for VLM captioning.

Uses pymupdf (fitz) for PDFs and python-docx for DOCX files.
Compatible with pymupdf 1.28+ where get_image_rects() requires an xref argument.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum image area ratio (5% of page) to consider a page for VLM
_MIN_IMAGE_AREA_RATIO = 0.05

# Minimum vector drawing count to consider a page as having visual content
# (charts/diagrams drawn with PDF vector primitives)
_MIN_VECTOR_DRAWINGS = 5

# Minimum drawing area ratio (3% of page) to consider as visual content
_MIN_DRAWING_AREA_RATIO = 0.03

# DPI for rendering full pages to PNG when captioning
_RENDER_DPI = 150


class VisualExtractionError(RuntimeError):
    """Known visual content could not be completely extracted."""


@dataclass
class ImageRegion:
    """A visual region extracted from a document, ready for VLM captioning.

    Attributes:
        image_bytes: PNG bytes of the visual content.
        page_num: 0-indexed page number in the source document.
        source_type: What kind of visual content this represents.
            One of: "page_render", "embedded_image", "table", "chart".
        width: Image width in pixels.
        height: Image height in pixels.
        metadata: Extra metadata for the chunk (page, source_file, etc.).
    """

    image_bytes: bytes
    page_num: int
    source_type: str = "page_render"
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_images_from_pdf(
    doc: Any,
    max_area_ratio: float = 1.0,
    render_dpi: int = _RENDER_DPI,
) -> list[ImageRegion]:
    """Extract visual regions from a pymupdf PDF document.

    Three strategies:
    1. For pages with significant embedded raster images (≥5% area),
       render the entire page to PNG and caption it.
    2. For pages with many vector drawings (charts, diagrams, flow charts
       drawn with PDF primitives), render the page and caption it.
    3. For large standalone embedded images, also extract them individually.

    Args:
        doc: pymupdf.Document object (open).
        max_area_ratio: Maximum image area ratio threshold (default: 1.0 = allow all).
        render_dpi: DPI for page rendering (default: 150).

    Returns:
        List of ImageRegion objects, one per visual region found.
    """
    regions: list[ImageRegion] = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc):
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            continue

        has_visual_content = False
        area_ratio = 0.0

        # ── Strategy 1: Check for embedded raster images ────────────
        images = page.get_images(full=True)
        has_image_masks = any(
            (len(item) > 1 and item[1])
            or doc.xref_get_key(item[0], 'ImageMask') == ('bool', 'true')
            or doc.xref_get_key(item[0], 'Mask')[0] != 'null'
            for item in images)
        # Inline images have no extractable XObject. They still require a
        # whole-page render and must not disappear behind a partial fallback.
        visible_images = page.get_image_info(xrefs=True)
        has_inline_images = any(not item.get('xref') for item in visible_images)
        required_xrefs = {item['xref'] for item in visible_images if item.get('xref')}
        has_annotations = (next(page.annots(), None) is not None
                           or next(page.widgets(), None) is not None)
        image_rects_by_xref: dict[int, list[Any]] = {}
        total_image_area = sum(
            max(0, item['bbox'][2] - item['bbox'][0]) * max(0, item['bbox'][3] - item['bbox'][1])
            for item in visible_images if not item.get('xref'))
        if images:
            for img_info in images:
                xref = img_info[0] if len(img_info) > 0 else 0
                try:
                    # pymupdf 1.28+: get_image_rects requires xref argument
                    rects = page.get_image_rects(xref)
                except (TypeError, ValueError):
                    # Older pymupdf versions: call without xref
                    rects = page.get_image_rects()
                except Exception as exc:
                    raise VisualExtractionError(f"Could not locate PDF image {xref} on page {page_num + 1}") from exc

                if rects:
                    image_rects_by_xref[xref] = rects
                    for r in rects:
                        if r and r.width > 0 and r.height > 0:
                            total_image_area += r.width * r.height

        area_ratio = total_image_area / page_area

        if area_ratio >= _MIN_IMAGE_AREA_RATIO:
            has_visual_content = True

        # ── Strategy 2: Check for vector drawings (charts/diagrams) ─
        try:
            drawings = page.get_drawings()
            meaningful = [d for d in drawings if d.get("type") in ("f", "s", "fs")]
            # Check vectors even on raster pages: embedded-image fallback
            # cannot recover a chart that was drawn on top of those images.
            draw_area = 0.0
            for d in drawings:
                rect = d.get("rect")
                if rect and rect.width > 0 and rect.height > 0:
                    draw_area += rect.width * rect.height
            draw_area_ratio = min(draw_area / page_area, 1.0)

            if (len(meaningful) >= _MIN_VECTOR_DRAWINGS or
                    draw_area_ratio >= _MIN_DRAWING_AREA_RATIO):
                if not has_visual_content:
                    has_visual_content = True
                    area_ratio = draw_area_ratio
        except Exception as exc:
            raise VisualExtractionError(f"Could not inspect PDF drawings on page {page_num + 1}") from exc

        if not has_visual_content:
            continue

        # ── Render the page to PNG for VLM captioning ───────────────
        page_rendered = False
        render_error = None
        if area_ratio <= max_area_ratio or max_area_ratio >= 1.0:
            try:
                pix = page.get_pixmap(dpi=render_dpi)
                img_bytes = pix.tobytes("png")
                regions.append(ImageRegion(
                    image_bytes=img_bytes,
                    page_num=page_num,
                    source_type="page_render",
                    width=pix.width,
                    height=pix.height,
                    metadata={
                        "area_ratio": round(area_ratio, 3),
                        "render_dpi": render_dpi,
                        "detection": "raster" if images else "vector",
                    },
                ))
                page_rendered = True
            except Exception as e:
                render_error = e
                logger.warning("Failed to render page %d: %s", page_num + 1, e)

        # Only also extract standalone images if the whole page was NOT
        # rendered — otherwise we'd caption the same content twice.
        if not page_rendered:
            fallback_images = 0
            for xref, rects in image_rects_by_xref.items():
                if xref in seen_xrefs:
                    fallback_images += 1
                    continue
                for r in rects:
                    if r and r.width > 0 and r.height > 0:
                        img_area = r.width * r.height
                        # Only extract standalone if it covers >10% of the page
                        if render_error is not None or img_area / page_area > 0.10:
                            try:
                                extracted = doc.extract_image(xref)
                                if not extracted or not extracted.get("image"):
                                    raise VisualExtractionError(f"PDF image {xref} has no image data")
                                if extracted and "image" in extracted:
                                    regions.append(ImageRegion(
                                        image_bytes=extracted["image"],
                                        page_num=page_num,
                                        source_type="embedded_image",
                                        width=extracted.get("width", 0),
                                        height=extracted.get("height", 0),
                                        metadata={
                                            "xref": xref,
                                            "ext": extracted.get("ext", ""),
                                        },
                                    ))
                                    seen_xrefs.add(xref)
                                    fallback_images += 1
                                    break  # One extraction per xref
                            except Exception as e:
                                raise VisualExtractionError(
                                    f"Could not extract PDF image {xref} on page {page_num + 1}"
                                ) from e
            if render_error is not None and (
                    drawings or has_inline_images or has_annotations or has_image_masks
                    or not required_xrefs.issubset(seen_xrefs) or not fallback_images):
                raise VisualExtractionError(f"Could not recover visual content on PDF page {page_num + 1}") from render_error

    return regions


def extract_images_from_docx(file_path: str) -> list[ImageRegion]:
    """Extract embedded images from a .docx file.

    DOCX files store images as separate parts in the zip archive.
    This extracts all image parts and returns them as ImageRegions.

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of ImageRegion objects, one per embedded image.
    """
    regions: list[ImageRegion] = []

    try:
        from docx import Document
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
    except ImportError:
        raise

    try:
        doc = Document(file_path)
        # Access the document part to get embedded images
        for rel in doc.part.rels.values():
            if rel.reltype == RT.IMAGE:
                try:
                    image_part = rel.target_part
                    image_bytes = image_part.blob
                    # Determine page/section number from paragraph position
                    # For now, just use a sequential index
                    page_num = len(regions)
                    regions.append(ImageRegion(
                        image_bytes=image_bytes,
                        page_num=page_num,
                        source_type="embedded_image",
                        width=0,  # Unknown without parsing
                        height=0,
                        metadata={
                            "rel_id": rel.rId,
                            "content_type": image_part.content_type,
                        },
                    ))
                except Exception as e:
                    raise VisualExtractionError(f"Failed to extract DOCX image {rel.rId}") from e
    except Exception as e:
        raise VisualExtractionError(f"Incomplete DOCX image extraction: {e}") from e

    return regions


def detect_image_pages(
    doc: Any,
    max_area_ratio: float = 1.0,
) -> list[int]:
    """Find pages in a pymupdf document that contain significant visual content.

    Detects both raster images and vector drawings (charts, diagrams).
    This is a lightweight version of extract_images_from_pdf that only
    returns page indices, for use by callers that want to decide whether
    to render pages themselves.

    Args:
        doc: pymupdf.Document object.
        max_area_ratio: Maximum image area ratio threshold.

    Returns:
        List of 0-indexed page numbers with significant visual content.
    """
    image_pages: list[int] = []

    for page_num, page in enumerate(doc):
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            continue

        has_visual = False

        # Check for raster images
        images = page.get_images(full=True)
        if images:
            total_image_area = 0.0
            for img_info in images:
                xref = img_info[0] if len(img_info) > 0 else 0
                try:
                    rects = page.get_image_rects(xref)
                except (TypeError, ValueError):
                    rects = page.get_image_rects()
                except Exception:
                    rects = []

                for r in rects:
                    if r and r.width > 0 and r.height > 0:
                        total_image_area += r.width * r.height

            area_ratio = total_image_area / page_area if page_area > 0 else 0
            if area_ratio >= _MIN_IMAGE_AREA_RATIO:
                if area_ratio <= max_area_ratio or max_area_ratio >= 1.0:
                    has_visual = True

        # Check for vector drawings (charts, diagrams)
        if not has_visual:
            try:
                drawings = page.get_drawings()
                meaningful = [d for d in drawings if d.get("type") in ("f", "s", "fs")]
                draw_area = 0.0
                for d in drawings:
                    rect = d.get("rect")
                    if rect and rect.width > 0 and rect.height > 0:
                        draw_area += rect.width * rect.height
                draw_area_ratio = min(draw_area / page_area, 1.0) if page_area > 0 else 0

                if (len(meaningful) >= _MIN_VECTOR_DRAWINGS or
                        draw_area_ratio >= _MIN_DRAWING_AREA_RATIO):
                    has_visual = True
            except Exception:
                pass

        if has_visual:
            image_pages.append(page_num)

    return image_pages
