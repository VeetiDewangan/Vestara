"""
End-to-end clothing background removal orchestrator  (Section 7 - Part 4).

Chains the Section 2-6 service implementations into a single pipeline call.
No logic is duplicated; every step delegates to the original service.

Pipeline:
    raw bytes  (from HTTP upload)
        |
        v  process_upload()          -- Section 2 (validate + persist)
        |
        v  preprocess_image_bytes()  -- Section 3 (decode, normalise, resize)
        |
        v  SegmentationService       -- Section 4 (U2-Net via rembg)
        |
        v  analyze_mask()            -- Section 5 (statistics, no mutation)
        |
        v  refine_mask()             -- Section 6 (morphological cleanup)
        |
        v  compose_rgba()            -- Section 7 (RGB + alpha = RGBA)
        |
        v  crop_to_content()         -- Section 7 (optional tight crop)
        |
        v  save_png()                -- Section 7 (write to output_images/)
        |
        v  PipelineResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from app.services.background_removal import (
    BackgroundRemovalResult,
    compose_rgba,
    crop_to_content,
    save_png,
)
from app.services.image_processing import FileTooLargeError, process_upload
from app.services.mask_analysis import MaskAnalysisResult, analyze_mask
from app.services.mask_refinement import RefinedMaskResult, refine_mask
from app.services.preprocessing import PreprocessedImage, preprocess_image_bytes
from app.services.segmentation import SegmentationService


# ---------------------------------------------------------------------------
# Shared singleton segmentation service
# (loaded lazily; model weights are ~176 MB)
# ---------------------------------------------------------------------------

_seg_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """Return the shared SegmentationService instance, creating it on demand."""
    global _seg_service
    if _seg_service is None:
        _seg_service = SegmentationService()
    return _seg_service


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Full result of the clothing background-removal pipeline.

    Only fields safe to expose through the API are marked 'api_safe'.
    The output_path field must never appear in API responses.
    """
    # api_safe
    filename: str           # output PNG basename
    width: int              # final image width (after crop)
    height: int             # final image height (after crop)
    original_width: int     # width before crop
    original_height: int    # height before crop
    processing_time_s: float

    # api_safe summary stats
    raw_foreground_pct: float
    refined_foreground_pct: float
    components_removed: int
    holes_filled: int

    # internal only - not returned to client
    output_path: str        # absolute path; do NOT expose via API


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    data: bytes,
    original_filename: str,
    apply_crop: bool = True,
) -> PipelineResult:
    """
    Execute the complete background-removal pipeline on uploaded image bytes.

    Args:
        data:              Raw image bytes from the HTTP upload.
        original_filename: Client-supplied filename (used only for extension
                           selection when writing to input_images/).
        apply_crop:        Whether to apply tight crop after composition.

    Returns:
        PipelineResult with API-safe fields and the internal output path.

    Raises:
        FileTooLargeError: if bytes exceed MAX_UPLOAD_SIZE.
        ValueError:        for invalid/corrupt/unsupported images.
    """
    t_start = time.perf_counter()

    # --- Step 1: validate + persist upload (Section 2) --------------------
    process_upload(data, original_filename)

    # --- Step 2: preprocess (Section 3) -----------------------------------
    preprocessed: PreprocessedImage = preprocess_image_bytes(data)
    orig_h, orig_w = preprocessed.image.shape[:2]

    # --- Step 3: U2-Net segmentation (Section 4) --------------------------
    svc = get_segmentation_service()
    seg_result = svc.segment(preprocessed.image)
    raw_mask = seg_result.mask

    # --- Step 4: raw mask analysis (Section 5) ----------------------------
    raw_analysis: MaskAnalysisResult = analyze_mask(raw_mask)

    # --- Step 5: mask refinement (Section 6) ------------------------------
    refined: RefinedMaskResult = refine_mask(raw_mask)
    refined_mask = refined.mask

    # --- Step 6: alpha composition (Section 7) ----------------------------
    rgba = compose_rgba(preprocessed.image, refined_mask)

    # --- Step 7: optional tight crop (Section 7) --------------------------
    if apply_crop:
        rgba = crop_to_content(rgba)

    final_h, final_w = rgba.shape[:2]

    # --- Step 8: save PNG (Section 7) -------------------------------------
    save_result: BackgroundRemovalResult = save_png(rgba)

    t_end = time.perf_counter()

    return PipelineResult(
        filename=save_result.filename,
        width=final_w,
        height=final_h,
        original_width=orig_w,
        original_height=orig_h,
        processing_time_s=round(t_end - t_start, 3),
        raw_foreground_pct=raw_analysis.foreground_percentage,
        refined_foreground_pct=refined.refined_foreground_percentage,
        components_removed=refined.components_removed,
        holes_filled=refined.holes_filled,
        output_path=save_result.output_path,
    )
