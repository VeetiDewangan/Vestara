"""
Mask refinement service  (Section 6).

Applies conservative OpenCV-based post-processing to the raw soft mask
produced by Section 4 (U2-Net).  The goal is to remove small isolated
noise, fill small holes, and smooth minor irregularities while preserving
clothing edges and soft foreground confidence values.

Key design decisions
--------------------
1. The RAW mask is NEVER modified.
   All operations begin from an internal copy.

2. Soft-mask preservation strategy:
   a. Binarise the raw mask with MASK_REFINEMENT_THRESHOLD -> binary_mask.
   b. Apply morphological opening, closing, small-component removal, and
      small-hole filling ONLY to binary_mask.
   c. Use the resulting clean_binary as a gate over the original soft values:
        refined_soft = raw_mask * clean_binary
   The output is still a uint8 soft mask, NOT a flat 0/255 image.

3. Conservative kernel sizes (3x3 open, 5x5 close) are used by default.
   Clothing edges (collars, straps, sleeves) are easily eroded by large kernels.

4. Small-component removal uses an area threshold (MASK_MIN_COMPONENT_AREA).
   It removes obvious speckle but never removes the largest foreground region.

5. Small-hole filling targets isolated background blobs enclosed inside
   foreground, bounded by MASK_MAX_HOLE_AREA.  The actual photo background
   is too large to be accidentally filled.

Known limitations
-----------------
- Morphological refinement operates on the binary representation of U2-Net's
  output.  It cannot understand clothing semantics.
- A fundamentally incorrect raw mask (wrong object, wrong region) will remain
  wrong after refinement.
- Refinement improves quality for masks that are mostly correct; it is not a
  substitute for a clothing-specific segmentation model.

Pipeline position:
    SegmentationResult.mask  (H x W uint8, soft values 0-255)
        |
        v
    refine_mask()
        |
        v
    RefinedMaskResult  (H x W uint8, soft values, raw mask unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import (
    MASK_REFINEMENT_THRESHOLD,
    MASK_OPENING_KERNEL_SIZE,
    MASK_CLOSING_KERNEL_SIZE,
    MASK_MIN_COMPONENT_AREA,
    MASK_MAX_HOLE_AREA,
)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class RefinedMaskResult:
    """
    Output of the mask refinement pipeline.

    Attributes
    ----------
    mask :
        Refined mask.  Same shape (H x W) and dtype (uint8) as the input.
        Soft values from the original mask are preserved where the refined
        binary foreground retained the pixel; removed regions are set to 0.

    raw_foreground_percentage :
        Foreground percentage of the raw input mask (pixels >= threshold /
        total pixels * 100).

    refined_foreground_percentage :
        Foreground percentage of the refined mask (same calculation).

    components_removed :
        Number of small connected foreground components that were discarded
        during the small-component removal step.

    holes_filled :
        Number of small background holes that were filled.
    """
    mask: np.ndarray
    raw_foreground_percentage: float
    refined_foreground_percentage: float
    components_removed: int
    holes_filled: int


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_mask(mask: np.ndarray) -> None:
    """
    Ensure the mask is a 2-D uint8 NumPy array with non-zero dimensions.

    Raises:
        ValueError: with a clear description of what was wrong.
    """
    if mask is None:
        raise ValueError(
            "mask must not be None. "
            "Pass the H x W uint8 ndarray from SegmentationResult.mask."
        )
    if not isinstance(mask, np.ndarray):
        raise ValueError(
            f"mask must be a NumPy ndarray, got {type(mask).__name__}."
        )
    if mask.dtype != np.uint8:
        raise ValueError(
            f"mask dtype must be uint8, got {mask.dtype}. "
            "The segmentation service always returns uint8 masks."
        )
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-dimensional (H x W), got shape {mask.shape}. "
            "Pass the raw 2-D mask, not an RGB or RGBA image."
        )
    if mask.size == 0:
        raise ValueError(
            f"mask must not be empty (zero elements). Got shape {mask.shape}."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_binary_mask(
    mask: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """
    Binarise a soft mask at the given threshold.

    Pixels >= threshold -> 255  (foreground)
    Pixels <  threshold -> 0    (background)

    Returns a new uint8 array; the input is unchanged.
    """
    return np.where(mask >= threshold, np.uint8(255), np.uint8(0))


def _apply_opening(
    binary: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Apply morphological OPENING (erosion -> dilation) to remove small noise.

    A small kernel (default 3 x 3) removes isolated foreground specks without
    significantly eroding clothing boundaries.

    Args:
        binary:      H x W uint8 binary mask (0 / 255).
        kernel_size: Side length of the square structuring element.

    Returns:
        New uint8 binary mask after opening.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def _apply_closing(
    binary: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """
    Apply morphological CLOSING (dilation -> erosion) to fill small gaps.

    A moderate kernel (default 5 x 5) bridges narrow gaps and connects
    nearby foreground pixels without merging distant components.

    Args:
        binary:      H x W uint8 binary mask (0 / 255).
        kernel_size: Side length of the square structuring element.

    Returns:
        New uint8 binary mask after closing.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _remove_small_components(
    binary: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, int]:
    """
    Remove foreground connected components smaller than min_area pixels.

    Uses 8-connectivity.  The largest component is always preserved
    regardless of min_area.

    Args:
        binary:   H x W uint8 binary mask (0 / 255).
        min_area: Minimum foreground component area in pixels.
                  Components strictly below this size are removed.

    Returns:
        (cleaned_binary, n_removed)
        cleaned_binary: H x W uint8 mask with small components zeroed.
        n_removed:      number of components that were removed.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    n_fg = num_labels - 1  # label 0 is background
    if n_fg == 0:
        return binary.copy(), 0

    # Collect (area, label) for foreground components
    fg_areas = [
        (int(stats[lbl, cv2.CC_STAT_AREA]), lbl)
        for lbl in range(1, num_labels)
    ]

    # Find the largest area so we never remove it
    max_area = max(a for a, _ in fg_areas)

    cleaned = binary.copy()
    n_removed = 0
    for area, lbl in fg_areas:
        if area < min_area and area < max_area:
            cleaned[labels == lbl] = 0
            n_removed += 1

    return cleaned, n_removed


def _fill_small_holes(
    binary: np.ndarray,
    max_hole_area: int,
) -> tuple[np.ndarray, int]:
    """
    Fill small background holes enclosed within foreground regions.

    Strategy:
      1. Invert the binary mask so background becomes foreground.
      2. Run connected-component analysis on the inverted mask.
      3. The largest inverted component is the actual photo background
         (connected to the image border) — do NOT fill it.
      4. Fill inverted components smaller than max_hole_area.

    This reliably fills only enclosed background holes, never the global
    background, because the global background always has a very large area.

    Args:
        binary:       H x W uint8 binary mask (0 / 255).
        max_hole_area: Maximum hole size (pixels) that will be filled.

    Returns:
        (filled_binary, n_holes_filled)
    """
    inverted = cv2.bitwise_not(binary)   # background -> white, fg -> black

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=4   # 4-connectivity for holes (more conservative)
    )

    if num_labels <= 1:
        return binary.copy(), 0

    # The actual photo background component is connected to the image border.
    # Identify it as the component with the largest area among those that
    # touch the border, or simply the overall largest background component.
    bg_areas = [(int(stats[lbl, cv2.CC_STAT_AREA]), lbl)
                for lbl in range(1, num_labels)]
    if not bg_areas:
        return binary.copy(), 0

    max_bg_area = max(a for a, _ in bg_areas)

    filled = binary.copy()
    n_filled = 0
    for area, lbl in bg_areas:
        # Skip the real background (it will always be the largest)
        if area == max_bg_area:
            continue
        if area <= max_hole_area:
            filled[labels == lbl] = 255
            n_filled += 1

    return filled, n_filled


def _restore_soft_mask(
    raw_mask: np.ndarray,
    clean_binary: np.ndarray,
) -> np.ndarray:
    """
    Gate the original soft mask with the refined binary mask.

    Where clean_binary is 255 (retained foreground):
        keep the original soft value from raw_mask.
    Where clean_binary is 0 (removed by refinement):
        set to 0.

    This preserves the gradient/confidence information at clothing edges
    rather than returning a hard 0/255 image.

    Args:
        raw_mask:     Original H x W uint8 soft mask (unchanged throughout).
        clean_binary: H x W uint8 binary mask (0 or 255) after all morphology.

    Returns:
        H x W uint8 soft mask — the refined output.
    """
    binary_01 = (clean_binary > 0).astype(np.uint8)
    return (raw_mask * binary_01).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_mask(
    raw_mask: np.ndarray,
    threshold: int = MASK_REFINEMENT_THRESHOLD,
    opening_kernel_size: int = MASK_OPENING_KERNEL_SIZE,
    closing_kernel_size: int = MASK_CLOSING_KERNEL_SIZE,
    min_component_area: int = MASK_MIN_COMPONENT_AREA,
    max_hole_area: int = MASK_MAX_HOLE_AREA,
) -> RefinedMaskResult:
    """
    Refine a raw U2-Net segmentation mask with conservative OpenCV operations.

    Pipeline (all operations on internal copies; raw_mask is NEVER modified):
        1. Binarise raw_mask at threshold.
        2. Morphological opening  — remove small isolated specks.
        3. Morphological closing  — fill narrow gaps.
        4. Remove small foreground components (area < min_component_area).
        5. Fill small background holes         (area <= max_hole_area).
        6. Gate the original soft mask with the cleaned binary.

    Args:
        raw_mask:            H x W uint8 ndarray from SegmentationResult.mask.
        threshold:           Binarisation threshold (default: config value).
        opening_kernel_size: Kernel side length for morphological opening.
        closing_kernel_size: Kernel side length for morphological closing.
        min_component_area:  Components below this size are removed.
        max_hole_area:       Holes up to this size are filled.

    Returns:
        RefinedMaskResult with the refined uint8 soft mask and diagnostics.

    Raises:
        ValueError: if raw_mask fails input validation.
    """
    _validate_mask(raw_mask)

    height, width = raw_mask.shape
    total_pixels = height * width

    # --- Raw foreground percentage (for comparison) ----------------------
    raw_fg_count = int((raw_mask >= threshold).sum())
    raw_fg_pct   = raw_fg_count / total_pixels * 100.0

    # --- Step 1: binarise (internal copy, raw_mask untouched) ------------
    binary = _create_binary_mask(raw_mask, threshold)

    # --- Step 2: opening — remove small isolated specks ------------------
    binary = _apply_opening(binary, opening_kernel_size)

    # --- Step 3: closing — fill narrow gaps ------------------------------
    binary = _apply_closing(binary, closing_kernel_size)

    # --- Step 4: remove small foreground components ----------------------
    binary, n_removed = _remove_small_components(binary, min_component_area)

    # --- Step 5: fill small background holes inside foreground -----------
    binary, n_filled = _fill_small_holes(binary, max_hole_area)

    # --- Step 6: restore soft values gated by clean binary ---------------
    refined = _restore_soft_mask(raw_mask, binary)

    refined_fg_count = int((refined >= threshold).sum())
    refined_fg_pct   = refined_fg_count / total_pixels * 100.0

    return RefinedMaskResult(
        mask=refined,
        raw_foreground_percentage=raw_fg_pct,
        refined_foreground_percentage=refined_fg_pct,
        components_removed=n_removed,
        holes_filled=n_filled,
    )
