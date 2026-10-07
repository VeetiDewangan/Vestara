"""
Mask analysis service  (Section 5).

Inspects the raw soft segmentation mask produced by Section 4 and returns
objective measurements about its contents.  This is a purely analytical
step — the original mask is NEVER modified.

Key design decisions
--------------------
- All threshold-based analysis uses a temporary binary copy of the mask.
  The caller's array is left byte-for-byte identical after analysis.
- No morphological operations, thresholding of the original, or pixel-value
  changes happen here.  Those belong to Section 6 (mask refinement).
- The service is stateless; no model loading is required.

Pipeline position:
    SegmentationResult.mask  (H×W uint8, soft values 0-255)
        |
        v
    analyze_mask()
        |
        v
    MaskAnalysisResult  (statistics only -- mask unchanged)

Foreground definition (for analysis only):
    pixel value >= MASK_ANALYSIS_THRESHOLD (default 128) -> foreground
    pixel value <  MASK_ANALYSIS_THRESHOLD               -> background
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import MASK_ANALYSIS_THRESHOLD


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box of the analytical foreground region.

    Convention: pixel-coordinate rectangle where:
        x_min  = leftmost  foreground column (inclusive)
        y_min  = topmost   foreground row    (inclusive)
        x_max  = rightmost foreground column (inclusive)
        y_max  = bottommost foreground row   (inclusive)

    All values are zero-indexed, matching NumPy/OpenCV conventions.
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        """Width of the bounding box in pixels."""
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        """Height of the bounding box in pixels."""
        return self.y_max - self.y_min + 1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x_min, y_min, x_max, y_max) as a plain tuple."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass
class MaskAnalysisResult:
    """
    Objective measurements of a segmentation mask.

    All statistics are derived from the original soft mask values or from
    an analytical binary view (threshold applied only to a temporary copy).
    The original mask is never modified.

    Attributes
    ----------
    width, height:
        Dimensions of the mask in pixels.

    min_value, max_value:
        Minimum and maximum pixel values in the ORIGINAL soft mask.

    mean_value:
        Mean pixel value of the ORIGINAL soft mask (float).

    foreground_pixel_count:
        Number of pixels whose value is >= threshold_used.

    foreground_percentage:
        foreground_pixel_count / total_pixels * 100  (float, 0.0-100.0).

    bounding_box:
        Tight bounding box around all foreground pixels, or None when
        no foreground pixels exist.

    connected_component_count:
        Number of distinct foreground connected regions (8-connectivity).
        The background component is NOT counted.

    largest_component_area:
        Pixel count of the largest foreground connected component.
        0 when there are no foreground pixels.

    largest_component_percentage:
        largest_component_area / total_pixels * 100  (float, 0.0-100.0).

    touches_top, touches_bottom, touches_left, touches_right:
        True when at least one foreground pixel touches that image edge.

    is_empty:
        True when foreground_pixel_count == 0.

    is_almost_empty:
        True when 0 < foreground_percentage < 1.0.

    is_almost_full:
        True when foreground_percentage > 95.0.

    has_multiple_components:
        True when connected_component_count > 1.

    threshold_used:
        The threshold value used during this analysis run.
        Recorded so results are self-documenting.
    """
    # Dimensions
    width: int
    height: int

    # Soft-mask statistics (original values, no threshold applied)
    min_value: int
    max_value: int
    mean_value: float

    # Foreground statistics (threshold applied to temporary copy only)
    foreground_pixel_count: int
    foreground_percentage: float

    # Spatial statistics
    bounding_box: Optional[BoundingBox]
    connected_component_count: int
    largest_component_area: int
    largest_component_percentage: float

    # Border contact
    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool

    # Descriptive flags
    is_empty: bool
    is_almost_empty: bool
    is_almost_full: bool
    has_multiple_components: bool

    # Provenance
    threshold_used: int


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
# Internal analysis helpers
# ---------------------------------------------------------------------------

def _compute_bounding_box(binary: np.ndarray) -> Optional[BoundingBox]:
    """
    Return the tight bounding box of foreground pixels in the binary mask.

    Args:
        binary: H x W bool/uint8 array where non-zero = foreground.

    Returns:
        BoundingBox or None when no foreground pixels exist.
    """
    rows = np.any(binary, axis=1)   # True for each row containing foreground
    cols = np.any(binary, axis=0)   # True for each col containing foreground

    if not rows.any():
        return None

    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _compute_connected_components(
    binary: np.ndarray,
) -> tuple[int, int]:
    """
    Run OpenCV connected-component analysis on the binary foreground mask.

    Uses 8-connectivity so diagonally touching pixels form one component.

    Args:
        binary: H x W uint8 array (non-zero = foreground, 0 = background).

    Returns:
        (component_count, largest_component_area)

        component_count:        number of foreground components (background
                                component excluded).
        largest_component_area: pixel count of the largest foreground component
                                (0 when no foreground exists).
    """
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Label 0 is always the background; foreground starts at label 1.
    foreground_count = num_labels - 1

    if foreground_count == 0:
        return 0, 0

    # stats row 0 is the background; slice it off and read the AREA column.
    foreground_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(foreground_areas.max())

    return foreground_count, largest_area


def _compute_border_contact(
    binary: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    """
    Determine whether foreground pixels touch each image edge.

    Returns:
        (touches_top, touches_bottom, touches_left, touches_right)
    """
    touches_top    = bool(binary[0, :].any())
    touches_bottom = bool(binary[-1, :].any())
    touches_left   = bool(binary[:, 0].any())
    touches_right  = bool(binary[:, -1].any())
    return touches_top, touches_bottom, touches_left, touches_right


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_mask(
    mask: np.ndarray,
    threshold: int = MASK_ANALYSIS_THRESHOLD,
) -> MaskAnalysisResult:
    """
    Analyze a raw segmentation mask and return objective statistics.

    The original mask array is NEVER modified.  All threshold-based analysis
    operates on an internal temporary copy.

    Args:
        mask:      H x W uint8 ndarray from SegmentationResult.mask.
                   Soft values 0-255 are expected; pure binary masks also work.
        threshold: Analysis-only foreground cutoff (default MASK_ANALYSIS_THRESHOLD).
                   Pixels >= threshold are considered foreground.
                   This threshold does NOT alter the original mask.

    Returns:
        MaskAnalysisResult with all statistics populated.

    Raises:
        ValueError: if the mask fails input validation.
    """
    _validate_mask(mask)

    height, width = mask.shape
    total_pixels = height * width

    # --- Soft-mask statistics (read directly from the original) ----------
    min_value  = int(mask.min())
    max_value  = int(mask.max())
    mean_value = float(mask.mean())

    # --- Analytical binary view (temporary copy, original untouched) -----
    binary = (mask >= threshold).astype(np.uint8)

    foreground_pixel_count = int(binary.sum())
    foreground_percentage  = foreground_pixel_count / total_pixels * 100.0

    # --- Spatial statistics -----------------------------------------------
    bounding_box           = _compute_bounding_box(binary)
    cc_count, largest_area = _compute_connected_components(binary)
    largest_pct            = largest_area / total_pixels * 100.0

    # --- Border contact ---------------------------------------------------
    t_top, t_bot, t_left, t_right = _compute_border_contact(binary)

    # --- Descriptive flags ------------------------------------------------
    is_empty        = foreground_pixel_count == 0
    is_almost_empty = 0 < foreground_percentage < 1.0
    is_almost_full  = foreground_percentage > 95.0
    has_multiple    = cc_count > 1

    return MaskAnalysisResult(
        width=width,
        height=height,
        min_value=min_value,
        max_value=max_value,
        mean_value=mean_value,
        foreground_pixel_count=foreground_pixel_count,
        foreground_percentage=foreground_percentage,
        bounding_box=bounding_box,
        connected_component_count=cc_count,
        largest_component_area=largest_area,
        largest_component_percentage=largest_pct,
        touches_top=t_top,
        touches_bottom=t_bot,
        touches_left=t_left,
        touches_right=t_right,
        is_empty=is_empty,
        is_almost_empty=is_almost_empty,
        is_almost_full=is_almost_full,
        has_multiple_components=has_multiple,
        threshold_used=threshold,
    )
