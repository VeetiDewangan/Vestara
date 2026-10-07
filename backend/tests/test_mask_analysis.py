"""
Tests for Section 5: Raw Mask Analysis.

Coverage:
  1.  Basic mask statistics (width, height, min, max, mean)
  2.  Min and max values across the value range
  3.  Mean value accuracy
  4.  Foreground percentage calculation
  5.  Bounding box — standard case
  6.  Empty mask (no foreground pixels)
  7.  Single connected component
  8.  Multiple connected components
  9.  Largest component percentage
  10. Border contact — top, bottom, left, right
  11. Threshold behaviour (custom threshold)
  12. Original mask is NOT modified after analysis
  13. Invalid input — None
  14. Invalid input — 3-channel RGB array
  15. Invalid input — wrong dtype (float32)
  16. Invalid input — empty NumPy array
  17. Descriptive flags (is_empty, is_almost_full, has_multiple_components)
  18. Real segmentation integration test (S3 -> S4 -> S5)

All synthetic masks are created in memory.
No files are written to input_images/ or output_images/.
The integration test (test 18) uses the real U2-Net model but loads it only
once via a module-level shared service instance.
"""

import numpy as np
import pytest

from app.services.mask_analysis import (
    BoundingBox,
    MaskAnalysisResult,
    analyze_mask,
    _compute_bounding_box,
    _compute_connected_components,
    _compute_border_contact,
)
from app.config import MASK_ANALYSIS_THRESHOLD


# ===========================================================================
# Helpers
# ===========================================================================

def _zeros(h: int = 100, w: int = 100) -> np.ndarray:
    """All-background mask."""
    return np.zeros((h, w), dtype=np.uint8)


def _full(h: int = 100, w: int = 100, value: int = 255) -> np.ndarray:
    """Uniform foreground mask."""
    return np.full((h, w), value, dtype=np.uint8)


def _rect_mask(
    h: int = 100, w: int = 100,
    y1: int = 10, y2: int = 50,
    x1: int = 20, x2: int = 70,
    fg: int = 255,
) -> np.ndarray:
    """Mask with a solid foreground rectangle."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = fg
    return m


# ===========================================================================
# Test 1 — Basic statistics
# ===========================================================================

class TestBasicStatistics:
    def test_width_and_height(self):
        mask = _zeros(80, 120)
        result = analyze_mask(mask)
        assert result.width == 120
        assert result.height == 80

    def test_threshold_recorded(self):
        mask = _zeros()
        result = analyze_mask(mask, threshold=100)
        assert result.threshold_used == 100

    def test_default_threshold_matches_config(self):
        mask = _zeros()
        result = analyze_mask(mask)
        assert result.threshold_used == MASK_ANALYSIS_THRESHOLD


# ===========================================================================
# Test 2 — Min and max values
# ===========================================================================

class TestMinMax:
    def test_all_zero_mask(self):
        mask = _zeros()
        result = analyze_mask(mask)
        assert result.min_value == 0
        assert result.max_value == 0

    def test_all_255_mask(self):
        mask = _full(value=255)
        result = analyze_mask(mask)
        assert result.min_value == 255
        assert result.max_value == 255

    def test_mixed_mask(self):
        mask = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
        result = analyze_mask(mask)
        assert result.min_value == 0
        assert result.max_value == 255

    def test_soft_mask_values_preserved(self):
        """Soft values (not 0 or 255) must be read from the original."""
        mask = np.full((50, 50), 180, dtype=np.uint8)
        result = analyze_mask(mask)
        assert result.min_value == 180
        assert result.max_value == 180


# ===========================================================================
# Test 3 — Mean value
# ===========================================================================

class TestMeanValue:
    def test_all_zero_mean(self):
        result = analyze_mask(_zeros())
        assert result.mean_value == pytest.approx(0.0)

    def test_all_255_mean(self):
        result = analyze_mask(_full(value=255))
        assert result.mean_value == pytest.approx(255.0)

    def test_half_foreground_mean(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255   # top half = 255
        expected = 255 * 50 / 100
        result = analyze_mask(mask)
        assert result.mean_value == pytest.approx(expected, rel=1e-4)

    def test_mean_is_float(self):
        mask = np.array([[0, 1]], dtype=np.uint8)
        result = analyze_mask(mask)
        assert isinstance(result.mean_value, float)


# ===========================================================================
# Test 4 — Foreground percentage
# ===========================================================================

class TestForegroundPercentage:
    def test_zero_foreground(self):
        result = analyze_mask(_zeros())
        assert result.foreground_percentage == pytest.approx(0.0)
        assert result.foreground_pixel_count == 0

    def test_full_foreground(self):
        result = analyze_mask(_full(value=255))
        assert result.foreground_percentage == pytest.approx(100.0)
        assert result.foreground_pixel_count == 100 * 100

    def test_half_foreground(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255   # 50 foreground pixels out of 100
        result = analyze_mask(mask)
        assert result.foreground_percentage == pytest.approx(50.0)
        assert result.foreground_pixel_count == 50

    def test_threshold_boundary(self):
        """Pixels exactly at the threshold must count as foreground."""
        mask = np.full((10, 10), MASK_ANALYSIS_THRESHOLD, dtype=np.uint8)
        result = analyze_mask(mask)
        assert result.foreground_pixel_count == 100

    def test_just_below_threshold_is_background(self):
        mask = np.full((10, 10), MASK_ANALYSIS_THRESHOLD - 1, dtype=np.uint8)
        result = analyze_mask(mask)
        assert result.foreground_pixel_count == 0


# ===========================================================================
# Test 5 — Bounding box
# ===========================================================================

class TestBoundingBox:
    def test_full_mask_bbox(self):
        mask = _full(h=50, w=80)
        result = analyze_mask(mask)
        bb = result.bounding_box
        assert bb is not None
        assert bb.x_min == 0
        assert bb.y_min == 0
        assert bb.x_max == 79
        assert bb.y_max == 49

    def test_rect_bbox(self):
        # 100x100 mask, foreground at rows 10-49, cols 20-69
        mask = _rect_mask(y1=10, y2=50, x1=20, x2=70)
        result = analyze_mask(mask)
        bb = result.bounding_box
        assert bb is not None
        assert bb.x_min == 20
        assert bb.y_min == 10
        assert bb.x_max == 69
        assert bb.y_max == 49

    def test_bbox_width_height(self):
        mask = _rect_mask(y1=10, y2=30, x1=5, x2=25)  # 20 rows, 20 cols
        result = analyze_mask(mask)
        bb = result.bounding_box
        assert bb.width == 20
        assert bb.height == 20

    def test_bbox_as_tuple(self):
        mask = _rect_mask(y1=5, y2=15, x1=3, x2=13)
        result = analyze_mask(mask)
        bb = result.bounding_box
        assert bb.as_tuple() == (3, 5, 12, 14)

    def test_single_pixel_bbox(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 30] = 255
        result = analyze_mask(mask)
        bb = result.bounding_box
        assert bb.x_min == bb.x_max == 30
        assert bb.y_min == bb.y_max == 25


# ===========================================================================
# Test 6 — Empty mask
# ===========================================================================

class TestEmptyMask:
    def test_all_zero_is_empty(self):
        result = analyze_mask(_zeros())
        assert result.is_empty is True

    def test_empty_mask_bbox_is_none(self):
        result = analyze_mask(_zeros())
        assert result.bounding_box is None

    def test_empty_mask_component_count(self):
        result = analyze_mask(_zeros())
        assert result.connected_component_count == 0

    def test_empty_mask_largest_area(self):
        result = analyze_mask(_zeros())
        assert result.largest_component_area == 0
        assert result.largest_component_percentage == pytest.approx(0.0)

    def test_empty_mask_no_border_contact(self):
        result = analyze_mask(_zeros())
        assert result.touches_top is False
        assert result.touches_bottom is False
        assert result.touches_left is False
        assert result.touches_right is False


# ===========================================================================
# Test 7 — Single connected component
# ===========================================================================

class TestSingleComponent:
    def test_one_component_count(self):
        mask = _rect_mask()
        result = analyze_mask(mask)
        assert result.connected_component_count == 1
        assert result.has_multiple_components is False

    def test_one_component_area(self):
        # Foreground: rows 10-49, cols 20-69  => 40 rows * 50 cols = 2000 px
        mask = _rect_mask(y1=10, y2=50, x1=20, x2=70)
        result = analyze_mask(mask)
        assert result.largest_component_area == 2000

    def test_one_component_percentage(self):
        # 2000 fg out of 10000 total => 20%
        mask = _rect_mask(h=100, w=100, y1=10, y2=50, x1=20, x2=70)
        result = analyze_mask(mask)
        assert result.largest_component_percentage == pytest.approx(20.0)


# ===========================================================================
# Test 8 — Multiple connected components
# ===========================================================================

class TestMultipleComponents:
    def _two_rect_mask(self) -> np.ndarray:
        """Mask with two separate rectangles (separated by at least 2 px)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255   # rect A: 20x20 = 400 px
        mask[60:80, 60:80] = 255   # rect B: 20x20 = 400 px
        return mask

    def test_two_components_count(self):
        result = analyze_mask(self._two_rect_mask())
        assert result.connected_component_count == 2

    def test_has_multiple_components_flag(self):
        result = analyze_mask(self._two_rect_mask())
        assert result.has_multiple_components is True

    def test_two_components_equal_largest(self):
        result = analyze_mask(self._two_rect_mask())
        assert result.largest_component_area == 400

    def test_three_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:15,  5:15]  = 255
        mask[40:50, 40:50] = 255
        mask[75:85, 75:85] = 255
        result = analyze_mask(mask)
        assert result.connected_component_count == 3
        assert result.has_multiple_components is True


# ===========================================================================
# Test 9 — Largest component percentage
# ===========================================================================

class TestLargestComponentPercentage:
    def test_largest_percentage_unequal_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:60, 10:60] = 255   # 50x50 = 2500 px
        mask[70:80, 70:80] = 255   # 10x10 = 100 px
        result = analyze_mask(mask)
        assert result.largest_component_area == 2500
        assert result.largest_component_percentage == pytest.approx(25.0)


# ===========================================================================
# Test 10 — Border contact
# ===========================================================================

class TestBorderContact:
    def test_center_rect_no_contact(self):
        # Foreground only in the center
        mask = _rect_mask(h=100, w=100, y1=20, y2=80, x1=20, x2=80)
        result = analyze_mask(mask)
        assert result.touches_top    is False
        assert result.touches_bottom is False
        assert result.touches_left   is False
        assert result.touches_right  is False

    def test_touches_top(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0, 50] = 255   # single pixel on top row
        result = analyze_mask(mask)
        assert result.touches_top is True
        assert result.touches_bottom is False

    def test_touches_bottom(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[99, 50] = 255
        result = analyze_mask(mask)
        assert result.touches_bottom is True
        assert result.touches_top is False

    def test_touches_left(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 0] = 255
        result = analyze_mask(mask)
        assert result.touches_left is True
        assert result.touches_right is False

    def test_touches_right(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 99] = 255
        result = analyze_mask(mask)
        assert result.touches_right is True
        assert result.touches_left is False

    def test_touches_all_edges(self):
        mask = _full(value=255)   # entire mask is foreground
        result = analyze_mask(mask)
        assert result.touches_top    is True
        assert result.touches_bottom is True
        assert result.touches_left   is True
        assert result.touches_right  is True


# ===========================================================================
# Test 11 — Custom threshold behaviour
# ===========================================================================

class TestThresholdBehaviour:
    def test_high_threshold_reduces_foreground(self):
        mask = np.full((10, 10), 100, dtype=np.uint8)
        # Threshold 128 -> all 100 < 128 -> no foreground
        result = analyze_mask(mask, threshold=128)
        assert result.foreground_pixel_count == 0

    def test_low_threshold_increases_foreground(self):
        mask = np.full((10, 10), 100, dtype=np.uint8)
        # Threshold 50 -> all 100 >= 50 -> full foreground
        result = analyze_mask(mask, threshold=50)
        assert result.foreground_pixel_count == 100

    def test_threshold_1_counts_all_nonzero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1   # 16 pixels with value 1
        result = analyze_mask(mask, threshold=1)
        assert result.foreground_pixel_count == 16

    def test_threshold_255_only_counts_max(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 255
        mask[1, 1] = 254  # should NOT count at threshold 255
        result = analyze_mask(mask, threshold=255)
        assert result.foreground_pixel_count == 1


# ===========================================================================
# Test 12 — Original mask is NOT modified
# ===========================================================================

class TestMaskImmutability:
    def test_soft_mask_unchanged_after_analysis(self):
        """analyze_mask must not alter any pixel of the input array."""
        original_values = [0, 50, 100, 150, 200, 255]
        mask = np.array(original_values * 10, dtype=np.uint8).reshape(10, 6)
        snapshot = mask.copy()
        analyze_mask(mask)
        np.testing.assert_array_equal(
            mask, snapshot,
            err_msg="analyze_mask modified the input mask array"
        )

    def test_full_mask_unchanged(self):
        mask = _full(value=200)
        snapshot = mask.copy()
        analyze_mask(mask)
        np.testing.assert_array_equal(mask, snapshot)

    def test_custom_threshold_does_not_modify(self):
        mask = np.arange(256, dtype=np.uint8).reshape(16, 16)
        snapshot = mask.copy()
        analyze_mask(mask, threshold=64)
        np.testing.assert_array_equal(mask, snapshot)


# ===========================================================================
# Tests 13-16 — Invalid input
# ===========================================================================

class TestInvalidInput:
    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="None"):
            analyze_mask(None)

    def test_3channel_rgb_raises_value_error(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-dimensional"):
            analyze_mask(rgb)

    def test_4channel_rgba_raises_value_error(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-dimensional"):
            analyze_mask(rgba)

    def test_wrong_dtype_float32_raises(self):
        bad = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            analyze_mask(bad)

    def test_wrong_dtype_int32_raises(self):
        bad = np.zeros((100, 100), dtype=np.int32)
        with pytest.raises(ValueError, match="uint8"):
            analyze_mask(bad)

    def test_empty_array_raises(self):
        empty = np.array([], dtype=np.uint8).reshape(0, 100)
        with pytest.raises(ValueError, match="empty"):
            analyze_mask(empty)


# ===========================================================================
# Test 17 — Descriptive flags
# ===========================================================================

class TestDescriptiveFlags:
    def test_is_empty_flag(self):
        assert analyze_mask(_zeros()).is_empty is True
        assert analyze_mask(_full()).is_empty is False

    def test_is_almost_empty_flag(self):
        # One pixel in 10000 total = 0.01% -> almost_empty
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        result = analyze_mask(mask)
        assert result.is_almost_empty is True

    def test_is_almost_full_flag(self):
        # 96% foreground -> almost_full
        mask = _full(value=255)
        mask[0:2, :] = 0   # zero out 2 rows (200/10000 = 2% bg -> 98% fg)
        result = analyze_mask(mask)
        assert result.is_almost_full is True

    def test_not_almost_full_when_50_percent(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255
        result = analyze_mask(mask)
        assert result.is_almost_full is False

    def test_has_multiple_components_false(self):
        result = analyze_mask(_rect_mask())
        assert result.has_multiple_components is False

    def test_has_multiple_components_true(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        mask[70:80, 70:80] = 255
        result = analyze_mask(mask)
        assert result.has_multiple_components is True


# ===========================================================================
# Test 18 — Real segmentation integration test (S3 -> S4 -> S5)
# ===========================================================================

# Shared service — loaded once, not re-downloaded per test.
_shared_seg_service = None


def _get_seg_service():
    global _shared_seg_service
    if _shared_seg_service is None:
        from app.services.segmentation import SegmentationService
        _shared_seg_service = SegmentationService()
    return _shared_seg_service


class TestRealSegmentationIntegration:
    """
    End-to-end integration: synthetic image -> preprocessing -> U2-Net -> mask analysis.
    Uses the real pretrained model (model must be cached from Section 4 tests).
    """

    def test_full_pipeline_s3_s4_s5(self, capsys):
        """
        Run the full pipeline on a synthetic two-region image and verify
        that MaskAnalysisResult is correctly populated.
        """
        import cv2 as cv2_import

        # Build a simple synthetic image: light top half, dark bottom half
        h, w = 320, 240
        img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        img_rgb[: h // 2, :] = (220, 220, 220)   # light region (foreground-ish)
        img_rgb[h // 2 :, :] = (20, 20, 20)      # dark region  (background-ish)

        # Section 4: run segmentation
        svc = _get_seg_service()
        seg_result = svc.segment(img_rgb)
        mask = seg_result.mask

        # Section 5: analyze the mask
        result = analyze_mask(mask)

        # Structural checks
        assert result.width  == w
        assert result.height == h
        assert result.min_value  >= 0
        assert result.max_value  <= 255
        assert 0.0 <= result.foreground_percentage <= 100.0
        assert isinstance(result.connected_component_count, int)
        assert result.connected_component_count >= 0
        assert 0.0 <= result.largest_component_percentage <= 100.0
        assert isinstance(result.touches_top, bool)
        assert isinstance(result.touches_bottom, bool)
        assert isinstance(result.touches_left, bool)
        assert isinstance(result.touches_right, bool)

        # Mask must not be modified
        original_copy = mask.copy()
        analyze_mask(mask)
        np.testing.assert_array_equal(mask, original_copy)

        # Print a human-readable report
        with capsys.disabled():
            bb = result.bounding_box
            print("\n--- Section 5 Mask Analysis (integration) ---")
            print(f"  Image dimensions     : {w} x {h}")
            print(f"  Mask dimensions      : {result.width} x {result.height}")
            print(f"  Min mask value       : {result.min_value}")
            print(f"  Max mask value       : {result.max_value}")
            print(f"  Mean mask value      : {result.mean_value:.2f}")
            print(f"  Foreground %         : {result.foreground_percentage:.2f}%")
            print(f"  Foreground px count  : {result.foreground_pixel_count}")
            print(f"  Bounding box         : {bb.as_tuple() if bb else None}")
            print(f"  Component count      : {result.connected_component_count}")
            print(f"  Largest component %  : {result.largest_component_percentage:.2f}%")
            print(f"  Touches top/bot/l/r  : "
                  f"{result.touches_top}/{result.touches_bottom}/"
                  f"{result.touches_left}/{result.touches_right}")
            print(f"  is_empty             : {result.is_empty}")
            print(f"  is_almost_empty      : {result.is_almost_empty}")
            print(f"  is_almost_full       : {result.is_almost_full}")
            print(f"  has_multiple_comps   : {result.has_multiple_components}")
            print(f"  Threshold used       : {result.threshold_used}")
            print("---------------------------------------------")
