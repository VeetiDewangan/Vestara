"""
Tests for Section 3: image preprocessing.

Coverage:
  1.  Normal JPEG  → correct RGB output, 3 channels, uint8
  2.  Image ≤ 2048 in both dimensions  → dimensions unchanged
  3.  Large image (largest dim > 2048)  → downscaled, aspect ratio preserved
  4.  Small image  → NOT upscaled
  5.  Grayscale image  → converted to 3-channel RGB
  6.  RGBA PNG  → converted to 3-channel RGB (alpha dropped)
  7.  Color conversion  → BGR input correctly becomes RGB output
  8.  Original image bytes are NOT modified by preprocessing
  9.  Invalid image data  → ValueError raised cleanly

All images are generated in memory; no files are written to input_images/ or
output_images/.
"""

import io

import cv2
import numpy as np
import pytest

from app.services.preprocessing import (
    PreprocessedImage,
    _compute_resize_dims,
    preprocess_image_bytes,
)
from app.config import MAX_PREPROCESSING_DIMENSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_jpeg(arr: np.ndarray, quality: int = 85) -> bytes:
    """Encode an ndarray as JPEG bytes (BGR input, as OpenCV expects)."""
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "OpenCV failed to encode JPEG in test helper"
    return buf.tobytes()


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode an ndarray as PNG bytes."""
    ok, buf = cv2.imencode(".png", arr)
    assert ok, "OpenCV failed to encode PNG in test helper"
    return buf.tobytes()


def _solid_bgr(width: int, height: int, b: int, g: int, r: int) -> np.ndarray:
    """Create a solid-colour BGR image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (b, g, r)
    return img


# ---------------------------------------------------------------------------
# Test 1 — Normal JPEG: correct output type and shape
# ---------------------------------------------------------------------------

class TestNormalJpeg:
    def test_output_is_preprocessed_image(self):
        """preprocess_image_bytes must return a PreprocessedImage."""
        data = _encode_jpeg(_solid_bgr(512, 512, 100, 149, 237))
        result = preprocess_image_bytes(data)
        assert isinstance(result, PreprocessedImage)

    def test_image_is_ndarray(self):
        data = _encode_jpeg(_solid_bgr(512, 512, 100, 149, 237))
        result = preprocess_image_bytes(data)
        assert isinstance(result.image, np.ndarray)

    def test_three_channels(self):
        """Output image must have 3 channels."""
        data = _encode_jpeg(_solid_bgr(512, 512, 100, 149, 237))
        result = preprocess_image_bytes(data)
        assert result.image.ndim == 3
        assert result.image.shape[2] == 3

    def test_dtype_uint8(self):
        """Output image dtype must be uint8."""
        data = _encode_jpeg(_solid_bgr(512, 512, 100, 149, 237))
        result = preprocess_image_bytes(data)
        assert result.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 2 — Image ≤ 2048 in both dims: no resize
# ---------------------------------------------------------------------------

class TestNoDimensionChange:
    def test_small_image_dimensions_unchanged(self):
        """A 640×480 image is well within limits and must not be resized."""
        w, h = 640, 480
        data = _encode_jpeg(_solid_bgr(w, h, 50, 100, 200))
        result = preprocess_image_bytes(data)
        assert result.processed_width == w
        assert result.processed_height == h
        assert result.was_resized is False

    def test_exactly_at_limit_unchanged(self):
        """An image whose largest dim equals MAX_PREPROCESSING_DIMENSION exactly."""
        # JPEG compression can shift dimensions by 1 for non-MCU-aligned sizes,
        # so use a safe round dimension.
        dim = MAX_PREPROCESSING_DIMENSION  # 2048
        w, h = dim, dim // 2              # 2048 × 1024
        data = _encode_jpeg(_solid_bgr(w, h, 0, 128, 255))
        result = preprocess_image_bytes(data)
        # Largest dim should be ≤ MAX (JPEG rounding may shrink slightly)
        assert max(result.processed_width, result.processed_height) <= dim
        assert result.was_resized is False


# ---------------------------------------------------------------------------
# Test 3 — Large image: downscaled, aspect ratio preserved
# ---------------------------------------------------------------------------

class TestLargeImageResize:
    def test_was_resized_flag(self):
        """An image wider than 2048 must be flagged as resized."""
        # Use PNG to avoid JPEG MCU-alignment rounding for exact dimension tests
        w, h = 4000, 3000
        arr = _solid_bgr(w, h, 0, 200, 100)
        data = _encode_png(arr)
        result = preprocess_image_bytes(data)
        assert result.was_resized is True

    def test_largest_dimension_becomes_max(self):
        """After resizing, the largest dimension must equal MAX_PREPROCESSING_DIMENSION."""
        w, h = 4000, 3000
        arr = _solid_bgr(w, h, 0, 200, 100)
        data = _encode_png(arr)
        result = preprocess_image_bytes(data)
        assert max(result.processed_width, result.processed_height) == MAX_PREPROCESSING_DIMENSION

    def test_aspect_ratio_preserved(self):
        """After resizing, the aspect ratio must match the original within 1 px."""
        w, h = 4000, 3000  # ratio = 4:3
        arr = _solid_bgr(w, h, 0, 200, 100)
        data = _encode_png(arr)
        result = preprocess_image_bytes(data)
        original_ratio = w / h
        processed_ratio = result.processed_width / result.processed_height
        assert abs(original_ratio - processed_ratio) < 0.01

    def test_portrait_resize(self):
        """Portrait image (height > width): height becomes MAX_PREPROCESSING_DIMENSION."""
        w, h = 3000, 5000
        arr = _solid_bgr(w, h, 120, 80, 40)
        data = _encode_png(arr)
        result = preprocess_image_bytes(data)
        assert result.processed_height == MAX_PREPROCESSING_DIMENSION
        assert result.processed_width < MAX_PREPROCESSING_DIMENSION

    def test_original_dims_recorded_correctly(self):
        """original_width/height must reflect the uploaded image, not the resized one."""
        w, h = 4000, 3000
        arr = _solid_bgr(w, h, 0, 200, 100)
        data = _encode_png(arr)
        result = preprocess_image_bytes(data)
        assert result.original_width == w
        assert result.original_height == h


# ---------------------------------------------------------------------------
# Test 4 — Small image: NOT upscaled
# ---------------------------------------------------------------------------

class TestSmallImageNotUpscaled:
    def test_small_image_not_enlarged(self):
        """A 300×300 image must not be upscaled."""
        w, h = 300, 300
        data = _encode_jpeg(_solid_bgr(w, h, 10, 20, 30))
        result = preprocess_image_bytes(data)
        assert result.processed_width <= w
        assert result.processed_height <= h
        assert result.was_resized is False

    def test_compute_resize_dims_no_upscale(self):
        """_compute_resize_dims must never return dims larger than the input."""
        new_w, new_h = _compute_resize_dims(300, 300, MAX_PREPROCESSING_DIMENSION)
        assert new_w == 300
        assert new_h == 300


# ---------------------------------------------------------------------------
# Test 5 — Grayscale image: converted to 3-channel RGB
# ---------------------------------------------------------------------------

class TestGrayscaleConversion:
    def test_grayscale_png_becomes_3_channel(self):
        """A grayscale PNG must be converted to a 3-channel output."""
        gray = np.full((512, 512), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        result = preprocess_image_bytes(buf.tobytes())
        assert result.image.ndim == 3
        assert result.image.shape[2] == 3

    def test_grayscale_dtype_uint8(self):
        gray = np.full((512, 512), 200, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", gray)
        assert ok
        result = preprocess_image_bytes(buf.tobytes())
        assert result.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 6 — RGBA image: converted to 3-channel RGB (alpha dropped)
# ---------------------------------------------------------------------------

class TestRGBAConversion:
    def _make_rgba_png(self, width: int = 512, height: int = 512) -> bytes:
        """Create a 4-channel BGRA PNG encoded in memory."""
        bgra = np.zeros((height, width, 4), dtype=np.uint8)
        bgra[:, :, 0] = 50   # B
        bgra[:, :, 1] = 100  # G
        bgra[:, :, 2] = 200  # R
        bgra[:, :, 3] = 128  # A (semi-transparent)
        ok, buf = cv2.imencode(".png", bgra)
        assert ok, "OpenCV failed to encode RGBA PNG"
        return buf.tobytes()

    def test_rgba_becomes_3_channel(self):
        """RGBA image must be reduced to 3 channels."""
        result = preprocess_image_bytes(self._make_rgba_png())
        assert result.image.ndim == 3
        assert result.image.shape[2] == 3

    def test_rgba_dtype_uint8(self):
        result = preprocess_image_bytes(self._make_rgba_png())
        assert result.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# Test 7 — Color conversion: BGR input → correct RGB output
# ---------------------------------------------------------------------------

class TestColorConversion:
    def test_red_channel_is_preserved_correctly(self):
        """
        Create a solid pure-red image in BGR (B=0, G=0, R=255).
        After preprocessing, the RGB output pixel should be (R=255, G=0, B=0).
        We use PNG to avoid JPEG chroma rounding.
        """
        # Pure red in BGR (OpenCV convention)
        bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # R channel in BGR
        ok, buf = cv2.imencode(".png", bgr)
        assert ok

        result = preprocess_image_bytes(buf.tobytes())
        center_pixel = result.image[256, 256]  # (R, G, B) after BGR→RGB
        r, g, b = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])

        assert r == 255, f"Expected R=255, got R={r} (possible BGR→RGB bug)"
        assert g == 0,   f"Expected G=0, got G={g}"
        assert b == 0,   f"Expected B=0, got B={b}"

    def test_blue_channel_is_preserved_correctly(self):
        """
        Create a solid pure-blue image in BGR (B=255, G=0, R=0).
        After BGR→RGB, channel index 2 should hold 255.
        """
        bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # B channel in BGR
        ok, buf = cv2.imencode(".png", bgr)
        assert ok

        result = preprocess_image_bytes(buf.tobytes())
        center_pixel = result.image[256, 256]
        r, g, b = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])

        assert r == 0,   f"Expected R=0, got R={r}"
        assert g == 0,   f"Expected G=0, got G={g}"
        assert b == 255, f"Expected B=255, got B={b} (possible BGR→RGB bug)"


# ---------------------------------------------------------------------------
# Test 8 — Original image bytes are NOT modified
# ---------------------------------------------------------------------------

class TestOriginalUnchanged:
    def test_original_bytes_unchanged(self):
        """preprocess_image_bytes must not mutate the input bytes object."""
        data = _encode_jpeg(_solid_bgr(512, 512, 100, 149, 237))
        original_copy = bytes(data)
        preprocess_image_bytes(data)
        assert data == original_copy, "Input bytes were mutated by preprocessing"

    def test_original_ndarray_unchanged(self):
        """
        If the caller converts bytes to ndarray themselves before calling,
        preprocessing should not modify the original array.
        """
        bgr = _solid_bgr(512, 512, 100, 149, 237)
        original_bgr = bgr.copy()
        # Encode then preprocess (preprocessing works on bytes, not the array)
        data = _encode_png(bgr)
        preprocess_image_bytes(data)
        # The original ndarray should be untouched
        np.testing.assert_array_equal(bgr, original_bgr)


# ---------------------------------------------------------------------------
# Test 9 — Invalid image data: ValueError raised cleanly
# ---------------------------------------------------------------------------

class TestInvalidData:
    def test_empty_bytes_raises(self):
        """Empty bytes must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            preprocess_image_bytes(b"")

    def test_random_bytes_raises(self):
        """Random bytes that are not a valid image must raise ValueError."""
        import os
        garbage = os.urandom(1024)
        with pytest.raises(ValueError):
            preprocess_image_bytes(garbage)

    def test_truncated_jpeg_raises(self):
        """JPEG magic bytes followed by garbage must raise ValueError."""
        truncated = b"\xff\xd8\xff" + b"\x00" * 64
        with pytest.raises(ValueError):
            preprocess_image_bytes(truncated)
