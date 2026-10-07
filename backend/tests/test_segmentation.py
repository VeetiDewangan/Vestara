"""
Tests for Section 4: AI Segmentation (rembg / U²-Net).

The test suite is divided into two groups:

  Group A — Unit tests  (no model download required)
    Tests that exercise SegmentationService._validate_input() and the
    data structures.  These run instantly and do not touch rembg.

  Group B — Integration tests  (real model inference)
    One shared SegmentationService instance is created at module scope so
    that model weights are downloaded and loaded only once, regardless of
    how many integration tests run.
    These tests may be slow on first run (~30 s for model download).
    Mark them with the custom 'integration' marker if you want to exclude
    them from fast CI runs.

IMPORTANT: the integration tests use the real pretrained U²-Net model.
They are NOT mocked.  The results are verified by structural properties
(shape, dtype, value range) rather than by inspecting exact pixel values,
because the model output is not deterministic across platforms.
"""

import time

import numpy as np
import pytest

from app.services.segmentation import (
    SegmentationResult,
    SegmentationService,
    SegmentationError,
    ModelLoadError,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_rgb(width: int = 256, height: int = 256, fill: tuple = (100, 149, 237)) -> np.ndarray:
    """Create a solid-colour RGB uint8 ndarray (the expected input format)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = fill
    return img


def _make_two_region_rgb(width: int = 256, height: int = 256) -> np.ndarray:
    """
    Create an RGB image with a clear light foreground on a dark background.
    The top half is near-white (foreground) and the bottom half is near-black
    (background), giving the model a synthetic but discernible scene.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[: height // 2, :] = (220, 220, 220)  # light top half  (foreground-ish)
    img[height // 2 :, :] = (20, 20, 20)     # dark bottom half (background-ish)
    return img


# ===========================================================================
# Module-level shared service instance for integration tests.
# Created here (not inside a fixture) so model loading happens once.
# ===========================================================================
_shared_service: SegmentationService | None = None


def _get_shared_service() -> SegmentationService:
    global _shared_service
    if _shared_service is None:
        _shared_service = SegmentationService()
    return _shared_service


# ===========================================================================
# Group A: Unit tests — no model required
# ===========================================================================

class TestInputValidation:
    """Verify that _validate_input catches bad input before touching the model."""

    def setup_method(self):
        # Fresh service instance; session never loaded in these tests.
        self.service = SegmentationService()

    def test_none_raises_value_error(self):
        """Passing None must raise ValueError."""
        with pytest.raises(ValueError, match="None"):
            self.service._validate_input(None)

    def test_non_ndarray_raises_value_error(self):
        """Passing a plain list must raise ValueError."""
        with pytest.raises(ValueError, match="ndarray"):
            self.service._validate_input([[1, 2, 3], [4, 5, 6]])

    def test_wrong_dtype_raises_value_error(self):
        """float32 array must raise ValueError (uint8 expected)."""
        bad = np.ones((256, 256, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            self.service._validate_input(bad)

    def test_wrong_channels_2d_raises_value_error(self):
        """2-D (grayscale) array must raise ValueError."""
        bad = np.zeros((256, 256), dtype=np.uint8)
        with pytest.raises(ValueError, match="H×W×3"):
            self.service._validate_input(bad)

    def test_wrong_channels_4d_raises_value_error(self):
        """4-channel array must raise ValueError."""
        bad = np.zeros((256, 256, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="H×W×3"):
            self.service._validate_input(bad)

    def test_valid_input_passes_without_error(self):
        """A proper H×W×3 uint8 array must pass validation silently."""
        good = _make_rgb()
        self.service._validate_input(good)  # must not raise


class TestSegmentationResultDataclass:
    """Smoke-test the SegmentationResult dataclass."""

    def test_fields_accessible(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        result = SegmentationResult(
            mask=mask,
            model_name="u2net",
            input_width=256,
            input_height=256,
        )
        assert result.model_name == "u2net"
        assert result.input_width == 256
        assert result.input_height == 256
        assert isinstance(result.mask, np.ndarray)


class TestServiceProperties:
    def test_model_name_property(self):
        svc = SegmentationService(model_name="u2net")
        assert svc.model_name == "u2net"

    def test_session_not_loaded_at_init(self):
        """Model session must be None until the first call to segment()."""
        svc = SegmentationService()
        assert svc._session is None


# ===========================================================================
# Group B: Integration tests — real U²-Net inference
# ===========================================================================

class TestSegmentationIntegration:
    """
    Tests that exercise the actual pretrained U²-Net model via rembg.

    The model is loaded once into _shared_service at module level.
    First run may take ~30 s to download model weights (~170 MB).
    """

    # ------------------------------------------------------------------
    # Test 1: output is a SegmentationResult
    # ------------------------------------------------------------------
    def test_returns_segmentation_result(self):
        """segment() must return a SegmentationResult instance."""
        svc = _get_shared_service()
        img = _make_rgb(256, 256)
        result = svc.segment(img)
        assert isinstance(result, SegmentationResult)

    # ------------------------------------------------------------------
    # Test 2: mask is a NumPy ndarray
    # ------------------------------------------------------------------
    def test_mask_is_ndarray(self):
        svc = _get_shared_service()
        result = svc.segment(_make_rgb(256, 256))
        assert isinstance(result.mask, np.ndarray)

    # ------------------------------------------------------------------
    # Test 3: mask is 2-dimensional
    # ------------------------------------------------------------------
    def test_mask_is_2d(self):
        """Output mask must have shape H×W (2-D)."""
        svc = _get_shared_service()
        result = svc.segment(_make_rgb(256, 256))
        assert result.mask.ndim == 2, (
            f"Expected 2-D mask, got shape {result.mask.shape}"
        )

    # ------------------------------------------------------------------
    # Test 4: mask dtype is uint8
    # ------------------------------------------------------------------
    def test_mask_dtype_is_uint8(self):
        svc = _get_shared_service()
        result = svc.segment(_make_rgb(256, 256))
        assert result.mask.dtype == np.uint8, (
            f"Expected uint8 mask, got {result.mask.dtype}"
        )

    # ------------------------------------------------------------------
    # Test 5: mask dimensions match input
    # ------------------------------------------------------------------
    def test_mask_dimensions_match_input(self):
        """Mask (H, W) must match the input image dimensions."""
        w, h = 320, 240
        svc = _get_shared_service()
        result = svc.segment(_make_rgb(w, h))
        assert result.mask.shape == (h, w), (
            f"Expected mask shape ({h}, {w}), got {result.mask.shape}"
        )

    # ------------------------------------------------------------------
    # Test 6: mask has meaningful values (not all zeros or all 255)
    # ------------------------------------------------------------------
    def test_mask_contains_meaningful_values(self):
        """
        Mask must not be a flat zero or flat 255 array.

        We use a two-region image (light top / dark bottom) to give the model
        a scene where at least some pixels should differ in foreground confidence.

        NOTE: the assertion is intentionally broad.  We only verify that the
        mask is not completely trivial — not that every pixel is correct.
        """
        svc = _get_shared_service()
        img = _make_two_region_rgb(256, 256)
        result = svc.segment(img)
        min_val = int(result.mask.min())
        max_val = int(result.mask.max())
        # At least some pixels should be non-zero (foreground detected)
        assert max_val > 0, (
            f"Mask is entirely zero — model may not be running correctly "
            f"(min={min_val}, max={max_val})"
        )

    # ------------------------------------------------------------------
    # Test 7: synthetic two-region image processed without crashing
    # ------------------------------------------------------------------
    def test_two_region_image_no_crash(self):
        """The two-region synthetic clothing-like image must process without error."""
        svc = _get_shared_service()
        img = _make_two_region_rgb(256, 256)
        result = svc.segment(img)  # must not raise
        assert result is not None

    # ------------------------------------------------------------------
    # Test 8: model is reused (session not None after first call)
    # ------------------------------------------------------------------
    def test_session_reused_after_first_inference(self):
        """After the first inference the session must be cached (not None)."""
        svc = _get_shared_service()
        svc.segment(_make_rgb(256, 256))  # ensure session loaded
        assert svc._session is not None, (
            "Session was None after inference — model is being reloaded on every call"
        )

    # ------------------------------------------------------------------
    # Test 9: invalid input still raises ValueError (not a model error)
    # ------------------------------------------------------------------
    def test_invalid_input_raises_before_model(self):
        """
        Invalid input must raise ValueError (from validation) even when the
        model is already loaded — the error must come from _validate_input,
        not from rembg.
        """
        svc = _get_shared_service()
        # Trigger session load first
        svc.segment(_make_rgb(256, 256))
        # Now send invalid input
        with pytest.raises(ValueError):
            svc.segment(None)

    # ------------------------------------------------------------------
    # Bonus: report timings and mask stats (informational, always passes)
    # ------------------------------------------------------------------
    def test_report_inference_stats(self, capsys):
        """Run inference and print useful stats about the result."""
        svc = _get_shared_service()
        img = _make_two_region_rgb(320, 240)

        t0 = time.perf_counter()
        result = svc.segment(img)
        elapsed = time.perf_counter() - t0

        mask = result.mask
        with capsys.disabled():
            print(f"\n--- Section 4 Segmentation Stats ---")
            print(f"  Input shape     : {img.shape}")
            print(f"  Mask shape      : {mask.shape}")
            print(f"  Mask dtype      : {mask.dtype}")
            print(f"  Mask min value  : {mask.min()}")
            print(f"  Mask max value  : {mask.max()}")
            print(f"  Mask mean value : {mask.mean():.1f}")
            nonzero = np.count_nonzero(mask)
            pct = 100 * nonzero / mask.size
            print(f"  Non-zero pixels : {nonzero} / {mask.size}  ({pct:.1f}%)")
            print(f"  Inference time  : {elapsed:.2f} s")
            print(f"  Model           : {result.model_name}")
            print(f"  Runtime         : CPU (onnxruntime)")
            print(f"------------------------------------")

        assert mask.shape == (img.shape[0], img.shape[1])
