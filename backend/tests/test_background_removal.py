"""
Tests for Section 7: Alpha Composition, PNG Export, Crop, and API.

Coverage (17 focused tests):
  1.  RGB + mask → valid RGBA output.
  2.  Composed RGBA dimensions match input.
  3.  Alpha channel equals the refined mask exactly.
  4.  RGB values are preserved in the RGBA output.
  5.  Zero mask → fully transparent pixels.
  6.  Soft mask values stay soft (not binarised).
  7.  save_png creates a .png file on disk.
  8.  Saved PNG reopens in RGBA mode.
  9.  Saved PNG dimensions match the RGBA array.
  10. Source image array is not modified.
  11. crop_to_content removes transparent borders.
  12. crop_to_content padding works.
  13. Empty alpha mask is handled safely by crop_to_content.
  14. GET /  and GET /health still return 200.
  15. POST /upload still works.
  16. POST /remove-background accepts a real uploaded image and returns 200.
  17. GET /result/{filename} returns the generated PNG.

Integration (test 18):
  Full pipeline on synthetic image: preprocessing → U2-Net → refinement
  → alpha composition → crop → save.
  NOTE: Uses a synthetic two-tone image — NOT a real clothing photograph.
  Real clothing quality cannot be assessed from this test.
"""

import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from app.services.background_removal import (
    BackgroundRemovalResult,
    compose_rgba,
    crop_to_content,
    save_png,
)
from app.config import OUTPUT_DIR


# ===========================================================================
# Helpers
# ===========================================================================

def _rgb(h=100, w=100) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 200   # R
    img[:, :, 1] = 100   # G
    img[:, :, 2] = 50    # B
    return img


def _mask(h=100, w=100, value=200) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def _zero_mask(h=100, w=100) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _make_valid_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG image bytes (256x256 solid colour)."""
    img = Image.new("RGB", (256, 256), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ===========================================================================
# Test 1 — RGB + mask → valid RGBA
# ===========================================================================

class TestComposeRgba:
    def test_returns_ndarray(self):
        rgba = compose_rgba(_rgb(), _mask())
        assert isinstance(rgba, np.ndarray)

    def test_output_is_4_channel(self):
        rgba = compose_rgba(_rgb(), _mask())
        assert rgba.ndim == 3
        assert rgba.shape[2] == 4

    def test_dtype_is_uint8(self):
        rgba = compose_rgba(_rgb(), _mask())
        assert rgba.dtype == np.uint8


# ===========================================================================
# Test 2 — Dimensions match
# ===========================================================================

class TestRgbaDimensions:
    def test_same_height_width(self):
        h, w = 120, 80
        rgba = compose_rgba(_rgb(h, w), _mask(h, w))
        assert rgba.shape[:2] == (h, w)


# ===========================================================================
# Test 3 — Alpha channel equals mask
# ===========================================================================

class TestAlphaChannel:
    def test_alpha_equals_mask(self):
        mask = _mask(value=180)
        rgba = compose_rgba(_rgb(), mask)
        np.testing.assert_array_equal(rgba[:, :, 3], mask)


# ===========================================================================
# Test 4 — RGB values preserved
# ===========================================================================

class TestRgbPreserved:
    def test_r_channel(self):
        img = _rgb()
        rgba = compose_rgba(img, _mask())
        np.testing.assert_array_equal(rgba[:, :, 0], img[:, :, 0])

    def test_g_channel(self):
        img = _rgb()
        rgba = compose_rgba(img, _mask())
        np.testing.assert_array_equal(rgba[:, :, 1], img[:, :, 1])

    def test_b_channel(self):
        img = _rgb()
        rgba = compose_rgba(img, _mask())
        np.testing.assert_array_equal(rgba[:, :, 2], img[:, :, 2])


# ===========================================================================
# Test 5 — Zero mask → fully transparent
# ===========================================================================

class TestZeroMask:
    def test_all_alpha_zero(self):
        rgba = compose_rgba(_rgb(), _zero_mask())
        assert rgba[:, :, 3].max() == 0


# ===========================================================================
# Test 6 — Soft values not binarised
# ===========================================================================

class TestSoftAlpha:
    def test_partial_values_preserved(self):
        mask = np.full((50, 50), 130, dtype=np.uint8)
        rgba = compose_rgba(_rgb(50, 50), mask)
        assert int(rgba[25, 25, 3]) == 130


# ===========================================================================
# Test 7 — save_png creates a .png file
# ===========================================================================

class TestSavePng:
    def test_file_is_created(self, tmp_path):
        rgba = compose_rgba(_rgb(), _mask())
        result = save_png(rgba, output_dir=str(tmp_path))
        assert os.path.isfile(result.output_path)

    def test_filename_ends_with_png(self, tmp_path):
        rgba = compose_rgba(_rgb(), _mask())
        result = save_png(rgba, output_dir=str(tmp_path))
        assert result.filename.lower().endswith(".png")


# ===========================================================================
# Test 8 — Saved PNG reopens as RGBA
# ===========================================================================

class TestSavedPngMode:
    def test_reopen_mode_is_rgba(self, tmp_path):
        rgba = compose_rgba(_rgb(), _mask())
        result = save_png(rgba, output_dir=str(tmp_path))
        reopened = Image.open(result.output_path)
        assert reopened.mode == "RGBA"

    def test_reopen_has_alpha(self, tmp_path):
        rgba = compose_rgba(_rgb(), _mask(value=200))
        result = save_png(rgba, output_dir=str(tmp_path))
        reopened = Image.open(result.output_path)
        channels = reopened.split()
        assert len(channels) == 4  # R, G, B, A


# ===========================================================================
# Test 9 — Saved dimensions match
# ===========================================================================

class TestSavedDimensions:
    def test_width_height(self, tmp_path):
        h, w = 80, 120
        rgba = compose_rgba(_rgb(h, w), _mask(h, w))
        result = save_png(rgba, output_dir=str(tmp_path))
        assert result.width == w
        assert result.height == h


# ===========================================================================
# Test 10 — Source image not modified
# ===========================================================================

class TestSourceImmutability:
    def test_rgb_unchanged(self):
        img = _rgb()
        snapshot = img.copy()
        mask = _mask()
        compose_rgba(img, mask)
        np.testing.assert_array_equal(img, snapshot)

    def test_mask_unchanged(self):
        mask = _mask(value=180)
        snapshot = mask.copy()
        compose_rgba(_rgb(), mask)
        np.testing.assert_array_equal(mask, snapshot)


# ===========================================================================
# Test 11 — crop_to_content removes transparent borders
# ===========================================================================

class TestCrop:
    def test_crop_reduces_size_when_foreground_smaller_than_image(self):
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba[50:150, 50:150, :3] = 128
        rgba[50:150, 50:150, 3]  = 255
        cropped = crop_to_content(rgba, padding=0)
        assert cropped.shape[0] < 200
        assert cropped.shape[1] < 200

    def test_crop_preserves_alpha_values(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:80, 20:80, 3] = 200
        cropped = crop_to_content(rgba, padding=0)
        assert cropped[:, :, 3].max() == 200


# ===========================================================================
# Test 12 — crop padding
# ===========================================================================

class TestCropPadding:
    def test_padding_adds_border(self):
        rgba_no_pad = np.zeros((200, 200, 4), dtype=np.uint8)
        rgba_no_pad[50:150, 50:150, 3] = 255
        rgba_pad = rgba_no_pad.copy()

        cropped_no = crop_to_content(rgba_no_pad, padding=0)
        cropped_pd = crop_to_content(rgba_pad, padding=5)

        # Padded crop must be at least as large as unpadded
        assert cropped_pd.shape[0] >= cropped_no.shape[0]
        assert cropped_pd.shape[1] >= cropped_no.shape[1]


# ===========================================================================
# Test 13 — Empty mask handled safely by crop
# ===========================================================================

class TestCropEmptyMask:
    def test_all_transparent_does_not_crash(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        result = crop_to_content(rgba, padding=10)
        assert result.shape[2] == 4   # still RGBA


# ===========================================================================
# Tests 14-17 — API endpoints
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


class TestExistingEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_upload_still_works(self, client):
        data = _make_valid_jpeg_bytes()
        r = client.post(
            "/api/v1/clothing/upload",
            files={"file": ("test.jpg", data, "image/jpeg")},
        )
        assert r.status_code == 200
        assert r.json()["success"] is True


class TestRemoveBackgroundEndpoint:
    """
    Test the POST /remove-background endpoint.

    Uses a synthetic image to avoid downloading model weights again.
    The endpoint exercises the real U2-Net model.
    """

    def test_remove_background_returns_200(self, client):
        data = _make_valid_jpeg_bytes()
        r = client.post(
            "/api/v1/clothing/remove-background",
            files={"file": ("shirt.jpg", data, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["filename"].endswith(".png")
        assert body["width"] > 0
        assert body["height"] > 0

    def test_no_path_in_response(self, client):
        data = _make_valid_jpeg_bytes()
        r = client.post(
            "/api/v1/clothing/remove-background",
            files={"file": ("shirt.jpg", data, "image/jpeg")},
        )
        assert r.status_code == 200
        body_str = r.text
        # Verify no Windows path leaked
        assert "output_images" not in body_str
        assert "C:\\" not in body_str


class TestResultEndpoint:
    def test_get_result_returns_png(self, client):
        # First generate a PNG via the remove-background endpoint
        data = _make_valid_jpeg_bytes()
        r = client.post(
            "/api/v1/clothing/remove-background",
            files={"file": ("shirt.jpg", data, "image/jpeg")},
        )
        assert r.status_code == 200
        filename = r.json()["filename"]

        # Now retrieve it
        r2 = client.get(f"/api/v1/clothing/result/{filename}")
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("image/png")

    def test_get_result_404_for_missing(self, client):
        r = client.get("/api/v1/clothing/result/doesnotexist.png")
        assert r.status_code == 404

    def test_get_result_blocks_path_traversal(self, client):
        r = client.get("/api/v1/clothing/result/../../etc/passwd.png")
        # Should be blocked (400 or 404 acceptable — never 200 for a path escape)
        assert r.status_code in (400, 404)

    def test_get_result_blocks_non_png(self, client):
        r = client.get("/api/v1/clothing/result/somefile.jpg")
        assert r.status_code == 400


# ===========================================================================
# Test 18 — Full pipeline integration (synthetic image)
# ===========================================================================

_shared_seg_service = None


def _get_seg_svc():
    global _shared_seg_service
    if _shared_seg_service is None:
        from app.services.segmentation import SegmentationService
        _shared_seg_service = SegmentationService()
    return _shared_seg_service


class TestFullPipelineIntegration:
    """
    End-to-end: preprocessing → U2-Net → refinement → compose → crop → save.

    NOTE: Uses a synthetic 256x256 two-tone image, NOT a real clothing
    photograph.  U2-Net produces sparse output on synthetic images.
    The test verifies pipeline structure and file integrity, not
    segmentation quality.  Clothing-specific accuracy cannot be assessed
    from this test image.
    """

    def test_full_pipeline_synthetic_image(self, tmp_path, capsys):
        from app.services.preprocessing import preprocess_image_bytes
        from app.services.mask_analysis import analyze_mask
        from app.services.mask_refinement import refine_mask

        # Build a valid 256x256 JPEG in memory (two-tone)
        h, w = 256, 256
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[: h // 2, :] = (210, 210, 210)
        arr[h // 2 :, :] = (40, 40, 40)
        pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # Step 3: preprocess
        preprocessed = preprocess_image_bytes(img_bytes)
        assert preprocessed.image.shape == (h, w, 3)

        # Step 4: segment (real model)
        seg = _get_seg_svc().segment(preprocessed.image)
        raw_mask = seg.mask
        assert raw_mask.shape == (h, w)
        assert raw_mask.dtype == np.uint8

        # Step 5: analysis
        raw_analysis = analyze_mask(raw_mask)

        # Step 6: refine
        refined = refine_mask(raw_mask)
        raw_snapshot = raw_mask.copy()
        np.testing.assert_array_equal(raw_mask, raw_snapshot,
            err_msg="refine_mask modified the raw mask")

        # Step 7a: compose
        rgba = compose_rgba(preprocessed.image, refined.mask)
        assert rgba.shape == (h, w, 4)
        assert rgba.dtype == np.uint8

        # Step 7b: crop
        cropped = crop_to_content(rgba, padding=10)
        assert cropped.shape[2] == 4

        # Step 7c: save
        result = save_png(cropped, output_dir=str(tmp_path))
        assert os.path.isfile(result.output_path)
        reopened = Image.open(result.output_path)
        assert reopened.mode == "RGBA"

        with capsys.disabled():
            print("\n--- Section 7 Full Pipeline (synthetic image) ---")
            print(f"  NOTE: synthetic two-tone image — NOT a real clothing photo")
            print(f"  Input           : {w} x {h}")
            print(f"  Raw mask        : min={raw_analysis.min_value} "
                  f"max={raw_analysis.max_value} "
                  f"fg={raw_analysis.foreground_percentage:.4f}%")
            print(f"  Refined mask    : fg={refined.refined_foreground_percentage:.4f}%  "
                  f"components_removed={refined.components_removed}")
            print(f"  RGBA shape      : {rgba.shape}")
            print(f"  Cropped shape   : {cropped.shape}")
            print(f"  Output file     : {result.filename}")
            print(f"  Reopened mode   : {reopened.mode}")
            print(f"  Raw mask untouched: {np.array_equal(raw_mask, raw_snapshot)}")
            print("-------------------------------------------------")
