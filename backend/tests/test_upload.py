"""
Tests for Section 2: image upload and validation.

Coverage:
  1.  Valid JPEG upload
  2.  Valid PNG upload
  3.  Valid WEBP upload
  4.  Empty file upload
  5.  Random bytes (not a real image)
  6.  Unsupported format (BMP magic bytes)
  7.  Image smaller than 256x256
  8.  File larger than 10 MB
  9.  Filename with unsafe path characters (path traversal attempt)
  10. GET / still works (Section 1 regression)
  11. GET /health still works (Section 1 regression)

All test images are generated programmatically in memory.
Valid uploads are cleaned up after each test so input_images/ is not polluted.
"""

import io
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import INPUT_DIR, MAX_UPLOAD_SIZE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 512, height: int = 512) -> bytes:
    """Return valid JPEG bytes for a solid-colour test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (100, 149, 237)  # cornflower blue
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok, "OpenCV failed to encode test JPEG"
    return buf.tobytes()


def _make_png(width: int = 512, height: int = 512) -> bytes:
    """Return valid PNG bytes for a solid-colour test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)  # forest green
    ok, buf = cv2.imencode(".png", img)
    assert ok, "OpenCV failed to encode test PNG"
    return buf.tobytes()


def _make_webp(width: int = 512, height: int = 512) -> bytes:
    """Return valid WEBP bytes for a solid-colour test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (255, 165, 0)  # orange
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 80])
    assert ok, "OpenCV failed to encode test WEBP"
    return buf.tobytes()


UPLOAD_URL = "/api/v1/clothing/upload"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Shared TestClient for all tests in this module."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def cleanup_input_images():
    """
    After every test, delete any files added to input_images/ during that
    test so the directory stays clean.  Files present before the test are
    left untouched.
    """
    before = set(os.listdir(INPUT_DIR)) if os.path.isdir(INPUT_DIR) else set()
    yield
    if os.path.isdir(INPUT_DIR):
        after = set(os.listdir(INPUT_DIR))
        for name in after - before:
            path = os.path.join(INPUT_DIR, name)
            try:
                os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Section 1 regression tests
# ---------------------------------------------------------------------------

class TestBootstrapEndpoints:
    def test_root_returns_200(self, client):
        """GET / must still work exactly as in Section 1."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Clothing Background Removal API is running"}

    def test_health_returns_200(self, client):
        """GET /health must still work exactly as in Section 1."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Valid uploads
# ---------------------------------------------------------------------------

class TestValidUploads:
    def test_valid_jpeg(self, client):
        """Test 1 — Valid JPEG: must return 200 with correct metadata."""
        data = _make_jpeg()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("shirt.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["format"] == "JPEG"
        assert body["width"] == 512
        assert body["height"] == 512
        assert body["file_size"] == len(data)
        # Filename must be UUID-based, never the original
        assert "shirt.jpg" not in body["filename"]
        assert body["filename"].endswith(".jpg") or body["filename"].endswith(".jpeg")

    def test_valid_png(self, client):
        """Test 2 — Valid PNG: must return 200 with correct metadata."""
        data = _make_png()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("jacket.png", io.BytesIO(data), "image/png")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["format"] == "PNG"
        assert body["width"] == 512
        assert body["height"] == 512

    def test_valid_webp(self, client):
        """Test 3 — Valid WEBP: must return 200 with correct metadata."""
        data = _make_webp()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("trousers.webp", io.BytesIO(data), "image/webp")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True
        assert body["format"] == "WEBP"

    def test_valid_image_is_stored_in_input_dir(self, client):
        """Successful upload must persist the file inside input_images/."""
        before = set(os.listdir(INPUT_DIR)) if os.path.isdir(INPUT_DIR) else set()
        data = _make_jpeg()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("tshirt.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert response.status_code == 200
        after = set(os.listdir(INPUT_DIR))
        new_files = after - before
        assert len(new_files) == 1, "Exactly one file should have been saved."
        stored_name = new_files.pop()
        assert response.json()["filename"] == stored_name


# ---------------------------------------------------------------------------
# Invalid uploads — must be rejected and NOT stored
# ---------------------------------------------------------------------------

class TestInvalidUploads:
    def _count_new_files(self, before: set, after: set) -> int:
        return len(after - before)

    def _snapshot(self):
        return set(os.listdir(INPUT_DIR)) if os.path.isdir(INPUT_DIR) else set()

    def test_empty_upload(self, client):
        """Test 4 — Empty file: must return 400."""
        snap = self._snapshot()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert response.status_code == 400, response.text
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_random_bytes_rejected(self, client):
        """Test 5 — Random bytes with .jpg extension: must return 400."""
        snap = self._snapshot()
        garbage = os.urandom(4096)
        response = client.post(
            UPLOAD_URL,
            files={"file": ("fake.jpg", io.BytesIO(garbage), "image/jpeg")},
        )
        assert response.status_code == 400, response.text
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_unsupported_format_bmp(self, client):
        """Test 6 — BMP format (not in allowed list): must return 400."""
        snap = self._snapshot()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".bmp", img)
        assert ok
        bmp_bytes = buf.tobytes()
        response = client.post(
            UPLOAD_URL,
            files={"file": ("image.bmp", io.BytesIO(bmp_bytes), "image/bmp")},
        )
        assert response.status_code == 400, response.text
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_image_too_small(self, client):
        """Test 7 — Image smaller than 256x256: must return 400."""
        snap = self._snapshot()
        data = _make_jpeg(width=128, height=128)
        response = client.post(
            UPLOAD_URL,
            files={"file": ("tiny.jpg", io.BytesIO(data), "image/jpeg")},
        )
        assert response.status_code == 400, response.text
        assert "too small" in response.json()["detail"].lower()
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_file_too_large(self, client):
        """Test 8 — File larger than 10 MB: must return 413."""
        snap = self._snapshot()
        # Build an oversized JPEG by creating a large raw image
        oversized = b"\xff\xd8\xff" + b"\x00" * (MAX_UPLOAD_SIZE + 1)
        response = client.post(
            UPLOAD_URL,
            files={"file": ("huge.jpg", io.BytesIO(oversized), "image/jpeg")},
        )
        assert response.status_code == 413, response.text
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_path_traversal_filename(self, client):
        """Test 9 — Unsafe filename (../../etc/passwd): must succeed (valid image)
        but store under a safe UUID name, never the traversal path."""
        data = _make_jpeg()
        response = client.post(
            UPLOAD_URL,
            files={
                "file": (
                    "../../etc/passwd",
                    io.BytesIO(data),
                    "image/jpeg",
                )
            },
        )
        # The image itself is valid; the endpoint should accept it...
        assert response.status_code == 200, response.text
        stored_name = response.json()["filename"]
        # ...but under a safe UUID-based name, not the traversal path
        assert ".." not in stored_name
        assert "/" not in stored_name
        assert "\\" not in stored_name

    def test_corrupted_jpeg_rejected(self, client):
        """JPEG magic bytes but corrupted body: must return 400."""
        snap = self._snapshot()
        corrupted = b"\xff\xd8\xff" + os.urandom(512)
        response = client.post(
            UPLOAD_URL,
            files={"file": ("corrupt.jpg", io.BytesIO(corrupted), "image/jpeg")},
        )
        assert response.status_code == 400, response.text
        assert self._count_new_files(snap, self._snapshot()) == 0

    def test_no_internal_path_in_error(self, client):
        """Error responses must not leak filesystem paths."""
        garbage = os.urandom(512)
        response = client.post(
            UPLOAD_URL,
            files={"file": ("bad.jpg", io.BytesIO(garbage), "image/jpeg")},
        )
        assert response.status_code == 400
        detail = response.json().get("detail", "")
        assert "input_images" not in detail
        assert "C:\\" not in detail
        assert "/home/" not in detail
