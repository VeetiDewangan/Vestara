"""
Image utilities.

Pure helper functions for image validation and safe filename generation.
All functions operate on raw bytes or plain Python types so they remain
fully independent of FastAPI and can be called from tests or other services
without spinning up the web layer.

Section 2 scope:
  - Magic-byte format detection
  - OpenCV decode + dimension check
  - Safe UUID-based filename generation

Functions for pre/post-processing and transparent-PNG export will be added
in later sections.
"""

import os
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_FORMATS,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageMetadata:
    """Validated image information returned to callers."""
    format: str     # e.g. "JPEG", "PNG", "WEBP"
    width: int
    height: int
    file_size: int  # raw byte length of the upload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_format(data: bytes) -> str | None:
    """
    Identify the image format by inspecting leading magic bytes.

    Returns the format name (e.g. "JPEG") or None if unrecognised.
    Trusting magic bytes is more reliable than trusting the filename
    extension or the client-supplied Content-Type header.
    """
    for fmt, signatures in ALLOWED_FORMATS.items():
        for sig in signatures:
            if data[:len(sig)] == sig:
                return fmt
    return None


def _decode_with_opencv(data: bytes) -> np.ndarray | None:
    """
    Attempt to decode raw bytes into an OpenCV image array.

    Returns the decoded ndarray or None if OpenCV cannot parse the data.
    This catches truncated files, corrupted JPEG/PNG streams, and anything
    that merely looks like an image by its magic bytes but is not.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img  # None when decoding fails


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_image_bytes(data: bytes) -> ImageMetadata:
    """
    Validate raw image bytes and return its metadata.

    Raises:
        ValueError: with a human-readable reason when validation fails.

    The validation pipeline is:
        1. Non-empty check
        2. Magic-byte format detection (allowed formats only)
        3. OpenCV decode (catches corruption)
        4. Dimension check (>= MIN_IMAGE_WIDTH x MIN_IMAGE_HEIGHT)
    """
    if not data:
        raise ValueError("File is empty.")

    detected_format = _detect_format(data)
    if detected_format is None:
        raise ValueError(
            "Unsupported or unrecognised image format. "
            f"Accepted formats: {', '.join(ALLOWED_FORMATS)}."
        )

    img = _decode_with_opencv(data)
    if img is None:
        raise ValueError(
            "Image data is corrupted or could not be decoded. "
            "Please upload a valid, non-truncated image file."
        )

    height, width = img.shape[:2]

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise ValueError(
            f"Image is too small ({width}x{height} px). "
            f"Minimum required dimensions are "
            f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} px."
        )

    return ImageMetadata(
        format=detected_format,
        width=width,
        height=height,
        file_size=len(data),
    )


def safe_filename(original_name: str, detected_format: str) -> str:
    """
    Generate a collision-free, path-traversal-safe filename.

    The client-supplied name is intentionally discarded.  Instead, we
    generate a UUID4 stem and derive the extension from the *detected*
    format (not from whatever extension the client sent).

    Args:
        original_name:   The raw filename string sent by the client
                         (used only to attempt a best-effort extension
                         fallback for JPEG variants -- never used as a path).
        detected_format: The format string from validate_image_bytes,
                         e.g. "JPEG".

    Returns:
        A safe filename such as ``8d8a0c7e-1234-....jpg``.
    """
    FORMAT_TO_EXT: dict[str, str] = {
        "JPEG": ".jpg",
        "PNG":  ".png",
        "WEBP": ".webp",
    }
    ext = FORMAT_TO_EXT.get(detected_format, ".jpg")

    # If the client sent .jpeg instead of .jpg, keep that cosmetically
    # (both are equally safe; only do this for JPEG variants).
    client_ext = os.path.splitext(original_name)[1].lower()
    if client_ext in ALLOWED_EXTENSIONS and detected_format == "JPEG":
        ext = client_ext

    return f"{uuid.uuid4()}{ext}"

