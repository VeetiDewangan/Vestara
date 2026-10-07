"""
Image processing service.

Orchestrates the image-upload pipeline for Section 2:
  1. Enforce maximum file size before doing any decoding work.
  2. Delegate byte-level validation to image_utils (format, decode, dimensions).
  3. Persist only validated images to INPUT_DIR under a safe UUID filename.

Pre-processing, segmentation, mask refinement, background removal, and
transparent-PNG export will be added in later sections.
"""

import os
from dataclasses import dataclass

from app.config import INPUT_DIR, MAX_UPLOAD_SIZE
from app.utils.image_utils import ImageMetadata, safe_filename, validate_image_bytes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    """Everything the API route needs to build its success response."""
    filename: str       # UUID-based safe filename (no directory component)
    format: str
    width: int
    height: int
    file_size: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_upload(data: bytes, original_filename: str) -> UploadResult:
    """
    Validate uploaded image bytes and save the file if valid.

    Args:
        data:              Raw bytes read from the UploadFile.
        original_filename: The filename string provided by the client
                           (used only for cosmetic extension selection).

    Returns:
        UploadResult containing metadata about the stored image.

    Raises:
        ValueError: propagated from validate_image_bytes with a clear
                    human-readable message so the route can turn it into
                    an HTTP 400 response.
        FileTooLargeError: when the byte length exceeds MAX_UPLOAD_SIZE.
    """
    # --- 1. Size gate (fast, before any decoding work) --------------------
    if len(data) > MAX_UPLOAD_SIZE:
        raise FileTooLargeError(
            f"File size ({len(data):,} bytes) exceeds the "
            f"{MAX_UPLOAD_SIZE // (1024 * 1024)} MB limit."
        )

    # --- 2. Validate (format, decode, dimensions) -------------------------
    meta: ImageMetadata = validate_image_bytes(data)

    # --- 3. Persist -------------------------------------------------------
    filename = safe_filename(original_filename, meta.format)
    dest_path = os.path.join(INPUT_DIR, filename)

    # Ensure the directory exists (it should, but be defensive).
    os.makedirs(INPUT_DIR, exist_ok=True)

    with open(dest_path, "wb") as fh:
        fh.write(data)

    return UploadResult(
        filename=filename,
        format=meta.format,
        width=meta.width,
        height=meta.height,
        file_size=meta.file_size,
    )


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class FileTooLargeError(Exception):
    """Raised when the upload exceeds the configured size limit."""

