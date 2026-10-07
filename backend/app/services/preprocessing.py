"""
Preprocessing service  (Section 3).

Transforms a validated image into a consistent representation that can be
passed directly to the AI segmentation model in Section 4.

Pipeline (in-memory — original file on disk is never modified):

    uploaded bytes
        ↓
    OpenCV decode  →  BGR ndarray
        ↓
    channel normalisation  (grayscale / RGBA  → 3-channel BGR)
        ↓
    proportional downscale  (only when largest dim > MAX_PREPROCESSING_DIMENSION)
        ↓
    BGR → RGB  (one explicit conversion, in one place)
        ↓
    PreprocessedImage  (H × W × 3, uint8, RGB)

What this module intentionally does NOT do:
  - Pixel normalisation to [0, 1]  — that is model-specific (Section 4).
  - Padding / letterboxing          — that is model-specific (Section 4).
  - Any file I/O                    — preprocessing is purely in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import MAX_PREPROCESSING_DIMENSION


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedImage:
    """
    The output of the preprocessing pipeline.

    Attributes:
        image:            NumPy ndarray, shape H×W×3, dtype uint8, RGB colour order.
        original_width:   Width of the image as decoded from the uploaded bytes.
        original_height:  Height of the image as decoded from the uploaded bytes.
        processed_width:  Width after preprocessing (equals original when no resize).
        processed_height: Height after preprocessing (equals original when no resize).
        was_resized:      True when the image was scaled down during preprocessing.
    """
    image: np.ndarray
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    was_resized: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_three_channel_bgr(img: np.ndarray) -> np.ndarray:
    """
    Guarantee the image has exactly 3 channels in BGR order.

    Handles:
      - 2-D (grayscale, H×W)     → replicate to H×W×3 BGR
      - 3-D, 1 channel (H×W×1)  → replicate to H×W×3 BGR
      - 3-D, 3 channels (H×W×3) → returned as-is (assumed BGR from OpenCV)
      - 3-D, 4 channels (H×W×4) → drop alpha, keep BGR

    Raises:
        ValueError: for unexpected channel counts.
    """
    if img.ndim == 2:
        # Grayscale H×W → H×W×3 (BGR)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    channels = img.shape[2]

    if channels == 1:
        # H×W×1 → H×W×3
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if channels == 3:
        # Already BGR (standard OpenCV output)
        return img

    if channels == 4:
        # BGRA → BGR  (drop alpha channel)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(
        f"Unexpected number of image channels: {channels}. "
        "Expected 1, 3, or 4."
    )


def _compute_resize_dims(
    width: int,
    height: int,
    max_dim: int,
) -> tuple[int, int]:
    """
    Compute new (width, height) so that the largest dimension equals max_dim.

    The aspect ratio is preserved exactly.  If both dimensions are already
    within max_dim, the original values are returned unchanged (no upscaling).
    """
    if width <= max_dim and height <= max_dim:
        return width, height

    if width >= height:
        new_width = max_dim
        new_height = max(1, round(height * max_dim / width))
    else:
        new_height = max_dim
        new_width = max(1, round(width * max_dim / height))

    return new_width, new_height


def _resize_if_needed(
    img: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, bool]:
    """
    Scale the image down if its largest dimension exceeds max_dim.

    Returns:
        (image, was_resized) — the (possibly unchanged) image and a flag
        indicating whether a resize actually took place.
    """
    h, w = img.shape[:2]
    new_w, new_h = _compute_resize_dims(w, h, max_dim)

    if new_w == w and new_h == h:
        return img, False

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image_bytes(data: bytes) -> PreprocessedImage:
    """
    Run the full preprocessing pipeline on raw image bytes.

    This function is the single entry point for Section 3 preprocessing.
    It is intended to be called by the image-processing service after the
    Section 2 validation step has already confirmed that the bytes are a
    valid, supported image.

    Pipeline:
        1. Decode bytes with OpenCV  →  BGR ndarray
        2. Normalise channels        →  3-channel BGR
        3. Record original dimensions
        4. Proportional downscale    (only if largest dim > MAX_PREPROCESSING_DIMENSION)
        5. BGR → RGB conversion      (one explicit, clearly labelled step)
        6. Return PreprocessedImage

    Args:
        data: Raw bytes of a validated image (JPEG / PNG / WEBP).

    Returns:
        PreprocessedImage with a uint8 RGB ndarray ready for model inference.

    Raises:
        ValueError: if the bytes cannot be decoded or produce an unexpected
                    channel layout, with a clear human-readable message.
    """
    if not data:
        raise ValueError("Cannot preprocess empty image data.")

    # --- 1. Decode -----------------------------------------------------------
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if bgr is None:
        raise ValueError(
            "Image could not be decoded during preprocessing. "
            "Ensure the image passed Section 2 validation first."
        )

    # --- 2. Channel normalisation --------------------------------------------
    bgr = _ensure_three_channel_bgr(bgr)

    # --- 3. Record original dimensions (after channel fix, before resize) ---
    original_height, original_width = bgr.shape[:2]

    # --- 4. Proportional downscale -------------------------------------------
    bgr, was_resized = _resize_if_needed(bgr, MAX_PREPROCESSING_DIMENSION)

    processed_height, processed_width = bgr.shape[:2]

    # --- 5. BGR → RGB  (explicit, single conversion point) ------------------
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- 6. Sanity check: ensure output dtype is uint8 ----------------------
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    return PreprocessedImage(
        image=rgb,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
        was_resized=was_resized,
    )
