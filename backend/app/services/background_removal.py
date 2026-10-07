"""
Background removal service  (Section 7 - Part 1 & 2).

Combines a preprocessed RGB image with a refined soft mask to produce
a transparent PNG (RGBA).  Optionally applies a tight crop to remove
unnecessary transparent borders.

Key design decisions
--------------------
- RGB channels come directly from the preprocessed image, unchanged.
- Alpha channel = refined soft mask (values 0-255, soft edges preserved).
- No pixel values are invented; no hard binarisation of the alpha channel.
- The original RGB array and the mask are NEVER modified.
- PNG is the only output format because JPEG does not support transparency.
- Output filenames are UUID-based to prevent collisions and path traversal.
- Saved files land in OUTPUT_DIR only; no other directory is written to.

Pipeline position:
    preprocessed RGB  (H x W x 3 uint8)
    +
    refined mask      (H x W uint8, soft values)
        |
        v
    compose_rgba()
        |
        v
    crop_to_content()     <- optional
        |
        v
    save_png()
        |
        v
    BackgroundRemovalResult
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from app.config import FINAL_CROP_PADDING, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class BackgroundRemovalResult:
    """
    Output of the full background-removal pipeline.

    Attributes
    ----------
    filename :
        UUID-based PNG filename (basename only, no directory component).
        Safe to include in API responses.

    width, height :
        Dimensions of the saved PNG (after optional crop).

    original_width, original_height :
        Dimensions of the preprocessed input image (before crop).

    output_path :
        Absolute path to the saved file.  NOT returned to API clients.
    """
    filename: str
    width: int
    height: int
    original_width: int
    original_height: int
    output_path: str


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_rgb(image_rgb: np.ndarray) -> None:
    if image_rgb is None:
        raise ValueError("image_rgb must not be None.")
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError(f"image_rgb must be an ndarray, got {type(image_rgb).__name__}.")
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"image_rgb dtype must be uint8, got {image_rgb.dtype}.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"image_rgb must be H x W x 3, got shape {image_rgb.shape}."
        )
    if image_rgb.size == 0:
        raise ValueError("image_rgb must not be empty.")


def _validate_mask(mask: np.ndarray) -> None:
    if mask is None:
        raise ValueError("mask must not be None.")
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"mask must be an ndarray, got {type(mask).__name__}.")
    if mask.dtype != np.uint8:
        raise ValueError(f"mask dtype must be uint8, got {mask.dtype}.")
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-dimensional (H x W), got shape {mask.shape}."
        )
    if mask.size == 0:
        raise ValueError("mask must not be empty.")


def _validate_compatible(image_rgb: np.ndarray, mask: np.ndarray) -> None:
    ih, iw = image_rgb.shape[:2]
    mh, mw = mask.shape
    if (ih, iw) != (mh, mw):
        raise ValueError(
            f"image_rgb shape ({ih}x{iw}) does not match mask shape ({mh}x{mw})."
        )


# ---------------------------------------------------------------------------
# Alpha composition
# ---------------------------------------------------------------------------

def compose_rgba(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Combine an RGB image and a soft mask into an RGBA array.

    The alpha channel is taken directly from the mask — soft values
    (0-255) are preserved.  No hard binarisation is applied.

    Args:
        image_rgb : H x W x 3 uint8 RGB array.
        mask      : H x W uint8 soft mask (0 = transparent, 255 = opaque).

    Returns:
        H x W x 4 uint8 RGBA array.  The input arrays are not modified.

    Raises:
        ValueError: if inputs are invalid or shapes do not match.
    """
    _validate_rgb(image_rgb)
    _validate_mask(mask)
    _validate_compatible(image_rgb, mask)

    h, w = image_rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb          # copy RGB channels
    rgba[:, :, 3]  = mask               # soft mask as alpha
    return rgba


# ---------------------------------------------------------------------------
# Tight crop
# ---------------------------------------------------------------------------

def crop_to_content(
    rgba: np.ndarray,
    padding: int = FINAL_CROP_PADDING,
) -> np.ndarray:
    """
    Crop the RGBA image to the bounding box of non-zero alpha pixels,
    with a small configurable padding around the detected foreground.

    If the alpha channel is completely zero (empty mask), the original
    array is returned unchanged so that downstream code never crashes.

    Args:
        rgba    : H x W x 4 uint8 RGBA array.
        padding : Extra pixels added on each side of the bounding box.
                  Clamped to image dimensions.

    Returns:
        Cropped H' x W' x 4 uint8 RGBA array (or the original if empty).
        Alpha values are NOT modified during crop.
        The input array is NOT modified.
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be a H x W x 4 uint8 array.")

    alpha = rgba[:, :, 3]

    # Find rows and columns that contain at least one non-zero alpha pixel
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        # Completely transparent — return a copy to avoid aliasing
        return rgba.copy()

    h, w = rgba.shape[:2]
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))

    # Apply padding, clamped to image boundaries
    y_min = max(0, y_min - padding)
    y_max = min(h - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w - 1, x_max + padding)

    return rgba[y_min : y_max + 1, x_min : x_max + 1].copy()


# ---------------------------------------------------------------------------
# Save as PNG
# ---------------------------------------------------------------------------

def save_png(
    rgba: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> BackgroundRemovalResult:
    """
    Save an RGBA array as a PNG file in output_dir.

    Uses Pillow so that RGBA mode is explicitly preserved.  The output
    filename is UUID-based to prevent collisions.

    Args:
        rgba       : H x W x 4 uint8 RGBA array.
        output_dir : Directory to write the file to.  Created if absent.

    Returns:
        BackgroundRemovalResult with filename, dimensions, and output path.

    Raises:
        ValueError: if rgba is not a valid H x W x 4 uint8 array.
    """
    if rgba is None or not isinstance(rgba, np.ndarray):
        raise ValueError("rgba must be a NumPy ndarray.")
    if rgba.dtype != np.uint8:
        raise ValueError(f"rgba dtype must be uint8, got {rgba.dtype}.")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba must be H x W x 4, got shape {rgba.shape}.")

    os.makedirs(output_dir, exist_ok=True)

    filename  = f"{uuid.uuid4().hex}.png"
    out_path  = os.path.join(output_dir, filename)

    h, w = rgba.shape[:2]
    pil_image = Image.fromarray(rgba, mode="RGBA")
    pil_image.save(out_path, format="PNG")

    return BackgroundRemovalResult(
        filename=filename,
        width=w,
        height=h,
        original_width=w,    # caller can override after crop
        original_height=h,
        output_path=out_path,
    )
