"""
Application configuration.

Centralises all configurable values so they can be adjusted in one place
without touching business logic.  More settings (model paths, thresholds,
etc.) will be added here as the project grows through later sections.
"""

import os

# Load backend/.env first so HOST / PORT / GEMINI_* set there actually apply.
# (Without this, run.py reads HOST before .env is loaded and `HOST=0.0.0.0`
# would be silently ignored, which stops a real phone from connecting.)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:  # python-dotenv is optional
    pass

# ---------------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------------
API_TITLE = "Clothing Background Removal API"
API_DESCRIPTION = (
    "A computer-vision service that accepts a photograph of a clothing item, "
    "isolates the garment using AI-based image segmentation, refines the mask "
    "with OpenCV post-processing, and returns a clean transparent PNG.  "
    "Intended to serve as the CV module inside a wardrobe-management application."
)
API_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
MODELS_DIR = os.path.join(BASE_DIR, "models", "segmentation")

# ---------------------------------------------------------------------------
# Upload validation  (Section 2)
# ---------------------------------------------------------------------------
# Maximum allowed upload size in bytes (10 MB)
MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

# Minimum accepted image dimensions (pixels)
MIN_IMAGE_WIDTH: int = 256
MIN_IMAGE_HEIGHT: int = 256

# Formats validated by magic-byte detection (used by image_utils)
# Keys are the human-readable format name; values are the leading byte
# sequences that must appear at the start of the raw file data.
ALLOWED_FORMATS: dict = {
    "JPEG": [b"\xff\xd8\xff"],
    "PNG":  [b"\x89PNG\r\n\x1a\n"],
    "WEBP": [b"RIFF"],          # further validated by OpenCV decode
}

# Allowed file extensions that a client may send (lower-case, with dot).
# Used only for generating a safe stored extension — NOT as a security gate.
ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Preprocessing  (Section 3)
# ---------------------------------------------------------------------------
# If either dimension of a decoded image exceeds this value, the image is
# scaled down proportionally until the largest dimension equals this value.
# Images that are already within this limit are never upscaled.
MAX_PREPROCESSING_DIMENSION: int = 2048

# ---------------------------------------------------------------------------
# Segmentation  (Section 4)
# ---------------------------------------------------------------------------
# Identifier for the rembg model to use.
# "u2net" is the default general-purpose U²-Net model.
# Change this here (not in the service) when switching to a
# clothing-specific model in a later section.
SEGMENTATION_MODEL_NAME: str = "u2net"

# ---------------------------------------------------------------------------
# Mask Analysis  (Section 5)
# ---------------------------------------------------------------------------
# Pixel threshold used ONLY for analysis of the soft segmentation mask.
# Pixels >= this value are considered foreground for statistical purposes.
# The original soft mask is NEVER modified — this threshold is applied only
# to a temporary copy created during analysis.
MASK_ANALYSIS_THRESHOLD: int = 128

# ---------------------------------------------------------------------------
# Mask Refinement  (Section 6)
# ---------------------------------------------------------------------------
# Threshold used to binarise the soft mask before morphological operations.
# After operations the clean binary is used as a gate over the original soft
# values; the overall output remains uint8 soft mask, not hard binary.
MASK_REFINEMENT_THRESHOLD: int = 128

# Kernel size (px) for morphological OPENING (erosion then dilation).
# Opening removes small isolated foreground specks.
# Keep this small — large values erode clothing edges.
MASK_OPENING_KERNEL_SIZE: int = 3

# Kernel size (px) for morphological CLOSING (dilation then erosion).
# Closing fills small gaps/holes in the foreground region.
# Slightly larger than opening so nearby blobs close up before holes are filled.
MASK_CLOSING_KERNEL_SIZE: int = 5

# Minimum connected-component area in pixels.
# Foreground regions smaller than this value are removed as noise.
# Set conservatively: large enough to remove single-pixel speckle, small
# enough not to remove thin clothing features like straps or collars.
MASK_MIN_COMPONENT_AREA: int = 500

# Maximum area in pixels of a background hole that will be filled.
# Background connected components (holes) smaller than this inside the
# foreground convex hull are filled.  Large background regions (e.g. the
# actual background of the photo) are never filled.
MASK_MAX_HOLE_AREA: int = 2000

# ---------------------------------------------------------------------------
# Alpha Composition & Final Output  (Section 7)
# ---------------------------------------------------------------------------
# Padding added around the detected foreground when performing the tight crop.
# Pixels on each side.  Set to 0 to disable padding.
FINAL_CROP_PADDING: int = 10
