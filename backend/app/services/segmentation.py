"""
Segmentation service  (Section 4).

Wraps the rembg / U²-Net pretrained model behind a thin application-level
abstraction so that the rest of the project never imports rembg directly.
If we later switch to a clothing-specific segmentation model, only this
file needs to change.

Pipeline:
    PreprocessedImage (H×W×3 RGB uint8 ndarray)
        ↓
    SegmentationService.segment()
        ↓
    NumPy RGB → PIL Image  (rembg expects PIL)
        ↓
    rembg.remove()         (U²-Net inference, returns RGBA PIL Image)
        ↓
    Extract alpha channel  (foreground probability mask)
        ↓
    SegmentationResult.mask  (H×W uint8 ndarray)

Mask semantics:
    255 (high values) = foreground / clothing
    0   (low values)  = background

Soft/alpha values between 0 and 255 are intentionally preserved.
Binary thresholding and morphological cleanup belong to Section 5/6.

IMPORTANT KNOWN LIMITATION:
    U²-Net / rembg is a general foreground-background-removal model.
    It does NOT guarantee clothing-only segmentation.  A photo of a person
    wearing a shirt may produce a mask around the entire person, not just
    the garment.  The architecture is intentionally designed to allow the
    segmentation model to be swapped for a clothing-specific model in a
    later section without changing the rest of the application.

Model caching:
    rembg downloads the U²-Net weights (~170 MB) on first use and caches
    them in the OS user cache directory (typically ~/.u2net/).
    Subsequent calls reuse the cached weights automatically.
    Model weights must NOT be committed to Git.

CPU / GPU:
    The prototype is designed for CPU-only execution.  CUDA is not required.
    rembg / ONNX Runtime will use CPU when no GPU is available.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from app.config import SEGMENTATION_MODEL_NAME


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """
    Output of the segmentation pipeline.

    Attributes:
        mask:          H×W uint8 ndarray.
                       255 = foreground, 0 = background.
                       Soft values between 0–255 are preserved from the model.
        model_name:    The rembg model identifier that produced this mask.
        input_width:   Width of the image that was segmented.
        input_height:  Height of the image that was segmented.
    """
    mask: np.ndarray
    model_name: str
    input_width: int
    input_height: int


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class SegmentationError(Exception):
    """Raised when the segmentation model cannot process the input."""


class ModelLoadError(Exception):
    """Raised when the segmentation model fails to load."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SegmentationService:
    """
    Application-level wrapper around the rembg U²-Net segmentation model.

    The model session is initialised lazily on the first call to segment()
    and reused for all subsequent calls.  A threading lock ensures that
    concurrent requests do not trigger duplicate model downloads.

    Usage:
        service = SegmentationService()          # cheap — no model loaded yet
        result  = service.segment(rgb_ndarray)   # model loads here (first time)
    """

    def __init__(self, model_name: str = SEGMENTATION_MODEL_NAME) -> None:
        """
        Args:
            model_name: rembg model identifier, e.g. "u2net".
                        Defaults to the value in config.SEGMENTATION_MODEL_NAME.
        """
        self._model_name: str = model_name
        self._session = None          # lazy: loaded on first call to segment()
        self._lock = threading.Lock() # prevents duplicate model downloads

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The rembg model identifier this service was configured with."""
        return self._model_name

    def segment(self, image_rgb: np.ndarray) -> SegmentationResult:
        """
        Run AI segmentation on a preprocessed RGB image.

        Args:
            image_rgb: NumPy ndarray, shape H×W×3, dtype uint8, RGB colour
                       order.  This is the direct output of Section 3
                       preprocessing.  Do NOT pass BGR arrays here.

        Returns:
            SegmentationResult containing the raw foreground mask (H×W uint8).

        Raises:
            ValueError:        if image_rgb fails input validation.
            SegmentationError: if the model cannot process the image.
            ModelLoadError:    if the rembg session cannot be initialised.
        """
        self._validate_input(image_rgb)
        session = self._get_session()

        height, width = image_rgb.shape[:2]

        try:
            mask = self._run_inference(image_rgb, session)
        except Exception as exc:
            raise SegmentationError(
                f"Segmentation inference failed: {exc}"
            ) from exc

        return SegmentationResult(
            mask=mask,
            model_name=self._model_name,
            input_width=width,
            input_height=height,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(image_rgb: np.ndarray) -> None:
        """
        Validate that image_rgb is in the expected format.

        Raises:
            ValueError: with a descriptive message if the array is invalid.
        """
        if image_rgb is None:
            raise ValueError(
                "image_rgb must not be None. "
                "Pass a uint8 H×W×3 RGB ndarray from Section 3 preprocessing."
            )

        if not isinstance(image_rgb, np.ndarray):
            raise ValueError(
                f"image_rgb must be a NumPy ndarray, got {type(image_rgb).__name__}."
            )

        if image_rgb.dtype != np.uint8:
            raise ValueError(
                f"image_rgb dtype must be uint8, got {image_rgb.dtype}. "
                "Section 3 preprocessing always produces uint8 output."
            )

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            shape_str = str(image_rgb.shape)
            raise ValueError(
                f"image_rgb must have shape H×W×3, got {shape_str}. "
                "Pass the 3-channel RGB output from Section 3 preprocessing."
            )

    def _get_session(self):
        """
        Return the rembg model session, loading it lazily on first access.

        Thread-safe: uses a lock to prevent duplicate model loads.

        Raises:
            ModelLoadError: if the rembg session cannot be created.
        """
        if self._session is not None:
            return self._session

        with self._lock:
            # Double-check after acquiring the lock (another thread may have
            # loaded the model while we were waiting).
            if self._session is not None:
                return self._session

            try:
                from rembg import new_session  # imported here to keep startup fast
                self._session = new_session(self._model_name)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load segmentation model '{self._model_name}'. "
                    f"Ensure rembg is installed and the model can be downloaded. "
                    f"Detail: {exc}"
                ) from exc

        return self._session

    @staticmethod
    def _run_inference(image_rgb: np.ndarray, session) -> np.ndarray:
        """
        Run rembg inference and return the raw foreground mask.

        Conversion path:
            NumPy RGB ndarray
                → PIL Image (RGB)
                → rembg.remove()  → RGBA PIL Image
                → alpha channel extracted
                → H×W uint8 ndarray (the mask)

        The alpha channel from rembg encodes foreground probability:
            255 = foreground / clothing
            0   = background

        Soft values between 0 and 255 are preserved exactly as the model
        produces them.  Thresholding belongs to Section 5/6.
        """
        from rembg import remove as rembg_remove  # local import keeps startup fast

        # NumPy RGB → PIL Image  (explicit, clearly labelled conversion)
        pil_image = Image.fromarray(image_rgb, mode="RGB")

        # Run U²-Net inference.  rembg returns an RGBA PIL Image where the
        # alpha channel encodes the foreground/background segmentation.
        result_pil: Image.Image = rembg_remove(pil_image, session=session)

        # Extract the alpha channel as the raw mask.
        # shape: (H, W)  dtype: uint8
        mask = np.array(result_pil.split()[-1], dtype=np.uint8)

        return mask


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# A single shared service instance so that model weights are only loaded once
# per application process.  Import and use this directly:
#
#     from app.services.segmentation import segmentation_service
#     result = segmentation_service.segment(preprocessed.image)
#
# Tests that need an isolated instance can create their own:
#     svc = SegmentationService()
#
_segmentation_service: Optional[SegmentationService] = None


def get_segmentation_service() -> SegmentationService:
    """
    Return the module-level singleton SegmentationService.

    The service is created once and reused for the lifetime of the process.
    Model weights are loaded lazily on the first call to .segment().
    """
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = SegmentationService()
    return _segmentation_service

