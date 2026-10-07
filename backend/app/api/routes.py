"""
API routes — clothing image endpoints.

Section 2: POST /upload
  Accepts a multipart image upload, validates it through the service layer,
  saves valid images to input_images/, and returns JSON metadata.

Section 7: POST /remove-background
  Runs the full pipeline: validate → preprocess → U2-Net → refinement →
  alpha composition → transparent PNG.  Returns JSON metadata.

Section 8: POST /analyze
  Same pipeline as /remove-background, plus tagging (colors, category, pattern,
  style, season, ...).  Returns the cut-out URL and the tags in one response.

Section 7: GET /result/{filename}
  Returns a previously generated transparent PNG for download.
  Security: only files inside OUTPUT_DIR are served; path traversal is blocked.
"""

import os

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config import OUTPUT_DIR
from app.services.clothing_background_removal import run_pipeline
from app.services.image_processing import FileTooLargeError, process_upload
from app.services.tagging import generate_tags, make_thumbnail_b64

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /upload   (Section 2)
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    summary="Upload a clothing image",
    description=(
        "Upload a JPEG, PNG, or WEBP photograph of a clothing item. "
        "The image must be at least 256×256 px and no larger than 10 MB. "
        "A UUID-based filename is generated server-side; the original "
        "filename is never used as a path."
    ),
    status_code=status.HTTP_200_OK,
    tags=["Clothing"],
)
async def upload_image(file: UploadFile = File(...)) -> dict:
    """
    Validate and store an uploaded clothing image.

    Returns JSON with success metadata on a valid upload.
    Raises HTTP 400 for invalid/corrupt/unsupported images.
    Raises HTTP 413 when the file exceeds the 10 MB limit.
    """
    # Ensure the client actually attached a file
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided.",
        )

    # Read all bytes — the service layer will size-check before any work
    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        result = process_upload(data, file.filename or "upload")
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "success": True,
        "message": "Image uploaded and validated successfully.",
        "filename": result.filename,
        "format": result.format,
        "width": result.width,
        "height": result.height,
        "file_size": result.file_size,
    }


# ---------------------------------------------------------------------------
# POST /remove-background   (Section 7)
# ---------------------------------------------------------------------------

@router.post(
    "/remove-background",
    summary="Remove clothing image background",
    description=(
        "Upload a JPEG, PNG, or WEBP photograph of a clothing item. "
        "The full pipeline runs: validation → preprocessing → U2-Net "
        "segmentation → mask refinement → alpha composition → transparent PNG. "
        "Returns JSON metadata.  The output PNG can be retrieved with "
        "GET /result/{filename}."
    ),
    status_code=status.HTTP_200_OK,
    tags=["Clothing"],
)
async def remove_background(file: UploadFile = File(...)) -> dict:
    """
    Run the complete background-removal pipeline on an uploaded image.

    Returns JSON with success metadata; the transparent PNG is stored in
    output_images/ and can be retrieved via GET /result/{filename}.
    """
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        result = run_pipeline(data, file.filename or "upload")
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Do not expose internal details (stack traces, model paths, etc.)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Background removal failed.  Please try again.",
        ) from exc

    return {
        "success": True,
        "message": "Background removed successfully.",
        "filename": result.filename,
        "width": result.width,
        "height": result.height,
        "original_width": result.original_width,
        "original_height": result.original_height,
        "processing_time_s": result.processing_time_s,
        "raw_foreground_pct": round(result.raw_foreground_pct, 4),
        "refined_foreground_pct": round(result.refined_foreground_pct, 4),
        "components_removed": result.components_removed,
        "holes_filled": result.holes_filled,
    }


# ---------------------------------------------------------------------------
# POST /analyze   (Section 8)  -  cut-out + tags in one call
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Remove background AND generate tags",
    description=(
        "Runs the full background-removal pipeline, then tags the cut-out "
        "garment (category, colors, pattern, style, season, material, extra "
        "tags).  If AI tagging is unavailable (no API key, quota, network), "
        "the cut-out is still returned with color tags only and "
        "`tagging.error` explains why."
    ),
    status_code=status.HTTP_200_OK,
    tags=["Clothing"],
)
async def analyze(file: UploadFile = File(...)) -> dict:
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        # The pipeline is CPU-heavy and blocking; keep it off the event loop
        # so other requests (e.g. /health) stay responsive.
        result = await run_in_threadpool(run_pipeline, data, file.filename or "upload")
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Background removal failed.  Please try again.",
        ) from exc

    # Tag the finished cut-out.  generate_tags() never raises.
    try:
        with Image.open(result.output_path) as img:
            cutout = img.convert("RGBA")
            cutout.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the generated image.",
        ) from exc

    tagging = await run_in_threadpool(generate_tags, cutout)

    # Send a small copy of the cut-out INSIDE this reply, so the app can show
    # and save it without a second request (that second request is what used
    # to leave the picture blank when the phone could not fetch the image URL).
    try:
        image_b64, image_mime = await run_in_threadpool(make_thumbnail_b64, cutout)
    except Exception:
        image_b64, image_mime = None, None

    return {
        "success": True,
        "message": "Background removed and tags generated.",
        "filename": result.filename,
        "image_url": f"/api/v1/clothing/result/{result.filename}",
        "image_base64": image_b64,
        "image_mime": image_mime,
        "width": result.width,
        "height": result.height,
        "processing_time_s": result.processing_time_s,
        **tagging.to_dict(),
    }


# ---------------------------------------------------------------------------
# POST /retag/{filename}  -  run only the tagging step again
# ---------------------------------------------------------------------------

def _resolve_output_png(filename: str) -> str:
    """Return the absolute path of a generated PNG, or raise 400/404."""
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .png files can be retrieved from this endpoint.",
        )
    candidate = os.path.realpath(os.path.join(OUTPUT_DIR, safe_name))
    allowed = os.path.realpath(OUTPUT_DIR)
    if not candidate.startswith(allowed + os.sep) and candidate != allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )
    if not os.path.isfile(candidate):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )
    return candidate


@router.post(
    "/retag/{filename}",
    summary="Run AI tagging again on an existing cut-out",
    description=(
        "Re-runs only the tagging step for a cut-out that POST /analyze already "
        "produced (useful when AI tagging was temporarily unavailable). The "
        "background is NOT removed again."
    ),
    tags=["Clothing"],
)
async def retag(filename: str) -> dict:
    path = _resolve_output_png(filename)
    try:
        with Image.open(path) as img:
            cutout = img.convert("RGBA")
            cutout.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the generated image.",
        ) from exc

    tagging = await run_in_threadpool(generate_tags, cutout)
    return {"success": True, "filename": os.path.basename(path), **tagging.to_dict()}


# ---------------------------------------------------------------------------
# GET /result/{filename}   (Section 7)
# ---------------------------------------------------------------------------

@router.get(
    "/result/{filename}",
    summary="Download a generated transparent PNG",
    description=(
        "Retrieve a transparent PNG generated by POST /remove-background. "
        "Only files inside the server's output directory are served. "
        "Path traversal is blocked; arbitrary filesystem access is not possible."
    ),
    tags=["Clothing"],
)
async def get_result(filename: str) -> FileResponse:
    """
    Serve a generated transparent PNG by its UUID-based filename.

    Security:
    - Strips any directory components from the filename.
    - Resolves the final path and confirms it is inside OUTPUT_DIR.
    - Returns 404 if the file does not exist.
    - Returns 400 if the path would escape OUTPUT_DIR.
    """
    candidate = _resolve_output_png(filename)
    safe_name = os.path.basename(candidate)

    # "inline" (not "attachment") so browsers and Flutter web display the
    # picture instead of treating it as a download.  File names are random
    # UUIDs and never change, so they can be cached for a long time.
    return FileResponse(
        path=candidate,
        media_type="image/png",
        filename=safe_name,
        content_disposition_type="inline",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
