"""
FastAPI application factory.

Only application-level initialisation and the two bootstrap endpoints
(/ and /health) live here.  All feature routes are registered through
app/api/routes.py and mounted under /api/v1/clothing.
"""

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as clothing_router
from app.config import API_DESCRIPTION, API_TITLE, API_VERSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model warm-up (opt-in: WARMUP_MODEL=1, which run.py enables by default).
# The U2-Net weights (~176 MB) are downloaded/loaded lazily on the first
# request, which would make the app's first photo hang.  Doing it in a
# background thread at server start means the first photo is as fast as the rest.
# Disabled by default so unit tests never trigger a download.
# ---------------------------------------------------------------------------

def warm_up_model() -> None:
    """Load the segmentation model now.  Never raises."""
    try:
        from app.services.clothing_background_removal import get_segmentation_service

        get_segmentation_service()._get_session()
        logger.info("Segmentation model is loaded and ready.")
    except Exception:  # the first real request will retry and report properly
        logger.exception("Model warm-up failed; it will be retried on first use.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("WARMUP_MODEL", "0").strip() == "1":
        threading.Thread(target=warm_up_model, name="model-warmup", daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS  (only needed when the Flutter app runs in a browser / Flutter web;
# native Android and iOS apps are not affected by CORS).
# Set CORS_ORIGINS="https://your-app.com,http://localhost:5000" to restrict it.
# ---------------------------------------------------------------------------
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Feature routers
# ---------------------------------------------------------------------------
app.include_router(clothing_router, prefix="/api/v1/clothing")


# ---------------------------------------------------------------------------
# Bootstrap endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
def root() -> dict:
    """Confirm the API is reachable."""
    return {"message": "Clothing Background Removal API is running"}


@app.get("/health", tags=["Health"])
def health() -> dict:
    """Lightweight liveness probe."""
    return {"status": "healthy"}

