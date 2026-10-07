"""
Tests for the pieces the Flutter app relies on:
  * CORS (Flutter web is a browser, so it sends preflight requests)
  * /analyze response shape (what lib/.../clothing_analysis.dart parses)
  * model warm-up never crashes the server
  * .env is honoured for HOST/PORT (needed so a real phone can connect)
"""
import importlib
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _png_bytes(size=(320, 320)) -> bytes:
    arr = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    arr[80:240, 80:240] = (200, 30, 40)  # a red square "garment"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_cors_preflight_allows_browser_upload(client):
    r = client.options(
        "/api/v1/clothing/analyze",
        headers={
            "Origin": "http://localhost:5000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:5000")


def test_analyze_response_has_every_field_the_app_reads(client, monkeypatch):
    # Use a tiny fake segmentation so this contract test is fast and offline.
    from app.services import clothing_background_removal as pipeline
    from app.services.segmentation import SegmentationResult

    class FakeSeg:
        def segment(self, rgb):
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            mask[80:240, 80:240] = 255
            return SegmentationResult(mask, "fake", rgb.shape[1], rgb.shape[0])

    monkeypatch.setattr(pipeline, "get_segmentation_service", lambda: FakeSeg())
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.services.tagging.os.getenv",
                        lambda k, d=None: "" if k == "GEMINI_API_KEY" else os.environ.get(k, d))

    r = client.post(
        "/api/v1/clothing/analyze",
        files={"file": ("shirt.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    for key in ("filename", "image_url", "width", "height", "processing_time_s",
                "tags", "tag_list", "tagging"):
        assert key in body, f"missing {key}"
    for key in ("category", "colors", "pattern", "style", "season",
                "material_guess", "extra_tags"):
        assert key in body["tags"], f"missing tags.{key}"
    for key in ("source", "model", "error"):
        assert key in body["tagging"], f"missing tagging.{key}"

    assert body["image_url"] == f"/api/v1/clothing/result/{body['filename']}"
    assert body["tagging"]["source"] == "local"          # no key -> fallback
    assert "red" in body["tags"]["colors"]               # measured from pixels

    img = client.get(body["image_url"])                  # the app downloads this
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(img.content)).mode == "RGBA"


def test_warm_up_never_raises(monkeypatch):
    from app import main as app_main
    from app.services import clothing_background_removal as pipeline

    def boom():
        raise RuntimeError("download failed")

    monkeypatch.setattr(pipeline, "get_segmentation_service", boom)
    app_main.warm_up_model()  # must swallow the error


def test_dotenv_host_is_applied(tmp_path, monkeypatch):
    import app.config as config

    env_file = os.path.join(os.path.dirname(os.path.dirname(config.__file__)), ".env")
    assert not os.path.exists(env_file), "refusing to overwrite a real .env"
    try:
        with open(env_file, "w") as fh:
            fh.write("HOST=0.0.0.0\nPORT=8123\n")
        monkeypatch.delenv("HOST", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        reloaded = importlib.reload(config)
        assert reloaded.HOST == "0.0.0.0"
        assert reloaded.PORT == 8123
    finally:
        os.remove(env_file)
        monkeypatch.delenv("HOST", raising=False)
        monkeypatch.delenv("PORT", raising=False)
        importlib.reload(config)
