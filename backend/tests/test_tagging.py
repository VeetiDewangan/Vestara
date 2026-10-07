"""
Tests for Section 8: clothing tagging and POST /analyze.

Everything external is mocked (Gemini HTTP calls and the U2-Net pipeline),
so these tests are fast, free, and need no API key or internet.

What is covered:
  - local color extraction (solid, two-tone, fully transparent)
  - JSON extraction from messy model text
  - normalisation of model output into the fixed vocabularies
  - flat tag list building
  - generate_tags: no key, success, model-not-found fallback, rate limit,
    bad JSON, unexpected exception (must never raise)
  - _call_gemini HTTP status mapping and request shape
  - POST /analyze end-to-end (mocked pipeline) and input validation
"""

import io
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.services import tagging
from app.services.tagging import (
    GeminiError,
    GeminiModelNotFound,
    build_tag_list,
    dominant_colors,
    extract_json,
    generate_tags,
    normalize_ai_tags,
)


# ===========================================================================
# Helpers
# ===========================================================================

def solid_rgba(color, size=(120, 120)) -> Image.Image:
    return Image.new("RGBA", size, (*color, 255))


def two_tone_rgba() -> Image.Image:
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[..., 3] = 255
    arr[:, :60] = (200, 30, 40, 255)     # red, 60 %
    arr[:, 60:] = (40, 90, 200, 255)     # blue, 40 %
    return Image.fromarray(arr, "RGBA")


GOOD_MODEL_JSON = (
    '{"category": "T-Shirt", "colors": ["dark blue", "white"], "pattern": "striped",'
    ' "style": ["casual", "boho"], "season": ["summer"], "material_guess": "Cotton",'
    ' "extra_tags": ["Crew Neck", "crew neck", "oversized"]}'
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    # forget remembered / discovered models between tests
    monkeypatch.setitem(tagging._model_state, "good", None)
    monkeypatch.setitem(tagging._model_state, "discovered", None)
    monkeypatch.setitem(tagging._model_state, "discovered_at", 0.0)

    monkeypatch.setattr(tagging, "_RETRY_DELAY_S", 0)

    # model discovery must never reach the real network in tests
    def offline(*a, **k):
        raise tagging.httpx.ConnectError("offline")

    monkeypatch.setattr(tagging.httpx, "get", offline)


# ===========================================================================
# Local color extraction
# ===========================================================================

def test_solid_red_is_named_red():
    assert dominant_colors(solid_rgba((200, 30, 40))) == ["red"]


def test_two_tone_orders_by_share():
    assert dominant_colors(two_tone_rgba())[:2] == ["red", "blue"]


def test_fully_transparent_image_has_no_colors():
    assert dominant_colors(Image.new("RGBA", (50, 50), (0, 0, 0, 0))) == []


def test_transparent_pixels_are_ignored():
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[:, :, 3] = 0                      # transparent background (black RGB)
    arr[20:80, 20:80] = (245, 245, 245, 255)
    assert dominant_colors(Image.fromarray(arr, "RGBA")) == ["white"]


# ===========================================================================
# JSON extraction + normalisation
# ===========================================================================

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert extract_json('Sure! Here you go: {"a": 1} Hope that helps.') == {"a": 1}


@pytest.mark.parametrize("bad", ["", "no json here", "[1, 2, 3]"])
def test_extract_json_rejects_non_objects(bad):
    with pytest.raises(ValueError):
        extract_json(bad)


def test_normalize_maps_aliases_and_vocab():
    tags = normalize_ai_tags(extract_json(GOOD_MODEL_JSON))
    assert tags["category"] == "t-shirt"
    assert tags["colors"] == ["navy", "white"]          # "dark blue" -> navy
    assert tags["pattern"] == "striped"
    assert tags["style"] == ["casual"]                  # "boho" not in vocab
    assert tags["season"] == ["summer"]
    assert tags["material_guess"] == "cotton"
    assert tags["extra_tags"] == ["crew neck", "oversized"]   # deduped, lowercased


def test_normalize_unknown_values_fall_back_safely():
    tags = normalize_ai_tags({"category": "spaceship", "pattern": "plaid-ish"})
    assert tags["category"] == "other"
    assert tags["pattern"] == "other"
    assert tags["colors"] == [] and tags["style"] == [] and tags["season"] == []
    assert tags["material_guess"] == "unknown"


def test_normalize_survives_wrong_types():
    tags = normalize_ai_tags({
        "category": 42, "colors": "red", "style": None,
        "extra_tags": {"x": 1}, "material_guess": None,
    })
    assert tags["colors"] == ["red"]         # a lone string becomes a list
    assert tags["style"] == []
    assert isinstance(tags["extra_tags"], list)


def test_normalize_caps_list_lengths():
    tags = normalize_ai_tags({
        "colors": ["red", "blue", "green", "black", "white"],
        "style": ["casual", "formal", "party"],
        "extra_tags": [f"tag{i}" for i in range(20)],
    })
    assert len(tags["colors"]) == 3
    assert len(tags["style"]) == 2
    assert len(tags["extra_tags"]) == 6


def test_normalize_strips_junk_characters_from_tags():
    tags = normalize_ai_tags({"extra_tags": ["<script>alert(1)</script>"]})
    assert all("<" not in t and ">" not in t and "(" not in t for t in tags["extra_tags"])


def test_build_tag_list_dedupes_and_skips_placeholders():
    tags = {
        "category": "other", "colors": ["red", "red"], "pattern": "other",
        "style": ["casual"], "season": [], "material_guess": "unknown",
        "extra_tags": ["red", "v-neck"],
    }
    assert build_tag_list(tags) == ["red", "casual", "v-neck"]


# ===========================================================================
# generate_tags
# ===========================================================================

def test_generate_tags_without_api_key_returns_local_colors():
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert result.source == "local"
    assert result.tags["colors"] == ["red"]
    assert result.tags["pattern"] == "solid"          # obvious from the pixels
    assert result.tag_list == ["red", "solid"]
    assert "GEMINI_API_KEY" in result.error


def test_generate_tags_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = {}

    def fake_call(image_b64, api_key, model, timeout):
        calls.update(api_key=api_key, model=model, has_image=bool(image_b64))
        return GOOD_MODEL_JSON

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((30, 40, 120)))

    assert result.source == "gemini" and result.error is None
    assert result.model == "gemini-2.5-flash"
    assert calls == {"api_key": "test-key", "model": "gemini-2.5-flash", "has_image": True}
    assert result.tags["category"] == "t-shirt"
    assert result.tag_list[:3] == ["t-shirt", "navy", "white"]


def test_generate_tags_falls_back_to_measured_colors_if_model_gives_none(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging, "_call_gemini",
                        lambda *a, **k: '{"category": "dress", "colors": []}')
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert result.source == "gemini"
    assert result.tags["colors"] == ["red"]


def test_generate_tags_tries_next_model_when_not_found(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "retired-model")
    tried = []

    def fake_call(image_b64, api_key, model, timeout):
        tried.append(model)
        if model == "retired-model":
            raise GeminiModelNotFound(model)
        return GOOD_MODEL_JSON

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((10, 10, 10)))
    assert tried == ["retired-model", "gemini-2.5-flash"]
    assert result.source == "gemini" and result.model == "gemini-2.5-flash"


def test_generate_tags_all_models_missing_reports_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def fake_call(image_b64, api_key, model, timeout):
        raise GeminiModelNotFound(model)

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert result.source == "local" and "not found" in result.error
    assert result.tags["colors"] == ["red"]


def test_generate_tags_rate_limit_tries_next_model_then_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    tried = []

    def fake_call(image_b64, api_key, model, timeout):
        tried.append(model)
        raise GeminiError("Gemini free-tier rate limit reached. Try again shortly.",
                          try_next_model=True)      # quota is per model

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert len(tried) == len(tagging._DEFAULT_MODELS)
    assert result.source == "local" and "rate limit" in result.error


def test_generate_tags_invalid_key_stops_immediately(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    tried = []

    def fake_call(image_b64, api_key, model, timeout):
        tried.append(model)
        raise GeminiError("Gemini rejected the API key (HTTP 400). Check GEMINI_API_KEY.")

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert len(tried) == 1                      # every model shares the key problem
    assert result.source == "local" and "GEMINI_API_KEY" in result.error


def test_rate_limited_model_falls_through_to_working_one(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def fake_call(image_b64, api_key, model, timeout):
        if model == "gemini-2.5-flash":
            raise GeminiError("rate limit", try_next_model=True)
        return GOOD_MODEL_JSON

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((30, 40, 120)))
    assert result.source == "gemini" and result.model == "gemini-2.5-flash-lite"


def test_overloaded_503_is_retried_then_newer_models_are_tried(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging.httpx, "get", lambda *a, **k: _ListResponse())
    calls = []

    def fake_call(image_b64, api_key, model, timeout):
        calls.append(model)
        if model in tagging._DEFAULT_MODELS:
            raise GeminiError("Gemini is overloaded right now (HTTP 503)",
                              try_next_model=True, transient=True)
        return GOOD_MODEL_JSON

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((30, 40, 120)))
    assert result.source == "gemini" and result.model == "gemini-3.5-flash"
    assert calls.count("gemini-2.5-flash") == 2          # one quick retry per busy model


class _ListResponse:
    status_code = 200

    def json(self):
        def m(name, methods=("generateContent",)):
            return {"name": f"models/{name}", "supportedGenerationMethods": list(methods)}
        return {"models": [
            m("gemini-2.5-flash"), m("gemini-3.5-flash"), m("gemini-3-flash-preview"),
            m("gemini-3.5-flash-image"), m("gemini-embedding-001", ("embedContent",)),
            m("gemini-live-flash", ("bidiGenerateContent",)), m("gemma-3-27b-it"),
        ]}


def test_discovers_working_model_when_builtin_names_are_retired(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging.httpx, "get", lambda *a, **k: _ListResponse())
    tried = []

    def fake_call(image_b64, api_key, model, timeout):
        tried.append(model)
        if model in tagging._DEFAULT_MODELS:
            raise GeminiModelNotFound(model)
        return GOOD_MODEL_JSON

    monkeypatch.setattr(tagging, "_call_gemini", fake_call)
    result = generate_tags(solid_rgba((30, 40, 120)))
    assert result.source == "gemini" and result.model == "gemini-3.5-flash"   # stable + newest first
    assert "gemini-3.5-flash-image" not in tried and "gemini-embedding-001" not in tried

    # the next photo goes straight to the model that worked
    tried.clear()
    generate_tags(solid_rgba((30, 40, 120)))
    assert tried == ["gemini-3.5-flash"]


def test_generate_tags_bad_json_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging, "_call_gemini", lambda *a, **k: "I cannot help with that")
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert result.source == "local" and result.error


def test_generate_tags_never_raises_on_unexpected_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def boom(*a, **k):
        raise RuntimeError("something weird")

    monkeypatch.setattr(tagging, "_call_gemini", boom)
    result = generate_tags(solid_rgba((200, 30, 40)))
    assert result.source == "local"
    assert "something weird" not in result.error       # no internals leaked


def test_api_key_never_appears_in_result(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(tagging, "_call_gemini", lambda *a, **k: GOOD_MODEL_JSON)
    assert "SECRET-KEY-123" not in str(generate_tags(solid_rgba((1, 2, 3))).to_dict())


# ===========================================================================
# Local (offline) guesses and the thumbnail sent to the app
# ===========================================================================

def _blank(w=300, h=400):
    return np.zeros((h, w, 4), dtype=np.uint8)


def _paint(arr, y0, y1, x0, x1, color):
    arr[y0:y1, x0:x1] = (*color, 255)


def test_local_guess_finds_trousers_by_two_legs():
    a = _blank()
    _paint(a, 20, 80, 60, 240, (30, 40, 120))
    _paint(a, 80, 380, 60, 140, (30, 40, 120))
    _paint(a, 80, 380, 160, 240, (30, 40, 120))
    assert tagging.local_guess(Image.fromarray(a, "RGBA"))["category"] == "trousers"


def test_local_guess_finds_shorts_when_legs_are_short():
    a = _blank(300, 260)
    _paint(a, 20, 60, 40, 260, (200, 180, 150))
    _paint(a, 60, 200, 40, 125, (200, 180, 150))
    _paint(a, 60, 200, 175, 260, (200, 180, 150))
    assert tagging.local_guess(Image.fromarray(a, "RGBA"))["category"] == "shorts"


def test_local_guess_does_not_invent_a_category_for_a_shirt():
    a = _blank()
    _paint(a, 30, 380, 90, 210, (10, 10, 10))
    _paint(a, 30, 130, 20, 90, (10, 10, 10))
    _paint(a, 30, 130, 210, 280, (10, 10, 10))
    guess = tagging.local_guess(Image.fromarray(a, "RGBA"))
    assert guess["category"] is None and guess["pattern"] == "solid"


def test_local_guess_does_not_call_stripes_solid():
    a = _blank()
    _paint(a, 30, 380, 90, 210, (10, 10, 10))
    for y in range(30, 380, 24):
        a[y:y + 12, 90:210, :3] = (240, 240, 240)
    assert tagging.local_guess(Image.fromarray(a, "RGBA"))["pattern"] is None


def test_local_guess_ignores_empty_image():
    assert tagging.local_guess(Image.new("RGBA", (200, 200), (0, 0, 0, 0))) == {
        "category": None, "pattern": None}


def test_thumbnail_is_small_keeps_transparency_and_decodes():
    import base64
    a = _blank(1200, 1600)
    _paint(a, 100, 1500, 200, 1000, (10, 10, 10))
    b64, mime = tagging.make_thumbnail_b64(Image.fromarray(a, "RGBA"), max_side=640)
    assert mime in ("image/webp", "image/png")
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) == 640 and img.mode == "RGBA"
    assert img.getpixel((0, 0))[3] == 0                   # corner stays transparent


# ===========================================================================
# _call_gemini (HTTP layer, mocked)
# ===========================================================================

class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok_payload(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_call_gemini_success_and_request_shape(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResponse(200, _ok_payload('{"category": "dress"}'))

    monkeypatch.setattr(tagging.httpx, "post", fake_post)
    text = tagging._call_gemini("QUJD", "my-key", "gemini-2.5-flash", 12.0)

    assert text == '{"category": "dress"}'
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "my-key"
    assert "my-key" not in seen["url"]                       # key goes in a header
    assert seen["timeout"] == 12.0
    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0]["inline_data"] == {"mime_type": "image/jpeg", "data": "QUJD"}
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("status,exc", [
    (404, GeminiModelNotFound),
    (429, GeminiError),
    (401, GeminiError),
    (403, GeminiError),
    (500, GeminiError),
])
def test_call_gemini_maps_http_errors(monkeypatch, status, exc):
    monkeypatch.setattr(tagging.httpx, "post", lambda *a, **k: FakeResponse(status))
    with pytest.raises(exc):
        tagging._call_gemini("QUJD", "k", "m", 5)


def test_call_gemini_handles_network_error(monkeypatch):
    def fail(*a, **k):
        raise tagging.httpx.ConnectError("down")

    monkeypatch.setattr(tagging.httpx, "post", fail)
    with pytest.raises(GeminiError):
        tagging._call_gemini("QUJD", "k", "m", 5)


def test_call_gemini_handles_blocked_or_empty_reply(monkeypatch):
    monkeypatch.setattr(tagging.httpx, "post",
                        lambda *a, **k: FakeResponse(200, {"candidates": []}))
    with pytest.raises(GeminiError):
        tagging._call_gemini("QUJD", "k", "m", 5)


# ===========================================================================
# POST /analyze  (pipeline mocked)
# ===========================================================================

@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture()
def fake_pipeline(monkeypatch, tmp_path):
    """Replace run_pipeline with a stub that 'produces' a red cut-out PNG."""
    out = tmp_path / "abc123.png"
    solid_rgba((200, 30, 40)).save(out)

    def stub(data, filename):
        return types.SimpleNamespace(
            filename="abc123.png", output_path=str(out),
            width=120, height=120, processing_time_s=0.5,
        )

    monkeypatch.setattr("app.api.routes.run_pipeline", stub)


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_returns_cutout_url_and_tags(client, fake_pipeline, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging, "_call_gemini", lambda *a, **k: GOOD_MODEL_JSON)

    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("shirt.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["image_url"] == "/api/v1/clothing/result/abc123.png"
    assert body["tags"]["category"] == "t-shirt"
    assert body["tag_list"][0] == "t-shirt"
    assert body["tagging"] == {"source": "gemini", "model": "gemini-2.5-flash", "error": None}
    assert "output_path" not in body                 # internal path never exposed


def test_analyze_still_works_without_api_key(client, fake_pipeline):
    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("shirt.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tagging"]["source"] == "local"
    assert body["tags"]["colors"] == ["red"]
    assert body["tag_list"] == ["red", "solid"]
    assert body["image_url"].endswith("abc123.png")


def test_analyze_embeds_the_cutout_so_the_app_needs_no_second_request(client, fake_pipeline):
    import base64
    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("shirt.png", _png_bytes(), "image/png")})
    body = resp.json()
    assert body["image_mime"] in ("image/webp", "image/png")
    img = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
    assert img.size == (120, 120)


def test_result_is_served_inline_and_cacheable(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.OUTPUT_DIR", str(tmp_path))
    solid_rgba((200, 30, 40)).save(tmp_path / "abc123.png")
    resp = client.get("/api/v1/clothing/result/abc123.png")
    assert resp.status_code == 200
    assert resp.headers["content-disposition"].startswith("inline")
    assert "max-age" in resp.headers["cache-control"]


def test_retag_runs_only_tagging(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.routes.OUTPUT_DIR", str(tmp_path))
    solid_rgba((200, 30, 40)).save(tmp_path / "abc123.png")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(tagging, "_call_gemini", lambda *a, **k: GOOD_MODEL_JSON)
    resp = client.post("/api/v1/clothing/retag/abc123.png")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tags"]["category"] == "t-shirt" and body["tagging"]["source"] == "gemini"


def test_retag_unknown_or_unsafe_names(client):
    assert client.post("/api/v1/clothing/retag/missing.png").status_code == 404
    assert client.post("/api/v1/clothing/retag/notes.txt").status_code == 400


def test_analyze_rejects_empty_file(client, fake_pipeline):
    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_analyze_requires_a_file(client):
    assert client.post("/api/v1/clothing/analyze").status_code == 422


def test_analyze_maps_invalid_image_to_400(client, monkeypatch):
    def bad(data, filename):
        raise ValueError("Not a valid image.")

    monkeypatch.setattr("app.api.routes.run_pipeline", bad)
    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("x.png", b"notanimage", "image/png")})
    assert resp.status_code == 400 and "valid image" in resp.json()["detail"]


def test_analyze_hides_internal_errors(client, monkeypatch):
    def crash(data, filename):
        raise RuntimeError("secret model path /home/x/.u2net")

    monkeypatch.setattr("app.api.routes.run_pipeline", crash)
    resp = client.post("/api/v1/clothing/analyze",
                       files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 500
    assert "secret" not in resp.text


def test_cors_headers_present_for_browser_clients(client):
    resp = client.get("/health", headers={"Origin": "http://localhost:5000"})
    assert resp.headers.get("access-control-allow-origin") in ("*", "http://localhost:5000")


def test_existing_endpoints_unaffected(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}
