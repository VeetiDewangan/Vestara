"""
Clothing tagging service.

Takes the transparent RGBA cut-out produced by the background-removal
pipeline and returns tags for it (category, colors, pattern, style, season,
occasions, material guess, fit, neckline, extra descriptive tags).

Two layers, so the app never breaks:

  1. LOCAL colors  - always computed with OpenCV k-means on the garment
                     pixels only.  Free, offline, no API key.
  2. AI tags       - a vision model (Google Gemini, free tier from Google AI
                     Studio) looks at the cut-out and describes it as JSON.

If there is no API key, the network is down, or the free quota is used up,
generate_tags() still returns the local colors and reports what went wrong
in `error`.  It never raises.

Configuration (environment variables or a .env file):
    GEMINI_API_KEY   required for AI tags  (https://aistudio.google.com/apikey)
    GEMINI_MODEL     optional; tried first.  Falls back to the defaults below
                     if Google reports the model does not exist.
    GEMINI_TIMEOUT_S optional; default 30

PRIVACY NOTE: on Google's free tier, the images you send may be used to
improve Google's products.  Only the cut-out garment image is sent - never
the original photo or any user identity.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import httpx
import numpy as np
from PIL import Image

try:  # optional: lets a local .env file provide GEMINI_API_KEY
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabularies (keep tags consistent so filtering in the app works)
# ---------------------------------------------------------------------------

CATEGORIES = [
    "t-shirt", "shirt", "blouse", "top", "sweater", "hoodie", "jacket", "coat",
    "dress", "skirt", "jeans", "trousers", "shorts", "leggings", "kurta",
    "saree", "jumpsuit", "other",
]
CATEGORY_ALIASES = {
    "tee": "t-shirt", "tshirt": "t-shirt", "t shirt": "t-shirt",
    "tank top": "top", "crop top": "top", "polo": "shirt", "polo shirt": "shirt",
    "pants": "trousers", "pant": "trousers", "chinos": "trousers",
    "jean": "jeans", "denim": "jeans", "cardigan": "sweater", "pullover": "sweater",
    "blazer": "jacket", "gown": "dress", "sari": "saree", "anarkali": "kurta",
    "salwar": "kurta", "kurti": "kurta", "lehenga": "skirt",
    "midi dress": "dress", "maxi dress": "dress", "mini dress": "dress",
    "ball gown": "dress", "evening gown": "dress",
}

PATTERNS = [
    "solid", "striped", "checked", "floral", "printed", "polka dot",
    "abstract", "animal print", "geometric", "embroidered", "lace", "other",
]
STYLES = [
    "casual", "formal", "party", "sporty", "ethnic", "streetwear",
    "bohemian", "vintage", "minimalist", "elegant",
]
SEASONS = ["summer", "winter", "monsoon", "spring", "autumn", "all-season"]

# Occasions: used to power outfit builder recommendations
OCCASIONS = [
    "daily wear", "office", "date night", "wedding", "party", "festive",
    "beach", "gym", "outdoor", "college", "travel", "cocktail", "formal event",
]

# Fit types
FITS = [
    "slim fit", "regular fit", "oversized", "relaxed fit", "bodycon",
    "a-line", "flared", "straight cut", "fitted",
]

# Necklines
NECKLINES = [
    "round neck", "v-neck", "collar", "off-shoulder", "halter", "square neck",
    "boat neck", "scoop neck", "turtleneck", "cowl neck", "sweetheart", "high neck",
]

# Extended color vocabulary — more granular so k-means gets a better match
COLORS = [
    "black", "white", "grey", "red", "pink", "rose", "coral", "orange",
    "yellow", "green", "olive", "teal", "blue", "navy", "purple", "lavender",
    "brown", "beige", "cream", "gold", "silver",
]
COLOR_ALIASES = {
    "gray": "grey", "dark blue": "navy", "navy blue": "navy",
    "light blue": "blue", "sky blue": "blue",
    "dark green": "green", "mint": "teal", "cyan": "teal",
    "maroon": "red", "burgundy": "red", "wine": "red",
    "off-white": "cream", "off white": "cream",
    "ivory": "cream", "tan": "beige", "khaki": "beige", "camel": "beige",
    "violet": "purple", "magenta": "pink", "fuchsia": "pink", "blush": "pink",
    "charcoal": "grey", "peach": "coral", "salmon": "coral",
    "mustard": "yellow", "amber": "orange",
    "rose gold": "gold", "bronze": "gold",
    "light pink": "pink", "baby pink": "pink", "dusty pink": "pink",
    "hot pink": "pink", "rose pink": "rose",
}

# Reference RGB values for naming k-means cluster centres.
# Extended and calibrated for better perceptual accuracy.
_PALETTE_RGB = {
    "black":   (15,  15,  15),
    "white":   (245, 245, 245),
    "grey":    (128, 128, 128),
    "red":     (210,  30,  30),
    "pink":    (240, 100, 150),   # vivid mid-pink (hot-pink range)
    "rose":    (210,  90, 120),   # deeper rose/blush
    "coral":   (240, 100,  80),
    "orange":  (240, 120,  20),
    "yellow":  (240, 210,  40),
    "green":   ( 40, 140,  60),
    "olive":   (100, 120,  40),
    "teal":    ( 20, 150, 140),
    "blue":    ( 40,  90, 200),
    "navy":    ( 15,  25,  80),
    "purple":  (110,  50, 160),
    "lavender":(175, 130, 220),
    "brown":   (110,  65,  30),
    "beige":   (210, 185, 150),   # shifted warmer/darker so pink ≠ beige
    "cream":   (240, 230, 210),
    "gold":    (210, 165,  30),
    "silver":  (180, 180, 190),
}

# Google retires models regularly.  These are only the FIRST guesses; if every
# one of them is "not found", the service asks Google which models the key can
# use (see _discover_models) and tries those instead.
_DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
]
_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_DISCOVERY_TTL_S = 3600
_MAX_DISCOVERED_TRIES = 3
_RETRY_DELAY_S = 2.0  # pause before retrying a model that answered 503 "overloaded"

# Remembers what worked, so later photos skip dead models.
_model_state: dict[str, Any] = {"good": None, "discovered": None, "discovered_at": 0.0}

PROMPT = f"""You are an expert fashion stylist tagging a single clothing item for a digital wardrobe app.
The image shows one garment on a white/transparent background.
Analyse the item carefully and reply with ONLY a JSON object (no markdown, no commentary) with exactly these keys:

{{
  "category": one of {CATEGORIES},
  "colors": [up to 3 main colors, most dominant first, each must be one of {COLORS}],
  "pattern": one of {PATTERNS},
  "style": [up to 2 of {STYLES}],
  "season": [up to 2 of {SEASONS}],
  "occasions": [up to 3 of {OCCASIONS} — where can this garment realistically be worn?],
  "fit": one of {FITS} or null,
  "neckline": one of {NECKLINES} or null,
  "material_guess": "one of: cotton, denim, wool, silk, polyester, chiffon, linen, velvet, leather, satin, net, georgette, or unknown",
  "extra_tags": [3 to 6 short lowercase descriptors, e.g. "maxi length", "puff sleeves", "embellished", "midi", "sleeveless"]
}}

Guidelines for accuracy:
- COLORS: Be precise. A pink/rose dress must be tagged "pink" or "rose", NOT beige/grey/brown.
  If the garment is floral, list the background color first.
- OCCASIONS: Think about real life — a formal ballgown → "wedding", "formal event", "cocktail".
  A printed tee → "daily wear", "college", "casual outing".
- CATEGORY: A full-length or midi dress is "dress", not "top" or "t-shirt".
- NECKLINE: Look at the neckline shape carefully.
- FIT: Describe the silhouette — flared skirt + fitted bodice → "a-line".
- EXTRA_TAGS: include length (maxi/midi/mini), sleeve type, special details like embroidery/ruffles/sequins.
Describe only what is clearly visible. If unsure about a field, choose the closest option."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TaggingResult:
    tags: dict[str, Any]
    tag_list: list[str]           # flat list, handy for UI chips
    source: str                   # "gemini" or "local"
    model: Optional[str] = None
    error: Optional[str] = None   # set when AI tagging did not happen

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": self.tags,
            "tag_list": self.tag_list,
            "tagging": {"source": self.source, "model": self.model, "error": self.error},
        }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _flatten_on_white(rgba: Image.Image) -> Image.Image:
    """Composite RGBA onto white (transparent pixels would turn black in RGB)."""
    rgba = rgba.convert("RGBA")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.split()[3])
    return white


def _image_to_b64_jpeg(rgba: Image.Image, max_side: int = 768) -> str:
    img = _flatten_on_white(rgba)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Local color extraction (no API needed)
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb) -> np.ndarray:
    px = np.uint8([[[int(rgb[0]), int(rgb[1]), int(rgb[2])]]])
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


_PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in _PALETTE_RGB.items()}


def _nearest_color_name(rgb) -> str:
    """Find the closest named color using perceptual Lab distance."""
    lab = _rgb_to_lab(rgb)
    return min(_PALETTE_LAB, key=lambda n: float(np.linalg.norm(_PALETTE_LAB[n] - lab)))


def dominant_colors(rgba: Image.Image, k: int = 4, min_share: float = 0.06) -> list[str]:
    """Named dominant colors of the garment pixels (alpha > 200), biggest first.

    Uses more clusters (k=4) and filters near-white cluster CENTRES to avoid
    mistaking white-composited background bleed for a garment color.  White
    garments are still detected correctly because a dominant near-white centre
    is kept whenever it represents the clear majority of garment pixels.
    """
    arr = np.array(rgba.convert("RGBA"))
    # Only keep pixels that are clearly part of the garment (alpha > 200)
    garment_mask = arr[..., 3] > 200
    px = arr[garment_mask][:, :3].astype(np.float32)
    if len(px) == 0:
        return []

    if len(px) > 8000:
        rng = np.random.default_rng(0)
        px = px[rng.choice(len(px), 8000, replace=False)]

    k = max(1, min(k, len(px)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(px, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    total = counts.sum()

    # Determine if the garment itself has a clear dominant near-white color
    # (e.g., a white dress) vs. bleed from alpha-compositing onto white background.
    # Rule: a near-white cluster is kept only when it is the largest cluster AND
    # represents > 40% of pixels.
    max_cluster_idx = int(counts.argmax())
    max_share = counts[max_cluster_idx] / total
    max_center_is_white = float(np.mean(centers[max_cluster_idx])) > 220

    names: list[str] = []
    for i in counts.argsort()[::-1]:
        share = counts[i] / total
        if share < min_share:
            continue
        center = centers[i]
        center_is_white = float(np.mean(center)) > 220
        # Skip near-white clusters that look like background bleed, unless
        # the garment genuinely is white (max cluster is white and dominant).
        if center_is_white and not (max_center_is_white and max_share > 0.40 and i == max_cluster_idx):
            continue
        name = _nearest_color_name(center)
        if name not in names:
            names.append(name)
    return names[:3]


# ---------------------------------------------------------------------------
# Local shape / pattern guess (no API needed, deliberately conservative)
# ---------------------------------------------------------------------------

def _runs(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Start/end of consecutive True stretches in a 1-D boolean array."""
    out, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def local_guess(rgba: Image.Image) -> dict[str, Optional[str]]:
    """
    Cheap offline hints from the cut-out, used when AI tagging is unavailable.

    Only returns something when it is fairly obvious, otherwise None:
      * category "trousers" / "shorts"  - two separate legs at the bottom
      * pattern  "solid"                - one uniform colour over the garment
    """
    result: dict[str, Optional[str]] = {"category": None, "pattern": None}
    arr = np.array(rgba.convert("RGBA"))
    alpha = arr[..., 3] > 128
    if int(alpha.sum()) < 400:
        return result

    ys, xs = np.where(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = alpha[y0:y1, x0:x1]
    h, w = sub.shape

    # Shape only means something when the background really was removed.
    if h >= 40 and w >= 40 and float(sub.mean()) < 0.97:
        bottom = sub[int(h * 0.85):int(h * 0.97)].any(axis=0)
        top = sub[int(h * 0.02):int(h * 0.10)].any(axis=0)
        legs = _runs(bottom, max(3, int(w * 0.12)))
        waist = _runs(top, max(3, int(w * 0.30)))
        if len(legs) == 2 and len(waist) == 1:
            gap = legs[1][0] - legs[0][1]
            if gap >= max(2, int(w * 0.03)):
                result["category"] = "trousers" if h / w >= 1.2 else "shorts"

    # Solid colour: low spread of Lab values well inside the garment.
    inner = cv2.erode(alpha.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3) > 0
    if int(inner.sum()) >= 500:
        lab = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)[inner]
        std_l, std_a, std_b = lab.std(axis=0)
        if std_l < 20 and std_a < 9 and std_b < 11:
            result["pattern"] = "solid"
    return result


def make_thumbnail_b64(rgba: Image.Image, max_side: int = 640) -> tuple[str, str]:
    """
    Small transparent copy of the cut-out, base64-encoded, for sending to the
    app inside the /analyze reply.  Returns (base64, mime_type).  Lossy WebP
    keeps it around 30-80 KB; PNG is the fallback if WebP is unavailable.
    """
    img = rgba.convert("RGBA")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    try:
        img.save(buf, format="WEBP", quality=85, method=4)
        mime = "image/webp"
    except Exception:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return base64.b64encode(buf.getvalue()).decode("ascii"), mime


# ---------------------------------------------------------------------------
# Cleaning the model's answer
# ---------------------------------------------------------------------------

def _clean_word(value: Any, max_len: int = 30) -> str:
    text = re.sub(r"[^a-z0-9 \-]", "", str(value).lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_len].strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _norm_category(value: Any) -> str:
    text = _clean_word(value)
    text = CATEGORY_ALIASES.get(text, text)
    return text if text in CATEGORIES else "other"


def _norm_color(value: Any) -> str:
    text = _clean_word(value, 20)
    text = COLOR_ALIASES.get(text, text)
    if text in COLORS:
        return text
    for name in COLORS:  # e.g. "bright red" -> "red"
        if re.search(rf"\b{name}\b", text):
            return name
    return ""


def _norm_choice(value: Any, allowed: list[str], default: Optional[str]) -> Optional[str]:
    text = _clean_word(value)
    if text in allowed:
        return text
    # partial match — e.g. "daily" -> "daily wear"
    for opt in allowed:
        if text and (text in opt or opt in text):
            return opt
    return default


def normalize_ai_tags(raw: dict[str, Any]) -> dict[str, Any]:
    """Force the model's JSON into our fixed shape and vocabularies."""
    colors = _dedupe([_norm_color(c) for c in _as_list(raw.get("colors"))])[:3]
    styles = _dedupe([_norm_choice(s, STYLES, "") or "" for s in _as_list(raw.get("style"))])[:2]
    seasons = _dedupe([_norm_choice(s, SEASONS, "") or "" for s in _as_list(raw.get("season"))])[:2]
    occasions = _dedupe([_norm_choice(o, OCCASIONS, "") or "" for o in _as_list(raw.get("occasions"))])[:3]
    fit = _norm_choice(raw.get("fit"), FITS, None)
    neckline = _norm_choice(raw.get("neckline"), NECKLINES, None)
    extra = _dedupe([_clean_word(t, 24) for t in _as_list(raw.get("extra_tags"))])[:6]
    material = _clean_word(raw.get("material_guess", ""), 30) or "unknown"
    return {
        "category": _norm_category(raw.get("category")),
        "colors": colors,
        "pattern": _norm_choice(raw.get("pattern"), PATTERNS, "other"),
        "style": [s for s in styles if s],
        "season": [s for s in seasons if s],
        "occasions": [o for o in occasions if o],
        "fit": fit,
        "neckline": neckline,
        "material_guess": material,
        "extra_tags": extra,
    }


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model text (tolerates ```json fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Model reply did not contain a JSON object.")
        obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model reply JSON was not an object.")
    return obj


def build_tag_list(tags: dict[str, Any]) -> list[str]:
    """Flatten the structured tags into one list of unique, display-ready tags."""
    items: list[str] = []
    if tags.get("category") and tags["category"] != "other":
        items.append(tags["category"])
    items += tags.get("colors", [])
    if tags.get("pattern") and tags["pattern"] != "other":
        items.append(tags["pattern"])
    items += tags.get("style", [])
    items += tags.get("season", [])
    items += tags.get("occasions", [])
    if tags.get("fit"):
        items.append(tags["fit"])
    if tags.get("neckline"):
        items.append(tags["neckline"])
    if tags.get("material_guess") and tags["material_guess"] != "unknown":
        items.append(tags["material_guess"])
    items += tags.get("extra_tags", [])
    return _dedupe(items)


# ---------------------------------------------------------------------------
# Gemini call (plain REST - no SDK needed)
# ---------------------------------------------------------------------------

class GeminiModelNotFound(Exception):
    """The requested model name does not exist (or was retired)."""


class GeminiError(Exception):
    """Any other API failure (bad key, quota, network, bad reply).

    ``try_next_model`` is True for problems that are specific to one model
    (its own quota, a temporary overload, an unsupported option), so another
    model may still work.  It is False for problems every model shares
    (invalid key, no network).
    """

    def __init__(self, message: str, *, try_next_model: bool = False, transient: bool = False):
        super().__init__(message)
        self.try_next_model = try_next_model
        self.transient = transient  # temporary overload (HTTP 5xx): worth one quick retry


def _model_candidates() -> list[str]:
    preferred = os.getenv("GEMINI_MODEL", "").strip()
    good = _model_state.get("good")
    return _dedupe(
        ([preferred] if preferred else []) + ([good] if good else []) + _DEFAULT_MODELS
    )


_NOT_FOR_TAGGING = (
    "image", "live", "audio", "tts", "native", "embedding", "robotics", "computer",
    "thinking", "exp", "learnlm", "gemma", "aqa", "imagen", "veo", "lyria",
    "deep-research", "customtools",
)


def _rank_model(name: str) -> tuple:
    """Sort key: stable before preview, newer before older, full before lite."""
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    version = float(m.group(1)) if m else 0.0
    return ("preview" in name, -version, "lite" in name, name)


def _discover_models(api_key: str, timeout: float) -> list[str]:
    """Ask Google which Gemini 'flash' models this key may call for images."""
    now = time.monotonic()
    cached = _model_state.get("discovered")
    if cached is not None and now - float(_model_state.get("discovered_at", 0)) < _DISCOVERY_TTL_S:
        return list(cached)

    names: list[str] = []
    try:
        resp = httpx.get(
            _MODELS_URL,
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 200},
            timeout=min(timeout, 15.0),
        )
        if resp.status_code != 200:
            return []
        for model in resp.json().get("models", []):
            name = str(model.get("name", "")).removeprefix("models/")
            methods = model.get("supportedGenerationMethods") or []
            if (
                name.startswith("gemini-")
                and "flash" in name
                and "generateContent" in methods
                and not any(bad in name for bad in _NOT_FOR_TAGGING)
            ):
                names.append(name)
    except Exception:  # network / JSON problems: just fall back to defaults
        logger.warning("Could not list Gemini models", exc_info=True)
        return []

    names = sorted(set(names), key=_rank_model)
    _model_state["discovered"] = names
    _model_state["discovered_at"] = now
    return list(names)


def _call_gemini(image_b64: str, api_key: str, model: str, timeout: float) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                {"text": PROMPT},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    try:
        resp = httpx.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise GeminiError(f"Network error talking to Gemini: {type(exc).__name__}") from exc

    if resp.status_code == 404:
        raise GeminiModelNotFound(model)
    if resp.status_code == 429:
        # Free-tier quota is tracked per model, so another model may still work.
        raise GeminiError("Gemini free-tier rate limit reached. Try again shortly.",
                          try_next_model=True)
    if resp.status_code in (400, 401, 403):
        detail = _api_error_message(resp, api_key)
        key_problem = resp.status_code == 401 or "api key" in detail.lower()
        if key_problem:
            raise GeminiError(
                f"Gemini rejected the API key (HTTP {resp.status_code}). Check GEMINI_API_KEY.")
        # e.g. this model does not accept the request / is not allowed for the key
        raise GeminiError(
            f"Gemini rejected the request (HTTP {resp.status_code}): {detail}".rstrip(": "),
            try_next_model=True)
    if resp.status_code >= 400:
        detail = _api_error_message(resp, api_key)
        busy = resp.status_code in (500, 502, 503, 504)
        label = "Gemini is overloaded right now" if busy else "Gemini error"
        raise GeminiError(
            f"{label} (HTTP {resp.status_code})" + (f": {detail}" if detail else "."),
            try_next_model=True, transient=busy)

    try:
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise GeminiError("Unexpected reply format from Gemini.", try_next_model=True) from exc
    if not text.strip():
        raise GeminiError("Gemini returned an empty reply.", try_next_model=True)
    return text


def _api_error_message(resp: Any, api_key: str, max_len: int = 140) -> str:
    """Short, key-free message from a Google error body ({"error": {"message": ...}})."""
    try:
        msg = str(resp.json().get("error", {}).get("message", ""))
    except Exception:
        return ""
    msg = " ".join(msg.replace(api_key, "***").split()) if api_key else " ".join(msg.split())
    return msg[:max_len]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _empty_tags(colors: list[str], guess: Optional[dict[str, Optional[str]]] = None) -> dict[str, Any]:
    guess = guess or {}
    return {
        "category": guess.get("category"), "colors": colors,
        "pattern": guess.get("pattern"), "style": [],
        "season": [], "occasions": [], "fit": None, "neckline": None,
        "material_guess": None, "extra_tags": [],
    }


def _run_models(models: list[str], image_b64: str, api_key: str, timeout: float):
    """
    Try each model in turn.  Returns (tags, model, last_error, only_not_found, fatal).
    ``tags`` is None when nothing worked; ``only_not_found`` is True when every
    failure was "this model does not exist"; ``fatal`` is True for problems
    every model shares (bad key, no network) so trying more models is pointless.
    """
    last_error = "No Gemini model available."
    only_not_found = True
    for model in models:
        for attempt in (1, 2):
            try:
                text = _call_gemini(image_b64, api_key, model, timeout)
                tags = normalize_ai_tags(extract_json(text))
                return tags, model, None, False, False
            except GeminiModelNotFound:
                last_error = f"Gemini model '{model}' was not found."
                break  # try the next candidate model
            except GeminiError as exc:
                only_not_found = False
                last_error = str(exc)
                if exc.transient and attempt == 1:
                    time.sleep(_RETRY_DELAY_S)  # brief overload: one more go
                    continue
                if exc.try_next_model:
                    break
                return None, None, last_error, False, True
            except ValueError as exc:  # reply was not usable JSON
                only_not_found = False
                last_error = str(exc)
                break
            except Exception:
                logger.exception("Unexpected tagging failure")
                return None, None, "Unexpected error during AI tagging.", False, True
    return None, None, last_error, only_not_found, False


def generate_tags(rgba: Image.Image) -> TaggingResult:
    """
    Tag a transparent clothing cut-out.  Never raises: on any AI failure the
    result contains local tags only (colors, plus a pattern / category when
    the shape makes it obvious), with `error` explaining why.
    """
    try:
        local_colors = dominant_colors(rgba)
    except Exception:  # extremely defensive; colors are a bonus
        logger.exception("Local color extraction failed")
        local_colors = []
    try:
        guess = local_guess(rgba)
    except Exception:
        logger.exception("Local shape/pattern guess failed")
        guess = {}

    def local_result(error: str) -> TaggingResult:
        tags = _empty_tags(local_colors, guess)
        return TaggingResult(tags, build_tag_list(tags), "local", error=error)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return local_result("GEMINI_API_KEY is not set; returning color tags only.")

    # Warn if the key does not look like a Gemini key (AIza prefix)
    if not api_key.startswith("AIza"):
        logger.warning(
            "GEMINI_API_KEY does not start with 'AIza' — it may not be a valid Gemini key. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    try:
        timeout = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
    except ValueError:
        timeout = 30.0

    try:
        image_b64 = _image_to_b64_jpeg(rgba)
    except Exception:
        logger.exception("Could not prepare image for tagging")
        return local_result("Could not prepare image for AI tagging.")

    tried = _model_candidates()
    tags, model, last_error, _only_missing, fatal = _run_models(tried, image_b64, api_key, timeout)

    if tags is None and not fatal:
        # The built-in models are retired, overloaded or out of quota.  Ask
        # Google what this key can actually use and try the newer ones too.
        found = [m for m in _discover_models(api_key, timeout) if m not in tried]
        if found:
            tags, model, found_error, _, _ = _run_models(
                found[:_MAX_DISCOVERED_TRIES], image_b64, api_key, timeout)
            if tags is None and found_error:
                last_error = found_error

    if tags is None:
        logger.warning("AI tagging unavailable: %s", last_error)
        return local_result(last_error or "AI tagging unavailable.")

    _model_state["good"] = model
    # Trust the model's colors, but fall back to measured ones.
    if not tags["colors"]:
        tags["colors"] = local_colors
    return TaggingResult(tags, build_tag_list(tags), "gemini", model=model)
