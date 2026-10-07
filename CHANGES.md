# What changed (picture not showing + tagging)

## 1. Picture not showing after upload
**Cause:** after saving, the app re-downloaded every picture with `Image.network` from a second URL on the backend.
On Flutter web / phones that request can fail silently (the dashboard card had no loading or error state, so it was
just blank). Nothing was stored on the device, so a restart or a changed backend address blanked everything.

**Fix**
- `POST /analyze` now returns a small transparent copy of the cut-out inside the reply (`image_base64`, ~30-80 KB WebP).
- The app shows that copy immediately and **saves it on the device** with the item (`WardrobeItem.imageData`).
- New `WardrobeImage` widget: draws saved bytes; for old items it downloads with `package:http`, caches, shows a
  spinner, and a tap-to-retry icon on failure.
- `GET /result/{file}` is now served `inline` (it was `attachment`) with long-lived cache headers.
- Flutter web: the default backend address is now the same host the page was opened from (port 8000), so a phone
  opening `http://<pc-ip>:5000` no longer tries to call its own `127.0.0.1`.

## 2. Tagging
**Cause (most likely):** the backend only tried `gemini-2.5-flash` and `gemini-2.5-flash-lite`, which Google is
shutting down on 16 Oct 2026; any retirement or per-model quota problem ended in "unavailable" with no explanation.

**Fix**
- Model discovery: if the built-in names are "not found", ask Google which models the key can use, try the newest
  stable flash model, and remember the one that works.
- Per-model rate limits / overloads now move on to the next model (before: gave up after the first).
- Thought-parts of the reply are ignored; empty replies are handled.
- Real, key-safe reason is passed to the app, which now explains what to do (missing key, bad key, quota, old model).
- New `POST /retag/{filename}` + **Try smart tags again** button (no re-upload, no second background removal).
- Offline fallback is better: `solid` pattern and `trousers`/`shorts` are detected locally; **Quick add** chips for
  style / season / pattern / material.

## Files
Backend: `app/services/tagging.py`, `app/api/routes.py`, `.env.example`, `tests/test_tagging.py`
App: `lib/core/network/api_config.dart`, `lib/features/wardrobe/{data,screens,widgets}/...`,
`lib/features/dashboard/screens/dashboard_screen.dart`, `test/*`
