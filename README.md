# Vestara – PocketFashion

A wardrobe app. Photograph a piece of clothing and it is **cut out** (background removed with U²-Net) and
**auto-tagged** (category, colors, pattern, style, season…) by an AI model, then saved to your wardrobe.

```
Flutter app (this folder)  ──HTTP──▶  FastAPI backend (backend/)
  pick photo                           POST /api/v1/clothing/analyze
  review category + tags    ◀──────    transparent PNG + tags (JSON)
  save to wardrobe                     GET  /api/v1/clothing/result/{file}
```

## Run it (two terminals)

### 1. Backend
Windows: double-click **`run_backend.bat`** (or run it in a terminal). macOS/Linux: `./run_backend.sh`.

First run creates a virtualenv, installs dependencies and creates `backend/.env`.
Open `backend/.env` and paste a free key from <https://aistudio.google.com/apikey>:

```
GEMINI_API_KEY=your-real-key
```

The server starts at <http://127.0.0.1:8000> (try <http://127.0.0.1:8000/docs>). The AI model (~176 MB) downloads
once on first start; the log says when it is ready. **No key?** The app still works, you just get color tags only
and an on-screen note.

### 2. App
```bash
flutter pub get
flutter run
```

| Running on | Command | Notes |
|---|---|---|
| Android emulator | `flutter run` | auto-uses `10.0.2.2:8000` |
| iOS simulator, Windows/macOS/Linux desktop | `flutter run` | auto-uses `127.0.0.1:8000` |
| **Flutter web opened on a phone** | `flutter run -d web-server --web-hostname 0.0.0.0 --web-port 5000`, then open `http://<PC-WIFI-IP>:5000` on the phone | the app now talks to `http://<PC-WIFI-IP>:8000` automatically (same host as the page); set `HOST=0.0.0.0` in `backend/.env` |
| **Real phone** | `flutter run --dart-define=API_BASE_URL=http://<PC-WIFI-IP>:8000` | also set `HOST=0.0.0.0` in `backend/.env`, same Wi-Fi, allow port 8000 in the firewall |

## Using it
Dashboard → **Items +** → tap the photo box → take/choose a photo → wait a few seconds → the cut-out appears and the
category/subcategory are pre-selected. Edit the tag chips if you like → tap **✓** to save. Saved items appear in the
dashboard *Items* row and are kept after restarting the app. **Long-press** an item to remove it.

## What was added
- `backend/` – the background-removal service + tagging update (merged), with fixes: `.env` is now read before
  `HOST`/`PORT` (so `HOST=0.0.0.0` really works), and the AI model is pre-loaded at startup.
- `lib/core/network/api_config.dart` – backend address (per-platform defaults, `--dart-define` override).
- `lib/features/wardrobe/data/` – API client, response model, category mapping, saved-item model, persistent store.
- `lib/features/wardrobe/screens/add_item_screen.dart` – now uploads the photo, shows the cut-out, tags, saves.
- `lib/features/dashboard/screens/dashboard_screen.dart` – *Items* row shows real saved items (was placeholders).
- Android/iOS/macOS permission config for network + camera/photos. `test/` – new tests.

Still placeholders (no backend for them yet): the weather card, *My Outfits* list and Create Outfit screen.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Couldn't reach the server" | Backend not running / wrong address. Open `http://127.0.0.1:8000/health` on the PC. On a real phone you must pass `API_BASE_URL` and set `HOST=0.0.0.0`. |
| Item picture is blank / shows a refresh icon | New items keep their picture on the device, so this only happens for items saved by an older version: tap the icon to re-download it (the backend must be running). |
| "Smart tagging is off" note | The note now says why: missing key, rejected key, quota used up, or outdated model. Fix it in `backend/.env`, restart the backend, then tap **Try smart tags again** (no need to re-upload the photo). |
| Works in debug, not in a release Android build | Needs the `INTERNET` permission, now in the main manifest. |

## How tagging works now
1. Gemini (needs `GEMINI_API_KEY`) describes the garment. Model names are no longer hard-wired: if the built-in names
   are retired the backend lists the models your key can use and picks one, then remembers it.
2. Without Gemini you still get colours, plus **solid** (when the cloth is one uniform colour) and
   **trousers/shorts** (when two separate legs are visible). Use the *Quick add* chips for style, season, material.

## Before shipping (prototype caveats)
Cut-outs are stored in `backend/output_images/` and never cleaned up, and the endpoints have no login. Use `https`
(the Android `usesCleartextTraffic` flag is for development), add authentication, and consider storing images per user.
On Google's free Gemini tier, images sent for tagging may be used to improve Google's products; the API key stays in
`backend/.env` and must never be put in the app.
