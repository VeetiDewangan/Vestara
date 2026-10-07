> **Note:** the update described below is already merged into this repository's `backend/` folder, so skip the "Copy the files" step. Everything about getting the Gemini key and the response shape still applies. See the root `README.md` for how to run the whole project.

# Tagging update (Section 8)

Adds `POST /api/v1/clothing/analyze`: removes the background **and** returns tags.

## 1. Copy the files
Unzip this archive **on top of** your `clothing-background-removal` folder and
allow it to overwrite. Files in this update:

    clothing-background-removal/
    ├── .env.example                  NEW   template for your API key
    ├── TAGGING_SETUP.md              NEW   this file
    ├── requirements.txt              CHANGED (adds httpx, python-dotenv)
    ├── app/
    │   ├── main.py                   CHANGED (adds CORS)
    │   ├── api/
    │   │   └── routes.py             CHANGED (adds POST /analyze)
    │   └── services/
    │       └── tagging.py            NEW   colors + Gemini AI tags
    └── tests/
        └── test_tagging.py           NEW   41 tests (all mocked, no key needed)

Nothing else is touched. `/upload`, `/remove-background`, `/result/{filename}`
behave exactly as before.

## 2. Get a free key
1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click "Create API key". No credit card needed.
3. Copy `.env.example` to `.env` (same folder as `run.py`) and paste the key:

       GEMINI_API_KEY=your-real-key

`.env` is already in your `.gitignore`. Never commit it and never put the key
in the Flutter app.

## 3. Install and run
    pip install -r requirements.txt
    python run.py

Open http://127.0.0.1:8000/docs, try **POST /api/v1/clothing/analyze** with an
image from `input_images/`.

## 4. Response shape
    {
      "success": true,
      "filename": "abc.png",
      "image_url": "/api/v1/clothing/result/abc.png",
      "width": 400, "height": 320, "processing_time_s": 3.1,
      "tags": {
        "category": "t-shirt",
        "colors": ["navy", "white"],
        "pattern": "striped",
        "style": ["casual"],
        "season": ["summer"],
        "material_guess": "cotton",
        "extra_tags": ["crew neck", "oversized"]
      },
      "tag_list": ["t-shirt", "navy", "white", "striped", "casual", "summer", "cotton", "crew neck", "oversized"],
      "tagging": { "source": "gemini", "model": "gemini-2.5-flash", "error": null }
    }

If there is no key, no internet, or the free quota is used up, you still get the
cut-out and color tags: `tagging.source` becomes `"local"` and `tagging.error`
says why. Show `tag_list` as chips in the app and let the user edit them.

## 5. Run the tests
    pip install pytest
    python -m pytest tests/test_tagging.py -q

## Notes
- Free-tier limits and model names change. If Gemini says a model is missing,
  set `GEMINI_MODEL` in `.env` to one listed in Google AI Studio.
- On Google's free tier, images you send may be used to improve Google's
  products. Only the cut-out garment is sent (not the original photo).
