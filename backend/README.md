# Clothing Background Removal

A computer-vision microservice that takes a photograph of a clothing item and
returns a clean transparent PNG with the background removed.

This is a functional **prototype** intended to become a CV module inside a
larger wardrobe-management application.

---

## Final Pipeline

```
Image Upload
  → Validation          (format, size, dimensions)
  → Preprocessing       (decode, channel normalise, resize to max 2048 px)
  → U²-Net              (soft segmentation mask via rembg)
  → Raw Mask Analysis   (objective statistics, no modification)
  → OpenCV Refinement   (opening, closing, small component removal, hole fill)
  → Alpha Composition   (RGB + soft mask → RGBA)
  → Transparent PNG     (saved to output_images/)
  → Optional Tight Crop (removes transparent borders)
  → JSON response       (filename, dimensions, processing time)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Liveness check |
| `GET`  | `/health` | Health probe |
| `POST` | `/api/v1/clothing/upload` | Validate and store a clothing image |
| `POST` | `/api/v1/clothing/remove-background` | Full pipeline — returns JSON with output filename |
| `GET`  | `/api/v1/clothing/result/{filename}` | Download the generated transparent PNG |

Interactive Swagger UI: **http://127.0.0.1:8000/docs**

---

## Final Validation

- Real clothing image tested: `input_images/shirt-m.jpg` (450 x 300).
- The output was valid RGBA PNG with genuine transparent pixels and visible garment pixels.
- The original input remained unchanged.
- The complete test suite passes: 194 tests.

The live result retained the shirt clearly, with a reasonable tight crop and no
major holes. Some wood texture remains around the garment. This is an expected
limitation of using general-purpose U2-Net: conservative morphological
refinement cannot correct a fundamentally imperfect segmentation mask.

Do not commit model weights, uploaded images, or processed outputs to version
control; these directories are git-ignored.
## Current Development Status

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Project Setup | ✅ Complete |
| 2 | Image Upload & Validation | 🔜 Planned |
| 3 | Pre-processing Pipeline | 🔜 Planned |
| 4 | AI Segmentation Integration | 🔜 Planned |
| 5 | Mask Refinement with OpenCV | 🔜 Planned |
| 6 | Background Removal & Export | 🔜 Planned |
| 7 | Testing & Optimisation | 🔜 Planned |

---

## Planned CV Pipeline

```
Image
  → Image Validation
  → Preprocessing
  → AI Segmentation
  → Mask Refinement with OpenCV
  → Background Removal
  → Cropping
  → Transparent PNG
```

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| Image processing | OpenCV, Pillow |
| Numerical computing | NumPy |
| File upload | python-multipart |
| AI segmentation | TBD (Section 4) |

---

## Project Structure

```
clothing-bg-removal/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application factory
│   ├── config.py            # Centralised configuration
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # Feature API routes (added from Section 2)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processing.py   # Pre/post-processing pipeline
│   │   └── segmentation.py       # AI segmentation wrapper
│   │
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py   # Shared image helpers
│
├── models/
│   └── segmentation/        # Model weights (git-ignored)
│
├── input_images/            # Uploaded images (git-ignored)
├── output_images/           # Processed results (git-ignored)
│
├── tests/
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── run.py                   # Development server entry point
```

---

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the development server

```bash
python run.py
```

The API will be available at **http://127.0.0.1:8000**.

Interactive docs: **http://127.0.0.1:8000/docs**

---

## API Endpoints (Section 1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Confirms the API is running |
| GET | `/health` | Liveness probe |

---

## Notes

- The segmentation model has **not** been selected yet and will be integrated in Section 4 after evaluating available options.
- Do not add model weights, uploaded images, or processed results to version control — they are git-ignored.
