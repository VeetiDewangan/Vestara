"""
Entry point.

Start the development server with:

    python run.py

The server reloads automatically on code changes (reload=True) so you do
not need to restart it manually while developing.
"""

import os

# Pre-load the AI model in the background when the dev server starts.
os.environ.setdefault("WARMUP_MODEL", "1")

import uvicorn  # noqa: E402
from app.config import HOST, PORT  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
    )
