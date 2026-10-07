#!/usr/bin/env bash
# Starts the Vestara backend (background removal + AI tagging) on macOS/Linux.
set -e
cd "$(dirname "$0")/backend"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Created backend/.env - paste your GEMINI_API_KEY there for smart tags."
fi
echo "Starting server on http://127.0.0.1:8000  (API docs: /docs)"
python run.py
