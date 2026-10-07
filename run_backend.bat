@echo off
REM Starts the Vestara backend (background removal + AI tagging) on Windows.
REM Double-click, or run from a terminal in the repo root.
cd /d "%~dp0backend"

if not exist venv (
  echo Creating Python virtual environment...
  python -m venv venv || goto :error
)
call venv\Scripts\activate || goto :error

echo Installing / checking dependencies (first time takes a few minutes)...
pip install -r requirements.txt || goto :error

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo *** Created backend\.env  -  open it and paste your GEMINI_API_KEY for smart tags.
  echo *** Without a key the app still works, but only detects colors.
  echo.
)

echo Starting server on http://127.0.0.1:8000  (API docs: http://127.0.0.1:8000/docs)
python run.py
goto :eof

:error
echo.
echo Setup failed - see the message above.
pause
