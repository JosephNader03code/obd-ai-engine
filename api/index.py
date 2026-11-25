# Vercel Python serverless entry point.
# Vercel serves files under /api as serverless functions; exposing an ASGI
# `app` here lets Vercel run the whole FastAPI app. vercel.json rewrites every
# path to this function, so FastAPI's own /api/* routes resolve normally.

import sys
from pathlib import Path

# Make the repo root importable so `app.main` resolves inside the function.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

# Vercel's Python runtime detects the module-level `app` ASGI application.
