# ============================================================================
# FILE: webapp.py
# Serve the built frontend from the API process.
# ============================================================================
"""One process, one port, no Node at runtime.

Why: running the app used to mean two servers, a Python API on :8000 and a Node
dev or preview server on :5173/:5180 proxying /api to it. That is fine on a
developer's machine and awkward everywhere else: it doubles what an installer has
to start and keep alive, and it is the main obstacle to a hosted instance or a
double-click bundle. Pointing FastAPI at the Vite build removes the Node process
entirely, so `uvicorn api:app` serves the whole application.

The build is `dist/` at the repository root (`pnpm build`). When it is absent the
API still starts and only the HTML routes 404, so backend development and the
test suite are unaffected; the dev server keeps proxying /api as before.

Routing rules, in order:
  /api/...        untouched, handled by the routers registered before this mount
  /assets/...     hashed build files, cached hard (a new build has new names)
  /favicon, etc.  any other real file under dist, served as-is
  anything else   index.html, because the frontend is a single-page app and a
                  deep link such as /?invite=TOKEN must reach the client router

One handler serves every real file rather than a StaticFiles mount, so the
cache rule and the containment check live in a single place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

# Paths that must never fall through to index.html: a wrong answer here turns a
# missing endpoint into a 200 page, which is far harder to diagnose than a 404.
API_PREFIXES = ("api/", "docs", "redoc", "openapi.json")
IMMUTABLE_MAX_AGE = 60 * 60 * 24 * 365


def dist_dir() -> Path:
    """Where the build lives. EE_DIST_DIR overrides it (a packaged app ships the
    files beside the executable rather than two directories up)."""
    env = os.getenv("EE_DIST_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "dist"


def frontend_status() -> dict:
    d = dist_dir()
    index = d / "index.html"
    return {"serving_frontend": index.is_file(), "dist": str(d),
            "hint": None if index.is_file() else "Run `pnpm build` to produce dist/, then restart."}


def mount_frontend(app: FastAPI, directory: Optional[Path] = None) -> bool:
    """Attach the built frontend. Call AFTER every API router is registered, since
    the catch-all would otherwise shadow them. Returns whether it mounted."""
    d = Path(directory) if directory else dist_dir()
    index = d / "index.html"
    if not index.is_file():
        return False

    def _index() -> FileResponse:
        # never cached: it names the hashed asset files, so a stale copy loads a
        # build that no longer exists
        return FileResponse(index, headers={"Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return _index()

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str, request: Request):
        if full_path.startswith(API_PREFIXES):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = (d / full_path).resolve()
        try:                      # refuse to serve outside dist (traversal)
            candidate.relative_to(d.resolve())
        except ValueError:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if candidate.is_file():
            headers = {"Cache-Control": f"public, max-age={IMMUTABLE_MAX_AGE}, immutable"} \
                if "/assets/" in f"/{full_path}" else {}
            return FileResponse(candidate, headers=headers)
        return _index()           # a client route, not a missing file

    return True
