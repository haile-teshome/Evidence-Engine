"""Is a newer Evidence Engine available?

The check lives in the backend rather than in the launcher so it works however
the app was started: the .app, launch.command, run.sh, or `uvicorn api:app`
directly. It is advisory only. Nothing here downloads or installs anything.

Every failure path returns "no update" instead of raising. A release check must
never be able to stop the app from starting, and the answer is cached on disk so
a reviewer working offline is not waiting on GitHub once a day.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from packaging.version import InvalidVersion, Version

# The project root holds package.json and the build's VERSION stamp.
_ROOT = Path(__file__).resolve().parent.parent

CACHE_TTL_SECONDS = 24 * 60 * 60
NOTES_LIMIT = 2000      # release bodies can be long; the banner only needs a summary
_TIMEOUT = 6


def repo() -> str:
    """owner/name of the repository whose releases are the source of updates."""
    return os.environ.get("EE_UPDATE_REPO", "haile-teshome/Evidence-Engine").strip()


def cache_path() -> Path:
    """Resolved per call so a test (or a sandboxed install) can redirect it."""
    override = os.environ.get("EE_UPDATE_CACHE", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".evidence-engine" / "update-check.json"


def enabled() -> bool:
    return os.environ.get("EE_UPDATE_CHECK", "1").strip().lower() not in ("0", "false", "no", "off")


def current_version() -> str:
    """The running version: the build stamp, else package.json, else 0.0.0.

    packaging/build.sh writes VERSION into the bundle, so a packaged install does
    not depend on package.json surviving the rsync. A source checkout falls
    through to package.json, which is the single place the version is edited.
    """
    for candidate in (
        os.environ.get("EE_VERSION", ""),
        _read_stamp(_ROOT / "VERSION"),
        _read_package_version(_ROOT / "package.json"),
    ):
        value = (candidate or "").strip().lstrip("vV")
        if value:
            return value
    return "0.0.0"


def _read_stamp(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_package_version(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("version", "") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def parse(value: Any) -> Optional[Version]:
    """A tag like "v1.2.0" as a comparable version, or None if it is not one."""
    try:
        return Version(str(value).strip().lstrip("vV"))
    except (InvalidVersion, TypeError):
        return None


def is_newer(latest: Any, current: Any) -> bool:
    """True only when both parse and latest is strictly greater.

    An unparseable tag is never treated as newer: a release named "nightly"
    should not nag every user forever.
    """
    new, old = parse(latest), parse(current)
    if new is None or old is None:
        return False
    return new > old


def fetch_latest() -> Dict[str, Any]:
    """The newest published release. Raises on any network or decode failure."""
    res = requests.get(
        f"https://api.github.com/repos/{repo()}/releases/latest",
        headers={
            "Accept": "application/vnd.github+json",
            # GitHub rejects requests that do not identify themselves.
            "User-Agent": "EvidenceEngine-update-check",
        },
        timeout=_TIMEOUT,
    )
    res.raise_for_status()
    body = res.json()
    return {
        "tag": str(body.get("tag_name") or ""),
        "name": str(body.get("name") or ""),
        "url": str(body.get("html_url") or ""),
        "notes": str(body.get("body") or "")[:NOTES_LIMIT],
        "published_at": str(body.get("published_at") or ""),
    }


def _read_cache() -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_cache(payload: Dict[str, Any]) -> None:
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass    # a cache we cannot write is slower, not broken


def check(force: bool = False) -> Dict[str, Any]:
    """Compare the running version against the latest release.

    `newer` is recomputed from the current version on every call, including on a
    cache hit, so installing the update clears the banner without waiting for
    the cache to expire.
    """
    current = current_version()
    out: Dict[str, Any] = {
        "current": current, "latest": "", "newer": False, "url": "", "name": "",
        "notes": "", "published_at": "", "checked_at": 0.0,
        "enabled": enabled(), "error": "",
    }
    if not out["enabled"]:
        return out

    cached = _read_cache() or {}
    known = cached.get("release") if isinstance(cached.get("release"), dict) else None
    cached_at = _as_float(cached.get("checked_at"))
    stale = (time.time() - cached_at) >= CACHE_TTL_SECONDS

    if known and not force and not stale:
        release, checked_at = known, cached_at
    else:
        try:
            release = fetch_latest()
            checked_at = time.time()
            _write_cache({"checked_at": checked_at, "release": release})
        except Exception as exc:
            # Offline, rate-limited, or no release published yet. Broad on
            # purpose: this function is called on a page load and must not throw.
            out["error"] = type(exc).__name__
            if not known:
                return out
            release, checked_at = known, cached_at    # last known answer beats none

    out.update({
        "latest": release.get("tag", ""),
        "name": release.get("name", ""),
        "url": release.get("url", ""),
        "notes": release.get("notes", ""),
        "published_at": release.get("published_at", ""),
        "checked_at": checked_at,
        "newer": is_newer(release.get("tag", ""), current),
    })
    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
