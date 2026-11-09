"""Download a release and stage it, so the app can replace itself.

Updating used to mean a browser, a 500 MB download, a drag onto Applications
and another trip through Gatekeeper. Every step of that is a place to give up,
and the Gatekeeper prompt recurs on every update because each download the
BROWSER makes is quarantined. A file this module downloads is not, so an update
applied from inside the app never meets that dialog at all.

Nothing here replaces anything. It downloads, checks the bytes against the
SHA256SUMS the release publishes, and unpacks the new app beside the reviewer's
data. The swap itself happens in the launcher, once the app has quit, because a
bundle cannot safely be replaced underneath the process running from it.

Two rules hold everywhere in this file:

  - Only ever write inside ~/.evidence-engine/updates. The reviews, the
    database and the cached article text sit beside that directory and are
    never touched, which is the whole reason an update can keep them.
  - Never stage bytes that failed their checksum. An unsigned build that
    arrived damaged is the one thing worse than no update.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

import updates

_TIMEOUT = 30
_CHUNK = 1 << 20          # 1 MiB: big enough that progress is cheap to report

# One in-flight update at a time, and one status the UI polls.
_lock = threading.Lock()
_state: Dict[str, Any] = {"state": "idle"}


def updates_dir() -> Path:
    """Where downloads and the staged app live: inside the data folder, in its
    own subdirectory, so the rest of the reviewer's work is never in reach."""
    override = os.environ.get("EE_UPDATE_DIR", "").strip()
    base = Path(override) if override else Path.home() / ".evidence-engine" / "updates"
    return base


def asset_name(version: str) -> str:
    """The release asset for the machine this is running on.

    Mac takes the disk image and Linux the zip, because that is what the app is
    distributed as. Windows takes the silent installer, which knows how to
    replace an install that is already there.
    """
    version = (version or "").lstrip("vV")
    system = platform.system().lower()
    if system == "darwin":
        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
        return f"EvidenceEngine-{version}-darwin-{arch}.dmg"
    if system == "windows":
        return f"EvidenceEngine-{version}-win-x64-Setup.exe"
    return f"EvidenceEngine-{version}-linux-x64.zip"


def _download_url(version: str, name: str) -> str:
    tag = version if version.startswith("v") else f"v{version}"
    return f"https://github.com/{updates.repo()}/releases/download/{tag}/{name}"


def expected_sha(version: str, name: str) -> str:
    """The checksum the release publishes for this asset, or "" if unavailable.

    The file is `<sha256>␣␣<filename>` per line, as sha256sum writes it.
    """
    try:
        res = requests.get(_download_url(version, "SHA256SUMS"),
                           headers={"User-Agent": "EvidenceEngine-update"}, timeout=_TIMEOUT)
        res.raise_for_status()
    except Exception as e:                      # noqa: BLE001 - advisory
        print(f"[self_update] could not read SHA256SUMS: {e}")
        return ""
    for line in res.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0].lower()
    return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def status() -> Dict[str, Any]:
    """What the in-flight update is doing, for the banner to poll."""
    with _lock:
        return dict(_state)


def _set(**fields: Any) -> None:
    with _lock:
        _state.update(fields)


def staged() -> Optional[Dict[str, Any]]:
    """The update waiting to be applied on the next launch, if there is one."""
    try:
        data = json.loads((updates_dir() / "staged.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def start(version: str) -> Dict[str, Any]:
    """Begin downloading `version` in the background. Returns the first status."""
    with _lock:
        if _state.get("state") in ("downloading", "staging"):
            return dict(_state)
        _state.clear()
        _state.update({"state": "downloading", "version": version,
                       "received": 0, "total": 0, "error": ""})
    threading.Thread(target=_run, args=(version,), daemon=True).start()
    return status()


def _run(version: str) -> None:
    try:
        name = asset_name(version)
        work = updates_dir()
        # A half-finished attempt from last time is not something to resume:
        # the bytes may be from a different release.
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        archive = work / name
        want = expected_sha(version, name)
        _download(_download_url(version, name), archive)

        _set(state="verifying")
        got = _sha256(archive)
        if want and got != want:
            raise RuntimeError("the download did not match the published checksum")
        if not want:
            print("[self_update] no published checksum for this asset; staging unverified")

        _set(state="staging")
        payload = _stage(archive, work)
        (work / "staged.json").write_text(json.dumps({
            "version": version, "path": str(payload), "asset": name, "sha256": got,
        }), encoding="utf-8")
        archive.unlink(missing_ok=True)
        _set(state="ready", path=str(payload))
    except Exception as e:                      # noqa: BLE001 - reported to the UI
        print(f"[self_update] {e}")
        _set(state="failed", error=str(e))


def _download(url: str, out: Path) -> None:
    with requests.get(url, stream=True, timeout=_TIMEOUT,
                      headers={"User-Agent": "EvidenceEngine-update"}) as res:
        res.raise_for_status()
        total = int(res.headers.get("content-length") or 0)
        _set(total=total)
        received = 0
        # Written to .part so a download cut off half way is never mistaken for
        # a complete one, the same rule the build scripts follow.
        part = out.with_suffix(out.suffix + ".part")
        with part.open("wb") as fh:
            for chunk in res.iter_content(_CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                received += len(chunk)
                _set(received=received)
        part.replace(out)


def _stage(archive: Path, work: Path) -> Path:
    """Unpack the download into something the launcher can move into place."""
    system = platform.system().lower()
    if system == "darwin":
        return _stage_dmg(archive, work)
    if system == "windows":
        return archive          # the installer IS what gets run
    target = work / "staged"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    inner = [p for p in target.iterdir() if p.is_dir()]
    return inner[0] if len(inner) == 1 else target


def _stage_dmg(archive: Path, work: Path) -> Path:
    """Copy the .app out of the disk image, then let the image go.

    ditto rather than shutil.copytree: it keeps the symlinks and the extended
    attributes a bundle needs, and a Python-level copy of a 637 MB tree with a
    Framework in it is both slower and lossier.
    """
    mount = work / "mnt"
    mount.mkdir(parents=True, exist_ok=True)
    subprocess.run(["hdiutil", "attach", str(archive), "-nobrowse", "-readonly",
                    "-mountpoint", str(mount), "-quiet"], check=True)
    try:
        apps = [p for p in mount.iterdir() if p.suffix == ".app"]
        if not apps:
            raise RuntimeError("the disk image held no application")
        out = work / apps[0].name
        shutil.rmtree(out, ignore_errors=True)
        subprocess.run(["ditto", str(apps[0]), str(out)], check=True)
    finally:
        subprocess.run(["hdiutil", "detach", str(mount), "-quiet"], check=False)
    shutil.rmtree(mount, ignore_errors=True)
    return out
