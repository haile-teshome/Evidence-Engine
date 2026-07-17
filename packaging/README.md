# Self-contained bundles

These scripts produce a **download-and-run** Evidence Engine: the end user needs
no system Node.js or Python. Each bundle ships:

- `runtime/node/` — a Node runtime (runs the launcher + serves the built frontend)
- `runtime/python/` — a relocatable [python-build-standalone](https://github.com/astral-sh/python-build-standalone) interpreter
- `Backend/wheels/` — vendored pip wheels, so the backend venv is built **offline** on first run
- the prebuilt `dist/` frontend and the project source

Only **Ollama + the ~4 GB model** download from the network on first run (too big
to ship in the download; the launcher fetches them automatically).

## Build

Run on (or matching) the target platform so the vendored wheels' compiled
extensions match the target's ABI.

```bash
packaging/build.sh darwin-arm64     # Apple Silicon
packaging/build.sh darwin-x64       # Intel Mac
packaging/build.sh linux-x64        # Linux
# Windows:
powershell -ExecutionPolicy Bypass -File packaging/build-windows.ps1
```

Output: `dist-bundle/EvidenceEngine-<target>.zip`.

## CI matrix

Because wheels + runtimes are platform-specific, build each target on its own
runner (GitHub Actions):

| Target        | Runner            |
|---------------|-------------------|
| darwin-arm64  | `macos-14` (arm)  |
| darwin-x64    | `macos-13` (intel)|
| linux-x64     | `ubuntu-latest`   |
| win-x64       | `windows-latest`  |

Each job: `npm ci` → run the matching build script → upload the zip as a release
asset.

## Code signing (optional)

Unsigned bundles work but trigger a one-time Gatekeeper/SmartScreen prompt
(macOS: System Settings → Privacy & Security → "Open Anyway"). To ship a clean
double-click, sign + notarize on macOS (needs an Apple Developer ID) and
Authenticode-sign on Windows. The bundle layout doesn't change when signing is added.

## Versions

Pinned in each script (`NODE_VERSION`, `PBS_TAG`, `PY_VERSION`) for reproducible
builds. Bump deliberately and keep `build.sh` and `build-windows.ps1` in sync.
