#!/usr/bin/env bash
#
# Build a SELF-CONTAINED Evidence Engine bundle: the end user needs no system
# Node.js or Python. Bundles a Node runtime + a relocatable python-build-standalone
# interpreter + vendored pip wheels, alongside the prebuilt frontend. On first run
# the launcher creates the backend venv from the vendored wheels (offline) and only
# Ollama + the ~4 GB model download from the network.
#
# Usage:
#   packaging/build.sh [target]
#     target ∈ darwin-arm64 (default) | darwin-x64 | linux-x64
#   Windows is built by packaging/build-windows.ps1 (bundles node.exe + python + wheels).
#
# Run this on (or matching) the target platform so the vendored wheels' compiled
# extensions (pydantic-core, lxml, numpy, pandas, ...) match. In CI use a matrix
# with one runner per target.
set -euo pipefail

TARGET="${1:-darwin-arm64}"

# --- pinned versions (bump deliberately for reproducible builds) --------------
NODE_VERSION="v22.11.0"
PBS_TAG="20241016"          # github.com/astral-sh/python-build-standalone release
PY_VERSION="3.12.7"

case "$TARGET" in
  darwin-arm64) NODE_ARCH="darwin-arm64"; PBS_TRIPLE="aarch64-apple-darwin" ;;
  darwin-x64)   NODE_ARCH="darwin-x64";   PBS_TRIPLE="x86_64-apple-darwin" ;;
  linux-x64)    NODE_ARCH="linux-x64";    PBS_TRIPLE="x86_64-unknown-linux-gnu" ;;
  *) echo "Unknown target '$TARGET' (use darwin-arm64 | darwin-x64 | linux-x64)"; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/dist-bundle/$TARGET"
STAGE="$OUT/Evidence Engine"
CACHE="$ROOT/.build-cache"
mkdir -p "$CACHE"

NODE_URL="https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-$NODE_ARCH.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/cpython-$PY_VERSION+$PBS_TAG-$PBS_TRIPLE-install_only.tar.gz"
NODE_TGZ="$CACHE/node-$NODE_VERSION-$NODE_ARCH.tar.gz"
PY_TGZ="$CACHE/cpython-$PY_VERSION-$PBS_TRIPLE.tar.gz"

echo "==> [1/6] Building the frontend (dist/) with the dev toolchain"
( cd "$ROOT" && npm run build >/dev/null )

echo "==> [2/6] Fetching Node $NODE_VERSION ($NODE_ARCH)"
[ -f "$NODE_TGZ" ] || curl -fL "$NODE_URL" -o "$NODE_TGZ"

echo "==> [3/6] Fetching python-build-standalone $PY_VERSION ($PBS_TRIPLE)"
[ -f "$PY_TGZ" ] || curl -fL "$PY_URL" -o "$PY_TGZ"

echo "==> [4/6] Assembling bundle → $STAGE"
rm -rf "$OUT"; mkdir -p "$STAGE"
rsync -a \
  --exclude '.git' --exclude '.build-cache' --exclude 'dist-bundle' \
  --exclude 'node_modules' \
  --exclude 'Backend/.venv' --exclude 'Backend/wheels' --exclude 'runtime' \
  --exclude '**/__pycache__' --exclude '.env.local' --exclude 'Backend/.env' \
  "$ROOT/" "$STAGE/"

mkdir -p "$STAGE/runtime/node"
tar -xzf "$NODE_TGZ" -C "$STAGE/runtime/node" --strip-components=1
tar -xzf "$PY_TGZ" -C "$STAGE/runtime"          # yields runtime/python/

echo "==> [5/6] Vendoring backend wheels (offline install source) with the bundled Python"
BPY="$STAGE/runtime/python/bin/python3"
"$BPY" -m pip download -r "$STAGE/Backend/requirements.txt" -d "$STAGE/Backend/wheels" >/dev/null

echo "==> [6/6] Zipping"
rm -f "$ROOT/dist-bundle/EvidenceEngine-$TARGET.zip"   # zip APPENDS to an existing archive; start fresh
( cd "$OUT" && zip -qr "../EvidenceEngine-$TARGET.zip" "Evidence Engine" )
echo "==> Done: $ROOT/dist-bundle/EvidenceEngine-$TARGET.zip"
echo "    Users unzip it and double-click 'Evidence Engine.app' (macOS) or run launch.command / launch.sh."
