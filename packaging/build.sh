#!/usr/bin/env bash
#
# Build a SELF-CONTAINED Evidence Engine: the end user needs no system Node.js or
# Python. Bundles a Node runtime + a relocatable python-build-standalone
# interpreter + vendored pip wheels, alongside the prebuilt frontend. On first run
# the launcher creates the backend venv from the vendored wheels (offline) and only
# Ollama + the ~4 GB model download from the network.
#
# Usage:
#   packaging/build.sh [target]
#     target ∈ darwin-arm64 (default) | darwin-x64 | linux-x64
#   Windows is built by packaging/build-windows.ps1.
#
# Output:
#   macOS  dist-bundle/EvidenceEngine-<version>-<target>.dmg   (drag to Applications)
#   Linux  dist-bundle/EvidenceEngine-<version>-linux-x64.zip  (unzip, run launch.sh)
#
# On macOS the whole app is placed INSIDE the .app bundle, at
# Contents/Resources/app. That is what lets it live in /Applications and be
# replaced wholesale to update it. Nothing is written inside the bundle at
# runtime: the venv, the env file and the database all live in ~/.evidence-engine.
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
  darwin-arm64) NODE_ARCH="darwin-arm64"; PBS_TRIPLE="aarch64-apple-darwin"; KIND="mac" ;;
  darwin-x64)   NODE_ARCH="darwin-x64";   PBS_TRIPLE="x86_64-apple-darwin";  KIND="mac" ;;
  linux-x64)    NODE_ARCH="linux-x64";    PBS_TRIPLE="x86_64-unknown-linux-gnu"; KIND="zip" ;;
  *) echo "Unknown target '$TARGET' (use darwin-arm64 | darwin-x64 | linux-x64)"; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/dist-bundle/$TARGET"
CACHE="$ROOT/.build-cache"
mkdir -p "$CACHE"

# package.json is the single place the version is edited. It is stamped into the
# payload as VERSION so a packaged install can report itself without depending on
# package.json surviving, and into the artifact name so a download says what it is.
VERSION="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$ROOT/package.json" | head -1)"
[ -n "$VERSION" ] || { echo "Could not read the version from package.json"; exit 1; }

# Where the app's own files go. On macOS that is inside the bundle.
if [ "$KIND" = "mac" ]; then
  APP="$OUT/Evidence Engine.app"
  PAYLOAD="$APP/Contents/Resources/app"
else
  PAYLOAD="$OUT/Evidence Engine"
fi

NODE_URL="https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-$NODE_ARCH.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/cpython-$PY_VERSION+$PBS_TAG-$PBS_TRIPLE-install_only.tar.gz"
NODE_TGZ="$CACHE/node-$NODE_VERSION-$NODE_ARCH.tar.gz"
PY_TGZ="$CACHE/cpython-$PY_VERSION-$PBS_TRIPLE.tar.gz"

echo "==> Evidence Engine $VERSION ($TARGET)"

echo "==> [1/7] Building the frontend (dist/) with the dev toolchain"
( cd "$ROOT" && npm run build >/dev/null )

echo "==> [2/7] Fetching Node $NODE_VERSION ($NODE_ARCH)"
[ -f "$NODE_TGZ" ] || curl -fL "$NODE_URL" -o "$NODE_TGZ"

echo "==> [3/7] Fetching python-build-standalone $PY_VERSION ($PBS_TRIPLE)"
[ -f "$PY_TGZ" ] || curl -fL "$PY_URL" -o "$PY_TGZ"

echo "==> [4/7] Assembling → $PAYLOAD"
rm -rf "$OUT"; mkdir -p "$PAYLOAD"
# Excludes are the difference between a 1.3 GB download and a reasonable one, and
# between shipping the developer's machine and shipping the app. Anything
# user-specific (caches, coverage, local tool state) or unrelated to running the
# app (the video project, test fixtures) stays out. Secrets are excluded twice
# over: by name here, and by .gitignore keeping them out of any clone.
# The two --include lines come first because rsync takes the first matching
# rule: without them '.env.*' would also drop .env.example, which the
# launcher needs to write the user's env file on first run.
rsync -a \
  --include 'Backend/.env.example' --include '.env.example' \
  --exclude '.git' --exclude '.build-cache' --exclude 'dist-bundle' \
  --exclude 'node_modules' \
  --exclude 'Backend/.venv' --exclude 'Backend/wheels' --exclude 'runtime' \
  --exclude '**/__pycache__' --exclude '.env.local' --exclude 'Backend/.env' \
  --exclude 'Backend/.env.*' --exclude '.env' --exclude '.env.*' \
  --exclude 'Backend/.rank_cache' --exclude 'Backend/.hf_cache' \
  --exclude 'Backend/textcache' --exclude 'Backend/pdfcache' \
  --exclude 'video' --exclude 'benchmark' --exclude 'reports' --exclude 'eval' \
  --exclude '.pytest_cache' --exclude '.coverage' --exclude 'coverage.json' \
  --exclude 'htmlcov' --exclude '.claude' --exclude '.githooks' \
  --exclude '.DS_Store' --exclude '._*' \
  --exclude 'Evidence Engine.app' \
  "$ROOT/" "$PAYLOAD/"

printf '%s\n' "$VERSION" > "$PAYLOAD/VERSION"

# A packaged bundle must never carry credentials or a developer's caches. Fail the
# build rather than ship one: a leaked key costs more than a failed build.
echo "==> [5/7] Auditing the staged payload"
LEAKS=$(find "$PAYLOAD" \( -name '.env' -o -name '.env.*' -o -name '*.bak-*' \) -not -name '.env.example' 2>/dev/null | head -5)
if [ -n "$LEAKS" ]; then echo "REFUSING TO BUILD: environment files staged:"; echo "$LEAKS"; exit 1; fi
if grep -rlEi 'sk-ant-[A-Za-z0-9_-]{20}|AIza[0-9A-Za-z_-]{30}|sk-[A-Za-z0-9]{40}' "$PAYLOAD" >/dev/null 2>&1; then
  echo "REFUSING TO BUILD: something in the bundle looks like an API key:"
  grep -rlEi 'sk-ant-[A-Za-z0-9_-]{20}|AIza[0-9A-Za-z_-]{30}|sk-[A-Za-z0-9]{40}' "$PAYLOAD" | head -5
  exit 1
fi
BIG=$(find "$PAYLOAD" -type d -name '.rank_cache' -o -type d -name 'video' 2>/dev/null | head -3)
[ -n "$BIG" ] && { echo "REFUSING TO BUILD: developer caches staged:"; echo "$BIG"; exit 1; }
echo "    no credentials or caches staged"

echo "==> [6/7] Adding the runtimes and vendoring backend wheels"
mkdir -p "$PAYLOAD/runtime/node"
tar -xzf "$NODE_TGZ" -C "$PAYLOAD/runtime/node" --strip-components=1
tar -xzf "$PY_TGZ" -C "$PAYLOAD/runtime"          # yields runtime/python/
BPY="$PAYLOAD/runtime/python/bin/python3"
"$BPY" -m pip download -r "$PAYLOAD/Backend/requirements.txt" -d "$PAYLOAD/Backend/wheels" >/dev/null

echo "==> [7/7] Packaging"
mkdir -p "$ROOT/dist-bundle"

if [ "$KIND" = "mac" ]; then
  # The .app wrapper: the launcher script, the icon, and a version-stamped plist.
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp "$ROOT/Evidence Engine.app/Contents/MacOS/EvidenceEngine" "$APP/Contents/MacOS/"
  cp "$ROOT/Evidence Engine.app/Contents/Resources/AppIcon.icns" "$APP/Contents/Resources/"
  chmod +x "$APP/Contents/MacOS/EvidenceEngine"
  sed -e "s|<key>CFBundleVersion</key><string>[^<]*</string>|<key>CFBundleVersion</key><string>$VERSION</string>|" \
      -e "s|<key>CFBundleShortVersionString</key><string>[^<]*</string>|<key>CFBundleShortVersionString</key><string>$VERSION</string>|" \
      "$ROOT/Evidence Engine.app/Contents/Info.plist" > "$APP/Contents/Info.plist"

  # Ad-hoc signature. It does not avoid Gatekeeper's "unidentified developer"
  # prompt (only a Developer ID and notarization do, see packaging/README.md),
  # but on Apple Silicon every binary must carry at least this to run at all.
  echo "    ad-hoc signing"
  codesign --force --deep --sign - "$APP" 2>/dev/null \
    || echo "    (codesign unavailable; the bundle is unsigned)"

  # A .dmg whose window holds the app and an Applications alias, so installing is
  # one drag. This is the whole reason for the restructure above.
  DMG="$ROOT/dist-bundle/EvidenceEngine-$VERSION-$TARGET.dmg"
  STAGEDMG="$OUT/dmg"
  rm -rf "$STAGEDMG"; mkdir -p "$STAGEDMG"
  cp -R "$APP" "$STAGEDMG/"
  ln -s /Applications "$STAGEDMG/Applications"
  cp "$ROOT/packaging/INSTALL-macOS.txt" "$STAGEDMG/Read Me First.txt" 2>/dev/null || true
  rm -f "$DMG"
  hdiutil create -volname "Evidence Engine $VERSION" -srcfolder "$STAGEDMG" \
    -ov -quiet -format UDZO "$DMG"
  rm -rf "$STAGEDMG"
  echo "==> Done: $DMG"
  echo "    Users open it and drag Evidence Engine to Applications."
else
  ZIP="$ROOT/dist-bundle/EvidenceEngine-$VERSION-$TARGET.zip"
  rm -f "$ZIP"   # zip APPENDS to an existing archive; start fresh
  ( cd "$OUT" && zip -qr "$ZIP" "Evidence Engine" )
  echo "==> Done: $ZIP"
  echo "    Users unzip it and run launch.sh."
fi
