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

# GitHub's release CDN answers these two assets with a 504 about half the time,
# and a single unlucky fetch killed a whole four-platform release: three builds
# green, one dead on the CPython tarball. Measured minutes apart, the arm64 and
# x86_64 tarballs failed and succeeded in turn, so the odds of eight attempts
# all failing are small. Written to a .part first, so a truncated download is
# never left behind to be taken for a complete one on the next run.
fetch() {
  url="$1"; out="$2"
  [ -f "$out" ] && return 0
  curl -fL --retry 8 --retry-delay 10 --retry-all-errors --connect-timeout 20 \
       "$url" -o "$out.part"
  mv "$out.part" "$out"
}

echo "==> [2/7] Fetching Node $NODE_VERSION ($NODE_ARCH)"
fetch "$NODE_URL" "$NODE_TGZ"

echo "==> [3/7] Fetching python-build-standalone $PY_VERSION ($PBS_TRIPLE)"
fetch "$PY_URL" "$PY_TGZ"

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

echo "==> [6/7] Adding the runtimes and installing the backend dependencies"
mkdir -p "$PAYLOAD/runtime/node"
tar -xzf "$NODE_TGZ" -C "$PAYLOAD/runtime/node" --strip-components=1
tar -xzf "$PY_TGZ" -C "$PAYLOAD/runtime"          # yields runtime/python/
BPY="$PAYLOAD/runtime/python/bin/python3"
# What this build stamps the bundle with, so the launcher can tell a user who
# downloaded the wrong one exactly that, instead of failing at the first thing
# that happens to need the right architecture.
echo "$TARGET" > "$PAYLOAD/runtime/TARGET"
# On Linux the default torch wheel drags in five CUDA packages (cudnn, nccl,
# cusparselt, nvshmem, triton) gated on platform_system == "Linux", which do not
# apply on macOS. That is gigabytes of GPU runtime for a desktop app whose
# ranker runs on CPU, and it would put the artifact over GitHub's 2 GB
# per-asset limit. Take torch from the CPU-only index there; PyPI still serves
# everything else.
# Spelled out rather than built as an array: expanding an EMPTY array under
# `set -u` is an error on bash 3.2, which is what macOS ships and what the Mac
# runners use, so an array here built fine on Linux and failed on both Macs.
# INSTALLED here, into the interpreter that ships with the app, rather than
# downloaded as wheels for the user's machine to install on first run. That
# first run was the whole problem: it needed pip to succeed on a stranger's
# machine, it took minutes, and anything that interrupted it left an
# environment that every later launch retried and failed on. A bundle now
# arrives with its dependencies already in it and has nothing to set up.
if [ "$TARGET" = "linux-x64" ]; then
  "$BPY" -m pip install --no-warn-script-location -r "$PAYLOAD/Backend/requirements.txt" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple >/dev/null
else
  "$BPY" -m pip install --no-warn-script-location \
    -r "$PAYLOAD/Backend/requirements.txt" >/dev/null
fi
# pip is not needed at runtime: nothing installs anything after this point.
# (`python -m venv` still works, from ensurepip in the standard library, which
# is the path a source checkout uses.)
rm -rf "$PAYLOAD"/runtime/python/lib/python3.*/site-packages/pip \
       "$PAYLOAD"/runtime/python/lib/python3.*/site-packages/pip-*.dist-info

# Prove the bundle has what it needs before it is wrapped up. A dependency that
# did not install is a build failure here, not something a reviewer discovers on
# their own laptop. The checker reads requirements.txt and honours its markers,
# so a package deliberately excluded on one platform (docling on Intel macs)
# counts as absent on purpose rather than failing the build.
# PYTHONDONTWRITEBYTECODE, because an import writes a .pyc beside every module
# it touches, and those are exactly what the next step is here to remove.
echo "    checking the bundle has everything the backend needs"
PYTHONDONTWRITEBYTECODE=1 "$BPY" "$ROOT/packaging/verify_bundle.py" "$PAYLOAD/Backend/requirements.txt" \
  || { echo "REFUSING TO PACKAGE: the bundled interpreter is missing something the backend needs"; exit 1; }

# No byte-compiled files ship, and this runs LAST because anything that imports
# a module writes one beside it. Compiling the lot added 373 MB to the download
# to save a one-off parse; the launcher points PYTHONPYCACHEPREFIX at the user's
# own cache instead, so they are written there on first run and an app folder
# nobody can write to stops mattering.
find "$PAYLOAD/runtime/python" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "    backend dependencies installed into the bundled interpreter"

echo "==> [7/7] Packaging"
mkdir -p "$ROOT/dist-bundle"

if [ "$KIND" = "mac" ]; then
  # The .app wrapper is an AppleScript applet, not a shell script, and that is
  # the whole point: an applet is a real app, so it owns a Dock tile with the
  # Evidence Engine icon and it is told when that tile is clicked. A shell
  # script is neither, which is why the icon used to vanish a second after
  # launch and leave a Chrome tile in its place. osacompile ships with macOS,
  # so this needs nothing installed.
  #
  # Built into a staging path first, because osacompile writes a whole bundle
  # and the payload has already been assembled inside $APP.
  echo "    building the Dock applet"
  APPLET="$OUT/applet.app"
  rm -rf "$APPLET"
  osacompile -s -o "$APPLET" "$ROOT/packaging/EvidenceEngine.applescript"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cp -R "$APPLET/Contents/MacOS/." "$APP/Contents/MacOS/"
  cp -R "$APPLET/Contents/Resources/." "$APP/Contents/Resources/"
  cp "$APPLET/Contents/Info.plist" "$APP/Contents/Info.plist"
  cp "$APPLET/Contents/PkgInfo" "$APP/Contents/PkgInfo" 2>/dev/null || true
  # The applet's own ad-hoc signature covers only the applet's files, and the
  # payload lands beside them; the bundle is re-signed as a whole below.
  rm -rf "$APP/Contents/_CodeSignature"
  rm -rf "$APPLET"
  # The applet's own generic icon is replaced by ours, under the name its
  # plist already points at.
  cp "$ROOT/Evidence Engine.app/Contents/Resources/AppIcon.icns" "$APP/Contents/Resources/applet.icns"
  cp "$ROOT/Evidence Engine.app/Contents/Resources/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
  # Identity: the name in the menu bar and the Dock, and the version macOS and
  # the update check report.
  /usr/libexec/PlistBuddy -c "Set :CFBundleName Evidence Engine" \
    -c "Set :CFBundleIconFile applet" \
    "$APP/Contents/Info.plist" >/dev/null 2>&1 || true
  for k in CFBundleDisplayName CFBundleIdentifier CFBundleShortVersionString CFBundleVersion; do
    case "$k" in
      CFBundleDisplayName) v="Evidence Engine" ;;
      CFBundleIdentifier)  v="com.evidenceengine.launcher" ;;
      *)                   v="$VERSION" ;;
    esac
    /usr/libexec/PlistBuddy -c "Set :$k $v" "$APP/Contents/Info.plist" >/dev/null 2>&1 \
      || /usr/libexec/PlistBuddy -c "Add :$k string $v" "$APP/Contents/Info.plist" >/dev/null 2>&1 || true
  done

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
