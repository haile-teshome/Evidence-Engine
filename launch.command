#!/bin/bash
# macOS launcher for Evidence Engine. Double-click to run.
# Self-locating: works no matter where this folder lives, on any Mac.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

if ! command -v node >/dev/null 2>&1; then
  osascript -e 'display alert "Node.js is required" message "Evidence Engine needs Node.js (LTS). Opening the download page — install it, then double-click this file again."' >/dev/null 2>&1
  open "https://nodejs.org/en/download"
  exit 1
fi

exec node "$DIR/launch.mjs"
