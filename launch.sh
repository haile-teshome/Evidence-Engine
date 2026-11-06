#!/bin/bash
# Linux launcher for Evidence Engine. Run ./launch.sh (or double-click it, if
# your file manager is set to run executables).
#
# Self-locating: works no matter where this folder lives.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the bundled Node runtime, so a packaged download needs nothing
# installed. Fall back to a system Node for a source checkout.
NODE="$DIR/runtime/node/bin/node"
if [ ! -x "$NODE" ]; then
  if command -v node >/dev/null 2>&1; then
    NODE="node"
  else
    echo "Node.js is required but was not found, and this copy has no bundled runtime."
    echo "Install it with your package manager, for example:"
    echo "  sudo apt install nodejs     # Debian / Ubuntu"
    echo "  sudo dnf install nodejs     # Fedora"
    echo "or download it from https://nodejs.org/en/download, then run this again."
    exit 1
  fi
fi

exec "$NODE" "$DIR/launch.mjs"
