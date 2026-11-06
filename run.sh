#!/usr/bin/env bash
# Start Evidence Engine: one process, one port, http://localhost:8000
#
#   ./run.sh              build the frontend if needed, then serve everything
#   ./run.sh --port 9000  serve on another port
#   ./run.sh --rebuild    force a fresh frontend build first
#
# No Node is needed at runtime: the API serves the built frontend. Node is only
# needed to produce dist/, which this script does for you if it is missing.
# Stop with Ctrl-C, or ./teardown.sh if it was started in the background.
set -euo pipefail
cd "$(dirname "$0")"
PORT=8000 REBUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2;;
    --rebuild) REBUILD=1; shift;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option: $1"; exit 1;;
  esac
done

# Either location: a source checkout builds Backend/.venv, while the packaged
# launcher builds it outside the app so updates can replace the app wholesale.
PY="$PWD/Backend/.venv/bin/python"
[ -x "$PY" ] || PY="$HOME/.evidence-engine/venv/bin/python"
[ -x "$PY" ] || { echo "No Python environment yet. Run ./setup.sh first."; exit 1; }

if [ "$REBUILD" = 1 ] || [ ! -f dist/index.html ]; then
  if command -v pnpm >/dev/null 2>&1; then
    echo "Building the frontend…"; pnpm build
  elif command -v npm >/dev/null 2>&1; then
    echo "Building the frontend with npm…"; npm run build
  else
    echo "dist/ is missing and neither pnpm nor npm is installed."
    echo "Install Node, or copy a prebuilt dist/ here, then re-run."
    exit 1
  fi
fi

echo "Evidence Engine → http://localhost:$PORT"
cd Backend
exec "$PY" -m uvicorn api:app --host 127.0.0.1 --port "$PORT"
