#!/usr/bin/env bash
# Start the Evidence Engine API (FastAPI + uvicorn).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found; copying from .env.example"
  cp .env.example .env
fi

exec uvicorn api:app --reload --host 0.0.0.0 --port "${PORT:-8000}"
