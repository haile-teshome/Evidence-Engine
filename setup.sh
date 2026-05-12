#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Evidence Engine — one-shot setup script
# Installs and configures: Backend (FastAPI), Frontend (React/Vite), and the
# Benchmark suite. Idempotent — safe to re-run.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ─── colors ────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; DIM="\033[2m"; RESET="\033[0m"
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; DIM=""; RESET=""
fi
log()    { printf "${BOLD}${GREEN}==>${RESET} %s\n" "$*"; }
warn()   { printf "${BOLD}${YELLOW}!! ${RESET} %s\n" "$*"; }
err()    { printf "${BOLD}${RED}xx ${RESET} %s\n" "$*" >&2; }
indent() { sed 's/^/    /'; }

# ─── arg parsing ───────────────────────────────────────────────────────────
INSTALL_BACKEND=1
INSTALL_FRONTEND=1
INSTALL_BENCHMARK=1
PULL_OLLAMA=0
HELP=0
for arg in "$@"; do
  case "$arg" in
    --backend-only)   INSTALL_BACKEND=1; INSTALL_FRONTEND=0; INSTALL_BENCHMARK=0 ;;
    --frontend-only)  INSTALL_BACKEND=0; INSTALL_FRONTEND=1; INSTALL_BENCHMARK=0 ;;
    --benchmark-only) INSTALL_BACKEND=0; INSTALL_FRONTEND=0; INSTALL_BENCHMARK=1 ;;
    --no-backend)     INSTALL_BACKEND=0 ;;
    --no-frontend)    INSTALL_FRONTEND=0 ;;
    --no-benchmark)   INSTALL_BENCHMARK=0 ;;
    --pull-ollama)    PULL_OLLAMA=1 ;;
    -h|--help)        HELP=1 ;;
    *) err "Unknown argument: $arg"; HELP=1 ;;
  esac
done

if [ "$HELP" = "1" ]; then
  cat <<'EOF'
Usage: ./setup.sh [flags]

Installs and configures the Evidence Engine workspace.

Components:
  Backend    Python FastAPI server (api.py)
  Frontend   React + Vite UI (src/)
  Benchmark  Standalone screening-architecture benchmark (benchmark/)

Flags:
  --backend-only        Install only the FastAPI backend
  --frontend-only       Install only the React frontend
  --benchmark-only      Install only the benchmark suite
  --no-backend          Skip backend
  --no-frontend         Skip frontend
  --no-benchmark        Skip benchmark
  --pull-ollama         Pull the recommended Ollama models (large download)
  -h, --help            Show this help

After setup:
  - Backend: cd Backend && bash run_api.sh        (FastAPI on :8000)
  - Frontend: pnpm dev                            (Vite on :5173)
  - Benchmark: cd benchmark && python run_benchmark.py --datasets sample \
               --architectures single_combined --models small
EOF
  exit 0
fi

# ─── prerequisite checks ──────────────────────────────────────────────────
log "Checking prerequisites…"
MISSING=()
command -v python3 >/dev/null 2>&1 || MISSING+=("python3 (>= 3.10)")
command -v pip3 >/dev/null 2>&1 || command -v pip >/dev/null 2>&1 || MISSING+=("pip")
if [ "$INSTALL_FRONTEND" = "1" ]; then
  command -v node >/dev/null 2>&1 || MISSING+=("node (>= 18)")
  if ! command -v pnpm >/dev/null 2>&1; then
    warn "pnpm not found — will try to install via 'npm install -g pnpm' if npm is available."
    command -v npm >/dev/null 2>&1 || MISSING+=("npm or pnpm")
  fi
fi

if [ ${#MISSING[@]} -ne 0 ]; then
  err "Missing prerequisites:"
  for m in "${MISSING[@]}"; do echo "    - $m" >&2; done
  err "Install them, then re-run ./setup.sh"
  err "  macOS:   brew install python@3.11 node pnpm"
  err "  Ubuntu:  sudo apt install python3.11 python3-pip nodejs && sudo npm i -g pnpm"
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
log "python3 = $PY_VER, pip available."

# Optional: Ollama (recommended for benchmark + local-LLM Backend flows)
if command -v ollama >/dev/null 2>&1; then
  log "ollama detected — local LLMs available."
  HAVE_OLLAMA=1
else
  warn "ollama not detected. Install from https://ollama.com if you want local-LLM screening."
  HAVE_OLLAMA=0
fi

# ─── backend ──────────────────────────────────────────────────────────────
if [ "$INSTALL_BACKEND" = "1" ]; then
  log "Installing Backend (FastAPI) dependencies…"
  (
    cd "$REPO_ROOT/Backend"
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    deactivate
  )
  if [ ! -f Backend/.env ]; then
    log "Creating Backend/.env from Backend/.env.example"
    cp Backend/.env.example Backend/.env
    warn "Edit Backend/.env to add your ANTHROPIC_API_KEY / OPENAI_API_KEY / etc."
  else
    log "Backend/.env already exists — leaving as-is."
  fi
fi

# ─── frontend ─────────────────────────────────────────────────────────────
if [ "$INSTALL_FRONTEND" = "1" ]; then
  log "Installing frontend (pnpm) dependencies…"
  if ! command -v pnpm >/dev/null 2>&1; then
    log "Installing pnpm globally via npm…"
    npm install -g pnpm
  fi
  pnpm install
fi

# ─── benchmark ────────────────────────────────────────────────────────────
if [ "$INSTALL_BENCHMARK" = "1" ]; then
  log "Installing benchmark dependencies (separate venv: benchmark/.venv)…"
  (
    cd "$REPO_ROOT/benchmark"
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    deactivate
  )
fi

# ─── optional: pull Ollama models ─────────────────────────────────────────
if [ "$PULL_OLLAMA" = "1" ] && [ "$HAVE_OLLAMA" = "1" ]; then
  log "Pulling recommended Ollama models (this can be a multi-GB download)…"
  for model in \
      "llama3.2:3b" \
      "qwen2.5:7b" \
      "hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest"; do
    log "  ollama pull $model"
    ollama pull "$model" || warn "Failed to pull $model — continuing."
  done
  warn "Heavier models (medgemma:27b, qwen3.5:27b) are optional — pull manually if needed."
elif [ "$PULL_OLLAMA" = "1" ] && [ "$HAVE_OLLAMA" = "0" ]; then
  warn "--pull-ollama requested but Ollama is not installed. Skipping."
fi

# ─── summary ─────────────────────────────────────────────────────────────
echo
log "Setup complete."
echo
echo -e "${BOLD}Next steps:${RESET}"
if [ "$INSTALL_BACKEND" = "1" ]; then
  echo -e "  ${DIM}# Backend (FastAPI, port 8000)${RESET}"
  echo    "  cd Backend && source .venv/bin/activate && bash run_api.sh"
  echo
fi
if [ "$INSTALL_FRONTEND" = "1" ]; then
  echo -e "  ${DIM}# Frontend (Vite, port 5173)${RESET}"
  echo    "  pnpm dev"
  echo
fi
if [ "$INSTALL_BENCHMARK" = "1" ]; then
  echo -e "  ${DIM}# Benchmark smoke test (~30s on local Ollama)${RESET}"
  echo    "  cd benchmark && source .venv/bin/activate && \\"
  echo    "    python run_benchmark.py --datasets sample \\"
  echo    "      --architectures single_combined --models small"
  echo
fi

if [ "$INSTALL_BACKEND" = "1" ] && [ ! -s Backend/.env ] || \
   ( [ -f Backend/.env ] && ! grep -qE '^[A-Z_]+_API_KEY=.+' Backend/.env 2>/dev/null ); then
  warn "Backend/.env exists but contains no populated API keys. Add at least one of:"
  warn "  ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY (or set DEFAULT_MODEL to an Ollama model)."
fi

if [ "$HAVE_OLLAMA" = "0" ] && [ "$INSTALL_BENCHMARK" = "1" ]; then
  warn "Ollama isn't installed — the benchmark's open-weight tiers (small/medium/leads/etc.) will be skipped."
  warn "  Install: https://ollama.com  →  then run:  ./setup.sh --pull-ollama"
fi
