#!/usr/bin/env bash
# =============================================================================
# Evidence Engine — one-shot, non-interactive installer + launcher
#
# What it does (without prompting):
#   1. Installs system deps (Homebrew on macOS, then python@3.12, node, pnpm, ollama)
#   2. Sets up the Backend Python venv + installs requirements
#   3. Installs the frontend pnpm packages
#   4. Configures Backend/.env from .env.example (auto-fills ENTREZ_EMAIL from git)
#   5. Starts Ollama and pulls the LEADS-Mistral 7B + Qwen 2.5 7B models
#   6. Starts the FastAPI backend on :8000 and the Vite frontend on :5173
#   7. Health-checks both, prints the URL
#
# Re-running is safe — every step is idempotent. To stop everything: ./teardown.sh
#
# Flags:
#   --no-pull-models   Skip downloading the two Ollama models (~9 GB total)
#   --no-start         Install everything but don't launch the services
#   --backend-only     Skip frontend
#   --frontend-only    Skip backend
#   -h | --help        Show this help
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ─── color logging ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; DIM="\033[2m"; RESET="\033[0m"
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; DIM=""; RESET=""
fi
log()  { printf "${BOLD}${GREEN}==>${RESET} %s\n" "$*"; }
warn() { printf "${BOLD}${YELLOW}!!${RESET}  %s\n" "$*"; }
err()  { printf "${BOLD}${RED}xx${RESET}  %s\n" "$*" >&2; }
die()  { err "$@"; exit 1; }

# ─── args ───────────────────────────────────────────────────────────────────
SKIP_MODELS=0
START_SERVICES=1
INSTALL_BACKEND=1
INSTALL_FRONTEND=1
for arg in "$@"; do
  case "$arg" in
    --no-pull-models)   SKIP_MODELS=1 ;;
    --no-start)         START_SERVICES=0 ;;
    --backend-only)     INSTALL_FRONTEND=0 ;;
    --frontend-only)    INSTALL_BACKEND=0 ;;
    -h|--help)          sed -n '/^# What it does/,/^# ====/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown argument: $arg  (try --help)" ;;
  esac
done

# ─── OS detect ──────────────────────────────────────────────────────────────
OS=""
case "$(uname -s)" in
  Darwin*) OS=mac ;;
  Linux*)  OS=linux ;;
  *) die "Unsupported OS: $(uname -s). Only macOS and Linux are auto-supported." ;;
esac
log "Detected OS: $OS"

# ─── 1. system dependencies ─────────────────────────────────────────────────
install_mac_deps() {
  if ! command -v brew >/dev/null 2>&1; then
    log "Installing Homebrew (non-interactive)…"
    NONINTERACTIVE=1 /bin/bash -c \
      "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Newly-installed brew isn't on PATH for this shell yet
    if [ -x /opt/homebrew/bin/brew ]; then eval "$(/opt/homebrew/bin/brew shellenv)"; fi
    if [ -x /usr/local/bin/brew ];   then eval "$(/usr/local/bin/brew shellenv)"; fi
  fi

  local pkgs=()
  command -v python3 >/dev/null 2>&1 || pkgs+=("python@3.12")
  if [ "$INSTALL_FRONTEND" = 1 ]; then
    command -v node >/dev/null 2>&1 || pkgs+=("node")
    command -v pnpm >/dev/null 2>&1 || pkgs+=("pnpm")
  fi
  command -v ollama >/dev/null 2>&1 || pkgs+=("ollama")

  if [ "${#pkgs[@]}" -gt 0 ]; then
    log "Installing via Homebrew: ${pkgs[*]}"
    brew install "${pkgs[@]}"
  else
    log "All system dependencies already installed."
  fi
}

install_linux_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing system packages via apt…"
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv curl ca-certificates
    if [ "$INSTALL_FRONTEND" = 1 ] && ! command -v node >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
      sudo apt-get install -y -qq nodejs
    fi
    if [ "$INSTALL_FRONTEND" = 1 ] && ! command -v pnpm >/dev/null 2>&1; then
      sudo npm install -g pnpm
    fi
    if ! command -v ollama >/dev/null 2>&1; then
      log "Installing Ollama (will run with sudo)…"
      curl -fsSL https://ollama.com/install.sh | sh
    fi
  else
    die "Linux: only apt is auto-supported. Install python3, node, pnpm, ollama manually, then re-run."
  fi
}

if [ "$OS" = mac ]; then
  install_mac_deps
else
  install_linux_deps
fi

# Verify the essentials
command -v python3 >/dev/null 2>&1 || die "python3 still missing after install attempt."
command -v ollama >/dev/null 2>&1 || die "ollama still missing after install attempt."
if [ "$INSTALL_FRONTEND" = 1 ]; then
  command -v pnpm >/dev/null 2>&1 || die "pnpm still missing after install attempt."
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
log "python3 = $PY_VER"

# ─── 2. backend (Python venv + deps) ────────────────────────────────────────
if [ "$INSTALL_BACKEND" = 1 ]; then
  log "Setting up Backend Python venv…"
  (
    cd "$REPO_ROOT/Backend"
    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    deactivate
  )
  log "Backend dependencies installed."

  # Backend/.env
  if [ ! -f "$REPO_ROOT/Backend/.env" ]; then
    log "Creating Backend/.env from .env.example"
    cp "$REPO_ROOT/Backend/.env.example" "$REPO_ROOT/Backend/.env"
    # Auto-fill ENTREZ_EMAIL from git config if available — NCBI requires a real address.
    GIT_EMAIL="$(git config --get user.email 2>/dev/null || true)"
    if [ -n "${GIT_EMAIL:-}" ]; then
      # Portable sed: macOS BSD sed needs -i ''; GNU sed accepts -i.
      if sed --version >/dev/null 2>&1; then
        sed -i "s|^ENTREZ_EMAIL=.*|ENTREZ_EMAIL=$GIT_EMAIL|" "$REPO_ROOT/Backend/.env"
      else
        sed -i '' "s|^ENTREZ_EMAIL=.*|ENTREZ_EMAIL=$GIT_EMAIL|" "$REPO_ROOT/Backend/.env"
      fi
      log "Set ENTREZ_EMAIL=$GIT_EMAIL  (from git config)"
    else
      warn "git user.email not set — leaving ENTREZ_EMAIL placeholder in Backend/.env."
      warn "NCBI/PubMed requires a real address; edit Backend/.env if you want PubMed retrieval to work cleanly."
    fi
  else
    log "Backend/.env already exists — leaving as-is."
  fi
fi

# ─── 3. frontend (pnpm) ─────────────────────────────────────────────────────
if [ "$INSTALL_FRONTEND" = 1 ]; then
  log "Installing frontend dependencies (pnpm)…"
  pnpm install --silent
fi

# ─── 4. Ollama service ──────────────────────────────────────────────────────
ensure_ollama_running() {
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Ollama already running."
    return
  fi
  log "Starting Ollama…"
  if [ "$OS" = mac ] && command -v brew >/dev/null 2>&1; then
    brew services start ollama >/dev/null 2>&1 || true
  fi
  # If brew services didn't take, fall back to a background process.
  sleep 1
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    mkdir -p "$REPO_ROOT/.runtime"
    nohup ollama serve >"$REPO_ROOT/.runtime/ollama.log" 2>&1 &
    echo $! > "$REPO_ROOT/.runtime/ollama.pid"
  fi
  # Wait up to 20 s for the API to respond.
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
      log "Ollama is up on :11434."
      return
    fi
    sleep 1
  done
  warn "Ollama did not come up within 20 s. Check ${REPO_ROOT}/.runtime/ollama.log"
}

ensure_ollama_running

# ─── 5. pull models ─────────────────────────────────────────────────────────
LEADS_TAG="hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest"
THINKING_TAG="qwen2.5:7b"

have_model() {
  ollama list 2>/dev/null | awk '{print $1}' | grep -Fxq "$1"
}

if [ "$SKIP_MODELS" = 0 ]; then
  if have_model "$LEADS_TAG"; then
    log "LEADS-Mistral 7B already pulled."
  else
    log "Pulling LEADS-Mistral 7B (~4 GB, this is the screening model)…"
    ollama pull "$LEADS_TAG"
  fi

  if have_model "$THINKING_TAG"; then
    log "Qwen 2.5 7B already pulled."
  else
    log "Pulling Qwen 2.5 7B (~5 GB, used for non-screening tasks)…"
    ollama pull "$THINKING_TAG"
  fi
else
  warn "--no-pull-models set: skipping Ollama model downloads."
  warn "  You'll need to pull them manually:"
  warn "    ollama pull $LEADS_TAG"
  warn "    ollama pull $THINKING_TAG"
fi

# ─── 6. start backend + frontend ────────────────────────────────────────────
mkdir -p "$REPO_ROOT/.runtime"

wait_for_url() {
  local url="$1"; local label="$2"; local timeout="${3:-30}"
  for _ in $(seq 1 "$timeout"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      log "$label is ready."
      return 0
    fi
    sleep 1
  done
  warn "$label did not come up within ${timeout}s."
  return 1
}

port_in_use() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

if [ "$START_SERVICES" = 1 ]; then
  if [ "$INSTALL_BACKEND" = 1 ]; then
    if port_in_use 8000; then
      log "Port 8000 already in use — assuming backend is already running."
    else
      log "Starting backend (FastAPI) on :8000…"
      (
        cd "$REPO_ROOT/Backend"
        # shellcheck source=/dev/null
        source .venv/bin/activate
        nohup uvicorn api:app --host 0.0.0.0 --port 8000 \
          >"$REPO_ROOT/.runtime/backend.log" 2>&1 &
        echo $! > "$REPO_ROOT/.runtime/backend.pid"
      )
      wait_for_url "http://localhost:8000/api/health" "Backend" 30 || \
        warn "Backend health check failed — see .runtime/backend.log"
    fi
  fi

  if [ "$INSTALL_FRONTEND" = 1 ]; then
    if port_in_use 5173; then
      log "Port 5173 already in use — assuming frontend is already running."
    else
      log "Starting frontend (Vite) on :5173…"
      nohup pnpm dev --host >"$REPO_ROOT/.runtime/frontend.log" 2>&1 &
      echo $! > "$REPO_ROOT/.runtime/frontend.pid"
      wait_for_url "http://localhost:5173" "Frontend" 30 || \
        warn "Frontend did not respond — see .runtime/frontend.log"
    fi
  fi
fi

# ─── 7. summary ─────────────────────────────────────────────────────────────
echo
log "Setup complete."
echo
echo -e "${BOLD}URLs${RESET}"
echo "  Frontend:    http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/api/health"
echo
echo -e "${BOLD}Models pulled to Ollama${RESET}"
if ! have_model "$LEADS_TAG" 2>/dev/null; then
  warn "  LEADS-Mistral 7B  — NOT pulled (rerun without --no-pull-models, or: ollama pull $LEADS_TAG)"
else
  echo "  LEADS-Mistral 7B  (screening — paper's default)"
fi
if ! have_model "$THINKING_TAG" 2>/dev/null; then
  warn "  Qwen 2.5 7B       — NOT pulled (rerun without --no-pull-models, or: ollama pull $THINKING_TAG)"
else
  echo "  Qwen 2.5 7B       (PICO, query gen, summary, meta-analysis extraction)"
fi
echo
if [ "$START_SERVICES" = 1 ]; then
  echo -e "${BOLD}Service logs${RESET}"
  [ -f "$REPO_ROOT/.runtime/backend.log" ]   && echo "  Backend:   $REPO_ROOT/.runtime/backend.log"
  [ -f "$REPO_ROOT/.runtime/frontend.log" ]  && echo "  Frontend:  $REPO_ROOT/.runtime/frontend.log"
  [ -f "$REPO_ROOT/.runtime/ollama.log" ]    && echo "  Ollama:    $REPO_ROOT/.runtime/ollama.log"
  echo
  echo -e "${BOLD}To stop everything${RESET}"
  echo "  ./teardown.sh"
fi
echo
echo -e "${DIM}Tip: To use a frontier model instead of the local Ollama defaults,${RESET}"
echo -e "${DIM}     add ANTHROPIC_API_KEY=… (or OPENAI_API_KEY=… or GEMINI_API_KEY=…) to Backend/.env${RESET}"
echo -e "${DIM}     and pick the model in the sidebar after the frontend loads.${RESET}"
