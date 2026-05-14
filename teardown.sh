#!/usr/bin/env bash
# =============================================================================
# Evidence Engine — stop everything started by setup.sh
#
# Kills the backend, frontend, and (optionally) Ollama processes. Idempotent:
# safe to re-run when nothing is running.
#
# Flags:
#   --keep-ollama   Leave Ollama running (useful when other tools use it)
#   --hard          Force-kill via pkill if the PID file approach misses
#   -h | --help     Show this help
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

if [ -t 1 ]; then
  BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; DIM="\033[2m"; RESET="\033[0m"
else
  BOLD=""; GREEN=""; YELLOW=""; DIM=""; RESET=""
fi
log()  { printf "${BOLD}${GREEN}==>${RESET} %s\n" "$*"; }
warn() { printf "${BOLD}${YELLOW}!!${RESET}  %s\n" "$*"; }

KEEP_OLLAMA=0
HARD=0
for arg in "$@"; do
  case "$arg" in
    --keep-ollama) KEEP_OLLAMA=1 ;;
    --hard)        HARD=1 ;;
    -h|--help)
      sed -n '/^# Evidence Engine/,/^# ====/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) warn "Ignoring unknown flag: $arg" ;;
  esac
done

stop_pidfile() {
  local label="$1"; local pidfile="$2"
  if [ -f "$pidfile" ]; then
    local pid; pid="$(cat "$pidfile")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      log "Stopping $label (PID $pid)…"
      kill "$pid" 2>/dev/null || true
      # Give it a moment; SIGKILL only on retry
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    else
      log "$label PID file present but process not running."
    fi
    rm -f "$pidfile"
  else
    log "$label not started by this setup (no PID file)."
  fi
}

stop_pidfile "Backend (uvicorn)" "$REPO_ROOT/.runtime/backend.pid"
stop_pidfile "Frontend (Vite)"   "$REPO_ROOT/.runtime/frontend.pid"

# Hard fallback: also kill any orphaned processes by pattern.
if [ "$HARD" = 1 ]; then
  log "Hard-killing any leftover uvicorn / vite processes…"
  pkill -f "uvicorn api:app" 2>/dev/null || true
  pkill -f "vite"            2>/dev/null || true
fi

if [ "$KEEP_OLLAMA" = 0 ]; then
  # Only stop Ollama if we started it (PID file present) — otherwise leave alone,
  # since the user may have started it themselves via brew services.
  if [ -f "$REPO_ROOT/.runtime/ollama.pid" ]; then
    stop_pidfile "Ollama" "$REPO_ROOT/.runtime/ollama.pid"
  else
    log "Ollama wasn't started by this script — leaving it alone."
    log "  (use 'brew services stop ollama' on macOS if you want to stop it manually)"
  fi
fi

echo
log "Teardown complete."
echo -e "${DIM}Logs preserved at ${REPO_ROOT}/.runtime/*.log — delete manually if not needed.${RESET}"
