#!/usr/bin/env bash
# Configure Ollama for high-throughput on M-series Macs with plenty of RAM.
# Idempotent: re-running it is safe.
set -euo pipefail

# Hardware-aware defaults for M4 Max + 64GB unified memory.
#   - 4 concurrent requests per loaded model
#   - up to 4 models hot in memory simultaneously (small 2GB + medium 5GB + two 27B ~= 41GB)
#   - Metal flash attention + quantized KV cache for context efficiency
#   - keep models loaded for the duration of long benchmark sweeps
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-4}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"

echo "Ollama env configured:"
env | grep -E '^OLLAMA_' | sort

# If Ollama was started by brew services, it won't pick up new env vars without a bounce.
if command -v brew >/dev/null && brew services list 2>/dev/null | grep -q '^ollama'; then
  echo "Restarting Ollama service so env vars take effect…"
  brew services restart ollama
  sleep 2
fi

# Probe
curl -sf http://localhost:11434/api/tags >/dev/null \
  && echo "Ollama is reachable ✓" \
  || echo "Warning: Ollama not reachable on :11434"
