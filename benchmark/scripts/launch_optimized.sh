#!/usr/bin/env bash
# Optimized launch for M4 Max 64GB.
# Phase 1 = small + medium tiers (fast, runs all 7 archs × repeat=3)
# Phase 2 = specialized + large 27B tiers (slower, top 5 archs × repeat=1)
# Phase 1's two sub-tiers run concurrently in separate processes; phase 2's
# two 27B tiers also run concurrently. Each phase writes to its own run id.

set -euo pipefail
cd "$(dirname "$0")/.."

# Pick up Ollama parallelism env
source scripts/configure_ollama.sh

DATASET="${DATASET:-synergy/van_Dis_2020}"
PHASE="${1:-phase1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}-${PHASE}"

case "$PHASE" in
  phase1)
    # Run small + medium tiers in parallel processes — different models, so Ollama
    # can serve them concurrently from its loaded set.
    python3 run_benchmark.py \
        --datasets "$DATASET" --architectures all \
        --models small --repeat 3 --workers 4 --field-stratify \
        --run-id "${RUN_ID}-small" \
        > "reports/${RUN_ID}-small.log" 2>&1 &
    SMALL_PID=$!

    python3 run_benchmark.py \
        --datasets "$DATASET" --architectures all \
        --models medium --repeat 3 --workers 4 --field-stratify \
        --run-id "${RUN_ID}-medium" \
        > "reports/${RUN_ID}-medium.log" 2>&1 &
    MEDIUM_PID=$!

    echo "Phase 1 launched (PIDs: small=$SMALL_PID, medium=$MEDIUM_PID)"
    echo "Tail logs: tail -f reports/${RUN_ID}-{small,medium}.log"
    wait $SMALL_PID $MEDIUM_PID
    ;;

  phase2)
    # Heavy 27B tiers — top 5 archs only, repeat=1
    ARCHS=("single_combined" "cascade_triage" "decompose_match" "self_consistency" "multi_agent")
    python3 run_benchmark.py \
        --datasets "$DATASET" --architectures "${ARCHS[@]}" \
        --models specialized --repeat 1 --workers 2 --field-stratify \
        --run-id "${RUN_ID}-specialized" \
        > "reports/${RUN_ID}-specialized.log" 2>&1 &
    SPEC_PID=$!

    python3 run_benchmark.py \
        --datasets "$DATASET" --architectures "${ARCHS[@]}" \
        --models large --repeat 1 --workers 2 --field-stratify \
        --run-id "${RUN_ID}-large" \
        > "reports/${RUN_ID}-large.log" 2>&1 &
    LARGE_PID=$!

    echo "Phase 2 launched (PIDs: specialized=$SPEC_PID, large=$LARGE_PID)"
    echo "Tail logs: tail -f reports/${RUN_ID}-{specialized,large}.log"
    wait $SPEC_PID $LARGE_PID
    ;;

  *)
    echo "Usage: $0 [phase1|phase2]" >&2
    exit 2 ;;
esac

echo "$PHASE complete."
