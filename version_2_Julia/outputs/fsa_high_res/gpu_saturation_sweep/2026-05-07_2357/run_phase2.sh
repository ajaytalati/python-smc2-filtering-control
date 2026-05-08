#!/usr/bin/env bash
# Phase 2: closed-loop accuracy at the saturated cell from Phase 1.
#
# Three runs at --step-minutes 60 --replan-K 2 --T-days 14 --seed 42:
#   A: closed-loop @ saturated (N*, K*)   ← headline test
#   B: open-loop   @ saturated (N*, K*)   ← upper bound (controller alone)
#   C: closed-loop @ (N=32, K=400)        ← lower bound (current default)
#
# After each run, extract_metrics.jl reads data.jld2 and appends to
# PHASE2_METRICS.csv + writes per-run cum_A and phi CSVs.
# After all three, plot_area_comparison.jl builds the overlay PNG.
#
# Usage:
#   ./run_phase2.sh <N_star> <K_star>

set -u

if [ $# -ne 2 ]; then
    echo "Usage: $0 <N_star> <K_star>"
    exit 1
fi

N_STAR=$1
K_STAR=$2

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SWEEP_DIR/../../../../.." && pwd)"
V2_DIR="$REPO_DIR/version_2_Julia"

PHASE2_DIR="$SWEEP_DIR/closed_loop_at_saturation"
mkdir -p "$PHASE2_DIR"

cd "$V2_DIR" || exit 1

# Reset any stale metrics
rm -f "$PHASE2_DIR/PHASE2_METRICS.csv"

run_bench () {
    local LABEL=$1
    local N=$2
    local K=$3
    local OPEN_LOOP=$4
    local OUT="$PHASE2_DIR/${LABEL}"
    mkdir -p "$OUT"
    echo "=========================================================================="
    echo "[$(date '+%H:%M:%S')] Phase 2 — ${LABEL}: N=${N}, K=${K}, open-loop=${OPEN_LOOP}"
    echo "=========================================================================="

    local BENCH_CMD="JULIA_NUM_THREADS=4 julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl --step-minutes 60 --replan-K 2 --T-days 14 --N-smc $N --K-per-chain $K --seed 42 --open-loop $OPEN_LOOP --output-dir $OUT"
    script -qfc "$BENCH_CMD" "$OUT/bench.log"

    local DATA="$OUT/data.jld2"
    if [ ! -f "$DATA" ]; then
        echo "ERROR: $DATA not produced by ${LABEL} run"
        return 1
    fi

    julia --project=. "$SWEEP_DIR/extract_metrics.jl" "$LABEL" "$DATA" "$PHASE2_DIR"
}

# A: closed-loop @ saturated
run_bench "run_A" "$N_STAR" "$K_STAR" "false"
# B: open-loop @ saturated
run_bench "run_B" "$N_STAR" "$K_STAR" "true"
# C: closed-loop @ default
run_bench "run_C" 32 400 "false"

echo "=========================================================================="
echo "Phase 2 metrics:"
echo "=========================================================================="
cat "$PHASE2_DIR/PHASE2_METRICS.csv"

echo "=========================================================================="
echo "Building comparison plot..."
echo "=========================================================================="
julia --project=. "$SWEEP_DIR/plot_area_comparison.jl" "$PHASE2_DIR"

echo "Phase 2 done. Outputs in: $PHASE2_DIR"
