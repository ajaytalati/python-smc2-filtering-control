#!/usr/bin/env bash
# RTX 5090 saturation sweep — N x K grid at --step-minutes 60 --replan-K 2.
#
# For each (N, K) cell:
#   1. Start nvidia-smi poller in background, sampling every 2s into gpu_log.csv.
#   2. Start the bench in background, redirecting stderr to bench.log.
#   3. Watch bench.log for "[stride NN/NN]" lines; kill after 5 strides OR 8min.
#   4. Parse per-stride wall (median, ex-stride-1), median util %, peak mem MB.
#   5. Append one row to RESULTS.csv.
#
# Sequential — KernelAbstractions caches kernels per Julia process, parallel
# cells would each pay JIT and contend for GPU mem. Small cells first.

set -u

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SWEEP_DIR/../../../../.." && pwd)"     # → repo root
V2_DIR="$REPO_DIR/version_2_Julia"

cd "$V2_DIR" || exit 1

RESULTS_CSV="$SWEEP_DIR/RESULTS.csv"
echo "N,K,M_max,strides,util_median_pct,util_max_pct,peak_mem_mb,per_stride_wall_s,wall_total_s,status" > "$RESULTS_CSV"

# 8-minute hard timeout per cell.
TIMEOUT_S=480
# Stride trigger.
STRIDES_NEEDED=5

# nvidia-smi poll interval (seconds). 2s is fine.
POLL_S=2

run_cell() {
    local N=$1
    local K=$2
    local CELL_DIR="$SWEEP_DIR/N${N}_K${K}"
    mkdir -p "$CELL_DIR"

    echo "=========================================================================="
    echo "[$(date '+%H:%M:%S')] CELL: N=$N K=$K"
    echo "=========================================================================="

    local BENCH_LOG="$CELL_DIR/bench.log"
    local GPU_LOG="$CELL_DIR/gpu_log.csv"
    local BENCH_OUT="$CELL_DIR/bench_output"
    mkdir -p "$BENCH_OUT"

    : > "$BENCH_LOG"
    echo "wall_s,util_pct,mem_mb" > "$GPU_LOG"

    local M_MAX=$(( N * 61 ))     # = N * (1 + 2*30) for d=30 filter

    # ── Start nvidia-smi poller ──
    local START_TS=$(date +%s.%N)
    (
        while true; do
            local OUT
            OUT=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [ -n "$OUT" ]; then
                local UTIL=$(echo "$OUT" | awk -F',' '{gsub(/ /,"",$1); print $1}')
                local MEM=$(echo "$OUT" | awk -F',' '{gsub(/ /,"",$2); print $2}')
                local NOW
                NOW=$(date +%s.%N)
                local ELAPSED
                ELAPSED=$(awk -v a="$NOW" -v b="$START_TS" 'BEGIN { printf "%.2f", a - b }')
                echo "${ELAPSED},${UTIL},${MEM}" >> "$GPU_LOG"
            fi
            sleep $POLL_S
        done
    ) &
    local POLL_PID=$!

    # ── Start bench ──
    # Use `script -qc` to give Julia a pseudo-tty so its @info logger flushes
    # per message. Otherwise `2>&1` to a regular file block-buffers and we
    # never see [stride NN/NN] lines in real time.
    local BENCH_CMD="JULIA_NUM_THREADS=4 nice -n 5 julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl --step-minutes 60 --replan-K 2 --T-days 14 --N-smc $N --K-per-chain $K --seed 42 --output-dir $BENCH_OUT"
    script -qfc "$BENCH_CMD" "$BENCH_LOG" > /dev/null 2>&1 &
    local BENCH_PID=$!

    echo "  bench PID=$BENCH_PID  poller PID=$POLL_PID  M_max=$M_MAX"

    # ── Watch bench log ──
    local STATUS=""
    local DEADLINE=$(( $(date +%s) + TIMEOUT_S ))
    while true; do
        if ! kill -0 "$BENCH_PID" 2>/dev/null; then
            STATUS="bench_exited"
            break
        fi
        local STRIDE_COUNT
        STRIDE_COUNT=$(awk '/\[stride /{c++} END{print c+0}' "$BENCH_LOG" 2>/dev/null)
        [ -z "$STRIDE_COUNT" ] && STRIDE_COUNT=0
        if [ "$STRIDE_COUNT" -ge "$STRIDES_NEEDED" ]; then
            STATUS="strides_ok"
            break
        fi
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            STATUS="timeout"
            break
        fi
        sleep 2
    done

    # ── Kill ──
    kill "$BENCH_PID" 2>/dev/null
    sleep 0.5
    kill -9 "$BENCH_PID" 2>/dev/null
    kill "$POLL_PID" 2>/dev/null
    sleep 0.5
    kill -9 "$POLL_PID" 2>/dev/null
    wait 2>/dev/null

    local WALL_TOTAL
    WALL_TOTAL=$(awk -v end="$(date +%s.%N)" -v start="$START_TS" 'BEGIN { printf "%.1f", end - start }')

    # ── Parse stride wall times ──
    local STRIDES_DONE
    STRIDES_DONE=$(awk '/\[stride /{c++} END{print c+0}' "$BENCH_LOG" 2>/dev/null)
    [ -z "$STRIDES_DONE" ] && STRIDES_DONE=0
    local PER_STRIDE_WALL="-"
    if [ "$STRIDES_DONE" -ge 2 ]; then
        # Drop stride 1 (JIT + bigger window). Median of strides 2+.
        PER_STRIDE_WALL=$(grep '\[stride ' "$BENCH_LOG" \
            | tail -n +2 \
            | grep -oP '\[stride\s+\d+/\d+\]\s+\K[\d.]+' \
            | sort -n \
            | awk '{a[NR]=$1} END { if (NR==0) print "-"; else if (NR%2==1) print a[(NR+1)/2]; else printf "%.2f", (a[NR/2] + a[NR/2+1])/2 }')
    fi

    # ── Parse GPU log: drop first 60s (JIT), report median + max util, peak mem ──
    local UTIL_MED="-"
    local UTIL_MAX="-"
    local PEAK_MEM="-"
    if [ -s "$GPU_LOG" ]; then
        # median util excluding first 60s
        UTIL_MED=$(awk -F',' 'NR>1 && $1>=60 { print $2 }' "$GPU_LOG" \
            | sort -n \
            | awk '{a[NR]=$1} END { if (NR==0) print "-"; else if (NR%2==1) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2] + a[NR/2+1])/2 }')
        UTIL_MAX=$(awk -F',' 'NR>1 { print $2 }' "$GPU_LOG" | sort -n | tail -1)
        PEAK_MEM=$(awk -F',' 'NR>1 { print $3 }' "$GPU_LOG" | sort -n | tail -1)
    fi

    echo "${N},${K},${M_MAX},${STRIDES_DONE},${UTIL_MED},${UTIL_MAX},${PEAK_MEM},${PER_STRIDE_WALL},${WALL_TOTAL},${STATUS}" >> "$RESULTS_CSV"

    echo "  result: strides=${STRIDES_DONE} util_med=${UTIL_MED}% util_max=${UTIL_MAX}% peak_mem=${PEAK_MEM}MB per_stride=${PER_STRIDE_WALL}s wall=${WALL_TOTAL}s status=${STATUS}"
    echo
}

# ── Sweep ──
for N in 32 64 128 256; do
    for K in 400 800 1600; do
        run_cell "$N" "$K"
    done
done

echo "=========================================================================="
echo "Sweep complete. Results: $RESULTS_CSV"
echo "=========================================================================="
cat "$RESULTS_CSV"
