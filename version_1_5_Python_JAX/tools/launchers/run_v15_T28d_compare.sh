#!/usr/bin/env bash
# Phase C of the v1.5 closed-loop comparison plan
# (`claude_plans/FSA_v1_5_three_way_closed_loop_comparison_*.md`).
#
# Three-way comparison: Julia MPC vs Python+JAX MPC vs constant-Φ baseline.
# (Baseline data is produced inside both benches automatically.)
#
# Flow:
#   1. Microbenchmark profilers first (~30s each):
#        - Julia profile_gpu.jl
#        - Python+JAX profile_gpu.py
#      Both write their stdout to {julia,python}_profile.log under the
#      shared parent dir.
#   2. Closed-loop bench, Julia side, with `nvidia-smi` telemetry going
#      to gpu_telemetry.csv in the Julia run dir.
#   3. Closed-loop bench, Python side, same telemetry pattern.
#
# All artefacts land under
#   outputs/v15_julia_vs_python_T28d_seed${SEED}/{julia,python}/
# and the two profile logs under the shared parent.
#
# Usage:
#   bash tools/launchers/run_v15_T28d_compare.sh [T_DAYS] [SEED]
# Defaults: T_DAYS=28, SEED=42.
#
# Configurations are matched on every controller knob (Python now
# exposes the 7 knobs Julia already had; both default to the same
# values).

set -euo pipefail

# ── Args + paths ───────────────────────────────────────────────────────
T_DAYS="${1:-28}"
SEED="${2:-42}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JULIA_DIR="$REPO_ROOT/version_1_5_Julia"
PYJAX_DIR="$REPO_ROOT/version_1_5_Python_JAX"

PARENT_DIR="$REPO_ROOT/outputs/v15_julia_vs_python_T${T_DAYS}d_seed${SEED}"
JULIA_OUT="$PARENT_DIR/julia"
PYJAX_OUT="$PARENT_DIR/python"

mkdir -p "$JULIA_OUT" "$PYJAX_OUT"

echo "================================================================="
echo "  v1.5 closed-loop comparison: Julia vs Python+JAX vs baseline"
echo "  T_days=$T_DAYS, seed=$SEED"
echo "  parent dir: $PARENT_DIR"
echo "================================================================="

# ── Conda activate (lazy — sourced only if available) ──────────────────
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate comfyenv
fi

# ── Helper: start nvidia-smi in background, sample every 1 s ──────────
start_nvsmi() {
    local out_csv="$1"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
               --format=csv -l 1 > "$out_csv" 2>&1 &
    echo $!
}

stop_nvsmi() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# ── 1. Microbenchmark profilers ──
echo ""
echo "─── Phase 1: microbenchmark profilers ───"

echo ""
echo "  [1a] Julia profile_gpu.jl"
( cd "$JULIA_DIR" && julia --project=. tools/profile_gpu.jl --both ) \
    > "$PARENT_DIR/julia_profile.log" 2>&1 || \
    echo "  WARNING: Julia profiler exited non-zero (see julia_profile.log)"
echo "    → $PARENT_DIR/julia_profile.log"

echo ""
echo "  [1b] Python+JAX profile_gpu.py"
( cd "$PYJAX_DIR" && \
  JAX_ENABLE_X64=True PYTHONPATH=.:.. \
  python tools/profile_gpu.py --both ) \
    > "$PARENT_DIR/python_profile.log" 2>&1 || \
    echo "  WARNING: Python profiler exited non-zero (see python_profile.log)"
echo "    → $PARENT_DIR/python_profile.log"

# ── 2. Julia closed-loop bench with GPU telemetry ──
echo ""
echo "─── Phase 2: Julia closed-loop bench (T=${T_DAYS}d) ───"
JULIA_GPU_CSV="$JULIA_OUT/gpu_telemetry.csv"
NV_PID=$(start_nvsmi "$JULIA_GPU_CSV")
trap "stop_nvsmi $NV_PID" EXIT

# `--output-dir` lets the bench write into our shared parent
( cd "$JULIA_DIR" && \
  julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
        --T-days "$T_DAYS" \
        --seed "$SEED" \
        --output-dir "$JULIA_OUT" ) \
    2>&1 | tee "$JULIA_OUT/bench.log"

stop_nvsmi "$NV_PID"
trap - EXIT
echo "  → Julia outputs: $JULIA_OUT/"
echo "  → GPU telemetry: $JULIA_GPU_CSV"

# ── 3. Python+JAX closed-loop bench with GPU telemetry ──
echo ""
echo "─── Phase 3: Python+JAX closed-loop bench (T=${T_DAYS}d) ───"
PY_GPU_CSV="$PYJAX_OUT/gpu_telemetry.csv"
NV_PID=$(start_nvsmi "$PY_GPU_CSV")
trap "stop_nvsmi $NV_PID" EXIT

( cd "$PYJAX_DIR" && \
  JAX_ENABLE_X64=True PYTHONPATH=.:.. \
  python tools/bench_smc_full_mpc_fsa_v15.py \
        --T-days "$T_DAYS" \
        --seed "$SEED" \
        --out-dir "$PYJAX_OUT" ) \
    2>&1 | tee "$PYJAX_OUT/bench.log"

stop_nvsmi "$NV_PID"
trap - EXIT
echo "  → Python outputs: $PYJAX_OUT/"
echo "  → GPU telemetry: $PY_GPU_CSV"

# ── 4. Done ──
echo ""
echo "================================================================="
echo "  All runs complete."
echo "  Parent dir: $PARENT_DIR"
echo ""
echo "  Next: run the comparison script (Phase D)"
echo "    cd $PYJAX_DIR && JAX_ENABLE_X64=True PYTHONPATH=.:.. \\"
echo "      python tools/compare_v15_julia_vs_python.py $PARENT_DIR"
echo "================================================================="
