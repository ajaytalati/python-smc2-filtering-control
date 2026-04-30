#!/usr/bin/env bash
# Generic single-horizon launcher.
#
# Usage:  ./run_horizon.sh <T_days>
#
# Example:
#   tmux new -s t56 -d "$HOME/bench_logs/run_horizon.sh 56"
#   tmux new -s t84 -d "$HOME/bench_logs/run_horizon.sh 84"
#
# Designed to be safe to run multiple instances in parallel (different
# T) — each process grabs GPU memory lazily and writes to its own
# output dir under outputs/swat/g4_runs/. JAX_COMPILATION_CACHE
# is shared across processes so cold-compile is paid once per binary
# shape.

set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <T_days>" >&2
    exit 1
fi
T=$1

LOGDIR="$HOME/bench_logs"
REPO="$HOME/Repos/python-smc2-filtering-control"
mkdir -p "$LOGDIR"

cd "$REPO/version_2" || { echo "REPO not found"; exit 1; }

export JAX_ENABLE_X64=True
export PYTHONPATH=.:..
# JAX safety caps — when running 3 horizons in parallel we want
# generous lazy headroom but no preallocation. See
# GPU_TUNING_RTX5090.md §A.2.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Shared compile cache across parallel processes.
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_compilation_cache"

STEP_MIN=${STEP_MINUTES:-15}     # SWAT default: 15-min bins for sleep/wake identifiability
stamp=$(date +%Y%m%d_%H%M)
log="$LOGDIR/swat_t${T}_h${STEP_MIN}min_${stamp}.log"

echo "============================================================"
echo "  SWAT T=${T} h=${STEP_MIN}min launching at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  log: $log"
echo "  GPU mem cap: $XLA_PYTHON_CLIENT_MEM_FRACTION × total (per process)"
echo "============================================================"

python -u tools/bench_smc_full_mpc_swat.py "$T" --step-minutes "$STEP_MIN" \
    2>&1 | tee "$log"
rc=${PIPESTATUS[0]}

echo
echo "============================================================"
echo "  SWAT T=${T} finished at $(date '+%Y-%m-%d %H:%M:%S')  exit code: $rc"
echo "============================================================"
ls -la "$REPO/version_2/outputs/swat/swat_runs/swat_T${T}d_replanK2_h${STEP_MIN}min/" 2>/dev/null
exit "$rc"
