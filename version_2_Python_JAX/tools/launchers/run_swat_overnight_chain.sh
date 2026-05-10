#!/usr/bin/env bash
# SWAT Phase 3.5 chain — log-Gaussian wake-gated steps, Jacobi
# diffusion on Z & a, tightened init priors, particles back to
# 512/400 + 512/64.
#
# Sequence:
#   1. T=2 pathological smoke (~25 min) — flushes any wiring bugs.
#   2. T=7 pathological main test (~3 hours) — recovery from
#      collapsed state, headline scientific result for 3.5.
#
# After each run: auto-generate the param-trace plot.
# If any run fails, the chain stops (no point burning GPU on bad
# state).
#
# Launch via:
#   tmux new -s swat_overnight -d \
#     "$HOME/Repos/python-smc2-filtering-control/version_2/tools/launchers/run_swat_overnight_chain.sh"

set -u

LOGDIR="$HOME/bench_logs"
REPO="$HOME/Repos/python-smc2-filtering-control"
mkdir -p "$LOGDIR"

cd "$REPO/version_2" || { echo "REPO not found"; exit 1; }

export JAX_ENABLE_X64=True
export PYTHONPATH=.:..
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85    # raised from 0.60 — solo overnight, no compositor competition
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_compilation_cache"
export FSA_STEP_MINUTES=15

run_one() {
    local label=$1
    local T=$2
    local scenario=$3
    local stamp
    stamp=$(date +%Y%m%d_%H%M)
    local log="$LOGDIR/swat_${label}_${stamp}.log"
    echo
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] EXPERIMENT: $label (T=${T}d, scenario=${scenario})"
    echo "  log: $log"
    echo "============================================================"

    python -u tools/bench_smc_full_mpc_swat.py "$T" \
            --step-minutes 15 --scenario "$scenario" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] $label FAILED (exit $rc). Stopping chain." >&2
        return 1
    fi

    # Auto-generate param trace plot
    local out_dir="outputs/swat/swat_runs/swat_T${T}d_replanK2_h15min_${scenario}"
    if [ -f "$out_dir/manifest.json" ]; then
        echo "[$(date '+%H:%M:%S')] Generating param-trace plot for $label"
        python -m tools.plot_param_traces "$out_dir" 2>&1 | tail -3
    fi
    return 0
}

# ── 1. T=2 pathological smoke ──
run_one "t2_pathological" 2 pathological || exit 1

# ── 2. T=7 pathological main test ──
run_one "t7_pathological" 7 pathological || exit 2

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] ALL SWAT EXPERIMENTS COMPLETE"
echo "============================================================"
ls -la "$REPO/version_2/outputs/swat/swat_runs/" 2>/dev/null
