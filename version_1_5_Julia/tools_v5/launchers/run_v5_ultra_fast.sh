#!/usr/bin/env bash
# Julia v5 SMC²-MPC — Ultra-Fast Qualitative Sweep Launcher.
#
# Slashes runtime from 7.7 hours to ~50 minutes by reducing particle counts,
# while maintaining basin-discovery quality for T=42d horizons.
#
# Follows the central timestamped output convention.
# Writes to: outputs/bench_runs/YYYY-MM-DD_HHMMSS_T<N>d_UltraFast/

#
# Usage:
#   ./run_v5_ultra_fast.sh 42              # T=42 days (positional)
#   ./run_v5_ultra_fast.sh --seed 7        # Uses default T=42
#   ./run_v5_ultra_fast.sh 14 --seed 7     # T=14 days + flags
#   ./run_v5_ultra_fast.sh 100 --init-phi-B 1.0 --init-phi-S 1.0    # T=100 days + overtraining baseline - controller trajectory should ramp down from phi=1 to be in basins of attraction for the long horizon

set -u

# Robust T_DAYS detection: Use first arg if it's a number, otherwise default to 42.
T_DAYS="42"
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
    T_DAYS="$1"
    shift
fi

REPO="/home/ajay/Repos/python-smc2-filtering-control"
JULIA_DIR="$REPO/version_1_5_Julia"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

# Central Output Root
OUTPUT_BASE="$REPO/outputs/bench_runs"
RUN_DIR="$OUTPUT_BASE/${TIMESTAMP}_T${T_DAYS}d_UltraFast"

mkdir -p "$RUN_DIR"
cd "$JULIA_DIR" || exit 1

echo "============================================================"
echo "LAUNCHING ULTRA-FAST SWEEP (T=${T_DAYS}d)"
echo "Target Directory: $RUN_DIR"
echo "============================================================"

# Background 1 Hz nvidia-smi sampler
SMI_CSV="$RUN_DIR/nvidia_smi.csv"
{
    echo "timestamp,gpu_util_percent,memory_used_mib,power_w"
    while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
                   --format=csv,noheader,nounits 2>/dev/null \
            | head -1 \
            | awk -F', ' '{print $1","$2","$3","$4}'
        sleep 1
    done
} > "$SMI_CSV" &
SMI_PID=$!

# Ultra-Fast Config (slashes 1024 -> 128 ctrl particles)
ULTRA_FAST_ARGS="--filt-n-smc 16 --filt-k-per-chain 100 --filt-num-mcmc 1 --filt-hmc-leapfrog 2 --ctrl-n-smc 128 --ctrl-num-mcmc 2 --ctrl-n-inner 32 --ctrl-chees-max 128 --ctrl-n-anchors 8"

set +e
julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl \
    --T-days "$T_DAYS" \
    --output-dir "$RUN_DIR" \
    --tensorboard true \
    $ULTRA_FAST_ARGS \
    "$@"
rc=$?
set -e

# Stop the GPU sampler
kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

# Render v5 plots if successful
if [[ $rc -eq 0 && -f "$RUN_DIR/data.jld2" ]]; then
    echo "Rendering plots..."
    julia --project=. tools_v5/plot_state_traces_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_traces.png" >/dev/null 2>&1
    julia --project=. tools_v5/plot_param_traces_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_params.png" >/dev/null 2>&1
    julia --project=. tools_v5/plot_obs_channels_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_obs.png" >/dev/null 2>&1
fi

echo "============================================================"
echo "RUN COMPLETE"
echo "Exit Code: $rc"
echo "Results saved to: $RUN_DIR"
echo
echo "TO MONITOR WITH TENSORBOARD:"
echo "  tensorboard --logdir $OUTPUT_BASE"
echo "============================================================"

exit "$rc"
