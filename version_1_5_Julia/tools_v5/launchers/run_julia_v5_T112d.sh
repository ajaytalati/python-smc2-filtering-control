#!/usr/bin/env bash
# Julia v5 closed-loop SMC²-MPC bench — Saturated Production Launcher (T=112d).
#
# Follows the central timestamped output convention.
# Writes to: outputs/bench_runs/YYYY-MM-DD_HHMMSS_T<N>d_Saturated/

set -u

# Robust T_DAYS detection: Use first arg if it's a number, otherwise default to 112.
T_DAYS="112"
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
    T_DAYS="$1"
    shift
fi

REPO="/home/ajay/Repos/python-smc2-filtering-control"
JULIA_DIR="$REPO/version_1_5_Julia"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

# Central Output Root
OUTPUT_BASE="$REPO/outputs/bench_runs"
RUN_DIR="$OUTPUT_BASE/${TIMESTAMP}_T${T_DAYS}d_Saturated"

mkdir -p "$RUN_DIR"
cd "$JULIA_DIR" || exit 1

echo "============================================================"
echo "LAUNCHING SATURATED PRODUCTION RUN (T=${T_DAYS}d)"
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

# Run the bench
set +e
julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl \
    --T-days "$T_DAYS" \
    --output-dir "$RUN_DIR" \
    --tensorboard true \
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
