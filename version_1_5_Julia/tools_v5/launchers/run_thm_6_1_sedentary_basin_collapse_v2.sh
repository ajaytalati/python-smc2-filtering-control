#!/usr/bin/env bash
# FSA v5 controllability test — Theorem 6.1 (Pathological sedentary basin).
#
# Tests the SMC²-MPC controller (full closed-loop stack: plant + obs + filter
# + controller, NO open-loop) starting from the sedentary state with the
# baseline policy pinned in the pathological corner Φ = (0.1, 0.1).
#
# Question: does the controller diagnose the sedentary corner and steer Φ
# toward the v2 healthy island? (Or stay near (0.1, 0.1) and let A collapse?)
#
# Cost: simplified J = -A_acc + lam_island * island_acc (all other terms 0).
# Truth params: TRUTH_PARAMS_V5_RECOMMENDED_V2.
# Reference deterministic constant-Φ outcome (PDF Thm 6.1): A(100) = 0.0000.
# The bench's BASELINE rollout reproduces this; the CLOSED-LOOP trajectory
# is what the controller actually does.
#
# Default T=100d; pass an integer first arg to override (e.g. T=2 for smoke).

set -u

T_DAYS="100"
if [[ $# -gt 0 && $1 =~ ^[0-9]+$ ]]; then
    T_DAYS="$1"
    shift
fi

REPO="/home/ajay/Repos/python-smc2-filtering-control"
JULIA_DIR="$REPO/version_1_5_Julia"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

OUTPUT_BASE="$REPO/outputs/bench_runs"
RUN_DIR="$OUTPUT_BASE/${TIMESTAMP}_thm_6_1_sedentary_collapse_v2_T${T_DAYS}d"

mkdir -p "$RUN_DIR"
cd "$JULIA_DIR" || exit 1

echo "============================================================"
echo "THM 6.1 — sedentary basin collapse (closed-loop, v2)  T=${T_DAYS}d"
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

# Ultra-Fast tier (matches run_v5_ultra_fast.sh)
ULTRA_FAST_ARGS="--filt-n-smc 16 --filt-k-per-chain 100 --filt-num-mcmc 1 --filt-hmc-leapfrog 2 --ctrl-n-smc 128 --ctrl-num-mcmc 2 --ctrl-n-inner 32 --ctrl-chees-max 128 --ctrl-n-anchors 8"

# Theorem-6.1-specific args:
#   - v2 parametrisation
#   - SEDENTARY_INIT plant
#   - baseline Φ pinned at the pathological corner (0.1, 0.1)
#   - simplified cost: only A reward 
THM_ARGS="\
  --truth-preset v2 \
  --init-preset SEDENTARY_INIT \
  --init-phi-B 0.1 --init-phi-S 0.1 \
  --open-loop false \
  --ctrl-lam-phi 0 --ctrl-lam-f 0 \
  --ctrl-lam-chance 0 --ctrl-lam-chance-b 0 --ctrl-lam-chance-s 0 \
  --ctrl-lam-a 1.0 --ctrl-lam-b 0 --ctrl-lam-s 0 \
  --ctrl-lam-island 0.0"

set +e
julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl \
    --T-days "$T_DAYS" \
    --output-dir "$RUN_DIR" \
    --tensorboard true \
    $ULTRA_FAST_ARGS \
    $THM_ARGS \
    "$@"
rc=$?
set -e

kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

if [[ $rc -eq 0 && -f "$RUN_DIR/data.jld2" ]]; then
    echo "Rendering plots..."
    julia --project=. tools_v5/plot_state_traces_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_traces.png" >/dev/null 2>&1
    julia --project=. tools_v5/plot_param_traces_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_params.png" >/dev/null 2>&1
    julia --project=. tools_v5/plot_obs_channels_v5.jl "$RUN_DIR/data.jld2" "$RUN_DIR/v5_obs.png" >/dev/null 2>&1
fi

echo "============================================================"
echo "RUN COMPLETE   exit=$rc"
echo "Results: $RUN_DIR"
echo "============================================================"
exit "$rc"
