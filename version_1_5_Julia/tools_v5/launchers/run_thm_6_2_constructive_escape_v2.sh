#!/usr/bin/env bash
# FSA v5 controllability test — Theorem 6.2 (Constructive escape from sedentary).
#
# Tests the SMC²-MPC controller (full closed-loop stack) starting from
# SEDENTARY_INIT with the DEFAULT initial Φ = (0.3, 0.3). The controller
# must DISCOVER the v2 healthy-island escape Φ on its own under the
# simplified cost. The most direct controller test of the v2 stack.
#
# Cost: simplified J = -A_acc + lam_island * island_acc.
# Truth params: TRUTH_PARAMS_V5_RECOMMENDED_V2.
# Reference deterministic outcome under PDF FIM witness Φ* = (0.86, 1.24):
# A(100) = 1.1605. The bench is NOT re-verifying this — it tests whether
# the controller finds the island from a generic sedentary start.
#
# Bonus check: does Φ-time-average track the FIM witness (0.86, 1.24)?
#
# Default T=100d; pass an integer first arg to override.

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
RUN_DIR="$OUTPUT_BASE/${TIMESTAMP}_thm_6_2_constructive_escape_v2_T${T_DAYS}d"

mkdir -p "$RUN_DIR"
cd "$JULIA_DIR" || exit 1

echo "============================================================"
echo "THM 6.2 — constructive escape (closed-loop, v2)  T=${T_DAYS}d"
echo "Target Directory: $RUN_DIR"
echo "============================================================"

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

ULTRA_FAST_ARGS="--filt-n-smc 16 --filt-k-per-chain 100 --filt-num-mcmc 1 --filt-hmc-leapfrog 2 --ctrl-n-smc 128 --ctrl-num-mcmc 2 --ctrl-n-inner 32 --ctrl-chees-max 128 --ctrl-n-anchors 8"

# Theorem-6.2-specific args:
#   - v2 parametrisation
#   - SEDENTARY_INIT plant
#   - DEFAULT initial Φ (no --init-phi-B / --init-phi-S overrides)
#   - simplified cost 
THM_ARGS="\
  --truth-preset v2 \
  --init-preset SEDENTARY_INIT \
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
