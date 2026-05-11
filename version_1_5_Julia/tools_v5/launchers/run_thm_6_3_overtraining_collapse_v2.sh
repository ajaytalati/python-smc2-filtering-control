#!/usr/bin/env bash
# FSA v5 controllability test — Theorem 6.3 (Pathological over-training basin).
#
# Tests the SMC²-MPC controller (full closed-loop stack) starting from the
# v2 trained-athlete state with the baseline policy pinned in the over-
# training corner Φ = (2.5, 2.5).
#
# Question: does the controller recognise over-training and pull Φ back to
# the v2 healthy island, or does the system collapse?
#
# Cost: simplified J = -A_acc + lam_island * island_acc.
# Truth params: TRUTH_PARAMS_V5_RECOMMENDED_V2.
# Reference deterministic constant-Φ outcome (PDF Thm 6.3): A(100) = 0.0000
# under Φ ≡ (2.5, 2.5). The bench's BASELINE rollout reproduces this;
# the CLOSED-LOOP trajectory is what the controller actually does.
#
# Note: Φ = (2.5, 2.5), NOT (2, 2). Under v2's autonomic-protective feedback
# a_F(A), Φ = (2, 2) does NOT collapse within T=100d (A(100) ≈ 0.49).
# See PDF §6.3 Remark 7.
#
# Default T=100d.

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
RUN_DIR="$OUTPUT_BASE/${TIMESTAMP}_thm_6_3_overtraining_collapse_v2_T${T_DAYS}d"

mkdir -p "$RUN_DIR"
cd "$JULIA_DIR" || exit 1

echo "============================================================"
echo "THM 6.3 — over-training collapse (closed-loop, v2)  T=${T_DAYS}d"
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

# Theorem-6.3-specific args:
#   - v2 parametrisation
#   - TRAINED_ATHLETE_INIT_V2 plant (v2 slow-manifold equilibrium)
#   - baseline Φ pinned at the over-training corner (2.5, 2.5)
#   - simplified cost with island pull
THM_ARGS="\
  --truth-preset v2 \
  --init-preset TRAINED_ATHLETE_INIT_V2 \
  --init-phi-B 2.5 --init-phi-S 2.5 \
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
