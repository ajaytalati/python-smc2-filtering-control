#!/usr/bin/env bash

set -u

# ==============================================================================
# Quality-Aware Automated Speed Sweep Launcher for T=42d
# This script extracts discovery metrics (Planned Phi) and difficulty metrics 
# (Tempering Levels) to evaluate basin discovery vs. run speed.
# ==============================================================================

export FSA_V5_SWEEP_TARGET_STRIDE=2

declare -A configs
configs["1_Baseline"]=""
configs["2_Fast_Filter"]="--filt-n-smc 16 --filt-k-per-chain 100 --filt-num-mcmc 1 --filt-hmc-leapfrog 2"
configs["3_Fast_Ctrl_A"]="--ctrl-n-smc 256 --ctrl-num-mcmc 4"
configs["4_Fast_Ctrl_B"]="--ctrl-n-smc 128 --ctrl-num-mcmc 2 --ctrl-n-inner 32 --ctrl-chees-max 128"
configs["5_Fast_Ctrl_C"]="--ctrl-n-anchors 8"
configs["6_Ultra_Fast"]="--filt-n-smc 16 --filt-k-per-chain 100 --filt-num-mcmc 1 --filt-hmc-leapfrog 2 --ctrl-n-smc 128 --ctrl-num-mcmc 2 --ctrl-n-inner 32 --ctrl-chees-max 128 --ctrl-n-anchors 8"

T_DAYS=42
TOTAL_STRIDES=85

cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia || exit 1

echo "==========================================================================================================="
echo "Starting Quality-Speed Sweep (T=42d). Evaluating discovery (Planned Phi) vs runtime."
printf "| %-15s | %-8s | %-7s | %-8s | %-8s | %-16s |\n" "Config" "Wall(s)" "Est(h)" "F-Levs" "C-Levs" "Plan Phi (B/S)"
echo "|-----------------|----------|---------|----------|----------|------------------|"

for key in $(echo "${!configs[@]}" | tr ' ' '\n' | sort); do
    args=${configs[$key]}
    temp_log=$(mktemp)
    
    julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl \
        --T-days "$T_DAYS" --seed 42 --output-dir "/tmp/dummy_run_$$" --tensorboard false $args > "$temp_log" 2>&1
    
    # Extract Metrics
    wall_s=$(grep -E "\[stride\s+2 phase walls \(s\)\]" "$temp_log" | grep -E -o "total=[0-9.]+" | cut -d= -f2 | tr -d '\r')
    f_levs=$(grep -E "\[stride\s+2/\s*[0-9]+\]" "$temp_log" | sed -n 's/.*s \+\([0-9]\+\) filter levels.*/\1/p' | tr -d '\r')
    c_levs=$(grep "replan @ stride" "$temp_log" | sed -n 's/.*ctrl_n_temp=\([0-9]\+\).*/\1/p' | tr -d '\r')
    phi_b=$(grep "replan @ stride" "$temp_log" | sed -n 's/.*Φ̄_B_plan=\([0-9.]\+\).*/\1/p' | tr -d '\r')
    phi_s=$(grep "replan @ stride" "$temp_log" | sed -n 's/.*Φ̄_S_plan=\([0-9.]\+\).*/\1/p' | tr -d '\r')

    if [ -n "$wall_s" ]; then
        est_hrs=$(echo "scale=2; $wall_s * $TOTAL_STRIDES / 3600" | bc)
        printf "| %-15s | %-8s | %-7s | %-8s | %-8s | %-16s |\n" "$key" "$wall_s" "$est_hrs" "$f_levs" "$c_levs" "B:$phi_b S:$phi_s"
    else
        printf "| %-15s | ERROR    | N/A     | N/A      | N/A      | N/A              |\n" "$key"
    fi
    
    rm -f "$temp_log"
    rm -rf "/tmp/dummy_run_$$"
done
echo "==========================================================================================================="
