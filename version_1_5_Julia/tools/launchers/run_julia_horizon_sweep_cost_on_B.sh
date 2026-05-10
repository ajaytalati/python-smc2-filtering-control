#!/usr/bin/env bash
# Julia v1.5 horizon sweep launcher — "cost on B" study (2026-05-09).
#
# IDENTICAL bench flags to run_julia_horizon_sweep_filter_hmc_unified.sh
# (filter HMC unified with framework ChEES; new 2026-05-09 defaults).
# THE ONE CHANGE for this study is in source, not in the CLI: the
# controller's cost kernel
# (version_1_5_Julia/models/fsa_high_res/gpu_control.jl::fsa_v1_cost_kernel!)
# has been edited so the negative integral is over B (chronic fitness)
# rather than A (autonomic amplitude):
#
#   J(Φ) = E[ −∫B·dt + λ_F · ∫max(F − F_max, 0)² · dt ]
#
# To recover cost-on-A, revert the kernel edit (1 comment + 3 lines).
#
# Default T_LIST is the 3-horizon test set: 14, 28, 42.

set -u

T_LIST=("$@")
if [ ${#T_LIST[@]} -eq 0 ]; then
    T_LIST=(14 28 42)
fi

REPO="$HOME/Repos/python-smc2-filtering-control"
SWEEP_ROOT="$REPO/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_cost_on_B"
PLOT_DIR="$REPO/compare_v15_julia_vs_python/src"
JULIA_DIR="$REPO/version_1_5_Julia"

mkdir -p "$SWEEP_ROOT"
cd "$JULIA_DIR" || { echo "FATAL: could not cd to $JULIA_DIR" >&2; exit 1; }

MANIFEST="$SWEEP_ROOT/SWEEP_MANIFEST.md"
RESULTS_CSV="$SWEEP_ROOT/horizons_results.csv"

{
  echo "# Julia v1.5 horizon sweep — cost-on-B (controller objective swapped from \\int A to \\int B) — $(date)"
  echo
  echo "Sweep root: \`$SWEEP_ROOT\`"
  echo
  echo "Configuration: filter HMC ON via framework's"
  echo "parallel_hmc_one_move_generic! + chees_pick_L_generic (same code"
  echo "path as the controller HMC); β=next_λ applied; ε=0.005,"
  echo "h_fd=0.01, ChEES list=[4,8,16,32,64]. All filter rejuvenation"
  echo "tricks ON. Fast (reduced) controller config."
  echo
  echo "## Per-horizon wall + exit code"
  echo
  echo "| T_days | start | wall_s | exit_code | dir |"
  echo "|---|---|---|---|---|"
} > "$MANIFEST"

for T in "${T_LIST[@]}"; do
    DIR="$SWEEP_ROOT/T${T}d_seed42"
    mkdir -p "$DIR"
    LOG="$DIR/bench.log"
    SMI_CSV="$DIR/nvidia_smi.csv"
    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    T0=$(date +%s)
    echo "============================================================"
    echo "[$START_TS]  T=${T}d  →  $DIR"
    echo "============================================================"

    # Background 1 Hz nvidia-smi sampler
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

    set +e
    julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
        --T-days "$T" --seed 42 \
        --num-mcmc 3 \
        --hmc-step-size 0.005 \
        --hmc-leapfrog 4 \
        --h-fd 0.01 \
        --filter-chees-min 4 \
        --filter-chees-max 64 \
        --liu-west-a 0.97 \
        --smooth-resample-bw 1.0 \
        --gaussian-bridge true \
        --N-smc 512 --K-per-chain 1000 \
        --ctrl-n-smc 1024 --ctrl-num-mcmc 8 \
        --collect-ctrl-diagnostics true \
        --output-dir "$DIR" \
        2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    set -u

    kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true

    if [ "$rc" -eq 0 ] && [ -f "$DIR/data.jld2" ]; then
        STATE_PNG="$DIR/v15_T${T}d_traces.png"
        julia --project=. -e "
            include(\"$PLOT_DIR/plot_state_traces.jl\")
            d = load_run_data(\"$DIR/data.jld2\")
            plot_state_traces(d; out_path=\"$STATE_PNG\")
            @info \"wrote \$(\"$STATE_PNG\")\"
        " 2>&1 | tee -a "$LOG" || \
            echo "  (state-traces plot failed — see $LOG — continuing)" | tee -a "$LOG"

        python3 "$PLOT_DIR/extract_horizon_results.py" \
            --horizon-dir "$DIR" --T-days "$T" \
            --sweep-csv "$RESULTS_CSV" \
            --bench-exit-code "$rc" 2>&1 | tee -a "$LOG" || \
            echo "  (results extraction failed — see $LOG — continuing)" | tee -a "$LOG"
    else
        echo "  (bench exit=$rc; skipping plot + extract for T=${T}d)"
    fi

    T1=$(date +%s)
    WALL=$((T1 - T0))
    echo "| $T | $START_TS | $WALL | $rc | T${T}d_seed42 |" >> "$MANIFEST"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  T=${T}d done in ${WALL}s, exit=$rc"
done

if [ -f "$RESULTS_CSV" ]; then
    {
        echo
        echo "## Per-horizon headline results"
        echo
        python3 -c "
import csv, sys
with open('$RESULTS_CSV') as f:
    rows = list(csv.reader(f))
if not rows:
    sys.exit(0)
hdr, *body = rows
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '|'.join(['---'] * len(hdr)) + '|')
for r in body:
    print('| ' + ' | '.join(r) + ' |')
"
    } >> "$MANIFEST"
fi

echo
echo "============================================================"
echo "Sweep complete. Sweep root:"
echo "  $SWEEP_ROOT"
echo "============================================================"
