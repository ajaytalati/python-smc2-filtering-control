#!/usr/bin/env bash
# Julia v1.5 multi-horizon sweep launcher
#
# Usage:
#   ./run_julia_horizon_sweep.sh                 # default: 14 28 42 56 70 84
#   ./run_julia_horizon_sweep.sh 14              # single horizon (smoke test)
#   ./run_julia_horizon_sweep.sh 28 42 56 70 84  # overnight chain (after T=14 smoke)
#
# Per horizon, writes everything into
#   compare_v15_julia_vs_python/example_run/julia_horizon_sweep_<DATE>/T<N>d_seed42/
#
# The script:
#   * Captures 1 Hz nvidia-smi telemetry (util %, VRAM MiB, power W).
#   * Runs the saturated-default Julia bench at the chosen --T-days.
#   * Calls the canonical state-traces plotter (the bench's internal
#     auto-plot path points at the deprecated v2_Julia/ tree which is
#     about to be removed; we don't rely on it).
#   * Calls the per-horizon results extractor → results.json +
#     appends a row to <sweep_root>/horizons_results.csv.
#   * Continues to the next horizon on non-zero exit so a single
#     failure doesn't abort the overnight chain.
#
# At the end, renders horizons_results.csv as a Markdown table at the
# bottom of SWEEP_MANIFEST.md so the sweep root is self-documenting.

set -u

T_LIST=("$@")
if [ ${#T_LIST[@]} -eq 0 ]; then
    T_LIST=(14 28 42 56 70 84)
fi

REPO="$HOME/Repos/python-smc2-filtering-control"
SWEEP_DATE=$(date +%Y-%m-%d)
SWEEP_ROOT="$REPO/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_${SWEEP_DATE}"
PLOT_DIR="$REPO/compare_v15_julia_vs_python/src"
JULIA_DIR="$REPO/version_1_5_Julia"

mkdir -p "$SWEEP_ROOT"
cd "$JULIA_DIR" || { echo "FATAL: could not cd to $JULIA_DIR" >&2; exit 1; }

MANIFEST="$SWEEP_ROOT/SWEEP_MANIFEST.md"
RESULTS_CSV="$SWEEP_ROOT/horizons_results.csv"

{
  echo "# Julia v1.5 horizon sweep — $(date)"
  echo
  echo "Sweep root: \`$SWEEP_ROOT\`"
  echo
  echo "## Per-horizon wall + exit code"
  echo
  echo "| T_days | start | wall_s | exit_code | dir |"
  echo "|---|---|---|---|---|"
} > "$MANIFEST"

# ── One-shot microbench profile (kernel walls don't depend on T-days) ──
PROFILE_OUT="$SWEEP_ROOT/microbench_profile.txt"
if [ ! -f "$PROFILE_OUT" ]; then
    echo "[$(date '+%H:%M:%S')] one-shot microbench profile → $PROFILE_OUT"
    julia --project=. "$PLOT_DIR/profile_gpu_julia_v15.jl" --both \
        > "$PROFILE_OUT" 2>&1 || \
        echo "  (microbench profile failed — see $PROFILE_OUT — continuing)"
fi

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

    # ── Background 1 Hz nvidia-smi sampler ──
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

    # ── Run the bench ──
    set +e
    julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
        --T-days "$T" --seed 42 --output-dir "$DIR" \
        2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    set -u

    # ── Stop the GPU sampler ──
    kill "$SMI_PID" 2>/dev/null || true
    wait "$SMI_PID" 2>/dev/null || true

    # ── Canonical state-traces plotter (the bench's auto-plot path
    # for state traces resolves under the deprecated v2_Julia tree
    # and currently @warn-skips). Param traces ARE produced by the
    # bench's internal _plot_param_traces_v15. ──
    if [ "$rc" -eq 0 ] && [ -f "$DIR/data.jld2" ]; then
        STATE_PNG="$DIR/v15_T${T}d_traces.png"
        julia --project=. -e "
            include(\"$PLOT_DIR/plot_state_traces.jl\")
            d = load_run_data(\"$DIR/data.jld2\")
            plot_state_traces(d; out_path=\"$STATE_PNG\")
            @info \"wrote \$(\"$STATE_PNG\")\"
        " 2>&1 | tee -a "$LOG" || \
            echo "  (state-traces plot failed — see $LOG — continuing)" | tee -a "$LOG"

        # ── Headline-metrics extractor → results.json + row in CSV ──
        python "$PLOT_DIR/extract_horizon_results.py" \
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

# ── Render horizons_results.csv as a Markdown table at the bottom
# of SWEEP_MANIFEST.md ──
if [ -f "$RESULTS_CSV" ]; then
    {
        echo
        echo "## Per-horizon headline results"
        echo
        python -c "
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
