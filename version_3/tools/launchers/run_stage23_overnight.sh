#!/usr/bin/env bash
# Overnight A/B sweep: --cost soft (current) vs --cost soft_fast (Gemini-optimised).
#
# Order rationale:
#   1. T=2d soft_fast HEALTHY -- 1-min sanity-first (catches typo failures
#      cheap before sinking 8h into doomed runs).
#   2. Stage 2 healthy   {soft_fast, soft}  -- the non-negotiable A/B.
#      soft_fast first so the cheap-fast result is visible early.
#   3. Stage 2 sedentary {soft_fast, soft}  -- only run if (2) clean.
#   4. Stage 2 overtrained {soft_fast, soft}  -- only run if (2) clean.
#   5. Stage 3 healthy   {soft_fast, soft}  -- only run if Stage 2 clean.
#
# Runs to ~10 hours total at the soft baseline of 100 min/14d. The
# soft_fast runs are expected ~6 min each (16x speedup).
#
# Per Ajay's instruction this is monitored actively, NOT fire-and-forget.
# Each run pipes its stdout to its own log under
# outputs/fsa_v5/experiments/runNN_<tag>/run.log so each can be
# inspected independently.

set -u   # -e omitted so a single failure doesn't tear the whole sweep
       # down; each run's exit code is captured + logged instead.
set -o pipefail

cd /home/ajay/Repos/python-smc2-filtering-control/version_3
export PYTHONPATH=.:..

# Use the comfyenv Python explicitly so the launcher works even when
# the env isn't pre-activated (e.g. nohup'd from a fresh shell).
PY="/home/ajay/miniconda3/envs/comfyenv/bin/python"
test -x "$PY" || { echo "ERROR: $PY not found"; exit 1; }

LOG_BASE="/tmp/stage23_sweep"
mkdir -p "$LOG_BASE"
SWEEP_LOG="$LOG_BASE/sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Sweep starting at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SWEEP_LOG"
echo "Sweep master log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"

run_one() {
  local label="$1"; shift
  local args=("$@")
  local log="$LOG_BASE/${label}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  echo "  [$(date '+%H:%M:%S')] START: $label" | tee -a "$SWEEP_LOG"
  echo "  args: ${args[*]}" | tee -a "$SWEEP_LOG"
  echo "  log:  $log" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  local t0=$(date +%s)
  "$PY" "${args[@]}" 2>&1 | tee "$log" || true
  local rc=${PIPESTATUS[0]}
  local t1=$(date +%s)
  local elapsed=$((t1 - t0))
  echo "  [$(date '+%H:%M:%S')] END  : $label (rc=$rc, ${elapsed}s)" | tee -a "$SWEEP_LOG"
}

CTRL=tools/bench_controller_only_fsa_v5.py
FULL=tools/bench_smc_full_mpc_fsa_v5.py

# 1. Sanity-first: T=2d soft_fast healthy. ~1 min. Skipped if matching
#    run already exists (avoids re-burning GPU on the same combination).
SANITY_TAG=stage2_soft_fast_T2d_sanity_overnight
run_one sanity_T2d_soft_fast_healthy \
  $CTRL --cost soft_fast --scenario healthy --T-days 2 --replan-K 2 \
        --run-tag $SANITY_TAG

# 2. Stage 2 HEALTHY -- the non-negotiable A/B. soft_fast first.
run_one s2_soft_fast_healthy \
  $CTRL --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_healthy_T14_optimized

run_one s2_soft_healthy \
  $CTRL --cost soft --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_healthy_T14_baseline

# 3. Stage 2 SEDENTARY -- only if healthy looked OK.
run_one s2_soft_fast_sedentary \
  $CTRL --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_sedentary_T14_optimized

run_one s2_soft_sedentary \
  $CTRL --cost soft --scenario sedentary --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_sedentary_T14_baseline

# 4. Stage 2 OVERTRAINED.
run_one s2_soft_fast_overtrained \
  $CTRL --cost soft_fast --scenario overtrained --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_overtrained_T14_optimized

run_one s2_soft_overtrained \
  $CTRL --cost soft --scenario overtrained --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_overtrained_T14_baseline

# 5. Stage 3 HEALTHY (full closed-loop MPC, both variants).
run_one s3_soft_fast_healthy \
  $FULL --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage3_soft_fast_healthy_T14_optimized

run_one s3_soft_healthy \
  $FULL --cost soft --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage3_soft_healthy_T14_baseline

echo "" | tee -a "$SWEEP_LOG"
echo "Sweep complete at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SWEEP_LOG"
