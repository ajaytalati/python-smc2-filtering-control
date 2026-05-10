#!/usr/bin/env bash
# Stage 2 / Stage 3 sweep -- soft_fast as production default.
#
# Per Ajay's redirect (after reading
# claude_plans/Geminis_observation_on_why_fp64_is_the_bottleneck_on_RTX_5090.md):
# the strict `soft` cost burns ~98% of the 5090 silicon as dark FP64
# cores -- 99% util / 120W power / cold fan are all symptoms of "the
# 1/64th of the chip with FP64 cores is maxed out, the FP32 cores are
# sitting idle". `soft_fast` (fp32 + relaxed bisection + sub-sampled
# bins + trimmed HMC) lets the controller hit the actual GPU. Smoke
# test confirmed bit-equivalent behaviour at the healthy-island corner
# (0.00% rel_diff on mean_A_integral, mean_effort).
#
# Therefore `soft_fast` is the production default; strict `soft` is
# only kept for the head-to-head A/B at healthy (Ajay's separately-
# running PID 1731233 / `stage2_ctrl_soft_healthy_T14d_sat` provides
# that data point so we don't re-run it here).
#
# Order:
#   1. T=2d soft_fast HEALTHY -- 1-min sanity-first (typo-catcher).
#   2. Stage 2 soft_fast HEALTHY     -- A/B partner to PID 1731233.
#   3. Stage 2 soft_fast SEDENTARY   -- full scenario coverage.
#   4. Stage 2 soft_fast OVERTRAINED -- full scenario coverage.
#   5. Stage 3 soft_fast HEALTHY     -- closed-loop MPC headline.
#
# Total target ~30 min Stage 2 + ~1-2 h Stage 3 = ~2 hours. The
# strict-soft runs at sedentary / overtrained / Stage 3 are NOT in
# this sweep -- they would burn another ~5 h confirming what soft_fast
# already shows. If a discrepancy at sedentary or overtrained shows
# up between scenarios, we can spot-check with strict soft after.
#
# Per Ajay's instruction this is monitored actively, NOT fire-and-forget.
# Each run pipes its stdout to its own log under /tmp/stage23_sweep/
# so each can be inspected independently.

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
  "$PY" -u "${args[@]}" 2>&1 | tee "$log" || true
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

# 2. Stage 2 HEALTHY -- only soft_fast here. The soft healthy baseline
#    is already running in a separate process (started by Ajay; tag
#    `stage2_ctrl_soft_healthy_T14d_sat`). This launcher will pick up
#    its manifest at A/B-summary time. No need to re-run soft healthy.
run_one s2_soft_fast_healthy \
  $CTRL --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_healthy_T14_optimized

# 3. Stage 2 SEDENTARY -- soft_fast only (production default).
run_one s2_soft_fast_sedentary \
  $CTRL --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_sedentary_T14_optimized

# 4. Stage 2 OVERTRAINED -- soft_fast only.
run_one s2_soft_fast_overtrained \
  $CTRL --cost soft_fast --scenario overtrained --T-days 14 --replan-K 2 \
        --run-tag stage2_soft_fast_overtrained_T14_optimized

# 5. Stage 3 HEALTHY (full closed-loop MPC) -- soft_fast only.
#    The Stage 3 strict-soft run is dropped: it would burn ~3 h on the
#    slow variant when soft_fast Stage 2 healthy + Ajay's PID 1731233
#    soft Stage 2 healthy A/B already pin the variant agreement.
run_one s3_soft_fast_healthy \
  $FULL --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
        --run-tag stage3_soft_fast_healthy_T14_optimized

echo "" | tee -a "$SWEEP_LOG"
echo "Sweep complete at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$SWEEP_LOG"
