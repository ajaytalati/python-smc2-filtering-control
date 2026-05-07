#!/usr/bin/env bash
# Follow-up to run_quality_speed_overnight.sh.
#
# Context (2026-05-06 01:13):
#   * Master launcher PID 1787048 is in flight running the soft_fast
#     sweep with full HMC. Will OOM at its Stage 3 step because that
#     step is missing the --n-smc 128 / --n-pf 200 / --n-inner 32 flags
#     (Run 11 already established that full-particle Stage 3 OOMs on
#     the 5090 -- 30.85 GB requested on a 32 GB device).
#   * Ajay's Run 14 (Stage 3 healthy soft_fast at smaller particles, PID
#     1784890) has been running concurrently. At 01:13 it's at Stride
#     24/28, ETA ~02:08.
#   * Stage 2 sedentary (Run 15, in master launcher) is at Stride 11/28
#     and will be bit-identical to Run 13 anyway (structural property
#     documented in Run 10 CHANGELOG entry); same for overtrained.
#
# This follow-up waits for BOTH the master launcher AND Run 14 to exit,
# then runs:
#   1. profile_cost_fn at multiple configs -- empirical RTX 5090
#      saturation data (each ~1-3 min once JIT'd).
#   2. Stage 3 sedentary soft_fast at smaller particles (~1.5-2 h).
#   3. (optional, only if 08:00 budget allows) Stage 3 overtrained
#      soft_fast at smaller particles (~1.5-2 h).
#
# Hard deadline: 08:00 London (2026-05-06 08:00). User's GPU budget.
# Anything launched here MUST finish before 08:00. Each step checks
# wall-clock-remaining and skips if not enough time.

set -u
set -o pipefail

REPO=/home/ajay/Repos/python-smc2-filtering-control
cd "$REPO/version_3"
export PYTHONPATH=.:..
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/ajay/miniconda3/envs/comfyenv/bin/python
test -x "$PY" || { echo "ERROR: $PY not found"; exit 1; }

LOG_DIR=/tmp/stage23_sweep
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/followup_overnight_$(date +%Y%m%d_%H%M).log"
STATE_FILE="$LOG_DIR/followup_state.json"
echo "Follow-up overnight starting at $(date '+%Y-%m-%d %H:%M:%S')" \
  | tee -a "$SWEEP_LOG"

# Hard deadline = 08:00 London local time today (2026-05-06).
DEADLINE_TS=$(date -d 'today 08:00' +%s)

write_state() {
  $PY - "$STATE_FILE" "$@" <<'PY' 2>/dev/null
import sys, json, os
from datetime import datetime
state_file = sys.argv[1]
update = dict(zip(sys.argv[2::2], sys.argv[3::2]))
data = {}
if os.path.exists(state_file):
    try: data = json.load(open(state_file))
    except: pass
data.setdefault('events', [])
data['events'].append({'t': datetime.now().isoformat(timespec='seconds'),
                        **update})
data.update(update)
json.dump(data, open(state_file, 'w'), indent=2)
PY
}

minutes_remaining() {
  local now=$(date +%s)
  local rem=$(( (DEADLINE_TS - now) / 60 ))
  echo "$rem"
}

run_one() {
  local label="$1"; shift
  local script="$1"; shift
  local args=("$@")
  local log="$LOG_DIR/${label}.log"
  echo "" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  echo "  [$(date '+%H:%M:%S')] START: $label" | tee -a "$SWEEP_LOG"
  echo "  log: $log" | tee -a "$SWEEP_LOG"
  echo "  remaining: $(minutes_remaining) min until 08:00" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  write_state current_run "$label" status running
  local t0=$(date +%s)
  "$PY" -u "$script" "${args[@]}" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local elapsed=$((t1 - t0))
  echo "  [$(date '+%H:%M:%S')] END  : $label (rc=$rc, ${elapsed}s)" | tee -a "$SWEEP_LOG"
  write_state current_run "$label" status finished elapsed_s "$elapsed" rc "$rc"
}

# --- step 1: wait for master launcher AND run14 -----------------------------
MASTER_PID=1787048
RUN14_PID=1784890
echo "" | tee -a "$SWEEP_LOG"
echo "Step 1: waiting for master launcher (PID $MASTER_PID) AND Run 14 (PID $RUN14_PID)" \
  | tee -a "$SWEEP_LOG"
write_state stage waiting

while true; do
  m_alive=0; r_alive=0
  kill -0 "$MASTER_PID" 2>/dev/null && m_alive=1
  kill -0 "$RUN14_PID" 2>/dev/null && r_alive=1
  if [ $m_alive -eq 0 ] && [ $r_alive -eq 0 ]; then
    echo "  [$(date '+%H:%M:%S')] Both exited" | tee -a "$SWEEP_LOG"
    break
  fi
  echo "  [$(date '+%H:%M:%S')] master=$m_alive run14=$r_alive remaining=$(minutes_remaining) min" \
    | tee -a "$SWEEP_LOG"
  sleep 300
done

# --- step 2: profile cost_fn (fast) -----------------------------------------
echo "" | tee -a "$SWEEP_LOG"
echo "Step 2: profile cost_fn at multiple configs" | tee -a "$SWEEP_LOG"
write_state stage profiling

# Each config is ~1-3 min once JIT'd. Total ~5-10 min.
run_one profile_soft_fast_n256_t1 tools/profile_cost_fn.py \
    --cost soft_fast --n-smc 256 --n-truth-particles 1 --n-iters 20 --no-trace

run_one profile_soft_fast_n256_t32 tools/profile_cost_fn.py \
    --cost soft_fast --n-smc 256 --n-truth-particles 32 --n-iters 20 --no-trace

run_one profile_soft_fast_n512_t1 tools/profile_cost_fn.py \
    --cost soft_fast --n-smc 512 --n-truth-particles 1 --n-iters 20 --no-trace

run_one profile_soft_n256_t1 tools/profile_cost_fn.py \
    --cost soft --n-smc 256 --n-truth-particles 1 --n-iters 20 --no-trace

# --- step 3: Stage 3 sedentary soft_fast (smaller particles) ----------------
REM=$(minutes_remaining)
if [ $REM -lt 130 ]; then
  echo "  [$(date '+%H:%M:%S')] Only $REM min until 08:00, skipping Stage 3 sedentary" \
    | tee -a "$SWEEP_LOG"
  write_state stage skipped_stage3_sedentary remaining_min "$REM"
else
  echo "" | tee -a "$SWEEP_LOG"
  echo "Step 3: Stage 3 sedentary soft_fast (smaller particles)" | tee -a "$SWEEP_LOG"
  write_state stage running_stage3_sedentary
  run_one s3_soft_fast_sedentary_smaller tools/bench_smc_full_mpc_fsa_v5.py \
      --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
      --n-smc 128 --n-pf 200 --n-inner 32 \
      --run-tag stage3_soft_fast_sedentary_T14_smaller
fi

# --- step 4: Stage 3 overtrained soft_fast (smaller particles) --------------
REM=$(minutes_remaining)
if [ $REM -lt 130 ]; then
  echo "  [$(date '+%H:%M:%S')] Only $REM min until 08:00, skipping Stage 3 overtrained" \
    | tee -a "$SWEEP_LOG"
  write_state stage skipped_stage3_overtrained remaining_min "$REM"
else
  echo "" | tee -a "$SWEEP_LOG"
  echo "Step 4: Stage 3 overtrained soft_fast (smaller particles)" | tee -a "$SWEEP_LOG"
  write_state stage running_stage3_overtrained
  run_one s3_soft_fast_overtrained_smaller tools/bench_smc_full_mpc_fsa_v5.py \
      --cost soft_fast --scenario overtrained --T-days 14 --replan-K 2 \
      --n-smc 128 --n-pf 200 --n-inner 32 \
      --run-tag stage3_soft_fast_overtrained_T14_smaller
fi

write_state stage finished
echo "" | tee -a "$SWEEP_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] FOLLOW-UP COMPLETE" | tee -a "$SWEEP_LOG"
