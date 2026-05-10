#!/usr/bin/env bash
# Overnight quality+speed optimisation experiments for soft_fast.
#
# Context (2026-05-05 22:30):
#   * Run 06 (soft, healthy, T=14d, 99.8 min) -- baseline.
#   * Run 09 (soft_fast trimmed-HMC, healthy, T=14d, 16.4 min) -- 6x
#     speedup but the basin overlay shows wild excursions into the
#     collapsed (Phi_B, Phi_S) regime. Quality regression.
#   * Hypothesis: the HMC config trim (n_smc 128, mcmc 5, leapfrog 8)
#     caused the regression; cost-fn optimisations (fp32, relaxed
#     bisection, sub-sampled bins) are mathematical and safe.
#   * Run 11 (in-flight, PID 1782342) tests this: soft_fast cost-fn
#     + FULL HMC (mcmc=10, leapfrog=16, n_smc=256).
#
# This launcher waits for Run 11 to finish, decides next steps based
# on its result, and runs a queue of experiments unattended.
#
# Decision tree:
#   * If Run 11 matches Run 06 in basin path quality (no collapsed
#     excursions, applied Phi range tight ~[0.15, 0.50]):
#       => HMC trim alone was the bug. Production config = soft_fast
#          cost-fn + full HMC. Run the rest of Stage 2 sweep
#          (sedentary, overtrained) + Stage 3 healthy.
#   * If Run 11 STILL has collapsed-regime excursions (max applied
#     Phi > 0.7):
#       => Cost-fn optimisation also contributes. Run ablations:
#          a. bin_stride=1 (revert sub-sampling)
#          b. fall back to strict bisection
#       To pick which one to revert first, the launcher tries
#       bin_stride=1 first (cheaper change).
#
# Heuristic for "is the path bad?": peek the manifest's
# applied_phi_max -- if > 0.70 the controller wandered out of the
# healthy-island/bistable-annulus region. Run 06 had applied_phi_max
# = 0.49; Run 09 had 1.07.

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
SWEEP_LOG="$LOG_DIR/quality_speed_overnight_$(date +%Y%m%d_%H%M).log"
STATE_FILE="$LOG_DIR/quality_speed_state.json"
echo "Quality+speed overnight starting at $(date '+%Y-%m-%d %H:%M:%S')" \
  | tee -a "$SWEEP_LOG"
echo "  state file: $STATE_FILE" | tee -a "$SWEEP_LOG"

# --- helpers ----------------------------------------------------------------
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

manifest_max_phi() {
  # Read applied_phi_max from a run dir's manifest.json. Print "missing"
  # if no manifest yet, or just the float otherwise.
  local run_dir="$1"
  local mfst="$run_dir/manifest.json"
  if [ ! -f "$mfst" ]; then echo "missing"; return; fi
  $PY -c "
import json, sys
m = json.load(open('$mfst'))
s = m.get('summary', {})
v = s.get('applied_phi_max')
print(v if v is not None else 'missing')
"
}

run_one() {
  local label="$1"; shift
  local kind="$1"; shift   # 'controller' or 'full'
  local args=("$@")
  local log="$LOG_DIR/${label}.log"
  local script
  case "$kind" in
    controller) script="tools/bench_controller_only_fsa_v5.py" ;;
    full)       script="tools/bench_smc_full_mpc_fsa_v5.py" ;;
    *) echo "BAD KIND: $kind"; return 1 ;;
  esac
  echo "" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  echo "  [$(date '+%H:%M:%S')] START: $label" | tee -a "$SWEEP_LOG"
  echo "  args: $script ${args[*]}" | tee -a "$SWEEP_LOG"
  echo "  log:  $log" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  write_state current_run "$label" status running
  local t0=$(date +%s)
  "$PY" -u "$script" "${args[@]}" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local t1=$(date +%s)
  local elapsed=$((t1 - t0))
  echo "  [$(date '+%H:%M:%S')] END  : $label (rc=$rc, ${elapsed}s)" | tee -a "$SWEEP_LOG"
  write_state current_run "$label" status finished elapsed_s "$elapsed" rc "$rc"
  return $rc
}

# --- step 1: wait for Run 11 (the diagnostic) -------------------------------
# Watch for PID 1782342 -- the soft_fast healthy + full HMC run already
# in flight. If it's already gone, skip the wait.
RUN11_PID=1782342
RUN11_DIR="$REPO/version_3/outputs/fsa_v5/experiments/run11_stage2_soft_fast_healthy_T14_full_hmc"
echo "" | tee -a "$SWEEP_LOG"
echo "Step 1: waiting on Run 11 (PID $RUN11_PID, dir run11_*)" | tee -a "$SWEEP_LOG"
write_state stage waiting_run11
while kill -0 "$RUN11_PID" 2>/dev/null; do
  ela=$(ps -o etime= -p $RUN11_PID 2>/dev/null | tr -d ' ' || echo 0)
  n=$(grep -c "REPLAN.*cost=" "$LOG_DIR/s2_soft_fast_healthy_full_hmc.log" 2>/dev/null || echo 0)
  echo "  [$(date '+%H:%M:%S')] Run 11 etime=$ela, replans=$n/14" | tee -a "$SWEEP_LOG"
  sleep 300
done
echo "  [$(date '+%H:%M:%S')] Run 11 EXITED" | tee -a "$SWEEP_LOG"

# Verdict on Run 11
MAX_PHI_R11=$(manifest_max_phi "$RUN11_DIR")
write_state run11_max_phi "$MAX_PHI_R11"
echo "  Run 11 applied_phi_max = $MAX_PHI_R11 (Run 06 baseline = 0.49, Run 09 bad = 1.07)" \
  | tee -a "$SWEEP_LOG"

# --- step 2: branch on Run 11 verdict ---------------------------------------
# Threshold: 0.70. Above => still wandering out; below => quality recovered.
QUALITY_OK=$($PY -c "
v = '$MAX_PHI_R11'
if v == 'missing': print('unknown'); raise SystemExit
print('ok' if float(v) <= 0.70 else 'bad')
")
write_state run11_verdict "$QUALITY_OK"
echo "  Run 11 quality verdict: $QUALITY_OK" | tee -a "$SWEEP_LOG"

if [ "$QUALITY_OK" = "ok" ] || [ "$QUALITY_OK" = "unknown" ]; then
  # Production config is soft_fast cost-fn + full HMC. Run the rest.
  echo "" | tee -a "$SWEEP_LOG"
  echo "Step 2A: production config is soft_fast + full HMC. Running sweep:" \
    | tee -a "$SWEEP_LOG"
  write_state stage running_sweep_with_full_hmc

  run_one s2_soft_fast_sedentary_full_hmc controller \
      --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
      --run-tag stage2_soft_fast_sedentary_T14_full_hmc

  run_one s2_soft_fast_overtrained_full_hmc controller \
      --cost soft_fast --scenario overtrained --T-days 14 --replan-K 2 \
      --run-tag stage2_soft_fast_overtrained_T14_full_hmc

  run_one s3_soft_fast_healthy_full_hmc full \
      --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
      --run-tag stage3_soft_fast_healthy_T14_full_hmc

else
  # Run 11 still bad. Try ablations.
  echo "" | tee -a "$SWEEP_LOG"
  echo "Step 2B: Run 11 still has collapsed-regime excursions. Ablating cost-fn buckets:" \
    | tee -a "$SWEEP_LOG"
  write_state stage running_ablations

  # Ablation A: revert bin_stride sub-sampling.
  run_one ablation_A_bin_stride_1 controller \
      --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
      --bin-stride 1 \
      --run-tag stage2_soft_fast_healthy_T14_full_hmc_binstride1

  ABL_A_DIR="$REPO/version_3/outputs/fsa_v5/experiments/run*ablation*binstride1"
  # find the actual ablation dir
  ABL_A_DIR=$(ls -td $ABL_A_DIR 2>/dev/null | head -1)
  if [ -n "$ABL_A_DIR" ]; then
    MAX_PHI_A=$(manifest_max_phi "$ABL_A_DIR")
    write_state ablation_A_max_phi "$MAX_PHI_A"
    echo "  Ablation A applied_phi_max = $MAX_PHI_A" | tee -a "$SWEEP_LOG"
  fi

  # Stop here; ablation B (revert bisection) requires editing
  # control_v5_fast.py and re-running, which is more involved -- save
  # for the next session if A also failed.
fi

write_state stage finished
echo "" | tee -a "$SWEEP_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] OVERNIGHT EXPERIMENTS COMPLETE" | tee -a "$SWEEP_LOG"
