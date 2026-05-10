#!/usr/bin/env bash
# Stage I+H multi-horizon h=1h sweep — runs outside VS Code in a tmux session.
#
# Usage (after closing VS Code):
#   tmux new -s sweep -d "$HOME/bench_logs/run_h1h_sweep.sh"
#   tmux attach -t sweep    # peek at progress
#   Ctrl-b d                 # detach (sweep keeps running)
#   tmux ls                  # list sessions
#   tmux attach -t sweep     # reattach
#
# When done (~7 hours), reopen VS Code, start a new Claude session, and
# point Claude at outputs/fsa_high_res/g4_runs/ — checkpoints will be
# loaded by tools/load_g4_run.py + tools/compare_g4_lqg.py.
#
# Safety env vars set below cap JAX at 60% of GPU memory (leaves ~13 GB
# for the compositor / desktop, in case the user opens VS Code again
# mid-sweep). Disables JAX preallocation so memory is claimed lazily.

set -u

LOGDIR="$HOME/bench_logs"
REPO="$HOME/Repos/python-smc2-filtering-control"
mkdir -p "$LOGDIR"

cd "$REPO/version_2" || { echo "REPO not found"; exit 1; }

export JAX_ENABLE_X64=True
export PYTHONPATH=.:..
# JAX-side safety caps — limit GPU memory share + lazy allocation so the
# Wayland/Xorg compositor always has headroom even if VS Code is reopened.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_one() {
    local T=$1
    local stamp
    stamp=$(date +%Y%m%d_%H%M)
    local log="$LOGDIR/g4_t${T}_h1h_${stamp}.log"
    echo
    echo "============================================================"
    echo "  starting T=${T} h=1h at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  log: $log"
    echo "============================================================"
    python -u tools/bench_smc_full_mpc_fsa.py "$T" --step-minutes 60 \
        2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "  T=${T} exit code: $rc, finished at $(date '+%Y-%m-%d %H:%M:%S')"
    return $rc
}

echo "============================================================"
echo "  Stage I+H h=1h sweep starting at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPU mem cap: $XLA_PYTHON_CLIENT_MEM_FRACTION × total"
echo "  preallocate: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "============================================================"

# T=14 first as sanity check; T=28 added for the catastrophic-failure
# regression check that the pre-M run flagged (was ratio 0.569).
run_one 14 || echo "  WARN: T=14 failed but continuing to T=28"
run_one 28 || echo "  WARN: T=28 failed but continuing to T=42 anyway"

# ── HARD GATE on T=28 ────────────────────────────────────────────────
# Pre-M T=28 (h=15min, BlackJAX, n_smc=128) gave ratio 0.569 and
# F-violation 75% — catastrophic. Stage M+N+J5 might fix this via
# bigger N → better SF-bridge q0 estimate. We don't run T=42/56/84
# until T=28 satisfies the basic acceptance gates.
#
# Gate logic lives in tools/check_t28_gate.py (NOT inline here) —
# inline bash gates are unsafe to hot-edit while a sweep is running,
# because bash streams the script from a file position and editing
# the file in place shifts byte offsets. The Python script is read
# once at gate-check time, so it can be edited safely.
T28_MANIFEST="$REPO/version_2/outputs/fsa_high_res/g4_runs/T28d_replanK2_h60min_no_infoaware/manifest.json"
if [ ! -f "$T28_MANIFEST" ]; then
    echo "  GATE FAIL: T=28 manifest missing at $T28_MANIFEST — aborting sweep" >&2
    exit 1
fi
python tools/check_t28_gate.py "$T28_MANIFEST"
GATE_RC=$?
if [ "$GATE_RC" -ne 0 ]; then
    echo "  Stopping sweep before T=42/56/84. Investigate T=28 first." >&2
    exit "$GATE_RC"
fi
echo "  Proceeding to T=42/56/84"

run_one 42 || echo "  WARN: T=42 failed but continuing to T=56"
run_one 56 || echo "  WARN: T=56 failed but continuing to T=84"
run_one 84 || echo "  WARN: T=84 failed"

echo
echo "============================================================"
echo "  sweep finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo "  checkpoints in: $REPO/version_2/outputs/fsa_high_res/g4_runs/"
echo "  logs in: $LOGDIR/"
ls -la "$REPO/version_2/outputs/fsa_high_res/g4_runs/" 2>/dev/null
