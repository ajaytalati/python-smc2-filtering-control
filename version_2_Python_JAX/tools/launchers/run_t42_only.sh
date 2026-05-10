#!/usr/bin/env bash
# T=42 single-horizon launcher (h=1h, no gate logic).
#
# T=14 + T=28 already gate-passed (see ~/bench_logs/run_h1h_sweep.sh
# log + outputs/fsa_high_res/g4_runs/{T14,T28}_replanK2_h60min_no_infoaware/).
# Run this from outside VS Code in a tmux session:
#
#   tmux new -s t42 -d "$HOME/bench_logs/run_t42_only.sh"
#   tmux ls; tmux attach -t t42; Ctrl-b d
#
# ETA at K=2, n_smc=1024, n_pf=800, h=1h: ~95 min.

set -u

LOGDIR="$HOME/bench_logs"
REPO="$HOME/Repos/python-smc2-filtering-control"
mkdir -p "$LOGDIR"

cd "$REPO/version_2" || { echo "REPO not found"; exit 1; }

export JAX_ENABLE_X64=True
export PYTHONPATH=.:..
# JAX-side safety caps — see GPU_TUNING_RTX5090.md §A.2.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_PREALLOCATE=false

stamp=$(date +%Y%m%d_%H%M)
log="$LOGDIR/g4_t42_h1h_${stamp}.log"

echo "============================================================"
echo "  T=42 h=1h launching at $(date '+%Y-%m-%d %H:%M:%S')"
echo "  log: $log"
echo "  GPU mem cap: $XLA_PYTHON_CLIENT_MEM_FRACTION × total"
echo "============================================================"

python -u tools/bench_smc_full_mpc_fsa.py 42 --step-minutes 60 \
    2>&1 | tee "$log"
rc=${PIPESTATUS[0]}

echo
echo "============================================================"
echo "  T=42 finished at $(date '+%Y-%m-%d %H:%M:%S')  exit code: $rc"
echo "============================================================"
ls -la "$REPO/version_2/outputs/fsa_high_res/g4_runs/T42d_replanK2_h60min_no_infoaware/" 2>/dev/null
exit "$rc"
