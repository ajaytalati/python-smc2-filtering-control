#!/usr/bin/env bash
# T=14d CPU launcher for the Julia FSA-v2 closed-loop bench.
#
# Mirrors `version_2/tools/launchers/run_horizon.sh` but for the Julia side.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export JULIA_NUM_THREADS=auto

julia --project=. tools/bench_smc_full_mpc_fsa.jl \
    --T-days 14 \
    --step-minutes 60 \
    --replan-K 2 \
    --N-smc 1024 \
    --N-pf 800 \
    --seed 42 \
    "$@"
