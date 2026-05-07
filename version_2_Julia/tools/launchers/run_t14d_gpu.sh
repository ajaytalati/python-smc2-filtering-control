#!/usr/bin/env bash
# T=14d GPU launcher (Stage B — scaffold only until gpu_pf.jl is written).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export JULIA_NUM_THREADS=auto
# Set CUDA_VISIBLE_DEVICES if needed.

julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days 14 \
    --step-minutes 60 \
    --replan-K 2 \
    --N-smc 1024 \
    --N-pf 800 \
    --seed 42 \
    "$@"
