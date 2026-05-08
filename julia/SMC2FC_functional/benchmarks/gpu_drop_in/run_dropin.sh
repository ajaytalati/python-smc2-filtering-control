#!/usr/bin/env bash
# Drop-in GPU bench runner.
#
# - Activates the version_1_5_Julia project so all the model deps
#   (CUDA, Match, StableRNGs, JLD2, JSON3, Plots, ...) are available.
# - Adds SMC2FC_functional to JULIA_LOAD_PATH so the bench's
#   `using SMC2FC_functional: run_tempered_smc_gpu` import resolves.
#
# This is the proper drop-in replacement test: same v1.5 GPU bench
# (bench_smc_full_mpc_fsa_gpu.jl), only the framework import line
# changed from SMC2FC -> SMC2FC_functional.
#
# Usage:
#   bash run_dropin.sh [--T-days 2 --replan-K 2 ...]
# (forwards all args to the bench)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../../.."
V1_5_PROJ="$REPO_ROOT/version_1_5_Julia"
SMC2FC_FN="$REPO_ROOT/julia/SMC2FC_functional"

cd "$SCRIPT_DIR"

JULIA_LOAD_PATH="@:$SMC2FC_FN:@stdlib" \
  julia --project="$V1_5_PROJ" "$SCRIPT_DIR/bench_dropin.jl" "$@"
