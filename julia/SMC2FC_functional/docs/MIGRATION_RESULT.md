# Julia framework migration result — SMC2FC → SMC2FC_functional

Date: 2026-05-08

This file records the result of migrating the v1.5 closed-loop SMC²-MPC
bench from the original `SMC2FC` framework to the new
`SMC2FC_functional` framework. The migration is the action item from
section 8.1 of
[`compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf`](../../../compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf):

> **1. Migrate to SMC2FC_functional.** `julia/SMC2FC_functional/` is
> intended to be the default Julia framework going forward. Rerun the
> comparison harness with the bench at
> `julia/SMC2FC_functional/benchmarks/...` to confirm it produces
> equivalent posteriors / control behaviour.

## What "migration" means here

The v1.5 GPU bench at
`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` imports from the
framework on **exactly one line**:

```julia
using SMC2FC: run_tempered_smc_gpu
```

Everything else in the bench is model code (lives in
`version_1_5_Julia/models/fsa_high_res/`) and is independent of the
framework choice. The "migration" is to swap that one line.

The drop-in artefact is at
[`julia/SMC2FC_functional/benchmarks/gpu_drop_in/bench_dropin.jl`](../benchmarks/gpu_drop_in/bench_dropin.jl)
— a fresh copy of the v1.5 bench with two surgical patches:

1. `REPO_ROOT` repointed at `version_1_5_Julia/` (since the file lives
    under `SMC2FC_functional/benchmarks/gpu_drop_in/`, not under
    `version_1_5_Julia/tools/`).
2. `using SMC2FC: run_tempered_smc_gpu` →
    `using SMC2FC_functional: run_tempered_smc_gpu`.

This drop-in was first set up during Gate 8 of the audit. It has now
been re-synced from the latest `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`
so it picks up today's GPU PF perf fixes
(per-segment GPU-resident accumulator, OT-rescue default flipped to
0.0; see commits `da847d9` and `099a14a`).

## Why we expect bit-identical numerics

`SMC2FC_functional/src/Control/GPUControlSMC.jl` was a verbatim
COPY+DOC of `SMC2FC/src/Control/GPUControlSMC.jl`. The only diff
between the two versions of the file is the file-header block-comment
being converted to a Julia docstring; the `run_tempered_smc_gpu`
function body and every kernel under it are byte-identical. So at
matched config, matched seed, the two frameworks must produce the same
output, modulo JIT-warmup wall-time variance.

## Test setup

```bash
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia

# Original (uses SMC2FC):
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days 2 --replan-K 1 --N-smc 16 --K-per-chain 100 \
    --num-mcmc 2 --max-temp-levels 8 \
    --output-dir /tmp/migrate_orig

# Drop-in (uses SMC2FC_functional):
JULIA_LOAD_PATH="@:/home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC_functional:@stdlib" \
  julia --project=/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia \
  /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC_functional/benchmarks/gpu_drop_in/bench_dropin.jl \
    --T-days 2 --replan-K 1 --N-smc 16 --K-per-chain 100 \
    --num-mcmc 2 --max-temp-levels 8 \
    --output-dir /tmp/migrate_dropin
```

Both invocations use the same config (T=2d, seed=42 default, replan
every stride). Each runs 3 strides on RTX 5090.

## Result — bit-identical numerics, equivalent wall time

| Quantity | Original `SMC2FC` | Drop-in `SMC2FC_functional` | max\|diff\| |
|---|---|---|---|
| Wall time | **12.5 s** | **12.9 s** | +0.41 s (JIT-warmup variance) |
| Trajectory B (state, all bins) | — | — | **0.000e+00** |
| Trajectory F (state, all bins) | — | — | **0.000e+00** |
| Trajectory A (state, all bins) | — | — | **0.000e+00** |
| Applied Φ per bin (MPC schedule) | — | — | **0.000e+00** |
| Baseline trajectory (Φ=1 control) | — | — | **0.000e+00** |
| `tau_F` posterior median | 8.3546 | 8.3546 | **0.000e+00** |
| `B_inf` posterior median | 0.51203 | 0.51203 | **0.000e+00** |
| `F_inf` posterior median | 0.30335 | 0.30335 | **0.000e+00** |
| `lambda_A` posterior median | 0.80115 | 0.80115 | **0.000e+00** |
| `mu_0` posterior median | 0.017354 | 0.017354 | **0.000e+00** |
| `mu_B` posterior median | 0.23021 | 0.23021 | **0.000e+00** |
| `mu_F` posterior median | 0.094195 | 0.094195 | **0.000e+00** |
| `sigma_B` posterior median | 0.014462 | 0.014462 | **0.000e+00** |
| `sigma_F` posterior median | 0.010993 | 0.010993 | **0.000e+00** |
| `sigma_A` posterior median | 0.015929 | 0.015929 | **0.000e+00** |

Per-particle posterior cloud comparison (over **all** posterior
particles in **all** windows where the filter ran):

```
Posterior particle clouds bit-identical?  true
Trajectory MPC bit-identical?              true
```

i.e. for every `(stride, particle, parameter)` slot, the two runs
agree exactly.

## Recommendation

`SMC2FC_functional` is now ready to be the default Julia framework
going forward. The migration on the bench side is one line
(`using SMC2FC: run_tempered_smc_gpu` →
`using SMC2FC_functional: run_tempered_smc_gpu`). The numerics are
bit-identical to the original framework on the production GPU
SMC²-MPC pipeline at the matched config tested.

The original `SMC2FC` framework at `julia/SMC2FC/` should be retained
as a comparison reference (the
[`benchmarks/compare_three_libraries.jl`](../benchmarks/compare_three_libraries.jl)
harness depends on having both libraries reachable).

## Pointers

- Drop-in bench: [`benchmarks/gpu_drop_in/bench_dropin.jl`](../benchmarks/gpu_drop_in/bench_dropin.jl)
- Drop-in runner: [`benchmarks/gpu_drop_in/run_dropin.sh`](../benchmarks/gpu_drop_in/run_dropin.sh)
- Earlier audit Gate 8 (the same drop-in idea, with an older config):
  [`docs/GATE_RESULTS.md`](GATE_RESULTS.md) (Gate 8 section)
- Original v1.5 bench (still valid; uses `SMC2FC`):
  `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`
- Pointer in the comparison writeup: section 8.1 of
  `compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf`
