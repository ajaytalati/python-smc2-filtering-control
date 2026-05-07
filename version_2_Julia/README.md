# version_2_Julia — FSA-v2 (high-res, Banister-coupled) Julia port

Julia port of `version_2/models/fsa_high_res/` and the closed-loop SMC²-MPC bench.

The framework lives in `../julia/SMC2FC/`. This directory only contains the
**FSA-specific** glue: dynamics, simulation, plant, control, estimation, tests,
benches, and plotters. The framework is *not* modified.

## Layout

```
models/fsa_high_res/
  _dynamics.jl        — drift + state-dependent diffusion (G1-reparametrized)
  _phi_burst.jl       — daily Φ → sub-daily Gamma envelope
  simulation.jl       — INIT_STATE, EXOGENOUS, 4 obs samplers + circadian
  _plant.jl           — StepwisePlant for closed-loop MPC
  control.jl          — RBF schedule + cost functional
  estimation.jl       — priors + locally-guided propagate (Pitt-Shephard)
  FSAHighRes.jl       — module aggregator
tests/
  test_*.jl           — port of version_2/tests/test_e2_plant.py + new
tools/
  bench_smc_full_mpc_fsa.jl       — T=14d CPU driver
  bench_smc_full_mpc_fsa_gpu.jl   — T=14d GPU driver (Stage B, scaffold)
  plot_param_traces.jl            — reproduce E5 30-panel plot
  compare_to_python.jl            — quantitative gate (5% on identifiable params)
outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/
  experiment_run.md, manifest.json, data.jld2, *.png
```

## Reference Python source of truth

`version_2/models/fsa_high_res/{_dynamics,simulation,_plant,control,estimation}.py`.

## Conventions

- **G1 reparametrization**: TRUTH_PARAMS use the effective values (κ_B^eff, τ_F^eff,
  μ_F^eff, μ_0^eff, F_typ-centered curvature) per `_dynamics.py`.
- **fp32 inner / fp64 outer**: hot-loop SDE / particle propagation in fp32; SMC
  log-weights / posteriors / mass-matrix in fp64. Anything fp64 in inner loops is
  flagged in `experiment_run.md`.
- **Time-grid env var**: `FSA_STEP_MINUTES=60` is read at module-import time to
  set `BINS_PER_DAY=24`. The bench drivers parse `--step-minutes` BEFORE
  importing model files (mirror Python).

## Running

```bash
conda activate comfyenv     # for the Python compare side; Julia is independent
cd version_2_Julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=auto julia --project=. tools/bench_smc_full_mpc_fsa.jl --T-days 14 --step-minutes 60 --replan-K 2
```
