# `version_3_julia/` — FSA-v5 Julia port

Julia port of `version_3/models/fsa_v5/` + `version_3/tools/` +
`version_3/tests/`. Sits next to the Python original; the two evolve
in parallel under the differential-test harness.

This package implements the FSA-v5 model on top of the
[`SMC2FC.jl`](../julia/SMC2FC) framework. Per the joint LEAN4-and-Julia
charter (`LaTex_docs/lean4_first_charter.pdf` Part II), Julia replaces
Python as the production stack for FSA-v5; the LEAN4 native binary at
`FSA_model_dev/lean/Fsa/V5/` remains the formal source of truth and
the differential-test oracle.

## Layout

```
version_3_julia/
├── Project.toml         # 11 deps; SMC2FC path-dev to ../julia/SMC2FC
├── src/
│   ├── FSAv5.jl         # umbrella module
│   └── FSAv5/
│       ├── Types.jl           # FSAv5State, BimodalPhi, DynParams, ObsParams
│       ├── DefaultParams.jl   # TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5,
│       │                       #   DEFAULT_INIT, SEDENTARY_INIT
│       ├── Dynamics.jl        # drift + diffusion (bit-equivalent to Python)
│       ├── Cost.jl            # mu_bar, find_a_sep, a_sep_grid (Bug 2 fix)
│       ├── Schedule.jl        # RBF schedule decoder
│       ├── PhiBurst.jl        # Gamma-pulse Phi envelope (BINS_PER_DAY=96)
│       ├── Plant.jl           # StepwisePlant (mutable; advance!)
│       ├── Obs.jl             # 5 channels: HR, sleep, stress, steps, VL
│       └── Estimation.jl      # PARAM_PRIORS_V5 (37) + EstimationModel
├── tools/
│   ├── _RunDir.jl              # atomic run-dir allocator (TOCTOU-safe)
│   ├── SummarizeRun.jl         # markdown summary of manifest.json
│   ├── DiagnoseViolationRate.jl  # 3-formulation violation diagnostic
│   ├── BenchSmcFilterOnly.jl   # Stage 1 driver
│   ├── BenchControllerOnly.jl  # Stage 2 driver
│   ├── BenchSmcFullMpc.jl      # Stage 3 driver (closed-loop MPC)
│   ├── PlotBasinOverlay.jl     # Φ_B / Φ_S basin overlay
│   └── PlotStage2ThetaTraces.jl  # 16-anchor θ trace panels
├── test/
│   ├── runtests.jl              # Pkg.test() entrypoint
│   ├── test_param_dict.jl       # 24 tests
│   ├── test_smoke.jl
│   ├── test_per_particle_separator.jl  # 9 tests; Bug 2 regression
│   ├── test_reconciliation.jl
│   ├── test_run_dir.jl          # 9 tests
│   ├── test_lean_diff.jl        # 40 tests; LEAN4 native binary @ 1e-6
│   └── test_python_diff.jl      # 38 tests; PythonCall in-process @ 1e-6
└── benchmarks/                  # (reserved for benchmark scripts)
```

## Quickstart

The package uses `comfyenv` for Python and the Julia toolchain pinned
in `Project.toml`. Run from this directory.

```bash
cd version_3_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # one-time
julia --project=. -e 'using Pkg; Pkg.test()'          # full suite
```

Expected: `169 / 169` tests pass in ~15 s on CPU.

## Differential testing — what's verified

Two oracles run on every test invocation (charter §15.7):

1. **LEAN4 native binary** at `FSA_model_dev/lean/Fsa/V5/build/`. The
   harness in `test/test_lean_diff.jl` opens a bidirectional pipe to
   the binary and round-trips every drift / `mu_bar` / `find_a_sep` call
   with random inputs in the physiological-bound box. Tolerance: 1e-6
   single-step, 1e-4 integrated. The binary is built by `lake build`
   from the LEAN4 spec — same file is both spec and reference impl.
2. **Python implementation** at `version_3/models/fsa_v5/`. The harness
   in `test/test_python_diff.jl` uses
   [`PythonCall.jl`](https://juliapack.github.io/PythonCall.jl/) to
   call the Python module directly in-process, no subprocess overhead.
   Tolerance: 1e-6.

Disagreement beyond tolerance is by construction a bug in the Julia
port (vs LEAN4) or in the Python port (vs LEAN4) — never in LEAN4.

## What's done

| Phase | Item | Status |
|-------|------|--------|
| 1 | `Project.toml` + `FSAv5.jl` umbrella | done |
| 2 | `Types.jl` (`FSAv5State`, `BimodalPhi`, `DynParams`, `ObsParams`) | done |
| 3 | `Dynamics.jl` + `Cost.jl` + `Schedule.jl` | done; bit-equivalent to Python |
| 4 | `PhiBurst.jl` + `Plant.jl` | done |
| 5 | `Obs.jl` (5 channels) | done |
| 6 | LEAN4 + Python differential tests (78 total) | done; all green at 1e-6 |
| 7 | `Estimation.jl` glue (37 priors, 14 frozen) | done |
| 8 | Tools: `_RunDir`, `SummarizeRun`, `DiagnoseViolationRate`, 3 bench skeletons, 2 plot tools | done |
| 9 | Test orchestrator (`runtests.jl`) | done; 169 tests green |

## What's deferred

The three bench drivers (`BenchSmcFilterOnly`, `BenchControllerOnly`,
`BenchSmcFullMpc`) are scaffolds: the synthetic-data plant path is
wired through `FSAv5.StepwisePlant` and verified end-to-end (see
`BenchSmcFilterOnly.simulate_synthetic_full` — 1344 bins runs cleanly),
but the GPU-side SMC² loops still need to be plumbed through the
`SMC2FC.run_smc_window` (filter) and `SMC2FC.run_tempered_smc_loop`
(controller) entry points. That integration is the closing item in
charter §15.7 and is its own separate piece of work.

The 5 observation channels in `Obs.jl` are the deterministic-mean
versions; obs samplers (Gaussian + Bernoulli noise on top of those
means) are left for a follow-up pass once the bench drivers exercise
them — same pattern as the existing `swat_model_factory/` separation.

## Known gotchas

* **`sigma_S` collision** (charter Bug 1): the Python port's `params`
  dict had two `sigma_S` keys; the second silently won. The Julia
  port carries two distinct `ComponentVector` instances:
  `DynParams.sigma_S` (latent-S Jacobi diffusion scale, ≈ 0.008) and
  `ObsParams.sigma_S_obs` (stress-channel obs noise, ≈ 4.0). They
  cannot collide.

* **Particle-0 separator template** (charter Bug 2): Python collapsed
  the SMC² ensemble to particle 0 before computing `A_sep`. Julia's
  `a_sep_grid(particles, schedule)` returns a `Matrix{Float64}` of
  shape `(n_particles, n_steps)` — the buggy 1-D collapse is a type
  error.

## Plan archive

The full per-function plan with rationale lives at
`claude_plans/Julia_port_FSA_v5_model_plan_2026-05-06_1921.md`.
