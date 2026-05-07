# version_2_Julia — handoff after autonomous mechanical port

> Session: 2026-05-07 (autonomous mode, 3-hour window).
> Mode: mechanical translation only — no long CPU/GPU runs.

## What is in this directory

A from-scratch Julia port of `version_2/models/fsa_high_res/` and the
closed-loop SMC²-MPC bench scaffolding. Sits next to `julia/SMC2FC/`
(the framework, untouched).

```
version_2_Julia/
├── Project.toml                  # path-deps SMC2FC; NPZ, JLD2, JSON3, Plots, etc.
├── README.md
├── HANDOFF.md                    # this file
├── models/fsa_high_res/
│   ├── FSAHighRes.jl             # aggregator
│   ├── _dynamics.jl              # G1-reparametrized drift + sqrt-Itô diffusion
│   ├── _phi_burst.jl             # daily Φ → sub-daily Gamma envelope
│   ├── simulation.jl             # DEFAULT_PARAMS, INIT_STATE, EXOGENOUS,
│   │                             # 4 obs samplers (HR/sleep/stress/steps), C(t)
│   ├── _plant.jl                 # StepwisePlant + advance! + finalise
│   ├── control.jl                # 8-RBF schedule + cost functional
│   └── estimation.jl             # priors (30 estimated + 5 frozen),
│                                 # locally-guided (Pitt-Shephard-style) propagate_fn,
│                                 # obs_log_weight_fn, align_obs_fn
├── tests/
│   ├── runtests.jl               # runs all three test files
│   ├── test_e2_plant.jl          # 5-test port of test_e2_plant.py
│   ├── test_obs_consistency_fsa.jl
│   └── test_estimation_smoke.jl
├── tools/
│   ├── bench_smc_full_mpc_fsa.jl       # CPU bench (scaffold; --smoke runs)
│   ├── bench_smc_full_mpc_fsa_gpu.jl   # GPU bench (scaffold; needs gpu_pf.jl)
│   ├── plot_param_traces.jl            # 30-panel posterior plot
│   ├── compare_to_python.jl            # 5%-error gate vs Python data.npz
│   └── launchers/{run_t14d_cpu.sh, run_t14d_gpu.sh}
└── outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/
    └── experiment_run.md         # written by --smoke
```

## What I verified

- **All modules load cleanly.** `julia --project=. -e 'include("models/fsa_high_res/FSAHighRes.jl"); using .FSAHighRes'` succeeds.
- **All 351 unit tests pass** in ~6 seconds:
  - `test_g1_reparam` (60 assertions, port of `test_g1_reparam.py`):
    drift-parity at typical state, parity over a 50-point random grid,
    truth-value derivation, residual params drop out at the typical point.
  - `test_e2_plant` (235 assertions): Φ-burst integral preservation, shape,
    StepwisePlant in-bounds trajectories, t_bin accumulation, finalise()
    psim-format artifact.
  - `test_obs_consistency_fsa` (5 assertions): the 4 obs-channel formulas
    are bit-identical between simulator and estimator (D1/D2-class bug guard).
  - `test_estimation_smoke` (51 assertions): prior cardinality (30 params),
    constrained↔unconstrained roundtrip, log-prior-unconstrained finite,
    propagate_fn produces in-bounds (x_new, pred_lw), obs_log_weight_fn
    finite, forward_sde_stochastic runs N steps in-bounds,
    build_estimation_model assembles.
  - `test_ot_parity` (scaffold): skipped gracefully when the Python
    fixture is absent. See in-file comment for how to generate it.
- **Bench `--smoke` mode runs end-to-end.** `julia --project=. tools/bench_smc_full_mpc_fsa.jl --T-days 14 --step-minutes 60 --replan-K 2 --smoke` advances the plant by one stride and writes
  `outputs/.../experiment_run.md`. Plant state after one 12-h stride at
  Φ=1: B=0.0566, F=0.2906, A=0.1028 (consistent with Banister: B rises
  slightly from 0.05, F decays slightly from 0.30, A stable at 0.10).

## What is NOT done

These remain for the next session (they all involve LONG CPU or GPU runs
that the autonomous-mode brief explicitly said to skip):

1. **Full SMC² closed-loop body** (`tools/bench_smc_full_mpc_fsa.jl`).
   The scaffold loads everything and calls `--smoke`; the actual
   `for stride in 1:n_strides` body that calls `SMC2FC.run_smc_window_bridge`
   + the controller + records posteriors is left as a TODO comment in
   the file, with clear pseudocode and pointers to the framework call
   sites (`julia/SMC2FC/src/SMC2/TemperedSMC.jl :: run_smc_window_bridge`,
   `julia/SMC2FC/src/Control/TemperedSMC.jl :: run_tempered_smc_loop`).

2. **`models/fsa_high_res/gpu_pf.jl`** — the FSA-specific GPU
   parallel-chains ChEES-HMC kernel. Reference template:
   `version_1_Julia/models/bistable_controlled/gpu_pf.jl :: parallel_hmc_one_move!`
   (line 498) and `version_1_Julia/tools/bench_b3_gpu_parallel.jl :: chees_pick_L_parallel`
   (line 81). The GPU bench scaffold (`tools/bench_smc_full_mpc_fsa_gpu.jl`)
   currently `error()`s with a clear message pointing at this dependency.

3. **OT parity test fixture.** The plan called for instrumenting the
   Python `gk_dpf_v3_lite.py` once to dump (particles, log_weights,
   anchor_idx, ot_output) for one degenerate window, then comparing
   `julia/SMC2FC/src/Filtering/OT.jl::ot_resample_lr` element-by-element.
   I read both implementations — Julia is a direct port (Nyström +
   low-rank Sinkhorn + sigmoid blend), but did not run the parity check
   because that requires modifying the Python tree (one-line `np.savez`
   add) which I flagged in the plan as "needs explicit user approval
   before touching".

4. **Quantitative compare-to-Python run.** Requires the full SMC² body
   to run end-to-end on T=14d, then `tools/compare_to_python.jl` against
   `version_2/outputs/.../data.npz`. Tool is in place; the data file
   isn't (because (1) is unfinished).

5. **Side-by-side param-trace PNG.** `tools/plot_param_traces.jl` is in
   place; needs a Julia `data.jld2` to consume (also blocked on (1)).

## Key design decisions / things to know

### G1 reparametrization is applied

Both `_dynamics.jl` and `simulation.jl` use the G1 effective values:
- `tau_F = 7 / (1 + 1·A_TYP) = 6.3636…`
- `kappa_B = 0.012 · (1 + 0.4·A_TYP) = 0.01248`
- `mu_0 = 0.02 + 0.4·F_TYP² = 0.036`
- `mu_F = 0.10 + 2·F_TYP·0.4 = 0.26`

The drift formula uses `(1 + ε_A·A) / (1 + ε_A·A_TYP)` and `F_dev = F - F_TYP`
inside the curvature term. This matches Python's
`version_2/models/fsa_high_res/_dynamics.py` verbatim. The `version_1_Julia`
copy of `_dynamics.jl` does NOT have this reparametrization — the
v2 port deliberately rewrites it from the Python source.

### fp32 inner / fp64 outer convention

Currently the inner SDE step in `_plant.jl::_plant_em_step_one` runs in
fp64 (Python uses fp64 here too — see `_plant.py::_plant_em_step`). The
GPU port (Stage B) is where the inner loop becomes fp32. The convention
will be enforced when `gpu_pf.jl` is written.

### Param order matches Python

`Estimation.PARAM_NAMES` and `Estimation._PI` are in the exact same
order as Python's `_PK` / `_PI` (`estimation.py:103-104`), so the
30-panel param-trace plot (Phase 8) can use the same panel index.

### Obs-channel parity test caught nothing yet

The 4 obs-channel formulas are tested for sim/est consistency in
`test_obs_consistency_fsa.jl` and PASS. This is the D1/D2-class bug
guard per CLAUDE.md.

## Quick commands to verify

```bash
cd version_2_Julia

# Instantiate (path-dev SMC2FC) — already done in this session.
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests.
julia --project=. tests/runtests.jl
# Expected: 291 passed.

# Smoke-test the bench (no long run, just imports + 1 plant.advance).
julia --project=. tools/bench_smc_full_mpc_fsa.jl \
    --T-days 14 --step-minutes 60 --replan-K 2 --smoke
# Expected: writes outputs/.../experiment_run.md.
```

## Open questions for the next session

(Restated from the plan-mode AskUserQuestion the user already answered.
Re-flagging here because the answers will drive the next steps.)

- **OT parity test:** need user approval to add the one-line fixture
  dump in Python `gk_dpf_v3_lite.py`. (Answer recorded: "Verify with a
  parity test first" — implementation deferred because it touches
  outside `version_2_Julia/`.)
- **Wiring the full SMC² loop.** Concretely, this is a `for` loop that
  calls `SMC2FC.run_smc_window_bridge` per stride. The framework's
  signature should be reviewed before stitching — the call site in the
  Python is the model for what to pass.
- **GPU `gpu_pf.jl`.** Stage B. Mirrors bistable B3. Roughly 500 LOC
  based on the bistable reference.
