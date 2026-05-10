# Julia port: FSA-v2 model + T=14d MPC reproduction

> Archived from plan mode: 2026-05-07 09:13.

## Context

You want a Julia version of the FSA-v2 (high-res) model and its closed-loop SMC²-MPC bench, sitting next to the existing Python pipeline. The success bar is reproducing
`version_2/outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/E5_full_mpc_T14d_param_traces.png` from a Julia run, with both visual parity on the 30-panel posterior plot and posterior means within 5% on the six identifiable params (HR_base, S_base, β_C_HR, k_F, κ_B_HR, μ_step0).

This is a translation task. The Julia framework (`julia/SMC2FC/`) is in place: inner particle filter, the optimal-transport rescue, the outer tempered SMC², the Schrödinger–Föllmer bridge, and the four CPU samplers (HMC / NUTS / MALA / AutoMALA + a ChEES helper). What's missing is the FSA-specific glue (plant, estimation, tests, bench driver, plotter) and a GPU parallel-chains ChEES-HMC kernel for FSA.

The principle: hot inner loops run on the GPU in fp32. Outer accumulators (SMC log-weights, ESS, mass-matrix Cholesky, posterior particle clouds) stay in fp64. Anything fp64 in inner loops or CPU-bound on the hot path must be flagged.

## Decisions confirmed with you

- **OT rescue:** verify with a parity test before any other work. The framework already has Nyström + Sinkhorn + sigmoid blend in `julia/SMC2FC/src/Filtering/OT.jl`. The parity test runs the Julia OT path against the Python OT output on the same particles/weights and reports divergence. Only if it diverges materially do I touch OT.
- **No copy from `version_1_Julia/`:** rewrite `_dynamics.jl`, `simulation.jl`, `control.jl`, `FSAHighRes.jl` from scratch in `version_2_Julia/` against the Python source. Treat `version_1_Julia/models/fsa_high_res/` as a reference only.
- **Stage A + Stage B together:** CPU end-to-end pipeline first to get a posterior baseline, then GPU parallel-chains ChEES-HMC for FSA modelled on `version_1_Julia/models/bistable_controlled/gpu_pf.jl`. Both delivered before sign-off.
- **Acceptance:** both visual and quantitative.
  - Visual: posterior bands and means match on the 30-panel `E5_full_mpc_T14d_param_traces.png`.
  - Quantitative: posterior means within 5% relative error vs Python on the six identifiable params; the four E5 gates (mean A vs baseline ≥ 0.95, F-violation ≤ 5%, ID-coverage ≥ 24/27 windows, compute ≤ 4 h) pass.

## Repo layout

```
version_2_Julia/
├── Project.toml
├── models/fsa_high_res/
│   ├── FSAHighRes.jl
│   ├── _dynamics.jl
│   ├── simulation.jl
│   ├── _phi_burst.jl
│   ├── control.jl
│   ├── _plant.jl
│   ├── estimation.jl
│   └── gpu_pf.jl
├── tests/
│   ├── test_ot_parity.jl
│   ├── test_e2_plant.jl
│   ├── test_obs_consistency_fsa.jl
│   ├── test_estimation_smoke.jl
│   ├── test_gpu_pf_smoke.jl
│   └── test_julia_vs_python_filter.jl
├── tools/
│   ├── bench_smc_full_mpc_fsa.jl
│   ├── bench_smc_full_mpc_fsa_gpu.jl
│   ├── plot_param_traces.jl
│   ├── compare_to_python.jl
│   └── launchers/{run_t14d_cpu.sh,run_t14d_gpu.sh}
└── outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/
    ├── experiment_run.md
    ├── manifest.json
    ├── data.jld2
    ├── E5_full_mpc_T14d_traces.png
    └── E5_full_mpc_T14d_param_traces.png
```

## Step-by-step work order

### Phase 0 — OT parity test (do first)
1. Instrument Python `gk_dpf_v3_lite.py` once to dump fixture `(particles, log_weights, stochastic_indices, ε, n_iter, rank, anchor_idx, ot_output)`.
2. In Julia, load fixture, call `ot_resample_lr`, compare elementwise. Tolerance 1e-5.

### Phase 1 — `version_2_Julia` skeleton + `Project.toml`
Path-dep on `../julia/SMC2FC`. Deps: NPZ, JLD2, CUDA, KernelAbstractions, Plots, Distributions, JSON3.

### Phase 2 — Model files (fresh ports)
- `_dynamics.jl` — TRUTH_PARAMS, drift, Jacobi/CIR diffusion, em_step_substepped (fp32 inner).
- `simulation.jl` — INIT_STATE, EXOGENOUS, 4 obs samplers (HR/sleep/stress/steps), C(t), simulate_em.
- `_phi_burst.jl` — Gamma-burst expansion (daily Φ → sub-daily; peak ~10:00, zero overnight).
- `control.jl` — 8-anchor RBF, schedule_from_theta_fsa, build_control, cost.
- `FSAHighRes.jl` — aggregator.

### Phase 3 — `_plant.jl` (StepwisePlant)
`advance!`, `finalise` writing manifest.json + npz files matching Python format.

### Phase 4 — `estimation.jl` (~600 LOC)
- Prior config (30 estimated + 5 frozen).
- align_obs_fn → Float32.
- propagate_fn = locally-guided sequential-scalar Kalman fusion (HR / stress / log_steps), Cholesky sample, return pred_lw.
- obs_log_weight_fn = Bernoulli sleep + residual.
- build_estimation_model() → EstimationModel.

### Phase 5 — Tests
test_ot_parity, test_e2_plant (5 cases), test_obs_consistency_fsa, test_estimation_smoke, test_julia_vs_python_filter.

### Phase 6 — CPU bench
T_total=14d, step_minutes=60, BINS_PER_DAY=24, WINDOW=24, STRIDE=12, n_strides=27, replan_K=2, n_replans=13, N_SMC=1024, N_PF=800, target_ess_frac=0.5, SF bridge, filter HMC (0.025/8), control HMC (0.2/16), seeds=42.

### Phase 7 — GPU bench
gpu_pf.jl mirroring bistable B3 pattern. ChEES-L per tempering level. fp32 inner / fp64 outer.

### Phase 8 — Plot + compare
30-panel param-trace; compare_to_python.jl tabulates posterior means, 5% gate.

### Phase 9 — Run, log, ship
For each run: execute, plot, compare, experiment_run.md (timestamp, git SHA, command, params, wall-time, gates, errors). Flag any inner-loop fp64.

## Verification
- Phase 0 gate: ot_parity within 1e-5.
- Unit gate: all six test files green.
- CPU end-to-end gate: 4 E5 gates pass, posterior means within 5%, visual match.
- GPU end-to-end gate: same gates, no inner-loop fp64, total wall-time < CPU.

## What I will NOT do without flagging
- Modify files outside `version_2_Julia/`. Framework only changes if Phase 0 surfaces a bug.
- Pad time. Surface blockers.
- Silently leave fp64 in inner GPU kernels.
