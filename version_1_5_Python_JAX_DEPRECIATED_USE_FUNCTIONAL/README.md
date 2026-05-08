# `version_1_5_Python_JAX/` — FSA v1.5 Python+JAX

Cross-validation Python+JAX implementation of FSA v1.5. Sister of
`../version_1_5_Julia/models/fsa_high_res/` — same math, three
languages (Lean4, Julia, Python+JAX) kept in lock-step by differential
tests.

**Purpose**: third independent implementation of the v1.5 model so the
Julia version (the production stack) can be cross-validated against
both Lean4 (formal source of truth) and Python+JAX (XLA-compiled GPU
backend).

The model files are mechanical translations of v1.5 Julia. The
closed-loop bench mirrors v2's `bench_smc_full_mpc_fsa.py`
structurally, swapped onto v1.5's reduced surface (3-state, 1 Φ,
3-channel direct-Gaussian obs, pinned obs noise).

## Layout

```
version_1_5_Python_JAX/
  README.md
  models/fsa_high_res/
    __init__.py
    _dynamics.py     # drift_jax, diffusion_state_dep, em_step_substepped
    simulation.py    # DEFAULT_PARAMS, INIT_STATE, PINNED_PARAMS,
                     #   params_v15_to_v1, fill_pinned, sample_obs_bfa
    _plant.py        # PlantState, plant_step, plant_rollout (purely-functional)
    estimation.py    # PARAM_PRIOR_CONFIG (10 lognormal), propagate, obs_log_weight,
                     #   HIGH_RES_FSA_V15_ESTIMATION (smc2fc EstimationModel)
    control.py       # RBF schedule, cost (∫A − λ_F·F-barrier),
                     #   build_control_spec → smc2fc ControlSpec
  diff_test/
    diff_main.py     # JSON-line CLI bridge — Julia driver pipes here
  tools/
    bench_smc_full_mpc_fsa_v15.py   # closed-loop SMC²-MPC bench
```

## Quickstart

Use the conda env `comfyenv` (already has JAX, BlackJAX, smc2fc deps).

```bash
# Activate
conda activate comfyenv

# Run the diff test from the Julia side (it pipes to diff_main.py)
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
julia --project=. diff_test/test_python_diff_v15.jl
# → Test Summary:                    | Pass  Total  Time
#     FSA v1.5 Julia ↔ Python+JAX diff |  121    121  2.0s

# Smoke-test the closed-loop bench (1 stride, ~2 s on GPU)
cd ../version_1_5_Python_JAX
JAX_ENABLE_X64=True PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v15.py --smoke
```

## What's verified

The Julia ↔ Python+JAX differential test exercises 121 randomised
cases at **1e-6 single-step / 1e-4 integrated** tolerance. Coverage
identical to the Lean4 ↔ Julia diff test:

| Function | Cases |
|---|---|
| `drift_jax` | 5 random |
| `diffusion_state_dep` | 5 random |
| `em_step_substepped` | 5 random + noise |
| `params_v15_to_v1` ★ | 14-field equality |
| `_reflect_unit` ★ | 10 boundary + 5 random |
| `apply_prior` ★ | 7 logNormal saturation + 5 normal |
| `plant_step` | 5 random (full B/F/A/obs round-trip) |
| `obs_log_weight_one` | 5 random |

(★ = explicit Match.jl `@match` site in the Julia source — the
mechanical Lean4-port bridge. Python uses `isinstance` dispatch /
`if/else` / `match-case` instead, but the result must be bit-equal.)

## Triangulation

| Pair | Tolerance | Status |
|---|---|---|
| Lean4 ↔ Julia | 1e-6 | green (`version_1_5_LEAN/diff_test`) |
| Julia ↔ Python+JAX | 1e-6 | green (this dir) |
| Lean4 ↔ Python+JAX | follows transitively | implied |

Any pair disagreeing beyond tolerance points to a bug in the
implementation that's the outlier in three.

## Closed-loop bench (`tools/bench_smc_full_mpc_fsa_v15.py`)

Mirrors v2's `bench_smc_full_mpc_fsa.py` structurally:

- Plant rollout uses pure `plant_rollout` (functional, matching Julia v1.5).
- Filter uses `smc2fc.core.jax_native_smc.run_smc_window_native` +
  `make_gk_dpf_v3_lite_log_density_compileonce` (compile-once factory).
- Controller uses `smc2fc.control.tempered_smc_loop.run_tempered_smc_loop_native`
  with a `ControlSpec` from `models.fsa_high_res.control.build_control_spec`.
- Per stride: plant.advance → accumulate obs → (every K) filter window →
  (every K) replan controller → splice new daily Φ plan.

`--smoke` runs 1 stride only as an end-to-end import + first-call
smoke test. Realistic 14-day runs need larger N_smc / K_pf (defaults
are deliberately small for the smoke).

## Out of scope

- Performance tuning (XLA/GPU). v1.5 Julia uses KernelAbstractions for
  the inner-PF + cost kernels; Python+JAX gets equivalent throughput
  via `jax.jit` + `vmap` over the framework's compile-once path.
- Convergence analysis on the closed-loop bench. The bench scaffolds
  the integration; the realistic 14- or 28-day MPC run is a follow-up.
- Filter / controller `acceptance_gates`. Mirroring v1's gate
  scaffolding is straightforward but not needed for cross-validation.

Plan archive: `claude_plans/FSA_v1_5_LEAN4_port_2026-05-08_1210.md`
(the Lean4 port plan; Python+JAX adopts the same goals applied to
a third implementation).
