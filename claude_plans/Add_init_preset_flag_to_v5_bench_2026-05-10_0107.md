# Add `--init-preset` flag to v5 bench (TRAINED_ATHLETE_INIT default, DEFAULT_INIT optional)

> Archived from plan mode: 2026-05-10 01:07.

## Context

Today, switching the v5 closed-loop bench from the deconditioned baseline (`DEFAULT_INIT`, tech guide §9.10) to the trained-athlete reference (`TRAINED_ATHLETE_INIT`, §8.1) requires hand-editing the bench source. The launcher header at [run_julia_v5_T42d.sh:21-29](../version_1_5_Julia/tools_v5/launchers/run_julia_v5_T42d.sh#L21-L29) literally tells the user to "switch the call inside `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl` from `init_plant_state_v5()` to `init_plant_state_trained()`".

This is friction: every preset switch is a source edit + risk of forgetting to revert. The two init functions and both named-tuple constants are already defined and exported; the only missing piece is a CLI flag that picks between them and threads the choice through every place that reaches for `DEFAULT_INIT`.

Goal: add `--init-preset {TRAINED_ATHLETE_INIT|DEFAULT_INIT}` to the bench (flag value is the literal constant name, no abstract labels), thread it through every site that depends on the patient's starting state, and update the launcher header + manifest line to reflect the new flag.

**Default flips from `DEFAULT_INIT` to `TRAINED_ATHLETE_INIT`** at the user's request. This is a behaviour change at the no-flag invocation — every prior bench run from this script used `DEFAULT_INIT`. To reproduce a pre-change baseline, you now have to pass `--init-preset DEFAULT_INIT` explicitly.

## Scope: every site that hardcodes the init

Verified via:

- `grep DEFAULT_INIT|TRAINED_ATHLETE_INIT|init_plant_state_v5|init_plant_state_trained` over the whole `version_1_5_Julia/` tree.
- `grep init_state =` and `grep last_xhat|xhat_init|prior_mean` to catch init-by-another-name.
- Same names searched against the framework dir `julia/SMC2FC_functional/` (only refs are inside the high-res `bench_dropin.jl`, a different model — out of scope).
- A literal-numeric search for the `DEFAULT_INIT` tuple `(0.05, 0.10, 0.30, 0.10, 0.030, 0.050)` to catch any site that hardcodes the values without using the symbol — none found.
- Inspected the bench's three included submodules end-to-end (`bench_glue_v5.jl`, `bench_controller.jl`, `bench_loop.jl`) and the postprocessor (`bench_postproc.jl`).

**Must follow the preset (5 sites):**

1. `bench_smc_full_mpc_fsa_v5_gpu.jl:201` — `base_plant_state = init_plant_state_v5()` (pre-roll baseline plant).
2. `bench_smc_full_mpc_fsa_v5_gpu.jl:211-215` — open-loop `s0` SVector built from `DEFAULT_INIT.{B,S,F,A,KFB,KFS}` (controller's open-loop initial plan).
3. `bench_smc_full_mpc_fsa_v5_gpu.jl:232` — `plant_state = init_plant_state_v5()` (closed-loop foldl initial plant).
4. `bench_smc_full_mpc_fsa_v5_gpu.jl:251-254` — `last_xhat` SVector from `DEFAULT_INIT.{...}` (filter's initial point estimate).
5. `bench_glue_v5.jl:107-115` — `window_grid_obs_v5` first-window fallback. Critical: with `--init-preset TRAINED_ATHLETE_INIT` and this site unchanged, the first window's `init_state` handed to `gpu_log_density_v5` would be the deconditioned tuple while the plant trajectory started trained.

**Cosmetic (still update for clarity):**

6. `bench_smc_full_mpc_fsa_v5_gpu.jl:23` — module-header docstring `"--open-loop true ... one up-front plan from DEFAULT_INIT+TRUTH_PARAMS_V5"`.
7. `bench_smc_full_mpc_fsa_v5_gpu.jl:210` — `@info "OPEN-LOOP: building one initial plan from DEFAULT_INIT+TRUTH_PARAMS_V5..."`.

**Also follows the preset (NaN-guard tracks the preset, per design decision):**

- `gpu_pf_v5.jl:213-221` — NaN-guard inside the JIT'd inner-PF kernel (tech guide §6.4). Verified the controller-side kernel (`gpu_control_v5.jl`) does NOT have a sister `isfinite` branch, so scope is just this one site + its plumbing.

**Confirmed NOT in scope:** `gpu_control_v5.jl:347-349` (function default, never fires from bench); `_plant_v5.jl:194-210` (the wrappers themselves); `tools_v5/bench/bench_controller.jl:60` (pure pass-through); `tools_v5/bench/bench_loop.jl` (threads through); `tools_v5/bench/bench_postproc.jl` (passive read); `tools_v5/bench/bench_filter.jl` (uses `prior_means` from param prior config); `diff_test/test_lean_diff_v5.jl` (separate Lean-vs-Julia test); `FSAv5.jl:44, 54` (exports only); framework `SMC2FC_functional/` (high-res only).

`A_TYP` and `F_TYP` are physical-model reference constants ("typical autonomic amplitude", "typical fatigue") used inside dynamics/cost kernels for parameter shaping. NOT patient initial state, NOT in scope.

## Files to modify

1. `version_1_5_Julia/tools_v5/bench/bench_args.jl` — add `--init-preset` flag.
2. `version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl` — import `TRAINED_ATHLETE_INIT`, dispatch to init function/tuple from the flag, replace 4 active sites + 2 cosmetic strings, pass selected tuple into the glue and target.
3. `version_1_5_Julia/models/fsa_v5/bench_glue_v5.jl` — give `window_grid_obs_v5` an `init_nt::NamedTuple = DEFAULT_INIT` keyword arg; use it in the `t0_bin == 0` branch.
4. `version_1_5_Julia/tools_v5/launchers/run_julia_v5_T42d.sh` — replace manual-edit instructions in the header with the new flag; update the manifest "Init:" line.
5. `version_1_5_Julia/tools_v5/launchers/run_julia_v5_T7d.sh` — same launcher updates.
6. `version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl` — add 6 NaN-fallback Float32 fields to `FSAv5GPUTarget` and 6 args to the kernel signature; replace `Float32(DEFAULT_INIT.{...})` in the NaN-guard branch with the kernel args. Mirror the existing `A_TYP_f32 / F_TYP_f32` plumbing.

## Concrete changes

### Change 1 — `bench/bench_args.jl`

Add one entry to `defaults`:

```julia
# Plant initial-state preset. The flag value is the literal name of
# the named-tuple constant in `models/fsa_v5/simulation_v5.jl`.
#
#   "TRAINED_ATHLETE_INIT" — const at `models/fsa_v5/simulation_v5.jl:167-174`
#                            (tech guide §8.1: "canonical test-scenario
#                             starting point"). DEFAULT.
#   "DEFAULT_INIT"         — const at `models/fsa_v5/simulation_v5.jl:158-165`
#                            (tech guide §9.10: "deconditioned but
#                             otherwise healthy"). Forward-sim from this
#                             under moderate Φ takes weeks to settle.
"init-preset"     => "TRAINED_ATHLETE_INIT",
```

### Change 2 — `bench_smc_full_mpc_fsa_v5_gpu.jl`

- Imports: extend the `SimulationV5` `using` block to include `TRAINED_ATHLETE_INIT`.
- Dispatch (just before the pre-roll block):

```julia
init_preset = string(args["init-preset"])
init_state_fn, init_nt = if init_preset == "TRAINED_ATHLETE_INIT"
    (init_plant_state_trained, TRAINED_ATHLETE_INIT)
elseif init_preset == "DEFAULT_INIT"
    (init_plant_state_v5, DEFAULT_INIT)
else
    error("--init-preset must be \"TRAINED_ATHLETE_INIT\" or " *
          "\"DEFAULT_INIT\", got: \"$init_preset\"")
end
@info "init preset: $init_preset"
```

- Replace 4 active sites with `init_state_fn()` / `init_nt.{B,S,...}`.
- Pass `init_nt = init_nt` to `window_grid_obs_v5` calls.
- Pass 6 `nan_fallback_*` kwargs to the `FSAv5GPUTarget(...)` call at line 132.
- Update cosmetic strings at line 23 and line 210.

### Change 3 — `bench_glue_v5.jl`

Add `init_nt::NamedTuple = DEFAULT_INIT` kwarg on `window_grid_obs_v5`; use it in the `t0_bin == 0` branch.

### Change 4 — launcher updates

In both `run_julia_v5_T42d.sh` and `run_julia_v5_T7d.sh`:

```bash
# Initial state used by the bench:
#   The plant defaults to `TRAINED_ATHLETE_INIT` (tech guide §8.1),
#   defined in `models/fsa_v5/simulation_v5.jl:167-174`. To start from
#   the deconditioned baseline `DEFAULT_INIT` (tech guide §9.10) defined
#   in `models/fsa_v5/simulation_v5.jl:158-165` instead — note that
#   forward-sim from DEFAULT_INIT under moderate Φ takes weeks to settle:
#
#       ./run_julia_v5_T42d.sh 42 --init-preset DEFAULT_INIT
```

Manifest line:

```bash
echo "Init:    TRAINED_ATHLETE_INIT (default; pass --init-preset DEFAULT_INIT to override)"
```

### Change 5 — `gpu_pf_v5.jl` (NaN-guard follows preset)

Mirror the `A_TYP_f32 / F_TYP_f32` plumbing already in this file:

- 6 `nan_fallback_{B,S,F,A,KFB,KFS}::Float32` fields on `FSAv5GPUTarget`.
- 6 kwargs on the constructor with `Float32(DEFAULT_INIT.{...})` defaults (back-compat for any caller outside the bench).
- 6 `nan_*::Float32` args on the kernel function signature.
- Replace `Float32(DEFAULT_INIT.{...})` in the NaN-guard branch with the kernel args.
- Pass `target.nan_fallback_*` at every kernel launch site (verify via grep; lines ~389 and ~680 may both be launches).
- Bench passes `nan_fallback_* = Float32(init_nt.{...})` when constructing the target.

## Verification

1. Default threading (no flag): expect `@info "init preset: TRAINED_ATHLETE_INIT"`; `traj_history[1, :]` matches `(0.50, 0.45, 0.20, 0.45, 0.06, 0.07)`.
2. Reproduce pre-change baseline with `--init-preset DEFAULT_INIT`: `traj_history[1, :]` matches `(0.05, 0.10, 0.30, 0.10, 0.030, 0.050)`.
3. Bad value (`--init-preset trained`) errors cleanly with the validation message.
4. Mixed flags still work.
5. `target.nan_fallback_B == Float32(init_nt.B)` `@assert` near the dispatch line.

## Out of scope

- More presets beyond the two existing constants.
- v1.5 (`tools/`) bench.
- Renaming the wrapper functions.
- Plotting changes.
