# `version_1_5_LEAN/` — FSA v1.5 Lean4 reference

Lean4 reference implementation of the FSA v1.5 model. Sister of
`../version_1_5_Julia/models/fsa_high_res/` — the same math, in two
languages, kept in lock-step by a differential-test harness.

Per the LEAN4-first charter
(`FSA_model_dev/LaTex_docs/lean4_first_charter.pdf`), Lean4 is the
formal source of truth; Julia is differentially tested at
1e-6 (single-step) / 1e-4 (integrated). Disagreement beyond tolerance
is by construction a Julia bug.

The Match.jl `@match` sites in the Julia source map line-for-line to
Lean4 `match … with` — that alignment is what makes the bridge
mechanical. Three sites in v1.5's non-GPU code:

| @match | Julia source | Lean4 |
|---|---|---|
| `params_v15_to_v1_nt` | `simulation.jl:110-143` | `Fsa.V15.Adapters.params_v15_to_v1` |
| `_reflect_unit` | `_plant.jl:57-61` | `Fsa.V15.Dynamics.reflect_unit` |
| `apply_prior` | (writeup §7.1) | `Fsa.V15.Estimation.apply_prior` |

## Layout

```
version_1_5_LEAN/
  lakefile.lean               # Mathlib-FREE Lake project
  lean-toolchain              # leanprover/lean4:v4.30.0-rc2
  Main.lean                   # JSON-line CLI bridge for the diff test
  Fsa.lean                    # umbrella
  Fsa/
    V15.lean                  # imports the V15 submodules
    V15/
      Types.lean              # PlantState, Obs, Params_v1, Params_v15,
                              #   ObsNoiseParams, EstimatedParams,
                              #   PinnedDynamics, ParamsForm, PriorKind
      Truth.lean              # TRUTH_PARAMS_V1, DEFAULT_PARAMS_V15,
                              #   DEFAULT_OBS_NOISE, DEFAULT_PINNED, INIT_STATE
      Dynamics.lean           # drift, diffusion_state_dep,
                              #   reflect_unit (★ @match #2), em_step_substepped
      Adapters.lean           # params_v15_to_v1 (★ @match #1), fill_pinned
      Plant.lean              # init_plant_state, plant_step, sample_obs_bfa
      Estimation.lean         # apply_prior (★ @match #3), obs_log_weight_one
```

## Build and test

```bash
# Build the Lean4 binary (no Mathlib pull; first build ~30s)
cd version_1_5_LEAN
lake build

# Smoke-test the binary
echo '{"fn":"reflectUnit","x":1.7}' | ./.lake/build/bin/fsa_v15_cli
# → {"x":0.300000}

# Run the Julia ↔ Lean4 differential test (121 cases at 1e-6)
cd ../version_1_5_Julia
julia --project=. diff_test/test_lean_diff_v15.jl
# → Test Summary:  | Pass  Total  Time
#     FSA v1.5 LEAN4 diff |  121    121  0.4s
```

## What's verified

| Function | Lean4 | Julia | Diff cases |
|---|---|---|---|
| `drift` | `Fsa.V15.drift` | `Dynamics.drift` | 5 random |
| `diffusion_state_dep` | `Fsa.V15.diffusion_state_dep` | `Dynamics.diffusion_state_dep` | 5 random |
| `reflect_unit` ★ | `Fsa.V15.reflect_unit` | `Plant._reflect_unit` | 10 boundary + 5 random |
| `em_step_substepped` | `Fsa.V15.em_step_substepped` | `Dynamics.em_step_substepped` | 5 random + noise |
| `params_v15_to_v1` ★ | `Fsa.V15.params_v15_to_v1_nt` | `Simulation.params_v15_to_v1_nt` | 14-field equality |
| `apply_prior` ★ | `Fsa.V15.apply_prior` | (logNormal/normal in `gpu_pf.jl` + `apply_prior`) | 7 logNormal + 5 normal |
| `plant_step` | `Fsa.V15.plant_step` | `Plant.plant_step` (noise pre-drawn) | 5 random |
| `obs_log_weight_one` | `Fsa.V15.obs_log_weight_one` | `Estimation.obs_log_weight` (per-particle) | 5 random |

Total: **121 cases, all green at 1e-6**.

## What's out of scope (still trusted as Julia)

- `gpu_control.jl` — fp32 KernelAbstractions cost kernel for the controller.
- `gpu_pf.jl` — segmented-PF GPU kernel.
- `tools/` — bench, FIM check, sanity test, smoke suite.

Per the writeup §7.2: "The GPU kernels (`propagate_segment_kernel!`,
`fsa_v1_cost_kernel!`) are NOT on the Lean4 port path. Formal verification
of fp32 GPU kernels is a separate, much harder problem."

## RNG handling — why we pass noise, not keys

Julia v1.5 derives sub-keys from a master `UInt64` via `hash((key, :obs))` /
`hash((key0, :step, k))`. Lean4 has no clean equivalent to Julia's `hash`,
so the diff test passes *pre-drawn standard-normal noise tuples* directly
to both implementations. Both stay deterministic and bit-identical for
the same noise input — the hash function never enters the diff-tested
math.

## Going forward

This is **bootstrap**: the v1.5 Julia exists first, and we translated it
to Lean4. From now on, per the charter, **Lean4 leads**: any new function
is written in Lean4 first, with the Julia version mechanically derived
via `@match` ↔ `match` translation.

Plan archive: `claude_plans/FSA_v1_5_LEAN4_port_2026-05-08_1210.md`.
