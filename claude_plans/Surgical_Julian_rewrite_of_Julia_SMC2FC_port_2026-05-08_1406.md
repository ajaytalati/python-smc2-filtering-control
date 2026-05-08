# Surgical, Julian rewrite of the Julia SMC2FC port — as a new sibling directory

> Archived from plan mode: 2026-05-08 14:06.

## Context

You have two parallel libraries:

- `smc2fc/` (Python + JAX) — original. Already mostly functional in shape because JAX requires it (frozen dataclasses, `jax.lax.scan`, closure factories, `jax.tree_util.Partial` for compile-once binding).
- `julia/SMC2FC/` (Julia) — earlier agent port. Not asked to be functional; not specifically tuned to Julia idioms.

You want a **new, parallel Julia library** written:
1. With a functional API surface (caller-visible mutation forbidden; pre-allocated buffers allowed inside hot loops).
2. More idiomatic Julian (multiple dispatch, immutability, dispatch-driven specialization, fewer per-call allocations).
3. With **Google-style docstrings** on every file and every function.
4. **Without touching the existing `julia/SMC2FC/`** — so you can run the new library, the existing library, and the Python library side-by-side and compare.

Your three earlier choices, baked in:
- **Style**: functional API, performant interior.
- **Scope**: surgical / targeted.
- **Numerics**: track the Python `smc2fc` reference; not required to match the existing Julia port bit-for-bit.

## New layout — sibling directory, no existing files touched

```
julia/
├── SMC2FC/                ← existing port, NOT TOUCHED
└── SMC2FC_functional/     ← new library, this plan's output
    ├── Project.toml       ← copied from existing, package name SMC2FC_functional
    ├── README.md          ← short, points at this plan
    ├── src/
    │   ├── SMC2FC_functional.jl
    │   ├── Types.jl
    │   ├── Config.jl
    │   ├── EstimationModel.jl
    │   ├── Transforms.jl
    │   ├── Filtering/{Bootstrap,Kernels,OT,GPUSegmentedPF}.jl
    │   ├── SMC2/{Bridge,HMC,MassMatrix,Sampling,Tempering,TemperedSMC}.jl
    │   ├── Control/{Calibration,GPUControlSMC,RBFSchedule,Spec,TemperedSMC}.jl
    │   └── Simulator/{Observations,SDEModel}.jl
    ├── test/
    │   ├── runtests.jl
    │   ├── fixtures.jl                 ← shared fixtures, new
    │   ├── test_transforms.jl
    │   ├── test_filtering.jl
    │   ├── test_smc2.jl
    │   ├── test_control.jl
    │   └── test_e2e_against_python.jl  ← regression vs Python snapshots
    └── benchmarks/
        └── compare_three_libraries.jl  ← runs old Julia, new Julia, and Python
```

The directory is **named differently and packaged differently** so both can be `Pkg.dev`'d in the same Julia session. That is what makes the side-by-side comparison work.

## Functional API contract (applies to every function in the new library)

- Top-level functions take immutable inputs and return immutable outputs.
- Pre-allocated buffers, where needed, are passed via a single `Workspace` argument that is an *immutable* container of mutable arrays. The caller never sees `!`-mutation.
- No exposed `!` functions on the public API.
- GPU kernels (`GPUSegmentedPF.jl`, `GPUControlSMC.jl`) are translated as-is — they are inherently imperative; the *wrapper* around them is functional.

## Documentation contract — Google-style docstrings on every file and function

Every `.jl` file starts with a module-level docstring of this form:

```julia
"""
Filtering/Bootstrap.jl

Bootstrap particle filter for SMC² inner-loop likelihood estimation.

This module implements the bootstrap PF that estimates the log-marginal-
likelihood log p(y_{1:T} | θ) for a fixed parameter vector θ. It is the
inner loop of the SMC² scheme: each outer particle (one θ-sample) calls
this filter once per window.

Mirrors `smc2fc/filtering/gk_dpf_v3_lite.py` (lite variant — no OT rescue;
that lives in `Filtering/OT.jl`).
"""
```

Every function has a Google-style docstring of this form:

```julia
"""
    bootstrap_log_likelihood(workspace, model, params, obs, key) -> Float64

Run a bootstrap particle filter and return the log-marginal-likelihood.

# Arguments
- `workspace::BootstrapWorkspace`: pre-allocated buffers. Caller-immutable
    container; this function may write into the inner arrays but the
    workspace identity is preserved.
- `model::EstimationModel`: filter contract — propagation, diffusion,
    observation log-weight functions and priors.
- `params::AbstractVector{<:Real}`: parameter vector in the **constrained**
    space (apply `unconstrained_to_constrained` first if working in u-space).
- `obs::AbstractMatrix{<:Real}`: observation matrix, shape `(n_obs, n_steps)`.
- `key::AbstractRNG`: random number generator. Threaded through; not mutated
    in a way the caller can observe (a fresh sub-key is split internally).

# Returns
- `Float64`: log-marginal-likelihood log p(obs | params, model).

# Notes
- Uses systematic resampling (no OT rescue here). For OT rescue see
    `OT.bootstrap_log_likelihood_with_rescue`.
- Allocates zero new arrays on the hot path; all storage lives in
    `workspace`.

# Example
```julia
ws = BootstrapWorkspace(model; n_particles=400)
ℓ = bootstrap_log_likelihood(ws, model, params, obs, MersenneTwister(0))
```
"""
function bootstrap_log_likelihood(workspace::BootstrapWorkspace,
                                  model::EstimationModel,
                                  params::AbstractVector{<:Real},
                                  obs::AbstractMatrix{<:Real},
                                  key::AbstractRNG)
    # ...
end
```

Sections used: `# Arguments`, `# Returns`, `# Throws` (where applicable), `# Notes`, `# Example`. This is the Google convention adapted to Julia's `Documenter.jl` syntax.

## What gets carried over vs reshaped

For each existing file, the new version is one of:
- **COPY+DOC**: translate verbatim, add file + function docstrings. No semantic change.
- **TIDY**: minor edits (replace allocating comprehensions, swap `mutable struct` for "immutable container of mutable arrays"), then doc.
- **REFACTOR**: API surface becomes pure (no `!` exposed), buffers go behind a `Workspace`, then doc.

| Existing file | Action | Notes |
|---|---|---|
| `Types.jl` | COPY+DOC | already idiomatic |
| `Config.jl` | COPY+DOC | already idiomatic |
| `EstimationModel.jl` | COPY+DOC | already idiomatic |
| `Transforms.jl` | TIDY | replace comprehensions on lines 93/105 with explicit pre-allocated loop; add tuple-of-priors fast path |
| `Filtering/Bootstrap.jl` | REFACTOR | `BootstrapBuffers` (mutable struct) → `BootstrapWorkspace` (immutable container of mutable arrays); `bootstrap_log_likelihood!` → `bootstrap_log_likelihood` |
| `Filtering/OT.jl` | REFACTOR | same `Workspace` pattern |
| `Filtering/Kernels.jl` | REFACTOR | same |
| `Filtering/GPUSegmentedPF.jl` | COPY+DOC | inherently imperative; wrap, don't rewrite |
| `SMC2/TemperedSMC.jl` | REFACTOR | already mostly functional per survey |
| `SMC2/{Bridge,HMC,MassMatrix,Sampling,Tempering}.jl` | COPY+DOC unless an obvious tidy pops up during translation |
| `Control/Spec.jl`, `Control/RBFSchedule.jl` | COPY+DOC |
| `Control/TemperedSMC.jl`, `Control/Calibration.jl` | REFACTOR | `Workspace` pattern |
| `Control/GPUControlSMC.jl` | COPY+DOC | GPU |
| `Simulator/{SDEModel,Observations}.jl` | COPY+DOC |

## Hour-scale phase plan

Total budget: **~2.5 hours**.

1. **Scaffold the new directory** (~10 min). `mkdir -p julia/SMC2FC_functional/{src,test,benchmarks}`. Copy `Project.toml` and `Manifest.toml`. Rename package to `SMC2FC_functional` in `Project.toml`. Write a 5-line `README.md` pointing at this plan archive.
2. **COPY+DOC pass on the simple files** (~30 min): `Types.jl`, `Config.jl`, `EstimationModel.jl`, all of `Simulator/`, `Control/Spec.jl`, `Control/RBFSchedule.jl`, `SMC2/{Bridge,HMC,MassMatrix,Sampling,Tempering}.jl`, the GPU files. Mostly mechanical — copy, prepend file docstring, add Google docstring above each function.
3. **TIDY `Transforms.jl`** (~15 min). Replace the two comprehensions with an explicit pre-allocated loop. Add a tuple-of-priors fast path. Doc both.
4. **REFACTOR filtering API surface** (~30 min). `Filtering/Bootstrap.jl`, `OT.jl`, `Kernels.jl`. `BootstrapBuffers` → `BootstrapWorkspace` (immutable container of mutable arrays), `!` removed from public names, doc everything.
5. **REFACTOR SMC² and control API surfaces** (~25 min). `SMC2/TemperedSMC.jl`, `Control/TemperedSMC.jl`, `Control/Calibration.jl`. Same workspace pattern, doc everything.
6. **Test fixtures + ported tests** (~20 min). `test/fixtures.jl` with `make_tiny_ou_model`, `make_tiny_bistable_model`, `make_tiny_fsa_model`, `make_tiny_smc_config`. Port the existing `test_phase*.jl` to the new module name and shared fixtures, collapsing into `test_transforms.jl` / `test_filtering.jl` / `test_smc2.jl` / `test_control.jl`.
7. **Three-library comparison harness** (~20 min). `benchmarks/compare_three_libraries.jl`:
   - imports `SMC2FC` (existing) and `SMC2FC_functional` (new) into the same session,
   - shells out to `python -m smc2fc.examples.tiny_ou` (or equivalent) to capture Python outputs to `.npz`,
   - runs the same tiny example through both Julia libraries,
   - prints a side-by-side table of posterior means / log-likelihoods / control schedules + the max abs / max rel difference on each row.

## Verification

End-to-end:
- `cd julia/SMC2FC_functional && julia --project=. -e 'using Pkg; Pkg.test()'` — green.
- `julia --project=. benchmarks/compare_three_libraries.jl` — prints the comparison table; new Julia must agree with Python within `rtol=1e-3` on means and `rtol=1e-2` on per-particle quantiles.
- Existing `julia/SMC2FC/` left strictly unchanged; verify with `git status` showing only additions under `julia/SMC2FC_functional/`.

## Critical files (new — none modified)

All paths below are **new**; they live under [julia/SMC2FC_functional/](julia/SMC2FC_functional/) and do not overwrite anything.

- [julia/SMC2FC_functional/Project.toml](julia/SMC2FC_functional/Project.toml)
- [julia/SMC2FC_functional/src/SMC2FC_functional.jl](julia/SMC2FC_functional/src/SMC2FC_functional.jl) — top-level module
- [julia/SMC2FC_functional/src/Types.jl](julia/SMC2FC_functional/src/Types.jl)
- [julia/SMC2FC_functional/src/Config.jl](julia/SMC2FC_functional/src/Config.jl)
- [julia/SMC2FC_functional/src/EstimationModel.jl](julia/SMC2FC_functional/src/EstimationModel.jl)
- [julia/SMC2FC_functional/src/Transforms.jl](julia/SMC2FC_functional/src/Transforms.jl)
- [julia/SMC2FC_functional/src/Filtering/Bootstrap.jl](julia/SMC2FC_functional/src/Filtering/Bootstrap.jl)
- [julia/SMC2FC_functional/src/Filtering/Kernels.jl](julia/SMC2FC_functional/src/Filtering/Kernels.jl)
- [julia/SMC2FC_functional/src/Filtering/OT.jl](julia/SMC2FC_functional/src/Filtering/OT.jl)
- [julia/SMC2FC_functional/src/Filtering/GPUSegmentedPF.jl](julia/SMC2FC_functional/src/Filtering/GPUSegmentedPF.jl)
- [julia/SMC2FC_functional/src/SMC2/{Bridge,HMC,MassMatrix,Sampling,Tempering,TemperedSMC}.jl](julia/SMC2FC_functional/src/SMC2/)
- [julia/SMC2FC_functional/src/Control/{Calibration,GPUControlSMC,RBFSchedule,Spec,TemperedSMC}.jl](julia/SMC2FC_functional/src/Control/)
- [julia/SMC2FC_functional/src/Simulator/{Observations,SDEModel}.jl](julia/SMC2FC_functional/src/Simulator/)
- [julia/SMC2FC_functional/test/runtests.jl](julia/SMC2FC_functional/test/runtests.jl)
- [julia/SMC2FC_functional/test/fixtures.jl](julia/SMC2FC_functional/test/fixtures.jl)
- [julia/SMC2FC_functional/test/test_transforms.jl](julia/SMC2FC_functional/test/test_transforms.jl)
- [julia/SMC2FC_functional/test/test_filtering.jl](julia/SMC2FC_functional/test/test_filtering.jl)
- [julia/SMC2FC_functional/test/test_smc2.jl](julia/SMC2FC_functional/test/test_smc2.jl)
- [julia/SMC2FC_functional/test/test_control.jl](julia/SMC2FC_functional/test/test_control.jl)
- [julia/SMC2FC_functional/test/test_e2e_against_python.jl](julia/SMC2FC_functional/test/test_e2e_against_python.jl)
- [julia/SMC2FC_functional/benchmarks/compare_three_libraries.jl](julia/SMC2FC_functional/benchmarks/compare_three_libraries.jl)

## Reuse

- Existing Julia port at [julia/SMC2FC/](julia/SMC2FC/) is the **structural template** — translate verbatim where action is COPY+DOC.
- Python reference at [smc2fc/](smc2fc/) is the **numerical template** — when in doubt about an algorithmic choice, mirror the Python.
- Existing tests at [julia/SMC2FC/test/](julia/SMC2FC/test/) are the **assertion template** — port their `@test` lines into the new fixture-based files.

## Plan archive

When this plan is approved, copy this file to `claude_plans/Surgical_Julian_rewrite_of_Julia_SMC2FC_port_<YYYY-MM-DD>_<HHMM>.md` per the project's CLAUDE.md archival rule.
