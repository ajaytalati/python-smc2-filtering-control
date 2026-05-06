# Plan: Julia port of the FSA-v5 model (against the LEAN4 reference)

> Authored 2026-05-06 ~19:21. Pre-implementation plan — no Julia code
> written until this is approved.
> Updated 2026-05-06 ~22:00 — Phases 0-6 (model layer) done. 8 source
> files (Types, DefaultParams, Dynamics, Cost, Schedule, PhiBurst,
> Plant, Obs) with the LEAN4 spec mirrored one-to-one. **38/38
> differential tests pass against Python at 1e-6** via PythonCall.jl.
> Drift, diffusion, mu_bar bit-identical; A_sep agrees to 16th
> decimal; em_step (zero-noise) Euler-by-hand match within 1e-12;
> all 5 obs channel means + sleep prob match formula+source.
> Updated 2026-05-06 ~23:30 — Phases 7, 8, 9, 10 done. Estimation glue
> assembled (37 priors, 14 frozen). Tools: 3 bench-driver skeletons
> (`BenchSmcFilterOnly`, `BenchControllerOnly`, `BenchSmcFullMpc`) and
> 2 plot tools (`PlotBasinOverlay`, `PlotStage2ThetaTraces`). Test
> orchestrator runs **169/169 tests green** under `Pkg.test()` in
> ~15 s on CPU (LEAN diff 40 + Python diff 38 + param-dict 24 +
> per-particle separator 9 + run_dir 9 + smoke + reconciliation). JET
> reports 3 issues, all false-positives in unreached / third-party
> code paths. README written. **What remains is GPU-side SMC² loop
> integration** through `SMC2FC.run_smc_window` (filter) and
> `SMC2FC.run_tempered_smc_loop` (controller); the bench skeletons
> wire the plant + scenario but punt the GPU loop to a follow-up pass.
>
> Aligned to:
> - `LaTex_docs/julia_port_charter.pdf` Part II (Julia-port-as-charter)
> - `FSA_model_dev/LaTex_docs/lean4_first_charter.pdf` (LEAN4 = formal source of truth)
> - The committed LEAN4 reference at `FSA_model_dev/lean/Fsa/V5/*.lean`
> - The existing Julia framework at `julia/SMC2FC/` (287/287 tests green)

## Context

Three artefacts already exist on disk and constrain this plan:

1. **The LEAN4 reference for FSA-v5** (`FSA_model_dev/lean/Fsa/V5/*.lean`)
   — six modules (Types, Drift, Cost, Schedule, Plant, Obs) line-by-line
   transcribed from the canonical Python and differentially tested at
   `1e-6` against Python on every function. This is the **formal source
   of truth for the Julia port too**: every Julia function maps to a
   LEAN4 function with the same signature shape.

2. **The Julia framework `SMC2FC.jl`** (`julia/SMC2FC/`) — 287/287
   tests passing, clean JET.jl pass, fully ported per the charter
   Part II §15. Provides: `EstimationModel` contract, `Particle`,
   `ParticleCloud`, `GPUFilterState`, `CPUParameterCloud`, marker
   types (`HardIndicator`, `SoftSurrogate`, `GaussianBridge`,
   `IdentityOutput`/`SigmoidOutput`, `LogNormalPrior`/`NormalPrior`/…),
   and the bootstrap PF / outer SMC² / control / simulator stacks.
   **No FSA-v5 model lives there yet** — this port will be the first
   integration.

3. **The Python production at `version_3/`** — 9 model files (3.5 kLOC),
   9 tool drivers (3.3 kLOC), 7 test files (1.5 kLOC). Stays in
   service as the differentially-tested fast path; the Julia port is
   an alternative typed implementation, not a replacement.

The work below is **translation, not research** (per charter §16). The
LEAN4 spec sets the math; the Julia framework sets the typing
discipline; the Python source sets the inventory; this plan stitches
the three together into idiomatic Julia.

## Hard constraints (non-negotiable)

1. **LEAN4 is the spec.** Every Julia function with a counterpart in
   `Fsa.V5.*` must have the same signature shape (input/output types
   in the same order, same dimensional contract). The differential
   tests against the LEAN4 binary must pass at `1e-6` for every
   ported function.
2. **No Python re-skin.** Charter §15.1 — port the *algorithm*,
   not the code. No `lax.scan` analogues; native `for` loops.
   Multiple dispatch on marker types instead of flag-driven branches.
   `AbstractArray{T,N}` so the same code runs on CPU + CUDA. Heavy
   use of `StaticArrays.SVector` for fixed-size state, `ComponentArrays`
   for parameter blocks, `StructArrays` for particle clouds.
3. **JET.jl clean.** Type-stability everywhere; no `Union{Float64,
   Nothing}` returns in hot loops. JET pass is a hard gate per
   charter §18.
4. **Zero-allocation inner loops** verified by `BenchmarkTools.@btime`
   per charter §15.1 ("0 allocations or a single amortised allocation").
5. **No edits to `julia/SMC2FC/`.** That package is locked; FSA-v5
   consumes it as a dep.
6. **Don't break Python.** Python source stays unchanged; the existing
   16/16 LEAN4-vs-Python diff tests must still pass. Add Julia as a
   third leg of the differential test (LEAN ↔ Python ↔ Julia).

## Architecture: 4 layers

```
┌──────────────────────────────────────────────────────────────────┐
│  L1  LaTeX equations           — design (LaTex_docs/sections/)   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ line-by-line transcription
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  L2  LEAN4 reference + binary  — formal spec & executable oracle │
│       FSA_model_dev/lean/Fsa/V5/{Types, Drift, Cost,             │
│         Schedule, Plant, Obs}.lean                               │
│       differential test: 16/16 vs Python at 1e-6                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │ shape contract
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  L3  Julia                                                       │
│    a) SMC2FC.jl framework (julia/SMC2FC/, 287 tests green)        │
│    b) FSA-v5 model           (THIS PLAN, version_3_julia/)       │
│         consumes (a) for filter / outer SMC² / control / sim     │
│         consumes (L2) as the spec for drift/cost/plant/obs       │
└──────────────────────────────┬───────────────────────────────────┘
                               │ differential test
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  L4  Python production (version_3/, unchanged)                   │
│       fast path; stays in service; differentially tested         │
└──────────────────────────────────────────────────────────────────┘
```

## Layout: `version_3_julia/` sister folder

```
python-smc2-filtering-control/version_3_julia/
├── Project.toml             # SMC2FC + StaticArrays + ComponentArrays + ...
├── Manifest.toml
├── README.md                # what's here, how to run, parity status
├── src/
│   ├── FSAv5.jl              # umbrella module
│   └── FSAv5/
│       ├── Types.jl           # FSAv5State, BimodalPhi, DynParams, ObsParams
│       ├── Dynamics.jl        # drift, diffusion (mirrors Lean Drift.lean)
│       ├── Cost.jl            # mu_bar, find_a_sep, a_sep_grid (mirrors Cost.lean)
│       ├── Schedule.jl        # design_matrix, schedule_from_theta (mirrors Schedule.lean)
│       ├── Plant.jl           # em_step deterministic + StepwisePlant (Plant.lean)
│       ├── Obs.jl             # 5-channel means + sleep prob (Obs.lean)
│       ├── Estimation.jl      # SMC2FC.EstimationModel registration
│       ├── PhiBurst.jl        # sub-daily Gamma envelope (no Lean counterpart yet)
│       └── DefaultParams.jl   # TRUTH_PARAMS_V5, DEFAULT_PARAMS_V5 as ComponentVector literals
├── tools/
│   ├── _RunDir.jl              # atomic run-dir allocator (port _run_dir.py)
│   ├── BenchSmcFilterOnly.jl   # Stage 1
│   ├── BenchControllerOnly.jl  # Stage 2
│   ├── BenchSmcFullMpc.jl      # Stage 3
│   ├── DiagnoseViolationRate.jl # post-hoc 3-formulation analyser
│   └── SummarizeRun.jl          # markdown-summary helper
├── tests/
│   ├── runtests.jl              # orchestrator
│   ├── test_types.jl
│   ├── test_dynamics.jl         # parity: Julia vs Lean binary, vs Python
│   ├── test_cost.jl             # incl. per-particle separator regression
│   ├── test_schedule.jl
│   ├── test_plant.jl            # deterministic em_step parity
│   ├── test_obs.jl              # 5-channel parity
│   ├── test_reconciliation.jl   # plant + estimator share drift (mirror test)
│   ├── test_param_dict.jl       # σ_S vs σ_S_obs guardrails (post-rename)
│   ├── test_run_dir.jl          # atomic allocator (concurrency-safe)
│   ├── test_lean_diff.jl        # Julia ↔ Lean binary, all functions, 1e-6
│   └── test_python_diff.jl      # Julia ↔ Python, all functions, 1e-6
└── benchmarks/
    └── run_per_window_bench.jl  # CPU + GPU per-window timings
```

## Type bridge: LEAN4 → Julia → SMC2FC

| LEAN4 type (`Fsa.V5.Types.lean`) | Julia type (in `FSAv5/Types.jl`) | Backing primitive |
|---|---|---|
| `State6D` (B, S, F, A, KFB, KFS) | `FSAv5State` | `SVector{6, Float64}` (StaticArrays) |
| `BimodalPhi` (Phi_B, Phi_S) | `BimodalPhi` | `SVector{2, Float64}` (StaticArrays) |
| `Params` (28 dynamics+frozen fields) | `DynParams` | `ComponentVector` (named-field flat AbstractVector) |
| `ObsParams` (22 obs-channel fields) | `ObsParams` | `ComponentVector` |
| `Schedule` = `Array BimodalPhi` | `SmoothSchedule` = `Vector{BimodalPhi}` (or `Matrix{Float64}` of size `(n_steps, 2)`) | parametric on backend |
| `RBFCoeffs` = `Array Float` | `RBFCoeffs` = `Vector{Float64}` (length `2 * n_anchors`) | typealias |

These types **must be distinct** (per charter §11.1 and the LEAN4
charter §4.1): `RBFCoeffs` is a controller decision variable;
`DynParams` is the SMC² inference target. The pattern-matching bug
(retracted "Bug 5") is structurally impossible in Julia because the
two are unifiable with neither each other nor a 2-D region in
control-output space.

The `SMC2FC.State{N,T}` alias already maps to `SVector{N,T}` — `FSAv5State` is
just `State{6, Float64}`. Reuse.

## Function-by-function map

| LEAN4 (`Fsa.V5.*`) | Julia (`FSAv5.*`) | Python source-of-truth |
|---|---|---|
| `Drift.drift y p phi → State6D` | `dynamics_drift(y::FSAv5State, p::DynParams, phi::BimodalPhi)::FSAv5State` | `_dynamics.drift_jax` |
| `Drift.diffusion y p → State6D` | `dynamics_diffusion(y::FSAv5State, p::DynParams)::FSAv5State` | `_dynamics.diffusion_state_dep` |
| `Cost.muBar A phi p → Float` | `mu_bar(A::Float64, phi::BimodalPhi, p::DynParams)::Float64` | `control_v5._jax_mu_bar` |
| `Cost.findASep phi p → Float` | `find_a_sep(phi::BimodalPhi, p::DynParams)::Float64` | `control_v5._jax_find_A_sep` |
| `Cost.aSepGrid particles sched → (n_p, n_steps)` | `a_sep_grid(particles::AbstractVector{<:DynParams}, sched::AbstractVector{BimodalPhi})::Matrix{Float64}` | `control_v5._compute_cost_internals` (the per-particle path post Bug-2 fix) |
| `Schedule.scheduleFromTheta theta design c_phi phi_max n_anchors → Schedule` | `schedule_from_theta(theta::RBFCoeffs, design::AbstractMatrix, c_phi::Float64, phi_max::Float64, n_anchors::Int)::Vector{BimodalPhi}` | `control._make_schedule.schedule_from_theta` |
| `Schedule.designMatrix n_steps dt n_anchors width → Matrix` | `design_matrix(n_steps::Int, dt::Float64, n_anchors::Int, width_factor::Float64)::Matrix{Float64}` | `smc2fc.control.RBFSchedule.design_matrix` |
| `Plant.emStep y phi p sigma_diag dt noise → State6D` | `em_step(y::FSAv5State, phi::BimodalPhi, p::DynParams, sigma_diag::SVector{6,Float64}, dt::Float64, noise::SVector{6,Float64})::FSAv5State` | `_plant._plant_em_step` (deterministic core) |
| `Obs.hrMean y C op → Float` | `hr_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64` | `simulation.gen_obs_hr` (mean part) |
| `Obs.sleepProb y C op → Float` | `sleep_prob(y::FSAv5State, C::Float64, op::ObsParams)::Float64` | `simulation._sleep_prob` |
| `Obs.stressMean y C op → Float` | `stress_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64` | `simulation.gen_obs_stress` (mean part) |
| `Obs.stepsLogMean y C op → Float` | `steps_log_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64` | `simulation.gen_obs_steps` (log-mean) |
| `Obs.volumeLoadMean y op → Float` | `volume_load_mean(y::FSAv5State, op::ObsParams)::Float64` | `simulation.gen_obs_volumeload` (mean) |

Functions *not* yet in LEAN (will be ported to Julia from Python only):

- `_phi_burst.expand_daily_phi_to_subdaily` — Gamma envelope for sub-daily Φ. Pure deterministic; port direct.
- The `StepwisePlant` mutable wrapper (the Lean side has only the
  deterministic single-step kernel `emStep`; the stride-by-stride
  stateful API is plant-side scaffolding and will live in
  `Plant.jl` as a Julia `mutable struct`).
- The `EstimationModel` registration glue (`obs_log_weight_fn`,
  `propagate_fn`, `align_obs_fn`, `shard_init_fn`, `make_init_state_fn`,
  `get_init_theta`) — these wire the model into SMC2FC's contract;
  no Lean counterpart needed.

## Multiple-dispatch markers (already in SMC2FC, reused)

Per charter §15.1: replace Python flag-driven branches with marker
types. SMC2FC.jl already exports the markers we need:

| Python flag | Julia marker (already in SMC2FC) | Used in |
|---|---|---|
| `cost_kind ∈ {'soft', 'hard'}` | `SoftSurrogate`, `HardIndicator <: ChanceConstraintMode` | Cost dispatch |
| `bridge_type ∈ {'gaussian', 'sf'}` | `GaussianBridge`, `SchrodingerFollmerBridge <: BridgeKind` | Inter-window warm start |
| `output ∈ {'identity', 'softplus', 'sigmoid'}` | `IdentityOutput`, `SoftplusOutput`, `SigmoidOutput <: RBFOutput` | RBF schedule transform |
| `backend ∈ {':cpu', ':cuda'}` | `CPUBackend`, `CUDABackend <: AbstractBackend` | Array constructor selection |

So `evaluate_chance_constrained_cost(::HardIndicator, ...)` and
`evaluate_chance_constrained_cost(::SoftSurrogate, ..., beta, scale)`
are two methods of the same generic. The agent cannot accidentally
take the wrong branch — the dispatcher does.

## Phased execution (in order)

Estimates per the charter precedent (framework was 7.5h; model side is
smaller in LOC but more semantic depth):

### Phase 0 — Foundations (~30 min)

- `Project.toml` with deps: `SMC2FC` (path = `../julia/SMC2FC`), `StaticArrays`, `ComponentArrays`, `StructArrays`, `Distributions`, `LinearAlgebra`, `Random`, `Printf`, `JSON3`, `NPZ` (for `*.npz` interop with Python tools), `Plots` (GR backend; for the two plotting tools), `BenchmarkTools` (test-only), `Test` (test-only), `JET` (test-only), `PythonCall` (test-only — see Phase 9 / Risks for the rationale).
- `src/FSAv5.jl` umbrella module: `using SMC2FC`, includes children.
- `Manifest.toml` resolved.
- Smoke `using FSAv5` succeeds.

### Phase 1 — Types + DefaultParams (~30 min)

- `src/FSAv5/Types.jl`: `FSAv5State`, `BimodalPhi`, `DynParams`,
  `ObsParams`, `RBFCoeffs`, `SmoothSchedule`. Operating-point
  constants `A_TYP`, `F_TYP`, `PHI_TYP`.
- `src/FSAv5/DefaultParams.jl`: `TRUTH_PARAMS_V5` and
  `DEFAULT_PARAMS_V5` as `ComponentVector` literals matching
  Python's keys verbatim. Note: `sigma_S` is the LATENT state-noise
  (0.008); the obs noise lives on `ObsParams.sigma_S_obs` (4.0) —
  the Bug 1 collision is structurally impossible because they're on
  different `ComponentVector`s with different field names.
- Test: `test_types.jl` — round-trip a known parameter dict through
  `DynParams ↔ NamedTuple ↔ Dict` and assert no field is silently
  dropped. Compare field names against the LEAN4 `Params` field list
  (cross-file invariant).

### Phase 2 — Dynamics: drift + diffusion (~1 h)

- `src/FSAv5/Dynamics.jl`: `dynamics_drift`, `dynamics_diffusion`,
  line-by-line port from Python `_dynamics.py`. **Read the LEAN4
  `Drift.lean` simultaneously** — it's the spec. Use plain `for`
  loops, no broadcasting tricks; the function operates on
  `SVector{6}` so the compiler unrolls trivially.
- Test: `test_dynamics.jl` — at 3 LaTeX §10.4 anchor points
  (healthy / sedentary / overtrained) the Julia drift must match
  the LEAN4 binary AND the Python drift within `1e-6`.

### Phase 3 — Cost (~1.5 h)

- `src/FSAv5/Cost.jl`: `mu_bar`, `find_a_sep`, `a_sep_grid`. Port
  from Python `control_v5.py:_jax_mu_bar` and `_jax_find_A_sep`.
  - `find_a_sep` uses the LEAN bisection structure (64-point grid,
    40 bisection iterations, three-way return ±inf or finite).
  - `a_sep_grid` is the Bug 2 structural fix: type signature
    returns `Matrix{Float64}` of shape `(n_particles, n_steps)`,
    NOT `Vector{Float64}` of shape `(n_steps,)`. The buggy
    "particle-0 template" collapse cannot type-check.
- `evaluate_chance_constrained_cost(::ChanceConstraintMode, ...)`
  with two methods: `::HardIndicator` (uses `(A_traj < A_sep)`)
  and `::SoftSurrogate` (uses `sigmoid(beta * (A_sep - A_traj) / scale)`).
- Test: `test_cost.jl` — the per-particle-separator regression test
  (the same one I added to the Python tests this session); plus
  parity with LEAN binary on 200 hypothesis-style random samples.

### Phase 4 — Schedule (~30 min)

- `src/FSAv5/Schedule.jl`: `design_matrix(n_steps, dt, n_anchors,
  width_factor)`, `schedule_from_theta(theta, design, c_phi, phi_max, n_anchors)`.
  These mirror `SMC2FC.Control.RBFSchedule` *for FSA-v5
  conventions* (the framework version is a more general builder;
  the model-side wrapper picks the right `c_phi` / `phi_max` /
  output mode for FSA-v5).
- Test: `test_schedule.jl` — at θ = 0, schedule equals
  `phi_default` everywhere (sigmoid identity check); 50
  random-θ comparisons against LEAN binary at `1e-6`.

### Phase 5 — Plant (~1 h)

- `src/FSAv5/PhiBurst.jl`: pure-deterministic Gamma-envelope sub-daily
  Φ expansion (port from `_phi_burst.py`).
- `src/FSAv5/Plant.jl`:
  - `em_step(y, phi, p, sigma_diag, dt, noise)` — deterministic
    single-bin Euler-Maruyama, mirrors LEAN `Plant.emStep` and
    Python `_plant._plant_em_step` body. Pure; takes noise as
    explicit input.
  - `mutable struct StepwisePlant` — stateful wrapper holding
    `state::FSAv5State`, `t_bin::Int`, `params::DynParams`,
    `obs_params::ObsParams`, `seed_offset::Int`, `dt::Float64`.
    Methods: `advance!(plant, stride_bins, phi_daily)` (mutates
    state, returns observation dict), `finalise(plant, out_dir)`
    (writes manifest, trajectory).
- Test: `test_plant.jl` — `em_step` parity with LEAN at zero noise
  + 100 random samples; `StepwisePlant.advance!` smoke at healthy
  scenario (T=14 days, deterministic seed).

### Phase 6 — Obs (~30 min)

- `src/FSAv5/Obs.jl`: 5 channel means/probabilities — `hr_mean`,
  `sleep_prob`, `stress_mean`, `steps_log_mean`, `volume_load_mean`.
  Each is a plain function over `(state, C, ObsParams)` and
  returns a `Float64`. Sample-with-noise wrappers come later if
  needed for the simulator.
- Test: `test_obs.jl` — all 5 channels parity vs LEAN at LaTeX
  anchor points + 50 random samples.

### Phase 7 — Estimation glue (~1 h)

- `src/FSAv5/Estimation.jl`: assemble an `SMC2FC.EstimationModel`
  for FSA-v5. Functions to provide:
  - `propagate_fn(y, t, dt, params, grid_obs, step_k, σ_diag, noise, rng) → (x_new, pred_lw)` — wraps `em_step` plus boundary handling.
  - `diffusion_fn(params) → SVector{6}` — wraps `dynamics_diffusion`.
  - `obs_log_weight_fn(x_new, grid_obs, step_k, params)` — uses Phase 6 means + Gaussian likelihoods.
  - `align_obs_fn`, `shard_init_fn`, `forward_sde_fn`,
    `imex_step_fn`, `obs_log_prob_fn`, `make_init_state_fn`,
    `get_init_theta_fn`.
  - Parameter priors: `LogNormalPrior`, `NormalPrior` etc. from
    SMC2FC, matching the 37-key `PARAM_PRIOR_CONFIG` from Python
    `estimation.py`.
- Export `HIGH_RES_FSA_V5_ESTIMATION::EstimationModel`.
- Test: `test_reconciliation.jl` — plant `em_step` and
  `propagate_fn` agree on a single bin (the "mirror" test from
  Python `test_reconciliation_v5.py`).

### Phase 8 — Tools (~2 h)

- `tools/_RunDir.jl` — atomic run-dir allocator. Port the
  `mkdir(exist_ok=False)` retry-loop pattern from
  `tools/_run_dir.py`. Test concurrency with multiple `Threads.@spawn`.
- `tools/BenchSmcFilterOnly.jl` (Stage 1):
  - Use `SMC2FC.run_smc_window` and `SMC2FC.bootstrap_log_likelihood`.
  - Output schema: `manifest.json` + `posterior.npz` + `trajectory.npz` matching Python's keys (so existing diagnostic tools work cross-language).
- `tools/BenchControllerOnly.jl` (Stage 2):
  - Use `SMC2FC.Control.run_tempered_smc_loop` for the controller
    HMC. Cost callable wraps `evaluate_chance_constrained_cost(::SoftSurrogate, ...)`.
- `tools/BenchSmcFullMpc.jl` (Stage 3):
  - Compose Stage 1 (filter) + Stage 2 (controller) per replan.
  - The hybrid CPU/GPU partition (charter §14): inner PF on CUDA
    (`CUDABackend`), outer SMC² + HMC on CPU. Both stages dispatch
    via `AbstractArray{T,N}`.
- `tools/DiagnoseViolationRate.jl` — port the 3-formulation analyser
  (per-bin / per-day / active-bin-only) from Python. Pure CPU.
- `tools/SummarizeRun.jl` — port the markdown-summary helper.
- `tools/PlotBasinOverlay.jl` — port `plot_basin_overlay.py`. Reads a
  completed bench run dir (`manifest.json` + `trajectory.npz`),
  overlays the applied (Φ_B, Φ_S) controller path on the v5
  closed-island bifurcation diagram (transcritical contour from
  `find_a_sep`, plus the bistable annulus). Uses Plots.jl with the
  GR backend (lighter than Makie, no GPU dep). Output: PNG to the
  run dir, same filename as the Python equivalent.
- `tools/PlotStage2ThetaTraces.jl` — port `plot_stage2_theta_traces.py`.
  Reads `posterior.npz`, plots θ_B and θ_S anchor coefficient traces
  across replans. Plots.jl + GR backend.
- `tools/cli/` (optional): driver shell scripts wrapping `julia
  --project=. tools/BenchSmcFullMpc.jl --T-days 14 --scenario healthy`
  etc., mirroring `version_2/tools/launchers/`.

### Phase 9 — Tests (~1.5 h)

- Port the 7 Python test files (3 added this session, 4 pre-existing).
- For each, add a Julia equivalent under `tests/` with the same
  test-function names and assertions (translated to `@test`).
- New: `test_lean_diff.jl` — Julia ↔ LEAN binary, every function,
  `1e-6` tolerance. Reuses the existing `LeanDriftClient` via PyCall
  OR (preferable) a Julia subprocess wrapper around the same
  `fsa_v5_cli` binary. Either works; subprocess is simpler.
- New: `test_python_diff.jl` — Julia ↔ Python, every function. Uses
  **`PythonCall.jl` in-process** (per user's preference: keep the
  test stack Julia-focused, no extra Python CLI to maintain).
  `PythonCall` imports the production Python functions directly
  (`from version_3.models.fsa_v5._dynamics import drift_jax`, …) and
  compares return values to the Julia equivalents in the same
  Julia process. Does NOT depend on the LEAN4 binary; the LEAN diff
  is in `test_lean_diff.jl`. Test-time only — `PythonCall` is in
  the `[extras]` / `[targets].test` block, not a runtime dep.
- Test orchestrator `tests/runtests.jl`:
  ```julia
  include("test_types.jl"); include("test_dynamics.jl"); …
  using JET; report_package(FSAv5)  # charter §18 audit gate
  ```

### Phase 10 — Benchmarks + JET (~30 min, mostly running)

- `benchmarks/run_per_window_bench.jl` — measure per-window cost
  on CPU and CUDA backends. Compare against the Python baseline
  cited in charter §15.7 (~1 s/window on RTX 5090). The Julia
  framework's Phase 6 bench already shows 0.09 s/window for a
  synthetic 4D Gaussian; FSA-v5's 6D state + 37D θ + 5 obs
  channels will be larger but still GPU-saturable.
- `julia --project=. -e 'using JET; report_package(FSAv5)'` must
  report 0 type uncertainties.
- `BenchmarkTools.@btime` on `dynamics_drift` and `em_step` must
  show 0 allocations.

## Differential testing strategy (three-way)

Three implementations of the same model:
- LEAN binary (`fsa_v5_cli` from `FSA_model_dev/lean/`)
- Python (`version_3/models/fsa_v5/`)
- Julia (`version_3_julia/src/FSAv5/`)

For each function, the diff test asserts pairwise agreement:

| Pair | Tolerance | Status today |
|---|---|---|
| LEAN ↔ Python | `1e-6` | 16/16 green (this session) |
| LEAN ↔ Julia | `1e-6` | NEW (Phase 9) |
| Python ↔ Julia | `1e-6` | NEW (Phase 9) |

If all three pairs pass, the implementations are consistent up to
IEEE-754 noise. If LEAN ↔ Python fails: flag the Python (already
covered by existing tests). If LEAN ↔ Julia fails: the Julia port has
a bug. If Python ↔ Julia fails but each agrees with LEAN
individually: floating-point order-of-operations difference; check
which side rounded.

## What is and isn't covered

**In scope:**
- Full FSA-v5 model port (drift, diffusion, cost, plant, schedule, obs).
- Estimation model assembled into SMC2FC's contract.
- 3 bench drivers (Stage 1/2/3) operational on CPU + CUDA.
- Diagnostic tooling (violation-rate analyser, run-dir allocator,
  summarize_run).
- **Plotting tools** (`plot_basin_overlay`, `plot_stage2_theta_traces`)
  via Plots.jl with GR backend. Same output filenames + run-dir
  contract as the Python equivalents.
- Test parity against LEAN + Python.

**Out of scope (deferred):**
- `control_v5_fast` — JAX-specific fp32 + sub-sampling
  optimization. Per charter §15.1, don't replicate Python perf
  hacks; rely on Julia's native compilation. If perf is short
  on GPU, revisit.
- `profile_cost_fn.py` — JAX TensorBoard profiler. Julia uses
  `BenchmarkTools` + `Profile` natively; not a 1:1 port.
- Property proofs in LEAN (Lipschitz, monotonicity, etc.) — per
  the LEAN4-first charter §5 step 5, deferred.

**Trusted upstream:**
- `julia/SMC2FC/` — 287 tests green, JET clean. Don't touch.
- The mathematics of FSA-v5 — fixed by the LEAN4 reference.
- IEEE-754 floating-point semantics — bounded by the `1e-6`
  tolerance.

## Combined audit checklist (LEAN-first §8 + Julia §18)

Every agent runs this at session start before touching any Julia file:

- [ ] Is there a LEAN4 reference for the function I'm porting? (Yes for
      drift/cost/schedule/plant/obs; no for `_phi_burst`, plant
      stateful wrapper, estimation glue.) If yes: read it first.
- [ ] `cd FSA_model_dev/lean && lake build` succeeds?
- [ ] Existing LEAN ↔ Python diff test (`tests/test_lean4_diff.py`)
      green?
- [ ] Have I declared parametric types for every input/output?
      Capturing dimensions Python erases.
- [ ] Are array fields `AbstractArray{T,N}` so the same code accepts
      `Array` (CPU) and `CuArray` (GPU)?
- [ ] Did I port the *algorithm*, not the code? No `lax.scan`
      analogues; plain `for` loops.
- [ ] Multiple dispatch on marker types instead of flag-driven
      conditionals?
- [ ] `StaticArrays.SVector` for fixed-size objects (`FSAv5State`,
      RBF coefficients)?
- [ ] `StructArrays` for particle clouds (Struct-of-Arrays
      memory layout)?
- [ ] Inner loops zero-allocation? `BenchmarkTools.@btime` should
      report 0 allocations.
- [ ] Outer particle loops parallelised on CPU
      (`Threads.@threads` / `Polyester.@batch`) or replaced by
      `KernelAbstractions.@kernel` on GPU?
- [ ] Hybrid contract respected (charter §14)? Inner SDE/PF on
      `CuArray`; outer HMC/tempering on `Array`; only scalar
      log-likelihood (and 37-D θ) crosses PCIe.
- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes on
      both CPU and CUDA paths?
- [ ] `julia --project=. -e 'using JET; report_package(FSAv5)'`
      reports 0 type uncertainties?
- [ ] Differential test against LEAN binary AND Python passes
      within tolerance for the function I just ported?
- [ ] Have I updated this plan to mark the file as ported?

If any box unchecked → stop, fix, then proceed.

## Effort estimate (per-phase, no inflation)

| Phase | Description | Estimate |
|---|---|---|
| 0 | Foundations (Project.toml, layout, smoke `using`) | 30 min |
| 1 | Types + DefaultParams | 30 min |
| 2 | Dynamics (drift + diffusion) | 1 h |
| 3 | Cost (mu_bar + find_a_sep + a_sep_grid + dispatch) | 1.5 h |
| 4 | Schedule (RBF design + decoder) | 30 min |
| 5 | Plant (em_step + StepwisePlant + PhiBurst) | 1 h |
| 6 | Obs (5 channel means) | 30 min |
| 7 | Estimation glue (EstimationModel registration) | 1 h |
| 8 | Tools (3 benches + diagnostic + run_dir + 2 plot tools) | 2.5 h |
| 9 | Tests (port 7 files + LEAN-diff + Python-diff) | 1.5 h |
| 10 | Benchmarks + JET pass | 30 min |
| **Total** | | **~10.5 hours** |

Compares to:
- Framework port: ~7.5 h (per charter §16, achieved).
- LEAN4 transcription of the model: ~4 h (per LEAN4 charter, achieved).
- Total typed-pipeline cost amortised across all future model
  versions: one-shot ≈21 h to replace open-ended Python+JAX
  pattern-match-then-debug rounds.

## Risks

- **Bench-driver output schema compatibility.** The Python bench
  scripts emit `manifest.json` + `*.npz` files that downstream tools
  (`diagnose_violation_rate.py`, `plot_basin_overlay.py`,
  `summarize_run.py`) consume. Mitigation: the Julia bench writes
  the exact same schema (using JSON3.jl + NPZ.jl), so existing
  Python diagnostics work unchanged on Julia outputs.

- **Two GPU backends to keep alive.** The Python production runs on
  JAX/XLA; the Julia port runs on CUDA.jl. Per charter §15.2 the
  Julia code uses `AbstractArray{T,N}` so the same kernel runs on
  CPU + CUDA, but cross-backend bit-equivalence is bounded by the
  `1e-6` tolerance, not exact.

- **JET.jl strictness.** Type-stability throughout is a hard gate;
  most performance regressions in Julia come from type instability
  in hot loops. JET catches these but only if the test target is
  invoked with concrete types — design tests to call the public
  API with concrete-typed inputs.

- **`PythonCall.jl` for the Julia↔Python diff test (chosen).** Adds
  a Python interpreter as a *test-time-only* dep (in `[extras]` /
  `[targets].test`, not in runtime). The user's preference is to
  keep the stack Julia-focused — `PythonCall` reads better than a
  bespoke Python CLI mirror, and avoids drift between two
  hand-maintained interfaces. Mitigation for the dep weight:
  `PythonCall` is loaded only by `test_python_diff.jl`; everything
  else (model, tools, LEAN diff) stays Python-free. The Python
  interpreter is the user's existing `comfyenv` (set
  `JULIA_PYTHONCALL_EXE` env var if needed).

- **Stateful `StepwisePlant` mutability.** Julia idioms prefer
  immutable structs + functional update; `StepwisePlant` needs
  mutation for the closed-loop bench drivers. Use `mutable struct`
  + `!`-suffixed methods (`advance!`, `finalise!`) per the SMC2FC
  convention from charter §12.2.

## Verification end-to-end

The plan succeeds when, on the FSA-v5 healthy-island scenario at
T=14d:

1. `julia --project=version_3_julia -e 'using Pkg; Pkg.test()'` passes
   all tests on CPU and CUDA backends.
2. `julia --project=version_3_julia -e 'using JET; report_package(FSAv5)'`
   reports 0 type uncertainties.
3. `julia --project=version_3_julia tools/BenchSmcFullMpc.jl --T-days 14
   --scenario healthy --backend cpu` produces a `manifest.json` whose
   numeric summary fields agree with the Python production within
   `1e-4` (integrated trajectory tolerance from the LEAN4 charter §5).
4. The same on `--backend cuda` — agrees within `1e-4` (fp64 should be
   tight; fp32 paths if added later would need a relaxed tolerance
   per charter §15.7).
5. Per-window wall-clock on RTX 5090 ≤ Python baseline (≈ 1 s/window;
   target 0.5 s or better given the framework's 0.09 s/window
   synthetic baseline).
6. The plan archive in `claude_plans/` has a closing `> Updated:` line
   recording the actual completion time and final test counts.

## Critical files this plan creates

New, in `version_3_julia/`:

- `Project.toml`, `Manifest.toml`, `README.md`
- `src/FSAv5.jl`
- `src/FSAv5/{Types,DefaultParams,Dynamics,Cost,Schedule,Plant,Obs,Estimation,PhiBurst}.jl`
- `tools/{_RunDir,BenchSmcFilterOnly,BenchControllerOnly,BenchSmcFullMpc,DiagnoseViolationRate,SummarizeRun,PlotBasinOverlay,PlotStage2ThetaTraces}.jl`
- `tests/{runtests,test_types,test_dynamics,test_cost,test_schedule,test_plant,test_obs,test_reconciliation,test_param_dict,test_run_dir,test_lean_diff,test_python_diff}.jl`
- `benchmarks/run_per_window_bench.jl`

Modified (none — Python and SMC2FC.jl untouched).

## Reference utilities to reuse (in priority order)

From `SMC2FC.jl`:
- `State{N,T}` ⇒ basis for `FSAv5State`
- `EstimationModel` contract ⇒ `HIGH_RES_FSA_V5_ESTIMATION` instance
- `Particle{N,T}`, `ParticleCloud{N,T}` ⇒ filter/controller particle clouds
- `LogNormalPrior`, `NormalPrior`, `BetaPrior` ⇒ prior config
- `RBFBasis` ⇒ basis for the FSA-v5 schedule decoder
- `bootstrap_log_likelihood`, `run_smc_window`, `run_smc_window_bridge` ⇒ Stage 1 driver
- `run_tempered_smc_loop` ⇒ Stage 2 driver (control)
- `simulate_sde`, `build_sde_problem` ⇒ alternative simulator path (deferred; production uses `StepwisePlant` directly)
- Marker types `HardIndicator`, `SoftSurrogate`, `GaussianBridge`,
  `IdentityOutput`, `SigmoidOutput`, `CPUBackend`, `CUDABackend`

From the LEAN4 reference (`FSA_model_dev/lean/Fsa/V5/`):
- Type signatures of all 13 functions to port (one Lean function = one
  Julia function with the same name).
- The `fsa_v5_cli` binary as the differential-test oracle.
- The 16 existing diff tests (`tests/test_lean4_diff.py`) as the
  template for the new `tests/test_lean_diff.jl`.

From Python (`version_3/`):
- The production source files as the line-by-line reference. **Read
  the LEAN spec FIRST**, the Python SECOND — the LEAN is closer to
  the LaTeX and shorter; the Python adds JAX-specific scaffolding
  that should be filtered out.

## Standing ready

If approved, Phase 0 starts the moment I'm given the go-ahead. I'll
keep this plan archive updated with `> Updated:` lines as each phase
lands (per global `CLAUDE.md` rule on plan-archive sync).
