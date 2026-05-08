# FSA v1.5 — open-loop v1 + simplest possible closed-loop add-ons

> Archived from plan mode: 2026-05-08 07:13.

> Plan only. Read-only on existing files; the only file edits are NEW files in a NEW directory `version_1_5_Julia/`.

## Context

There are two FSA Julia codebases today:

- **v1** (`version_1_Julia/models/fsa_high_res/`) — G0 (original, non-reparametrised) dynamics + GPU control kernel + open-loop SMC²-as-controller bench. Five files: `_dynamics.jl`, `simulation.jl`, `control.jl` (CPU spec), `gpu_control.jl` (GPU cost kernel), `FSAHighRes.jl`. **No plant. No filter. No closed-loop.**
- **v2** (`version_2_Julia/models/fsa_high_res/`) — G1 (reparametrised) dynamics + four-channel observation model (HR sleep-gated, sleep label, stress wake-gated, steps wake-gated) + StepwisePlant + GPU particle filter + closed-loop SMC²-MPC bench. Eleven files. The recent saturation sweep (Phase 2) just confirmed v2's closed-loop only delivers ~+3 % area gain (vs Python's +51 %), and the gap is **saturation-invariant** — diagnosed as a cost-surface or kernel bug (writeup §2.15 row 10/11), not sampling noise.

The user wants a **v1.5 bridge** that takes v1's open-loop pieces verbatim (dynamics, both control files) and adds the **minimum** machinery for closed-loop MPC: a tiny three-channel Gaussian observation layer on (B, F, A), a tiny plant glue, and a tiny filter that consumes those obs. Goal: a clean diagnostic environment where the closed-loop pipeline can be validated without v2's reparametrisation, fp32 cancellation issues, and multi-channel observation complexity. If v1.5's closed-loop hits Python's +51 % area gain, the v2 bug lives in its reparametrisation or its multi-channel obs. If v1.5 also under-shoots, the bug lives somewhere structural in the closed-loop pipeline itself.

## The three asks (verbatim from the user's notes)

1. Filtering code with **simple Gaussian noise on each of the three latents** (B, F, A) — the simplest possible observation model.
2. **Simple plant code** — glue for closed-loop MPC.
3. **Reuse v1's `gpu_control.jl` as-is** (the GPU kernel; `control.jl` is the CPU spec and is not needed for the closed-loop bench, which decodes RBF inline as v2 does).

---

## HIGHEST-PRIORITY DESIGN CONSTRAINT — pure functions, immutable data

v1 and v2 are imperative: mutable structs (`mutable struct StepwisePlant`), `!`-suffix in-place mutators (`advance_subdaily!`, `update_window_obs!`), history dicts `push!`-ed onto, RNG state threaded as a stateful `MersenneTwister`. They were Python ports — not designed in Julia idiom — and they're tangled. **v1.5 will not inherit that style.** v1.5 is a clean redesign in functional, stateless Julia. Performance is explicitly **not** a goal — clarity and composability are. The user has stated they accept any reasonable performance loss in exchange.

### What "purely functional" means here, concretely

1. **No mutable structs.** Use immutable `struct` (or `NamedTuple` records). State updates produce a NEW value, not a mutation of the old one. Use `Setfield.@set` / `Accessors.@set` if convenient.
2. **No `!`-suffix mutating methods in the public API.** Functions return new values: `plant_step(state, Φ_t, key) → (new_state, obs)`. The caller can build up a history by collecting returned values, or by `accumulate`/`foldl` over time bins.
3. **No hidden state in modules.** Module-level `const` immutable values are fine (`DEFAULT_PARAMS`, `INIT_STATE`, `BINS_PER_DAY`). Module-level mutable globals are not.
4. **Explicit RNG keys, JAX-style.** Pass an explicit `key::UInt64` (or a `StableRNGs.LehmerRNG` seeded by it) to every randomised function; the function takes the key and returns a deterministic result. No threaded `MersenneTwister`. Two callers with the same `(state, key)` get the same answer, period. Internal split-keys are derived deterministically from the input key (`hash((key, :obs))`, `hash((key, :sde))`, etc.).
5. **No I/O in business logic.** Reading `ENV[FSA_STEP_MINUTES]` happens once at module load to set a `const`. JLD2 / PNG writes happen only at the top-level bench script, as the single explicit I/O step at the end. Logging happens at the bench level, not inside `plant_step` / `propagate` / `obs_log_likelihood`.
6. **Composition over orchestration.** The 14-day bench is a `foldl` (or its for-loop equivalent) over per-stride pure functions, not an imperative loop pushing onto a history dict. The history is the sequence of returned values.
7. **Functional facade over the inherently imperative GPU kernel.** v1's `fsa_cost_kernel!` is reused verbatim — KernelAbstractions kernels are inherently mutating (write into preallocated CuArrays). v1.5 wraps every kernel-touching public function with a pure facade: `gpu_cost(theta, target_inputs) → Vector{Float64}` allocates a fresh output, dispatches the kernel, returns. The mutation is fully encapsulated; callers never see it.

### What this changes vs v1 / v2 (concrete name table)

| v1 / v2 (imperative)                        | v1.5 (functional)                                     |
|---|---|
| `mutable struct StepwisePlant`              | `struct PlantState` (immutable record)                |
| `advance_subdaily!(plant, Phi_subdaily)`    | `plant_rollout(state, Phi_subdaily, key) → (new_state, traj, obs)` |
| `plant.history[:trajectory] |> vcat(_...)`  | `traj` returned from `plant_rollout`, accumulated by the bench loop |
| `update_window_obs!(target, grid_obs)`      | dropped — pass `grid_obs` as a function argument to `gpu_log_density(target, U, grid_obs)` |
| `parallel_hmc_one_move!(U, target, ε, L, ...)` | `parallel_hmc_one_move(U, target, ε, L, key, ...) → (new_U, n_acc, new_key)` |
| `MersenneTwister(seed) ; randn(rng, ...)`   | `key::UInt64` threaded explicitly through every randomised function |
| `target.theta_per_chain .= ...` (in-place)  | a fresh `(theta_per_chain=..., cost_buffer=...)` input bundle is constructed per call |
| `for s in 1:n_strides; advance_subdaily!(plant, ...) ; ...; end` | `accumulate((acc, s) -> step(acc, s), 1:n_strides; init=acc0)` pulling out the history at the end |

### Where functional discipline does NOT extend (and why that's OK)

- **Inside `fsa_cost_kernel!` (v1's GPU kernel).** It's reused verbatim per the user's instruction; KernelAbstractions kernels are written against preallocated buffers. The mutation is wrapped inside a pure facade `gpu_cost(theta, …) → Vector{Float64}`.
- **CuArray buffer allocation.** A new CuArray is allocated per `gpu_cost` call (since perf is not a concern). The buffer is a private temporary inside the function — never visible to callers.
- **JLD2 / PNG writes at the bench top level.** These are explicit I/O effects, isolated to the bench's `main()` function. Everything before that point is pure.
- **`@info` logging.** Treated as a side-effect at the bench level, not in business logic.

### Library choices to support this

- `StaticArrays.SVector{3, Float64}` for the (B, F, A) state — immutable, stack-allocated, fast and clean.
- `Setfield.jl` or `Accessors.jl` for "update" syntax on immutable records (`@set state.t_bin = state.t_bin + 1`).
- `StableRNGs.LehmerRNG` keyed by `UInt64` for deterministic RNG-from-key.
- Built-in `accumulate`, `foldl`, `mapreduce` for sequence operations.
- Stick with the existing `JLD2`, `Plots`, `CUDA`, `KernelAbstractions` deps — same as v2.

This design constraint is THE highest priority. Every new file in v1.5 is judged against it: if a function mutates an argument or hides state, redesign it before merging.

## Approach — file-by-file

New directory: `/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/`. Mirror v2's layout.

### Files reused VERBATIM from v1 (no edits)

| New file (v1.5) | Source (v1) | Why verbatim |
|---|---|---|
| `models/fsa_high_res/_dynamics.jl` | [version_1_Julia/models/fsa_high_res/_dynamics.jl](version_1_Julia/models/fsa_high_res/_dynamics.jl) | G0 drift `µ = µ₀ + µ_B·B − µ_F·F − µ_FF·F²`, `dB/dt = κ_B·(1+ε_A·A)·Φ − B/τ_B`, `dF/dt = κ_F·Φ − (1+λ_A·A)/τ_F·F`. Numerically clean (no near-equal subtraction → no §2.7 fp32 cancellation). TRUTH_PARAMS evaluates identically to v2 at INIT_STATE. |
| `models/fsa_high_res/gpu_control.jl` | [version_1_Julia/models/fsa_high_res/gpu_control.jl](version_1_Julia/models/fsa_high_res/gpu_control.jl) | GPU cost kernel `fsa_cost_kernel!` with the v1 drift form (no Horner-form, no `a_typ_inv` precomputation, no F_dev² subtraction). Same Eq 37 cost (`−∫A + λ_F·∫max(F−F_max,0)²`). Public API: `FSAControlGPUTarget`, `gpu_cost_log_density_batched`, `make_log_density_fn`. |

(v1's `control.jl` — the CPU control spec — is intentionally NOT carried over per the user's instruction. v2's closed-loop bench decodes RBF θ → per-bin Φ inline, with no dependency on the CPU spec module; v1.5 mirrors that pattern.)

### Files copied from v2

None. v2's `_phi_burst.jl` (which defines `BINS_PER_DAY` plus the daily→subdaily morning-loaded Gamma burst-envelope) is **not** carried over — the burst envelope is exactly the kind of plant ↔ controller signal mismatch §2.10 of the writeup flagged. The closed-loop bench feeds per-bin Φ directly from the controller's RBF decoder into the plant; no daily-to-subdaily expansion is needed anywhere in v1.5.

`BINS_PER_DAY` will be defined directly at the top of v1.5's new `simulation.jl`, parsed from `ENV["FSA_STEP_MINUTES"]` at module load (the same env-var hook v2 uses, just without the burst-envelope file around it).

### NEW files (the smallest possible additions)

These are the three asks. Estimated total: ~400 LOC.

#### 1. `models/fsa_high_res/simulation.jl` (NEW; ~90 LOC)

Defines:
- `_STEP_MIN = parse(Int, get(ENV, "FSA_STEP_MINUTES", "60"))` — read at module load, identical hook to v2.
- `BINS_PER_DAY = (60 * 24) ÷ _STEP_MIN`, `DT_BIN_DAYS = 1.0 / BINS_PER_DAY`. Defined here (not in `_phi_burst.jl`, which we're not carrying over).
- `DEFAULT_PARAMS::Dict{Symbol,Float64}` — 14 dynamics params (`tau_B`, `tau_F`, `kappa_B`, `kappa_F`, `epsilon_A`, `lambda_A`, `mu_0`, `mu_B`, `mu_F`, `mu_FF`, `eta`, `sigma_B`, `sigma_F`, `sigma_A`) + 3 obs-noise params (`sigma_B_obs`, `sigma_F_obs`, `sigma_A_obs`, default 0.005 each). Values copied from v1's `Dynamics.TRUTH_PARAMS`.
- `INIT_STATE = (B = 0.05, F = 0.30, A = 0.10)` (same as v1 / v2).
- `gen_obs_bfa(traj, params, rng)` — returns `(obs_B, obs_F, obs_A)` as length-`stride_bins` `Float32` vectors. One-liner per channel: `obs_X[k] = X[k] + sigma_X_obs * randn(rng)`. Every channel observed every bin (no gating).
- (No HR / sleep / stress / steps. No circadian C(t). No EXOGENOUS. No burst envelope.)

#### 2. `models/fsa_high_res/_plant.jl` (NEW, written from first principles, FUNCTIONAL; ~100 LOC)

The plant is a pair of pure functions. No mutable struct, no history dict, no `!`-suffix.

**Immutable state record:**
```julia
struct PlantState
    bfa::SVector{3, Float64}     # (B, F, A) — stack-allocated, immutable
    t_bin::Int                   # global bin counter
end
```

`Params` (truth_params) and `dt` (= `1 / BINS_PER_DAY`) are passed as function arguments, not stored in the state. They don't change across the bench, so they live as `const`s in `simulation.jl` or as arguments in the bench's `main()` and are passed down explicitly. Threading them through avoids any "where did this value come from?" ambiguity.

**Pure single-bin step:**
```julia
function plant_step(s::PlantState, Φ_t::Float64, params, dt::Float64, key::UInt64)
    rng = StableRNG(key)              # deterministic from key
    d   = drift(s.bfa, params, Φ_t)            # v1's G0, pure
    σ   = diffusion_state_dep(s.bfa, params)   # v1's, pure
    ξ   = randn(rng, 3)
    y   = s.bfa .+ d .* dt .+ σ .* sqrt(dt) .* SVector{3}(ξ)
    y_r = SVector{3}(reflect_unit(y[1]), abs(y[2]), abs(y[3]))     # boundary policy
    obs = sample_obs_bfa(y_r, params, hash((key, :obs)))           # pure, see Simulation
    return (state = PlantState(y_r, s.t_bin + 1), obs = obs)
end
```

`reflect_unit(x) = x < 0 ? -x : (x > 1 ? 2 - x : x)`. Pure helper.

**Pure stride rollout** — folds `plant_step` over the per-bin Φ vector:
```julia
function plant_rollout(s0::PlantState, Φ_subdaily::AbstractVector, params, dt::Float64, key0::UInt64)
    states  = Vector{PlantState}(undef, length(Φ_subdaily) + 1)
    obses   = Vector{NamedTuple}(undef, length(Φ_subdaily))
    states[1] = s0
    for k in 1:length(Φ_subdaily)
        sub_key = hash((key0, :step, k))
        nxt = plant_step(states[k], Float64(Φ_subdaily[k]), params, dt, sub_key)
        states[k + 1] = nxt.state
        obses[k]      = nxt.obs
    end
    traj = stack(s -> Vector(s.bfa), states[2:end])    # (stride_bins, 3) Matrix{Float64}
    obs_B = Float32.(getfield.(obses, :obs_B))
    obs_F = Float32.(getfield.(obses, :obs_F))
    obs_A = Float32.(getfield.(obses, :obs_A))
    return (final_state = states[end], trajectory = traj,
            obs_B = obs_B, obs_F = obs_F, obs_A = obs_A,
            Phi = Float32.(Φ_subdaily))
end
```

The internal loop builds two temporary Vectors and could equivalently be written as `accumulate(...; init=s0)` — semantically identical, slightly heavier syntax for readers unfamiliar with `accumulate`. We keep the explicit loop because the body is short and the assignment-into-preallocated-vectors pattern is local (does not leak mutation outside the function). The function as a whole is referentially transparent.

**Pure bench-level history fold** (used by the bench's `main()`):
```julia
function bench_rollout(s0::PlantState, plan_phis::AbstractVector{<:AbstractVector},
                       params, dt::Float64, base_key::UInt64)
    # plan_phis is a vector of per-stride Φ vectors — shape ((n_strides,), each (stride_bins,))
    # Returns the concatenated trajectory + obs across all strides.
    function step(acc, args)
        (s_prev, traj_acc, obs_B_acc, obs_F_acc, obs_A_acc) = acc
        (stride_idx, Φ) = args
        out = plant_rollout(s_prev, Φ, params, dt, hash((base_key, :stride, stride_idx)))
        return (out.final_state,
                vcat(traj_acc,  out.trajectory),
                vcat(obs_B_acc, out.obs_B),
                vcat(obs_F_acc, out.obs_F),
                vcat(obs_A_acc, out.obs_A))
    end
    init = (s0, Matrix{Float64}(undef, 0, 3), Float32[], Float32[], Float32[])
    final = foldl(step, enumerate(plan_phis); init = init)
    return (final_state = final[1], trajectory = final[2],
            obs_B = final[3], obs_F = final[4], obs_A = final[5])
end
```

A clean `foldl`. The `vcat`s are O(n²) total because we copy on every step — performance loss accepted (the user explicitly said so). If this ever needs to scale, swap for an `accumulate` that returns the per-stride trajectories and concat at the end (one O(n) `vcat`).

**No `advance!`. No `advance_subdaily!`. No history dict. No `MersenneTwister`. No `mutable struct`. No `finalise`.** Bench writes its own JLD2 at the end as a single explicit I/O step.

#### 3. `models/fsa_high_res/estimation.jl` (NEW, FUNCTIONAL; ~80 LOC)

The filter side. All functions pure.

- `propagate(particles::Matrix{Float64}, Φ_t::Float64, params, dt::Float64, key::UInt64) → Matrix{Float64}` — prior-predictive. Returns a NEW particles matrix, doesn't mutate the input. From each particle's `(B, F, A)`, runs one v1-G0 SDE step. No locally-guided proposal needed — the obs is so informative (direct Gaussian on each latent) that prior-predictive doesn't degenerate.
- `obs_log_weight(particles::Matrix{Float64}, obs::NamedTuple, params) → Vector{Float64}` — pure. `Σ_X log N(obs_X; X_particle, σ_X_obs²)` for `X ∈ {B, F, A}`. Three `Normal.logpdf` lines, allocates one fresh result vector.
- `PARAM_PRIOR_CONFIG::Vector{Tuple{Symbol, AbstractPrior}}` — `const`, 14 lognormal/normal priors centred on truth (use the same prior-μ and prior-σ values v2 uses for each shared param). The 3 obs-noise params are **fixed at truth** by default (one less dimension for the filter to estimate); add to priors later if we want to test obs-noise estimation.
- `PARAM_NAMES::Vector{Symbol}` — `const`, symbol vector matching `PARAM_PRIOR_CONFIG`.

No `propagate_fn!`, no in-place updates, no module-level RNG.

#### 4. `models/fsa_high_res/gpu_pf.jl` (NEW, FUNCTIONAL FACADE; ~150 LOC)

GPU-batched filter target. The KernelAbstractions kernel is inherently mutating (writes into preallocated CuArrays); the public API is a pure facade around it.

**Immutable target record:**
```julia
struct FSAGPUTarget
    K_per_chain::Int
    M_max::Int
    T_steps::Int
    dt::Float32
    rbf_obs_buffers::CuArray  # zero-init at construction; stays the same shape
    propagate_kernel::Any     # KernelAbstractions kernel handle
    # No mutable fields. Buffers exist but their contents are written via kernels
    # invoked by the pure facade below; not visible to callers.
end

# Construction is pure — same target struct returned for the same inputs.
function FSAGPUTarget(; K_per_chain::Int, M_max::Int, T_steps::Int, dt::Real, ...)
    ...
end
```

**Pure log-density evaluator (the public API):**
```julia
function gpu_log_density(target::FSAGPUTarget,
                         U::Matrix{Float64},
                         grid_obs::NamedTuple,    # (obs_B::Vector{Float32}, obs_F::Vector{Float32}, obs_A::Vector{Float32}, B_init::Float64, F_init::Float64, A_init::Float64)
                         key::UInt64)             # CRN seed for inner-PF noise
    # Allocate fresh output buffer.
    log_lik = zeros(Float64, size(U, 1))
    # Dispatch the kernel — it mutates a private CuArray which is then copied
    # to log_lik via Array(view(...)). Caller does NOT see the CuArray.
    _run_segmented_smc(target, U, grid_obs, key, log_lik)
    return log_lik
end
```

`_run_segmented_smc` is private (underscore prefix). It calls `run_segmented_smc_step!` from `julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl` (model-agnostic, full reuse). The exclamation mark is because that's the name in the framework, but the framework call is hidden inside the pure facade.

No `update_window_obs!`. The window obs are passed as a function argument every call. Yes, this re-uploads the obs to the GPU per call — perf loss accepted, and the upload is small (3 × T_steps Float32 vs 30+ params × M chains, so a noise-level cost).

**One propagate-segment kernel** (`@kernel function propagate_segment_kernel!`) — written from scratch for v1.5's three-channel Gaussian obs. G0 drift + state-dep diffusion + three Gaussian log-pdfs added per bin. Significantly shorter than v2's (no sequential Kalman fusion / no Cholesky-3 / no Bernoulli sleep likelihood). The kernel itself is imperative (KernelAbstractions requirement); but it is called only from inside the pure facade.

**Pure gradient batcher:**
```julia
function gpu_grads(target::FSAGPUTarget, U::Matrix{Float64}, grid_obs::NamedTuple,
                   h::Float64, key::UInt64) → (vals::Vector{Float64}, grads::Matrix{Float64})
    # Builds the M(1+2d) perturbation matrix functionally (no in-place expansion of a buffer)
    # via a comprehension; calls gpu_log_density once on the full block; central-diffs out
    # the gradients. Returns fresh outputs. No mutation of inputs.
end
```

**Pure HMC move:**
```julia
function parallel_hmc_one_move(U::Matrix{Float64}, target::FSAGPUTarget,
                                grid_obs::NamedTuple,
                                ε::Float64, L::Int, prior_means, prior_sigmas,
                                key::UInt64) → (U_new::Matrix{Float64}, n_acc::Int)
    # Leapfrog with central-diff gradient via gpu_grads.
    # Acceptance computed from key-seeded uniform draws.
    # Returns a NEW U matrix (or U if rejected). No mutation.
end
```

The framework's `parallel_hmc_one_move!` exists but mutates U. v1.5 wraps it: copy U, run the framework's mutating version on the copy, return the copy. Perf-loss-accepted.

#### 5. `models/fsa_high_res/FSAHighRes.jl` (NEW; ~30 LOC)

Module aggregator. Copy v2's [FSAHighRes.jl](version_2_Julia/models/fsa_high_res/FSAHighRes.jl) layout, swap module includes:
```julia
include("_dynamics.jl")
include("simulation.jl")        # defines BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE, gen_obs_bfa
include("_plant.jl")            # depends on Simulation for BINS_PER_DAY + gen_obs_bfa
include("gpu_control.jl")       # v1's GPU kernel, only control file needed
include("estimation.jl")
include("gpu_pf.jl")
```
Re-export `TRUTH_PARAMS`, `INIT_STATE`, `BINS_PER_DAY`, `DT_BIN_DAYS`, `StepwisePlant`, `advance_subdaily!`, `FSAGPUTargetBatched`, `FSAControlGPUTarget`, `make_log_density_fn`, `propagate_fn`, `PARAM_NAMES`, etc. (No `advance!`, no `expand_daily_phi_to_subdaily`, no `BurstEnvelope` — those don't exist in v1.5.)

#### 6. `tools/bench_smc_full_mpc_fsa_gpu.jl` (NEW, FUNCTIONAL; ~400 LOC)

Closed-loop MPC bench, redesigned in functional style. The bench is the SINGLE place where I/O happens (JLD2 write, log lines, plot generation).

**Top-level shape:**
```julia
function main(args::NamedTuple)
    cfg = build_config(args)                                # pure
    s0  = PlantState(SVector{3}(0.05, 0.30, 0.10), 0)       # immutable initial state

    # The closed-loop bench is one big foldl over strides. Each iteration:
    #   - takes the current bench state (filter posterior, plan, plant_state)
    #   - returns a new bench state + a per-stride log entry
    # The final accumulator carries the full history (concatenated) — no global mutation.

    init = (
        plant_state    = s0,
        plan_phi       = fill(Float32(1.0), cfg.T_total_bins),   # baseline-Φ initially
        plan_offset    = 0,
        filter_post    = nothing,                                # cold-start
        traj_history   = Matrix{Float64}(undef, 0, 3),
        obs_history    = (B=Float32[], F=Float32[], A=Float32[]),
        per_stride_log = NamedTuple[],
    )
    final = foldl((acc, s) -> stride_step(acc, s, cfg), 1:cfg.n_strides; init=init)

    write_outputs(final, cfg)                              # the only I/O
end
```

**Pure per-stride step:**
```julia
function stride_step(acc, stride_idx::Int, cfg)
    # 1. Slice next stride's Φ from the current plan.
    phi = acc.plan_phi[acc.plan_offset+1 : acc.plan_offset+cfg.stride_bins]

    # 2. Plant rollout — pure
    p = plant_rollout(acc.plant_state, phi, cfg.params, cfg.dt,
                      hash((cfg.seed, :plant, stride_idx)))

    # 3. Filter window — pure
    grid_obs = window_grid_obs(p.obs_B, p.obs_F, p.obs_A, p.final_state)
    filter_out = run_outer_smc(cfg.target, cfg.n_smc, cfg.outer_smc_cfg, grid_obs,
                               acc.filter_post,
                               hash((cfg.seed, :filter, stride_idx)))

    # 4. Maybe replan (closed-loop only)
    new_plan, new_offset = if !cfg.open_loop && stride_idx % cfg.replan_K == 0
        post_params = posterior_mean(filter_out.U)
        plan = run_controller_smc(cfg.ctrl_target, post_params, cfg.ctrl_cfg,
                                  hash((cfg.seed, :ctrl, stride_idx)))
        (plan.phi_per_bin, 0)
    else
        (acc.plan_phi, acc.plan_offset + cfg.stride_bins)
    end

    return (
        plant_state    = p.final_state,
        plan_phi       = new_plan,
        plan_offset    = new_offset,
        filter_post    = filter_out.U,
        traj_history   = vcat(acc.traj_history, p.trajectory),
        obs_history    = (B = vcat(acc.obs_history.B, p.obs_B),
                          F = vcat(acc.obs_history.F, p.obs_F),
                          A = vcat(acc.obs_history.A, p.obs_A)),
        per_stride_log = vcat(acc.per_stride_log, [(stride=stride_idx, n_temp=filter_out.n_temp, ...)]),
    )
end
```

Each ingredient (`run_outer_smc`, `run_controller_smc`, `posterior_mean`, `window_grid_obs`) is a pure function returning new values. No `target` mutation, no `plant` mutation, no growing-history-dict. The final `foldl` accumulator IS the history.

`write_outputs(final, cfg)` — JLD2 dump of `final.traj_history`, `final.obs_history`, `final.per_stride_log`, `final.filter_post`. Plot regeneration via the existing `plot_state_traces.jl` (which is read-only here). One explicit I/O effect at the end.

CLI parsing the same as v2's bench (dict-based, set `FSA_STEP_MINUTES` BEFORE the model import). Same `--open-loop` switch (per the user's earlier callout). Same `script -qfc` tty-wrap recipe from the recent sweep work for live progress visibility.

The bench-script is materially shorter than v2's (~400 LOC vs ~700) because there's no `mutable mpc_plant` / `mutable base_plant` to manage state for, no per-channel obs slicing across HR/sleep/stress/steps, no posterior-particle pre-allocation matrix to mutate, and no `current_phi_per_bin` mutable buffer.

#### 7. `Project.toml`, `Manifest.toml`

Copy v2's verbatim. Same SMC2FC framework dep, same JLD2/Plots/CUDA/KernelAbstractions/etc.

## Dependency reuse — what comes from `julia/SMC2FC/` unchanged

The framework package is model-agnostic. v1.5 reuses without copy:

- `Filtering/GPUSegmentedPF.jl` — segmented PF, Liu-West shrinkage, OT rescue.
- `Filtering/Kernels.jl` — ESS, Silverman bandwidth.
- `Filtering/OT.jl` — Sinkhorn + low-rank OT.
- `Control/GPUControlSMC.jl` — `run_tempered_smc_gpu` (parallel-chains tempered SMC² + ChEES-HMC for the controller's RBF θ).
- `SMC2/HMC.jl`, `TemperedSMC.jl` — outer SMC² for the filter posterior.
- `Types.jl`, `EstimationModel.jl`, prior types.

This is the same reuse pattern v2 has — proves the framework boundaries are correctly drawn, no new framework code needed.

## Pros (why this is worth doing)

1. **Functional + stateless — readable, composable, testable.** Every piece of the closed-loop pipeline is a pure function with explicit inputs and outputs. Junior engineers can read the bench top-to-bottom and follow data flow without having to mentally track which mutable struct got mutated when. Each function is unit-testable in isolation (no need to set up a stateful plant + filter + RNG before testing one piece). This is the main reason for the redesign and supersedes the perf trade-off.
2. **Sidesteps v2's §2.7 fp32 cancellation bug entirely.** v1's drift has no `(F − F_typ)²` subtraction in fp32 — the cost surface is numerically clean by construction. The 0 % → 5 % → 10 % closed-loop oscillation seen in Phase 2 (mostly attributable to fp32 noise injecting basin-jumps) should drop dramatically.
2. **Sidesteps the §2.15 magnitude gap as far as we can isolate it.** Phase 2 just proved the gap is invariant under saturation. The remaining axes to vary are reparametrisation (G0 vs G1) and obs model (3-Gaussian vs 4-channel). v1.5 changes both at once. If v1.5 hits Python's +51 % gain, the bug is one of those two; if it doesn't, the bug is structural in the closed-loop pipeline machinery.
3. **Smallest possible obs model — maximum identifiability.** Three independent Gaussian channels on (B, F, A) every bin means every latent is directly observed with known noise. No gating, no missing-data, no compound observables. The filter posterior should shrink fast and stay tight.
4. **Mostly reuse, very little new code.** ~400 LOC of new model code; everything else (framework, kernels, OT rescue, bench wiring) is copied or imported. Minimal maintenance surface.
5. **Easy end-to-end verification against truth.** Generate obs from a truth-params plant; filter at saturated config; check posterior spread shrinks around truth across windows; check controller ∫A_MPC dt approaches the analytic optimum. No HR/sleep/stress/steps confounders. Each piece is independently checkable.
6. **A clean place to test the host-loop fixes from `HOST_LOOP_CATALOGUE.md`.** v1.5's filter is much simpler than v2's, so the per-chain host loops in `parallel_hmc_one_move!` and `gpu_log_density_batched` are easier to inspect and patch first in v1.5, then port to v2.

## Cons (where this falls short)

1. **Performance loss.** Functional discipline costs allocations: a fresh CuArray per `gpu_log_density` call, fresh `Matrix{Float64}` per HMC move (no in-place leapfrog), `vcat` history concatenation in `O(n²)` over strides. Per the user's instruction this is **acceptable** for v1.5 (which is a diagnostic harness, not a production target). If v1.5 graduates and perf matters, swap the `vcat`-fold for an `accumulate`-then-concat-once and add a buffer pool inside the GPU facade — neither change affects the public API.
2. **Some framework code is still imperative.** `julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl` has `!`-suffix mutating functions (`run_segmented_smc_step!`). v1.5 wraps these inside its own pure facade, but the framework itself isn't being rewritten — the host-loop catalogue from the recent sweep work still applies. v1.5's discipline is "no public API mutates"; the framework is allowed to remain imperative behind that wall.
3. **Synthetic-only.** Direct Gaussian observation of (B, F, A) is biologically unrealistic. v1.5 cannot validate against real wearable data — it's a clean-room test bed only. The path to real data still goes through v2's HR/sleep/stress/steps obs model.
2. **Risk of over-identification.** Three direct latent observations every bin makes the filter posterior much tighter than v2's (which has gated sparse compound obs). v1.5's closed-loop performance is therefore an upper bound on what's achievable; closing the gap to Python in v1.5 is necessary but not sufficient for closing it in v2.
3. **G0 may have its own filter problem.** Writeup §2.7 closing paragraph notes v2 went to G1 *because* of FIM-rank identifiability of the static parameters in the FILTER. If G0 is filter-degenerate on `(kappa_B, epsilon_A)` or `(mu_F, mu_FF)`, the v1.5 filter posterior won't shrink and the controller will be fed wide priors → controller behaves like a fixed-prior open-loop. Need to FIM-check at truth before committing to closed-loop runs (small, cheap, ~10 min of work).
4. **Three FSA codebases now (v1, v1.5, v2).** Each has its own dynamics form + cost form + bench scripts. Risk of skew if all three are kept current. Mitigation: declare v1 frozen (open-loop reference only); v1.5 is a debugging aid; v2 is the production target. Migrate fixes from v1.5 → v2 explicitly when validated.
5. **Doesn't directly fix v2.** Even if v1.5 closes the gap to Python, the fix has to migrate back to v2's reparametrisation and obs model — non-trivial, and not automatic. The justification for v1.5 is "fast diagnostic harness", not "production target".
6. **Cost form has a `λ_Φ·∫Φ²` term v2 doesn't.** v1's `control.jl` has three cost terms vs v2's two. For apples-to-apples comparison we set `λ_Φ = 0`, but the full kernel still does the multiplications. Tiny perf cost, no functional difference.

## Decisions locked in (user-confirmed)

A) **Obs noise: PINNED at truth (default 0.005 each).** The filter estimates the 14 dynamics params only — `σ_B_obs / σ_F_obs / σ_A_obs` are fixed across the whole codebase. This keeps the filter tight, the experiment controlled, and the v1.5 → v2 migration story simple. (We can flip to estimating obs noise later if needed; for v1.5 it's pinned.)

B) **FIM analysis is a HARD GATE before any closed-loop run.** The user has flagged this as very important. The v1.5 implementation order is:
   1. Build `_dynamics.jl`, `simulation.jl`, `_plant.jl`, `estimation.jl` (just enough to evaluate the obs likelihood at truth).
   2. Run the FIM analysis at TRUTH_PARAMS over a 1-day window, three Gaussian channels (closed-form FIM — no sampling needed for diagonal Gaussian obs, the FIM is `Σ_t J_t^T diag(1/σ_X_obs²) J_t` where `J_t = ∂(B,F,A)/∂params` is the sensitivity tensor at truth integrated through the SDE).
   3. Check the FIM rank and the spectrum of small eigenvalues / large condition number.
   4. **GATE**:
      - If FIM is full-rank with reasonable conditioning across all 14 dynamics params → **continue** with the rest of v1.5 (filter / GPU PF / bench).
      - If FIM is rank-deficient or has dramatic eigenvalue collapse on any subset → **STOP, report findings to the user, and wait.** The user has explicitly said: *"if this flags identifiability issues STOP and report to me and we will reparametrize"*. Do not push forward with a closed-loop bench on a non-identifiable parametrisation; that's how v2's posterior pathologies happen.

   This step gets a small dedicated tool: `tools/fim_check.jl` (~150 LOC, pure functional). It uses ForwardDiff to compute `∂(B,F,A)/∂params` along an EM trajectory under Φ=1 from `INIT_STATE` for one day at h=60min (24 bins), then assembles the 14×14 FIM matrix, prints rank + eigenvalue table + worst-conditioned parameter pair, and exits.

C) **Run at the FAST default config, NOT the saturated cell.** User: *"FAST runs at this stage are more important to catch bugs — and test that sensible results are produced."* So all v1.5 closed-loop bench runs use:
   ```
   --N-smc 32 --K-per-chain 400 --step-minutes 60 --replan-K 2 --T-days 14
   ```
   (~4 min wall per closed-loop run from Phase 1's measurements.) Saturation isn't the question for v1.5 — bug detection and qualitative correctness is. We can always re-run at `(N=256, K=1600)` later for a head-to-head against v2 if the qualitative picture is right.

## Verification (in this exact order)

Layered, cheapest first. Each step gates the next.

1. **psim sanity (~1 min).** Drive `plant_rollout` for 14 days at constant Φ=1 from `INIT_STATE`. Plot trajectory + obs samples. Should see B grow ~linearly, F oscillate, A drift down (fitness vs the de-trained-init baseline). Same qualitative shape as v1's open-loop sim. **PASS gate**: trajectory shapes look right qualitatively.

2. **FIM rank at truth (~5 min, HARD GATE).** Run `tools/fim_check.jl`. Closed-form 14×14 FIM at TRUTH_PARAMS over a 1-day Φ=1 window with three Gaussian channels.
   - **PASS gate**: FIM is full-rank with reasonable conditioning across all 14 dynamics params → continue to step 3.
   - **STOP gate**: FIM is rank-deficient or has eigenvalue collapse → halt v1.5 work, surface the degenerate parameter combination(s) to the user, wait for re-parametrisation guidance. Do not attempt to "work around" — the user has been explicit.

3. **Controller-only test (~5 min).** Adapt v2's `tools/test_max_A_only.jl` against v1.5's `gpu_control.jl`. The controller alone (no filter) should produce the recovery → overload schedule from `INIT_STATE + TRUTH_PARAMS`. Confirms the GPU cost kernel + RBF decoder are wired correctly.

4. **Filter-only test (~5 min).** Generate 14 days of obs from a truth-params `plant_rollout`; run the SMC² filter once across the full bench at the fast default config (N=32, K=400). Check posterior-mean params sit within the prior σ band of truth, and posterior spread shrinks across windows. Confirms `gpu_pf.jl` + segmented PF + `propagate` + `obs_log_weight` are wired.

5. **Bench open-loop test (~5 min).** Run `bench_smc_full_mpc_fsa_gpu.jl --open-loop true --T-days 14 --N-smc 32 --K-per-chain 400`. One up-front plan, applied for 14 days. Should match step 3's schedule structurally. Confirms the open-loop branch wiring before turning on replans.

6. **Closed-loop end-to-end (~5 min at fast default).** Run `bench_smc_full_mpc_fsa_gpu.jl --T-days 14 --N-smc 32 --K-per-chain 400 --step-minutes 60 --replan-K 2`. Headline metric: `(∫A_MPC − ∫A_baseline) / ∫A_baseline` over 14 days. Bug-catching first; head-to-head against v2's +3.04 % and Python's +51 % is reported but not the gating criterion. Sensible result = "B trajectory rises, A trajectory rises, F doesn't blow past F_max". If the bench produces qualitative nonsense, debug at the fast config before scaling.

Total verification budget: ~25 min wall if all gates pass cleanly. Steps 2 and 6 are where the project most likely halts and surfaces a finding to the user.

## Out of scope (deliberately)

- Real wearable obs. v1.5 is synthetic-only; that's the point.
- Information-aware filtering or adaptive obs scheduling.
- Reuse of v2's reparametrised dynamics. v1.5 commits to G0 specifically to test whether reparametrisation matters.
- Performance work on the host-loop ceiling. The §2.12 catalogue from the saturation sweep applies to v1.5 too, but fixing it is a separate task.
- Tests directory (`tests/`). v1.5 will use the layered verification above; pytest-style suite can come later if v1.5 graduates from "diagnostic" to "production-relevant".

## Critical files referenced

**Read-only sources:**
- [version_1_Julia/models/fsa_high_res/_dynamics.jl](version_1_Julia/models/fsa_high_res/_dynamics.jl) — G0 SDE.
- [version_1_Julia/models/fsa_high_res/control.jl](version_1_Julia/models/fsa_high_res/control.jl) — CPU control spec.
- [version_1_Julia/models/fsa_high_res/gpu_control.jl](version_1_Julia/models/fsa_high_res/gpu_control.jl) — GPU control kernel (Eq 37 cost).
- [version_1_Julia/tools/bench_smc_control_fsa_gpu.jl](version_1_Julia/tools/bench_smc_control_fsa_gpu.jl) — open-loop reference.
- [version_2_Julia/models/fsa_high_res/_plant.jl](version_2_Julia/models/fsa_high_res/_plant.jl) — plant *struct shape* reference only (the v1.5 plant is written fresh, not copied; v2's `advance!`/`advance_subdaily!` and the burst-envelope expansion are dropped).
- [version_2_Julia/models/fsa_high_res/gpu_pf.jl](version_2_Julia/models/fsa_high_res/gpu_pf.jl) — filter target template.
- [version_2_Julia/models/fsa_high_res/estimation.jl](version_2_Julia/models/fsa_high_res/estimation.jl) — propagate_fn / obs_log_weight_fn template.
- [version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl](version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) — closed-loop bench template.
- [julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl](julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl) — model-agnostic framework, full reuse.
- [version_2_Julia/docs/julia_fsa_writeup.pdf](version_2_Julia/docs/julia_fsa_writeup.pdf) §2.7 (G1 fp32 bug), §2.15 (magnitude gap row 10/11), §5 (running experiments).

**New files to add (all under `version_1_5_Julia/`):**

```
version_1_5_Julia/
├── Project.toml                                # copy from v2 + add Setfield, StableRNGs, StaticArrays, ForwardDiff
├── Manifest.toml                               # copy from v2 (will need re-resolve)
├── models/fsa_high_res/
│   ├── _dynamics.jl                            # copy from v1 (verbatim)
│   ├── simulation.jl                           # NEW (~90 LOC) — owns BINS_PER_DAY, pure
│   ├── _plant.jl                               # NEW (~100 LOC) — pure, immutable PlantState
│   ├── gpu_control.jl                          # copy from v1 (verbatim) — only control file
│   ├── estimation.jl                           # NEW (~80 LOC) — pure propagate / obs_log_weight
│   ├── gpu_pf.jl                               # NEW (~150 LOC) — pure facade over kernel
│   └── FSAHighRes.jl                           # NEW (~30 LOC)
├── tools/
│   ├── fim_check.jl                            # NEW (~150 LOC) — FIM gate (HARD)
│   └── bench_smc_full_mpc_fsa_gpu.jl           # NEW (~400 LOC) — closed-loop bench, foldl-based
└── outputs/
    └── fsa_high_res/                           # auto-created at run time
```

Total new LOC: ~1000 (model: 450, tools: 550 — bench 400, FIM 150).
Reused LOC (v1 verbatim): ~700 (`_dynamics.jl` + `gpu_control.jl`).
v2 LOC carried over: zero (no `_phi_burst.jl`, no v2 plant, no v2 estimation).
Framework reuse (no copy): all of `julia/SMC2FC/` — wrapped behind v1.5's pure facade.
