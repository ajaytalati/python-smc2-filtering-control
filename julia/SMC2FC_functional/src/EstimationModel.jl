"""
    EstimationModel.jl

The "filter contract" struct that the generic SMC² + particle-filter
machinery consumes. Mirrors `smc2fc/estimation_model.py` (Python reference)
and `julia/SMC2FC/src/EstimationModel.jl` (existing port).

Function-typed fields are parameterised so each call site is compile-time
specialised — equivalent to the JIT trace specialisation JAX gets for free.
"""

# ── EstimationModel struct ───────────────────────────────────────────────────

"""
    EstimationModel{Fp,Fd,Fo,Fa,Fs,...}

Frozen, immutable struct describing a stochastic state-space model to the
generic SMC² / particle-filter machinery.

# Fields (selected)
- `name::String`, `version::String`: human-readable identifiers.
- `n_states::Int`: state-space dimension.
- `n_stochastic::Int`: number of stochastic latent dimensions.
- `stochastic_indices::Vector{Int}`: which state components carry noise.
- `state_bounds::Vector{Tuple{Float64,Float64}}`: hard physical bounds
    (used for resample-clamping and chance constraints).
- `param_priors::Vector{Tuple{Symbol,PriorType}}`: estimated parameter
    priors. Order is significant — defines the layout of θ.
- `init_state_priors::Vector{Tuple{Symbol,PriorType}}`: priors for the
    estimated initial-state components. Concatenated after `param_priors`
    in the θ layout.
- `frozen_params::Dict{Symbol,Float64}`: parameters held fixed.
- `propagate_fn::Fp`: dynamics step.
    Signature: `(y, t, dt, params, grid_obs, step_k, σ_diag, noise, rng)
    -> (x_new, pred_lw)`.
- `diffusion_fn::Fd`: `(params) -> Vector(n_states)` of diffusion scales.
- `obs_log_weight_fn::Fo`: `(x_new, grid_obs, step_k, params) -> Float64`,
    log p(y_k | x_k, θ).

# Optional fields
The optional fields default to `nothing` and are checked at call sites.
They cover direct-scan log-density (`imex_step_fn`, `obs_log_prob_fn`,
`make_init_state_fn`), synthetic data (`obs_sample_fn`, `forward_sde_fn`),
EKF (`gaussian_obs_fn`, `init_cov_fn`), marginal SGR
(`dynamic_kernel_log_density_fn`, `proposal_log_density_fn`), I/O
(`get_init_theta_fn`), and GPU-batched forms (`propagate_batch_fn`,
`obs_log_weight_batch_fn`).

# Notes
- The 16+ type parameters encode the function signatures so each call
    site is compile-time specialised — no boxing in hot loops.
- All fields are immutable; the struct is functional by construction.
"""
Base.@kwdef struct EstimationModel{Fp,Fd,Fo,Fa,Fs,
                                   Fld,Fpld,Fls,Fobs,Fsamp,
                                   Ffwd,Fgo,Fic,Fdkl,Fpld2,
                                   Fpb,Fob}
    # ── Metadata ─────────────────────────────────────────────────────────────
    name::String
    version::String

    # ── State space ──────────────────────────────────────────────────────────
    n_states::Int
    n_stochastic::Int
    stochastic_indices::Vector{Int}
    state_bounds::Vector{Tuple{Float64,Float64}}

    # ── Parameters ───────────────────────────────────────────────────────────
    param_priors::Vector{Tuple{Symbol,PriorType}}
    init_state_priors::Vector{Tuple{Symbol,PriorType}}
    frozen_params::Dict{Symbol,Float64}

    # ── Dynamics ─────────────────────────────────────────────────────────────
    propagate_fn::Fp
    diffusion_fn::Fd

    # ── Observation model ────────────────────────────────────────────────────
    obs_log_weight_fn::Fo

    # ── Grid alignment ───────────────────────────────────────────────────────
    align_obs_fn::Fa
    shard_init_fn::Fs

    # ── Optional: direct-scan log-density ────────────────────────────────────
    imex_step_fn::Fld         = nothing
    obs_log_prob_fn::Fpld     = nothing
    make_init_state_fn::Fls   = nothing

    # ── Optional: synthetic-data sampling ────────────────────────────────────
    obs_sample_fn::Fobs       = nothing
    forward_sde_fn::Fsamp     = nothing

    # ── Optional: EKF ────────────────────────────────────────────────────────
    gaussian_obs_fn::Ffwd     = nothing
    init_cov_fn::Fgo          = nothing

    # ── Optional: marginal SGR ───────────────────────────────────────────────
    dynamic_kernel_log_density_fn::Fic   = nothing
    proposal_log_density_fn::Fdkl        = nothing

    # ── Optional: I/O ────────────────────────────────────────────────────────
    get_init_theta_fn::Fpld2   = nothing

    # ── Optional: GPU-batched forms ──────────────────────────────────────────
    propagate_batch_fn::Fpb           = nothing
    obs_log_weight_batch_fn::Fob      = nothing

    # ── Grid obs structure ───────────────────────────────────────────────────
    exogenous_keys::Vector{Symbol} = Symbol[]
end


# ── Derived properties ──────────────────────────────────────────────────────

"""
    n_params(m::EstimationModel) -> Int

Number of estimated parameters (`length(m.param_priors)`).

# Arguments
- `m::EstimationModel`: the model.

# Returns
- `Int`: estimated-parameter count.
"""
n_params(m::EstimationModel)      = length(m.param_priors)

"""
    n_init_states(m::EstimationModel) -> Int

Number of estimated initial-state components (`length(m.init_state_priors)`).

# Arguments
- `m::EstimationModel`: the model.

# Returns
- `Int`: estimated init-state count.
"""
n_init_states(m::EstimationModel) = length(m.init_state_priors)

"""
    n_dim(m::EstimationModel) -> Int

Total dimension of θ — estimated parameters plus estimated init-state
components.

# Arguments
- `m::EstimationModel`: the model.

# Returns
- `Int`: `n_params(m) + n_init_states(m)`.
"""
n_dim(m::EstimationModel)         = n_params(m) + n_init_states(m)

"""
    all_names(m::EstimationModel) -> Vector{Symbol}

Concatenated list of estimated-parameter names then estimated-init-state
names. The order matches the layout of θ used everywhere in the library.

# Arguments
- `m::EstimationModel`: the model.

# Returns
- `Vector{Symbol}`: names in θ order.
"""
function all_names(m::EstimationModel)
    return vcat(first.(m.param_priors), first.(m.init_state_priors))
end

"""
    all_priors(m::EstimationModel) -> Vector{PriorType}

Vector of `PriorType` instances for every estimated dimension, in θ order.
Used by `Transforms.jl` to build the bijection list.

# Arguments
- `m::EstimationModel`: the model.

# Returns
- `Vector{PriorType}`: priors in θ order.
"""
function all_priors(m::EstimationModel)
    return PriorType[last(p) for p in vcat(m.param_priors, m.init_state_priors)]
end
