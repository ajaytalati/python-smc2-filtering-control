# EstimationModel.jl — Julia version of `smc2fc/estimation_model.py`.
#
# A frozen struct holding the model-specific functions and prior specs the
# generic SMC²/PF + control machinery consumes. Mirrors the Python contract;
# the Julia version uses parametric function-typed fields so calls are
# compile-time devirtualised (no boxing).
#
# Charter: LaTex_docs/julia_port_charter.pdf §15.2.

# ── EstimationModel struct ───────────────────────────────────────────────────
# All fields are typed; optional fields default to `nothing` and are checked
# at call sites. Function-typed fields are parameterised so the JIT can specialise.

Base.@kwdef struct EstimationModel{Fp,Fd,Fo,Fa,Fs,
                                   Fld,Fpld,Fls,Fobs,Fsamp,
                                   Ffwd,Fgo,Fic,Fdkl,Fpld2}
    # ── Metadata ─────────────────────────────────────────────────────────────
    name::String
    version::String

    # ── State space ──────────────────────────────────────────────────────────
    n_states::Int
    n_stochastic::Int
    stochastic_indices::Vector{Int}
    state_bounds::Vector{Tuple{Float64,Float64}}

    # ── Parameters ───────────────────────────────────────────────────────────
    # `param_priors` and `init_state_priors` are vectors of (name, PriorType)
    # tuples. The order is significant — it defines the layout of θ.
    param_priors::Vector{Tuple{Symbol,PriorType}}
    init_state_priors::Vector{Tuple{Symbol,PriorType}}
    frozen_params::Dict{Symbol,Float64}

    # ── Dynamics ─────────────────────────────────────────────────────────────
    propagate_fn::Fp     # (y, t, dt, params, grid_obs, step_k, σ_diag, noise, rng) -> (x_new, pred_lw)
    diffusion_fn::Fd     # (params) -> Vector(n_states)

    # ── Observation model ────────────────────────────────────────────────────
    obs_log_weight_fn::Fo   # (x_new, grid_obs, step_k, params) -> scalar log-weight

    # ── Grid alignment (numpy-side, called once) ─────────────────────────────
    align_obs_fn::Fa        # (obs_data, t_steps, dt_hours) -> Dict
    shard_init_fn::Fs       # (time_offset, params, exogenous, global_init) -> init_states

    # ── Optional: direct-scan log-density (matches Python v6.0+) ─────────────
    imex_step_fn::Fld         = nothing       # (y, t, dt, params, grid_obs) -> y_next
    obs_log_prob_fn::Fpld     = nothing       # (y, grid_obs, k, params) -> scalar
    make_init_state_fn::Fls   = nothing       # (init_estimates, params) -> y0

    # ── Optional: synthetic-data sampling (matches Python v6.3) ──────────────
    obs_sample_fn::Fobs       = nothing       # (y, exog, k, params, rng_key) -> Dict
    forward_sde_fn::Fsamp     = nothing       # (init, params, exog, dt, n_steps) -> trajectory

    # ── Optional: EKF (matches Python v6.4) ──────────────────────────────────
    gaussian_obs_fn::Ffwd     = nothing
    init_cov_fn::Fgo          = nothing

    # ── Optional: marginal SGR (matches Python v6.4) ─────────────────────────
    dynamic_kernel_log_density_fn::Fic   = nothing
    proposal_log_density_fn::Fdkl        = nothing

    # ── Optional: I/O (matches Python) ───────────────────────────────────────
    get_init_theta_fn::Fpld2   = nothing

    # ── Grid obs structure ───────────────────────────────────────────────────
    exogenous_keys::Vector{Symbol} = Symbol[]
end


# ── Derived properties (mirror Python @property accessors) ───────────────────

n_params(m::EstimationModel)      = length(m.param_priors)
n_init_states(m::EstimationModel) = length(m.init_state_priors)
n_dim(m::EstimationModel)         = n_params(m) + n_init_states(m)

"""
    all_names(m::EstimationModel)

Concatenated list of estimated parameter names then estimated init-state names.
The order matches the layout of θ.
"""
function all_names(m::EstimationModel)
    return vcat(first.(m.param_priors), first.(m.init_state_priors))
end

"""
    all_priors(m::EstimationModel)

Vector of `PriorType` instances for every estimated dimension, in θ order.
Used by Transforms.jl to build the bijection list.
"""
function all_priors(m::EstimationModel)
    return PriorType[last(p) for p in vcat(m.param_priors, m.init_state_priors)]
end
