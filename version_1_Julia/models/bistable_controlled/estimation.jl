# SMC² EstimationModel for the controlled bistable model.
# Mirrors version_1/models/bistable_controlled/estimation.py.

module Estimation

using SMC2FC: EstimationModel, PriorType, LogNormalPrior, NormalPrior
import ..Dynamics
import ..Simulation: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A

export build_estimation_model, PARAM_PRIOR_CONFIG, INIT_STATE_PRIOR_CONFIG

# ── Priors centred at truth, log-normal σ = 0.5 ─────────────────────────────

const PARAM_PRIOR_CONFIG = Tuple{Symbol,PriorType}[
    (:alpha,     LogNormalPrior(log(1.0),  0.5)),
    (:a,         LogNormalPrior(log(1.0),  0.5)),
    (:sigma_x,   LogNormalPrior(log(0.10), 0.5)),
    (:gamma,     LogNormalPrior(log(2.0),  0.5)),
    (:sigma_u,   LogNormalPrior(log(0.05), 0.5)),
    (:sigma_obs, LogNormalPrior(log(0.20), 0.5)),
]

const INIT_STATE_PRIOR_CONFIG = Tuple{Symbol,PriorType}[
    (:x_0, NormalPrior(-1.0, 0.3)),
    (:u_0, NormalPrior( 0.0, 0.1)),
]

const _PI = Dict{Symbol,Int}(
    :alpha => 1, :a => 2, :sigma_x => 3,
    :gamma => 4, :sigma_u => 5, :sigma_obs => 6,
)

# ── Bootstrap propagate: hand-rolled EM step on (x, u) ──────────────────────
# Bistable dynamics are nonlinear (cubic in x); locally-optimal proposal is
# not closed-form, so we use the bootstrap proposal and rely on more
# particles + Liu-West smoothing to keep the PF healthy.

function propagate_fn(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng_)
    p_named = (alpha = params[_PI[:alpha]], a = params[_PI[:a]],
                gamma = params[_PI[:gamma]], sigma_x = params[_PI[:sigma_x]],
                sigma_u = params[_PI[:sigma_u]], sigma_obs = params[_PI[:sigma_obs]])
    T_i  = haskey(grid_obs, :T_intervention) ? grid_obs[:T_intervention] : EXOGENOUS_A.T_intervention
    u_on = haskey(grid_obs, :u_on)           ? grid_obs[:u_on]           : EXOGENOUS_A.u_on
    y_new = Dynamics.em_step(y_old, t, dt, p_named, T_i, u_on, ξ)
    return y_new, 0.0
end

diffusion_fn(params) = [
    sqrt(2.0 * params[_PI[:sigma_x]]),
    sqrt(2.0 * params[_PI[:sigma_u]]),
]

function obs_log_weight_fn(x_new, grid_obs, k, params)
    σ_obs   = params[_PI[:sigma_obs]]
    y_k     = grid_obs[:obs_value][k]
    present = haskey(grid_obs, :obs_present) ? grid_obs[:obs_present][k] : 1.0
    return present * Dynamics.obs_log_prob(x_new, y_k, σ_obs)
end

shard_init_fn(time_offset, params, exog, init) = init

function align_obs_fn(obs_data, t_steps, dt_hours)
    T = Int(t_steps)
    obs_value   = zeros(Float64, T)
    obs_present = ones(Float64,  T)
    if haskey(obs_data, :obs)
        ch = obs_data[:obs]
        for (j, val) in enumerate(ch[:obs_value])
            (1 ≤ j ≤ T) && (obs_value[j] = val)
        end
    end
    return Dict{Symbol,Any}(
        :obs_value      => obs_value,
        :obs_present    => obs_present,
        :has_any_obs    => obs_present,
        :T_intervention => EXOGENOUS_A.T_intervention,
        :u_on           => EXOGENOUS_A.u_on,
    )
end

# ── Batched (GPU-portable) propagate / obs ──────────────────────────────────
# Phase 6 follow-up #2 contract: when these are provided, the framework's
# `bootstrap_log_likelihood` calls them on the full `(K, n_states)` particle
# matrix in a single broadcast — works on `Array{Float64}` and `CuArray{Float32}`
# with the same source.

"""
    propagate_batch_fn(particles_in, t, dt, params, grid_obs, k, σ_diag, noise, rng)
        -> (particles_out::AbstractMatrix, pred_lw::AbstractVector)

Vectorised EM step on the bistable dynamics. `particles_in` is `(K, 2)`,
`noise` is `(K, 2)`. `pred_lw` is zero (bootstrap proposal == prior).
"""
function propagate_batch_fn(particles_in, t, dt, params, grid_obs, k,
                              σ_diag, noise, rng_)
    α  = params[_PI[:alpha]]
    a  = params[_PI[:a]]
    γ  = params[_PI[:gamma]]
    sx_sd = sqrt(2.0 * params[_PI[:sigma_x]])
    su_sd = sqrt(2.0 * params[_PI[:sigma_u]])
    sqrt_dt = sqrt(dt)

    T_i  = haskey(grid_obs, :T_intervention) ? grid_obs[:T_intervention] : EXOGENOUS_A.T_intervention
    u_on = haskey(grid_obs, :u_on)           ? grid_obs[:u_on]           : EXOGENOUS_A.u_on
    u_tgt = t < T_i ? 0.0 : u_on   # piecewise constant; same for all particles at step k

    x_old = @view particles_in[:, 1]
    u_old = @view particles_in[:, 2]
    nx    = @view noise[:, 1]
    nu    = @view noise[:, 2]

    x_new = x_old .+ dt .* (α .* x_old .* (a^2 .- x_old .^ 2) .+ u_old) .+
             sx_sd .* sqrt_dt .* nx
    u_new = u_old .+ dt .* (-γ .* (u_old .- u_tgt)) .+
             su_sd .* sqrt_dt .* nu
    out = hcat(x_new, u_new)

    pred_lw = similar(particles_in, eltype(particles_in), size(particles_in, 1))
    fill!(pred_lw, 0.0)
    return out, pred_lw
end

"""
    obs_log_weight_batch_fn(particles, grid_obs, k, params)

Vectorised Gaussian observation log-weight on x-channel only.
Returns a `(K,)` vector — works on `Array` + `CuArray`.
"""
function obs_log_weight_batch_fn(particles, grid_obs, k, params)
    σ_obs = params[_PI[:sigma_obs]]
    y_k   = grid_obs[:obs_value][k]
    HALF_LOG_2PI = 0.9189385332046727
    x_pred = @view particles[:, 1]
    ν      = y_k .- x_pred
    return -0.5 .* (ν ./ σ_obs) .^ 2 .- log(σ_obs) .- HALF_LOG_2PI
end


function build_estimation_model()
    return EstimationModel(
        name              = "bistable_controlled",
        version           = "v1_julia",
        n_states          = 2,
        n_stochastic      = 2,
        stochastic_indices = [1, 2],
        state_bounds      = [(-5.0, 5.0), (-5.0, 5.0)],
        param_priors      = PARAM_PRIOR_CONFIG,
        init_state_priors = INIT_STATE_PRIOR_CONFIG,
        frozen_params     = Dict{Symbol,Float64}(),
        propagate_fn      = propagate_fn,
        diffusion_fn      = diffusion_fn,
        obs_log_weight_fn = obs_log_weight_fn,
        propagate_batch_fn = propagate_batch_fn,
        obs_log_weight_batch_fn = obs_log_weight_batch_fn,
        align_obs_fn      = align_obs_fn,
        shard_init_fn     = shard_init_fn,
    )
end

end # module Estimation
