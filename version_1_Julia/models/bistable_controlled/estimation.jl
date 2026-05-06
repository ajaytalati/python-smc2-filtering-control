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
        align_obs_fn      = align_obs_fn,
        shard_init_fn     = shard_init_fn,
    )
end

end # module Estimation
