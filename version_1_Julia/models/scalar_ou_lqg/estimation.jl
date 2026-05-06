# SMC² EstimationModel for the scalar OU LQG model — Julia port of
# `version_1/models/scalar_ou_lqg/estimation.py`.
#
# Priors mirror the Python config exactly:
#   a, b, sigma_w, sigma_v ~ LogNormal centred at truth, σ = 0.5
#   x_0 ~ Normal(0, 1)

module Estimation

using SMC2FC: EstimationModel, PriorType, LogNormalPrior, NormalPrior
import ..Dynamics
import ..Simulation: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A

export build_estimation_model, PARAM_PRIOR_CONFIG, INIT_STATE_PRIOR_CONFIG

# ── Priors (centred at truth, σ = 0.5) ──────────────────────────────────────

const PARAM_PRIOR_CONFIG = Tuple{Symbol,PriorType}[
    (:a,       LogNormalPrior(log(1.0), 0.5)),
    (:b,       LogNormalPrior(log(1.0), 0.5)),
    (:sigma_w, LogNormalPrior(log(0.3), 0.5)),
    (:sigma_v, LogNormalPrior(log(0.2), 0.5)),
]

const INIT_STATE_PRIOR_CONFIG = Tuple{Symbol,PriorType}[
    (:x_0, NormalPrior(0.0, 1.0)),
]

# Param-name → index map (matches Python `_PI`).
const _PI = Dict{Symbol,Int}(
    :a => 1, :b => 2, :sigma_w => 3, :sigma_v => 4,
)

# ── Per-particle propagate / obs / shard_init ───────────────────────────────
#
# Locally-optimal Gaussian-fusion proposal (Pitt-Shephard tilt). For scalar
# linear-Gaussian dynamics with Gaussian observations this is the EXACT
# optimal proposal — bootstrap-PF marginal-LL has no Jensen-inequality
# bias and the estimator converges to the analytical Kalman log-likelihood
# at finite K. Matches Python `estimation.py:_propagate_fn` exactly.

"""
    propagate_fn(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng)

Pitt-Shephard locally-optimal proposal: fuses the Euler-step prior with
the Gaussian observation likelihood at step `k`. Returns `(y_new, pred_lw)`
where `pred_lw` is the *predictive* log-density log N(y_k | μ_prior,
σ_v² + var_prior). The obs likelihood is folded into the proposal so
`obs_log_weight_fn` returns 0.
"""
function propagate_fn(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng_)
    a, b       = params[_PI[:a]], params[_PI[:b]]
    σ_w, σ_v   = params[_PI[:sigma_w]], params[_PI[:sigma_v]]
    u_t        = haskey(grid_obs, :u_value) ? grid_obs[:u_value][k] : 0.0

    # Euler-step prior moments: μ_prior, var_prior
    μ_prior   = (1 - a * dt) * y_old[1] + b * dt * u_t
    var_prior = max(σ_w^2 * dt, 1e-12)

    # Observation availability
    obs_pres = haskey(grid_obs, :obs_present) ? grid_obs[:obs_present][k] : 1.0
    obs_val  = grid_obs[:obs_value][k]

    # Pitt-Shephard fusion (Kalman update on x given y_k):
    #   prec_post = 1/var_prior + obs_pres/σ_v²
    #   info_post = μ_prior/var_prior + obs_pres·y_k/σ_v²
    x_prec = 1.0 / var_prior + obs_pres / σ_v^2
    x_info = μ_prior / var_prior + obs_pres * obs_val / σ_v^2
    x_var  = 1.0 / x_prec
    x_mu   = x_var * x_info

    # Sample from the tilted proposal.
    x_new = x_mu + sqrt(x_var) * ξ[1]

    # Predictive log-weight: log N(y_k | μ_prior, σ_v² + var_prior).
    pred_var = σ_v^2 + var_prior
    lw = obs_pres * (
        -0.5 * (obs_val - μ_prior)^2 / pred_var
        - 0.5 * log(pred_var) - Dynamics.HALF_LOG_2PI
    )
    return [x_new], lw
end

diffusion_fn(params) = [params[_PI[:sigma_w]]]

# Obs likelihood is folded into the proposal — return 0.
obs_log_weight_fn(x_new, grid_obs, k, params) = 0.0

# Shard-init: use the estimated `x_0` from `init` (the Φ stack); the framework
# has already drawn it from the prior and bound it via `unconstrained_to_constrained`.
shard_init_fn(time_offset, params, exog, init) = init

# Grid alignment: simple wrapper that injects `obs_value` + (optional)
# `u_value` into the dict the PF consumes.
function align_obs_fn(obs_data, t_steps, dt_hours)
    T = Int(t_steps)
    obs_value = zeros(Float32, T)
    obs_present = zeros(Float32, T)

    if haskey(obs_data, :obs)
        ch = obs_data[:obs]
        idx = Int.(ch[:t_idx]) .+ 1   # Python is 0-indexed; Julia 1-indexed
        for (j, i) in enumerate(idx)
            if 1 ≤ i ≤ T
                obs_value[i]   = ch[:obs_value][j]
                obs_present[i] = 1.0
            end
        end
    end
    return Dict(:obs_value => obs_value, :obs_present => obs_present,
                :has_any_obs => obs_present)
end

# ── Factory ─────────────────────────────────────────────────────────────────

"""
    build_estimation_model() -> SMC2FC.EstimationModel

Build the full SMC2FC contract for scalar_ou_lqg. Bound to the Python-
identical priors + dynamics + observation model.
"""
function build_estimation_model()
    return EstimationModel(
        name              = "scalar_ou_lqg",
        version           = "v1_julia",
        n_states          = 1,
        n_stochastic      = 1,
        stochastic_indices = [1],
        state_bounds      = [(-50.0, 50.0)],   # generous bound; OU is unbounded
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
