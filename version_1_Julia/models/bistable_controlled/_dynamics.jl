# Pure-Julia dynamics for the controlled bistable model.
# Mirrors version_1/models/bistable_controlled/simulation.py:drift / diffusion.
#
# State:  y = [x, u]                              (2-D)
# Drift:  dx/dt = α·x·(a² − x²) + u
#         du/dt = -γ·(u − u_target(t))
# Diffusion: [√(2 σ_x),  √(2 σ_u)]                (additive)
# Obs:    y_k = x_k + N(0, σ_obs²)                (channel "obs")
# Schedule: u_target(t) = 0 if t < T_i else u_on  (piecewise-constant)
#
# Bifurcation: u_c = 2·α·a³ / (3√3) ≈ 0.385 for α = a = 1.
# u_on = 0.5 > u_c → supercritical → x transitions out of −a well.

module Dynamics

const HALF_LOG_2PI = 0.9189385332046727

"""u_target(t, T_i, u_on): piecewise-constant 2-phase schedule."""
@inline u_target(t::Real, T_i::Real, u_on::Real) = t < T_i ? 0.0 : u_on

"""drift([x, u], t, params, T_i, u_on)"""
@inline function drift(y::AbstractVector, t::Real, params, T_i::Real, u_on::Real)
    x, u   = y[1], y[2]
    u_tgt  = u_target(t, T_i, u_on)
    dx     = params[:alpha] * x * (params[:a]^2 - x^2) + u
    du     = -params[:gamma] * (u - u_tgt)
    return [dx, du]
end

"""diffusion_diagonal(params): additive σ scaling for [x, u]."""
@inline diffusion_diagonal(params) =
    [sqrt(2.0 * params[:sigma_x]), sqrt(2.0 * params[:sigma_u])]

"""em_step([x, u], t, dt, params, T_i, u_on, ξ::AbstractVector{2})"""
@inline function em_step(y::AbstractVector, t::Real, dt::Real, params,
                          T_i::Real, u_on::Real, ξ::AbstractVector)
    drift_y = drift(y, t, params, T_i, u_on)
    σ       = diffusion_diagonal(params)
    return y .+ dt .* drift_y .+ σ .* sqrt(dt) .* ξ
end

"""Gaussian log-pdf of obs at step k given state [x, u]."""
@inline function obs_log_prob(y::AbstractVector, y_obs::Real, σ_obs::Real)
    resid = y_obs - y[1]
    return -0.5 * (resid / σ_obs)^2 - log(σ_obs) - HALF_LOG_2PI
end

end # module Dynamics
