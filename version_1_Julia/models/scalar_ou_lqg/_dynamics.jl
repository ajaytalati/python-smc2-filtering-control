# Pure-Julia dynamics for the scalar OU LQG model.
# Mirrors version_1/models/scalar_ou_lqg/_dynamics.py line-by-line.
#
# State:     y = [x]                     (1-D)
# Drift:     dx/dt = -a x + b u(t)
# Diffusion: σ_w
# Obs:       y_k = x_k + N(0, σ_v²)
#
# Discrete-time Euler-Maruyama:
#     x_{k+1} = (1 - a·dt) x_k + b·dt·u_k + √dt·σ_w·ξ_k

module Dynamics

const HALF_LOG_2PI = 0.9189385332046727   # 0.5 * log(2π)

"""
    drift(y, t, params, u_t)

Scalar OU drift `dx/dt = -a·x + b·u`. `params` indexes `:a, :b`.
"""
@inline function drift(y::AbstractVector, t::Real, params, u_t::Real)
    return [-params[:a] * y[1] + params[:b] * u_t]
end

"""
    diffusion_diagonal(params)

Diagonal diffusion `[σ_w]` (single state).
"""
@inline diffusion_diagonal(params) = [params[:sigma_w]]

"""
    em_step(y, t, dt, params, u_t, ξ)

Explicit Euler-Maruyama step. Returns the new state vector.
"""
@inline function em_step(y::AbstractVector, t::Real, dt::Real, params,
                          u_t::Real, ξ::Real)
    drift_y = drift(y, t, params, u_t)
    σ       = diffusion_diagonal(params)
    return y .+ dt .* drift_y .+ σ .* sqrt(dt) .* ξ
end

"""
    obs_log_prob(y, y_obs::Real, σ_v::Real)

Gaussian log-likelihood of a scalar observation given state `y`.
"""
@inline function obs_log_prob(y::AbstractVector, y_obs::Real, σ_v::Real)
    resid = y_obs - y[1]
    return -0.5 * (resid / σ_v)^2 - log(σ_v) - HALF_LOG_2PI
end

end # module Dynamics
