# Analytical scalar Kalman filter — direct port of
# `version_1/models/scalar_ou_lqg/bench_kalman.py`.
#
# Discrete-time linear-Gaussian state space:
#   x_{k+1} = A x_k + B u_k + w_k,   w_k ~ N(0, Q)
#   y_k     = C x_k + v_k,           v_k ~ N(0, R)
# with A = 1 - a·dt, B = b·dt, Q = σ_w²·dt, C = 1, R = σ_v².

module Kalman

using Statistics: mean, var

struct KalmanResult
    means::Vector{Float64}        # posterior means x_{k|k}
    covars::Vector{Float64}       # posterior variances P_{k|k}
    pred_means::Vector{Float64}   # predictive means x_{k|k-1}
    pred_covars::Vector{Float64}  # predictive variances P_{k|k-1}
    log_likelihood::Float64
end

"""
    kalman_filter(; y, u, a, b, sigma_w, sigma_v, dt, x0_mean, x0_var)

Forward Kalman pass on the scalar OU LQG model. Returns a `KalmanResult`
with posterior moments at every step plus the cumulative marginal
log-likelihood.
"""
function kalman_filter(;
    y::AbstractVector{<:Real},
    u::AbstractVector{<:Real},
    a::Real, b::Real, sigma_w::Real, sigma_v::Real,
    dt::Real, x0_mean::Real, x0_var::Real,
)
    T = length(y)
    A = 1.0 - a * dt
    B = b * dt
    Q = sigma_w^2 * dt
    R = sigma_v^2

    means       = zeros(Float64, T)
    covars      = zeros(Float64, T)
    pred_means  = zeros(Float64, T)
    pred_covars = zeros(Float64, T)

    pred_means[1]  = x0_mean
    pred_covars[1] = x0_var
    S0 = pred_covars[1] + R
    K0 = pred_covars[1] / S0
    means[1]  = pred_means[1] + K0 * (y[1] - pred_means[1])
    covars[1] = (1.0 - K0) * pred_covars[1]
    log_lik   = -0.5 * (log(2π * S0) + (y[1] - pred_means[1])^2 / S0)

    for k in 2:T
        pred_means[k]  = A * means[k-1] + B * u[k-1]
        pred_covars[k] = A^2 * covars[k-1] + Q

        S = pred_covars[k] + R
        K = pred_covars[k] / S
        means[k]  = pred_means[k] + K * (y[k] - pred_means[k])
        covars[k] = (1.0 - K) * pred_covars[k]
        log_lik  += -0.5 * (log(2π * S) + (y[k] - pred_means[k])^2 / S)
    end

    return KalmanResult(means, covars, pred_means, pred_covars, log_lik)
end

"""
    kalman_log_likelihood(; y, u, a, b, sigma_w, sigma_v, dt, x0_mean, x0_var)

Convenience wrapper that returns just the marginal log-likelihood.
"""
function kalman_log_likelihood(; kwargs...)
    return kalman_filter(; kwargs...).log_likelihood
end

end # module Kalman
