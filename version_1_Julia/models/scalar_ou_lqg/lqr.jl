# LQR/LQG ground truth for scalar OU LQG — direct port of
# `version_1/models/scalar_ou_lqg/bench_lqr.py`. Closed-form Riccati,
# closed-form expected optimal cost, plus Monte Carlo LQG cost (which
# adds the Kalman-update penalty for partial observation).

module LQR

using Random: AbstractRNG, MersenneTwister
using Statistics: mean

export RiccatiResult, lqr_riccati, lqr_optimal_cost,
       lqg_optimal_cost_mc, open_loop_zero_control_cost_mc

struct RiccatiResult
    gains::Vector{Float64}     # K_0 .. K_{T-1}, T entries
    values::Vector{Float64}    # P_0 .. P_T,     T+1 entries
end

"""
    lqr_riccati(; a, b, q, r, s, sigma_w, dt, T) -> RiccatiResult

Backward Riccati recursion for the finite-horizon scalar LQR problem.
Returns gains and value-function coefficients. Optimal control at step k
is `u*_k = -K_k · x_k`.
"""
function lqr_riccati(; a::Real, b::Real, q::Real, r::Real, s::Real,
                       sigma_w::Real, dt::Real, T::Integer)
    A = 1.0 - a * dt
    B = b * dt
    P = zeros(Float64, T + 1)
    K = zeros(Float64, T)
    P[T + 1] = s
    for k in T:-1:1
        K[k] = (B * P[k + 1] * A) / (B^2 * P[k + 1] + r)
        P[k] = q + A^2 * P[k + 1] - A * P[k + 1] * B * K[k]
    end
    return RiccatiResult(K, P)
end

"""
    lqr_optimal_cost(; riccati, x0_mean, x0_var, sigma_w, dt, T)

Expected optimal LQR cost under perfect state observation.
"""
function lqr_optimal_cost(; riccati::RiccatiResult,
                            x0_mean::Real, x0_var::Real,
                            sigma_w::Real, dt::Real, T::Integer)
    Q = sigma_w^2 * dt
    init_term = (x0_mean^2 + x0_var) * riccati.values[1]
    noise_term = sum(riccati.values[2:T+1]) * Q
    return init_term + noise_term
end

"""
    lqg_optimal_cost_mc(; a, b, q, r, s, sigma_w, sigma_v, dt, T,
                          x0_mean, x0_var, n_trials, seed) -> NamedTuple

Monte Carlo LQG cost — Kalman estimator + LQR gain on the estimate.
Returns `(mean_cost, std_cost)`.
"""
function lqg_optimal_cost_mc(; a::Real, b::Real, q::Real, r::Real, s::Real,
                                sigma_w::Real, sigma_v::Real,
                                dt::Real, T::Integer,
                                x0_mean::Real, x0_var::Real,
                                n_trials::Integer = 5000,
                                seed::Integer = 0)
    A  = 1.0 - a * dt
    B  = b * dt
    Q  = sigma_w^2 * dt
    R  = sigma_v^2
    riccati = lqr_riccati(a = a, b = b, q = q, r = r, s = s,
                            sigma_w = sigma_w, dt = dt, T = T)

    rng = MersenneTwister(seed)
    costs = zeros(Float64, n_trials)
    for n in 1:n_trials
        x = x0_mean + sqrt(x0_var) * randn(rng)
        x_hat_mean = x0_mean
        x_hat_var  = x0_var
        cost = 0.0
        for k in 1:T
            y      = x + sigma_v * randn(rng)
            S      = x_hat_var + R
            G      = x_hat_var / S
            x_hat_mean_post = x_hat_mean + G * (y - x_hat_mean)
            x_hat_var_post  = (1.0 - G) * x_hat_var
            u_k    = -riccati.gains[k] * x_hat_mean_post
            cost  += q * x^2 + r * u_k^2
            # Advance
            x          = A * x + B * u_k + sqrt(Q) * randn(rng)
            x_hat_mean = A * x_hat_mean_post + B * u_k
            x_hat_var  = A^2 * x_hat_var_post + Q
        end
        costs[n] = cost + s * x^2
    end
    return (mean_cost = mean(costs), std_cost = sqrt(sum((costs .- mean(costs)).^2) / (n_trials - 1)))
end

"""
    open_loop_zero_control_cost_mc(; a, b, q, r, s, sigma_w, dt, T,
                                       x0_mean, x0_var, n_trials, seed)

Monte Carlo cost under u ≡ 0 (no control). Used as the open-loop
baseline for Stage A2's gate.
"""
function open_loop_zero_control_cost_mc(; a::Real, b::Real, q::Real, r::Real,
                                              s::Real, sigma_w::Real,
                                              dt::Real, T::Integer,
                                              x0_mean::Real, x0_var::Real,
                                              n_trials::Integer = 5000,
                                              seed::Integer = 0)
    A  = 1.0 - a * dt
    sw = sigma_w * sqrt(dt)
    rng = MersenneTwister(seed)
    costs = zeros(Float64, n_trials)
    for n in 1:n_trials
        x = x0_mean + sqrt(x0_var) * randn(rng)
        cost = 0.0
        for k in 1:T
            cost += q * x^2     # u = 0 → no r·u² contribution
            x = A * x + sw * randn(rng)
        end
        costs[n] = cost + s * x^2
    end
    return (mean_cost = mean(costs),
             std_cost = sqrt(sum((costs .- mean(costs)).^2) / (n_trials - 1)))
end

end # module LQR
