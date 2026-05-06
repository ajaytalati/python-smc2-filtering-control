# Cost evaluators for scalar OU LQG control — direct port of
# `version_1/models/scalar_ou_lqg/control.py`.
#
# Two cost functions:
#   - build_open_loop_cost_fn       (Stage A2): θ = 20-D raw-pulse schedule
#                                                u = (u_0, ..., u_{T-1})
#   - build_state_feedback_cost_fn  (Stage A3): θ = 20-D state-feedback gain
#                                                K = (K_0, ..., K_{T-1});
#                                                runs an inline Kalman filter

module Control

using Random: MersenneTwister
using Statistics: mean
import ..Simulation: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A
import ..LQR

export build_open_loop_cost_fn, build_state_feedback_cost_fn, TRUTH

const TRUTH = (
    a = PARAM_SET_A.a, b = PARAM_SET_A.b,
    sigma_w = PARAM_SET_A.sigma_w, sigma_v = PARAM_SET_A.sigma_v,
    q = PARAM_SET_A.q, r = PARAM_SET_A.r, s = PARAM_SET_A.s,
    dt = EXOGENOUS_A.dt, T = EXOGENOUS_A.T,
    x0_mean = INIT_STATE_A.x_0, x0_var = EXOGENOUS_A.x0_var,
)

# ── Stage A2: open-loop cost (raw u schedule) ──────────────────────────────

"""
    build_open_loop_cost_fn(; n_inner, seed) -> J(u)

Build a closure `J(u::Vector)` that returns the mean cost over `n_inner`
common-random-numbers Wiener trajectories. Schedule `u` is the raw 20-D
control pulse — interpreted as `u[k]` applied at step `k`.
"""
function build_open_loop_cost_fn(; n_inner::Integer = 64, seed::Integer = 42)
    A   = 1.0 - TRUTH.a * TRUTH.dt
    B   = TRUTH.b * TRUTH.dt
    sw  = TRUTH.sigma_w * sqrt(TRUTH.dt)
    Tn  = TRUTH.T

    # Pre-sample CRN noise (Wiener increments + initial conditions).
    rng_w  = MersenneTwister(seed)
    fixed_w = randn(rng_w, n_inner, Tn)
    rng_x  = MersenneTwister(seed + 1)
    fixed_x0 = TRUTH.x0_mean .+ sqrt(TRUTH.x0_var) .* randn(rng_x, n_inner)

    function J(u::AbstractVector{<:Real})
        total = 0.0
        for n in 1:n_inner
            x = fixed_x0[n]
            cost = 0.0
            for k in 1:Tn
                u_k = u[k]
                cost += TRUTH.q * x^2 + TRUTH.r * u_k^2
                x = A * x + B * u_k + sw * fixed_w[n, k]
            end
            cost += TRUTH.s * x^2
            total += cost
        end
        return total / n_inner
    end
    return J
end

# ── Stage A3: state-feedback cost (gain K) ─────────────────────────────────

"""
    build_state_feedback_cost_fn(; n_inner, seed) -> J(K)

`K` is the 20-D state-feedback gain vector. Per trial, an inline scalar
Kalman filter updates the state estimate from observations; the action
applied is `u_k = -K[k] · x̂_k`.
"""
function build_state_feedback_cost_fn(; n_inner::Integer = 64, seed::Integer = 42)
    A   = 1.0 - TRUTH.a * TRUTH.dt
    B   = TRUTH.b * TRUTH.dt
    sw  = TRUTH.sigma_w * sqrt(TRUTH.dt)
    Q_  = TRUTH.sigma_w^2 * TRUTH.dt
    R   = TRUTH.sigma_v^2
    Tn  = TRUTH.T

    rng_w  = MersenneTwister(seed)
    fixed_w = randn(rng_w, n_inner, Tn)
    rng_v  = MersenneTwister(seed + 100_001)
    fixed_v = randn(rng_v, n_inner, Tn)
    rng_x  = MersenneTwister(seed + 1)
    fixed_x0 = TRUTH.x0_mean .+ sqrt(TRUTH.x0_var) .* randn(rng_x, n_inner)

    function J(K::AbstractVector{<:Real})
        total = 0.0
        for n in 1:n_inner
            x  = fixed_x0[n]
            x̂ = TRUTH.x0_mean
            P̂ = TRUTH.x0_var
            cost = 0.0
            for k in 1:Tn
                # Kalman update on observation y_k = x_k + σ_v · v_k
                y = x + TRUTH.sigma_v * fixed_v[n, k]
                S = P̂ + R
                G = P̂ / S
                x̂_post = x̂ + G * (y - x̂)
                P̂_post = (1.0 - G) * P̂
                # State-feedback action
                u_k = -K[k] * x̂_post
                cost += TRUTH.q * x^2 + TRUTH.r * u_k^2
                # Advance true state and predicted state
                x  = A * x + B * u_k + sw * fixed_w[n, k]
                x̂ = A * x̂_post + B * u_k
                P̂ = A^2 * P̂_post + Q_
            end
            cost += TRUTH.s * x^2
            total += cost
        end
        return total / n_inner
    end
    return J
end

end # module Control
