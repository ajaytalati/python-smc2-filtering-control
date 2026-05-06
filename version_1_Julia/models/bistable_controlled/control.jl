# Cost evaluator for bistable control. Cost rewards being in the
# positive (healthy) basin: J(u_schedule) = -mean over MC rollouts of
# E[1{x_T > 0}], augmented by an L2 penalty on |u| to discourage
# excessive control. Mirrors `version_1/models/bistable_controlled/control.py`
# at the spec level.

module Control

using Random: MersenneTwister
using Statistics: mean
import ..Dynamics
import ..Simulation: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A

export build_basin_cost_fn, TRUTH

const TRUTH = (
    alpha = PARAM_SET_A.alpha, a = PARAM_SET_A.a,
    sigma_x = PARAM_SET_A.sigma_x, gamma = PARAM_SET_A.gamma,
    sigma_u = PARAM_SET_A.sigma_u, sigma_obs = PARAM_SET_A.sigma_obs,
    T_total = EXOGENOUS_A.T_total, dt = EXOGENOUS_A.dt,
    T_intervention = EXOGENOUS_A.T_intervention,
    x_0 = INIT_STATE_A.x_0, u_0 = INIT_STATE_A.u_0,
)

"""
    build_basin_cost_fn(; n_inner, seed, n_steps, lambda_u=0.1)
        -> J(u_schedule)

`u_schedule` is the RBF-decoded control signal evaluated at every step.
The cost penalises (1 − E[1{x_T > 0}]) — i.e. failure to leave the
unhealthy well by t = T_total — plus an L2 regulariser on the schedule.

Used as the Stage B2 cost target for SMC²-as-controller.
"""
function build_basin_cost_fn(; n_inner::Integer = 64,
                                seed::Integer    = 42,
                                n_steps::Integer = Int(round(TRUTH.T_total / TRUTH.dt)),
                                lambda_u::Real   = 0.1)
    α    = TRUTH.alpha
    a    = TRUTH.a
    γ    = TRUTH.gamma
    sx_sd = sqrt(2.0 * TRUTH.sigma_x)
    su_sd = sqrt(2.0 * TRUTH.sigma_u)
    dt    = TRUTH.dt

    # CRN noise grids for Wiener increments (x and u channels).
    rng_w = MersenneTwister(seed)
    fixed_wx = randn(rng_w, n_inner, n_steps)
    rng_w2 = MersenneTwister(seed + 100_001)
    fixed_wu = randn(rng_w2, n_inner, n_steps)

    function J(u_sched::AbstractVector{<:Real})
        T_eff = min(length(u_sched), n_steps)
        success = 0.0
        for n in 1:n_inner
            x = TRUTH.x_0
            u = TRUTH.u_0
            for k in 1:T_eff
                u_tgt = u_sched[k]
                dx = α * x * (a^2 - x^2) + u
                du = -γ * (u - u_tgt)
                x += dt * dx + sx_sd * sqrt(dt) * fixed_wx[n, k]
                u += dt * du + su_sd * sqrt(dt) * fixed_wu[n, k]
            end
            success += x > 0 ? 1.0 : 0.0
        end
        success_rate = success / n_inner
        # Penalise failure to transition + L2 on the schedule.
        return (1.0 - success_rate) + lambda_u * sum(abs2, u_sched) / T_eff
    end
    return J
end

end # module Control
