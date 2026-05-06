# Simulator for the controlled bistable model — Julia reference.
# Two paths: hand-rolled Euler-Maruyama + StochasticDiffEq.jl.

module Simulation

using Random: AbstractRNG, MersenneTwister
using StochasticDiffEq: SDEProblem, EM, solve

# ── Parameter sets (mirror the Python `PARAM_SET_A`, `INIT_STATE_A`,
# `EXOGENOUS_A` exactly) ──────────────────────────────────────────────────────

const PARAM_SET_A = (
    alpha     = 1.0,
    a         = 1.0,
    sigma_x   = 0.10,
    gamma     = 2.0,
    sigma_u   = 0.05,
    sigma_obs = 0.20,
)

const INIT_STATE_A = (x_0 = -1.0, u_0 = 0.0)

const EXOGENOUS_A = (
    T_intervention = 24.0,
    T_total        = 72.0,
    u_on           = 0.5,
    dt             = 1/6,    # 10 min in hours
)

@inline _u_target(t::Real, T_i::Real, u_on::Real) = t < T_i ? 0.0 : u_on

# Nice param-as-NamedTuple helper that mirrors the Python dict access pattern.
_p(p) = (alpha = p.alpha, a = p.a, sigma_x = p.sigma_x,
          gamma = p.gamma, sigma_u = p.sigma_u, sigma_obs = p.sigma_obs)

# ── Path 1: hand-rolled Euler-Maruyama ─────────────────────────────────────

"""
    simulate_em(; params, init_state, exogenous, seed=0)

Direct port of `simulation.py:simulate`-equivalent. Two-state EM step,
piecewise-constant `u_target` schedule, Gaussian observation channel.
"""
function simulate_em(; params = PARAM_SET_A,
                       init_state = INIT_STATE_A,
                       exogenous  = EXOGENOUS_A,
                       seed::Integer = 0)
    dt       = exogenous.dt
    T_total  = exogenous.T_total
    T_i      = exogenous.T_intervention
    u_on     = exogenous.u_on
    n_steps  = Int(round(T_total / dt))

    α    = params.alpha
    a    = params.a
    γ    = params.gamma
    sx_sd = sqrt(2.0 * params.sigma_x)
    su_sd = sqrt(2.0 * params.sigma_u)

    rng = MersenneTwister(seed)

    x = zeros(Float64, n_steps)
    u = zeros(Float64, n_steps)
    x[1] = init_state.x_0
    u[1] = init_state.u_0

    for k in 1:n_steps-1
        t      = (k - 1) * dt
        u_tgt  = _u_target(t, T_i, u_on)
        dx     = α * x[k] * (a^2 - x[k]^2) + u[k]
        du     = -γ * (u[k] - u_tgt)
        x[k+1] = x[k] + dt * dx + sx_sd * sqrt(dt) * randn(rng)
        u[k+1] = u[k] + dt * du + su_sd * sqrt(dt) * randn(rng)
    end

    obs = x .+ params.sigma_obs .* randn(rng, n_steps)
    u_target_grid = [_u_target((k - 1) * dt, T_i, u_on) for k in 1:n_steps]

    return (
        t_grid     = collect(0:n_steps-1) .* dt,
        trajectory = hcat(x, u),
        obs        = obs,
        u_target   = u_target_grid,
    )
end

# ── Path 2: StochasticDiffEq.jl ────────────────────────────────────────────

"""
    simulate_diffeq(; params, init_state, exogenous, seed=0)

Same dynamics through StochasticDiffEq.jl's `EM()` solver. Wiener
RNG is internal — agreement with `simulate_em` is statistical, not
bit-for-bit.
"""
function simulate_diffeq(; params = PARAM_SET_A,
                           init_state = INIT_STATE_A,
                           exogenous  = EXOGENOUS_A,
                           seed::Integer = 0)
    dt      = exogenous.dt
    T_total = exogenous.T_total
    T_i     = exogenous.T_intervention
    u_on    = exogenous.u_on

    function f!(du, y, p, t)
        x, u = y[1], y[2]
        u_tgt = _u_target(t, T_i, u_on)
        du[1] = params.alpha * x * (params.a^2 - x^2) + u
        du[2] = -params.gamma * (u - u_tgt)
        return nothing
    end
    function g!(du, y, p, t)
        du[1] = sqrt(2.0 * params.sigma_x)
        du[2] = sqrt(2.0 * params.sigma_u)
        return nothing
    end

    y0 = [init_state.x_0, init_state.u_0]
    prob = SDEProblem(f!, g!, y0, (0.0, T_total - dt), nothing)
    sol  = solve(prob, EM(); dt = dt, saveat = 0:dt:(T_total-dt), seed = seed)
    traj = reduce(hcat, sol.u)'

    rng_obs = MersenneTwister(seed + 1_000_003)
    obs = vec(traj[:, 1]) .+ params.sigma_obs .* randn(rng_obs, size(traj, 1))
    u_target_grid = [_u_target((k - 1) * dt, T_i, u_on) for k in 1:size(traj, 1)]

    return (
        t_grid     = collect(0:size(traj,1)-1) .* dt,
        trajectory = traj,
        obs        = obs,
        u_target   = u_target_grid,
    )
end

end # module Simulation
