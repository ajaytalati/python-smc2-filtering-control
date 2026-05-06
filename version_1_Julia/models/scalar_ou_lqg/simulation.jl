# Simulator for the scalar OU LQG model — Julia reference.
# Two paths:
#   1. `simulate_em` — hand-rolled Euler-Maruyama, line-for-line port
#      of `version_1/models/scalar_ou_lqg/simulation.py:simulate`.
#      Used for bit-equivalence checks against the Python reference.
#   2. `simulate_diffeq` — `StochasticDiffEq.jl` (production path).
#      Same EM solver at the same dt → numerically equivalent to (1).

module Simulation

using Random: AbstractRNG, MersenneTwister
using StochasticDiffEq: SDEProblem, EM, solve

# ── Parameter sets ──────────────────────────────────────────────────────────
# Match the Python `PARAM_SET_A` / `INIT_STATE_A` / `EXOGENOUS_A` exactly.

const PARAM_SET_A = (
    a       = 1.0,
    b       = 1.0,
    sigma_w = 0.3,
    sigma_v = 0.2,
    q       = 1.0,
    r       = 0.1,
    s       = 1.0,
)

const INIT_STATE_A = (x_0 = 0.0,)

const EXOGENOUS_A = (
    dt     = 0.05,
    T      = 20,
    x0_var = 1.0,
)

# ── Path 1: hand-rolled Euler-Maruyama (matches Python `simulate`) ──────────

"""
    simulate_em(; params, init_state, exogenous, u=nothing, seed=0)

Direct port of `simulation.py:simulate`. The recurrence
    x_{k+1} = A·x_k + B·u_k + sw·ξ_k,  ξ_k ~ N(0, 1)
with A = 1 - a·dt, B = b·dt, sw = σ_w·√dt — bit-equivalent to Python at the
same RNG seed (modulo Python's `np.random` vs Julia's `MersenneTwister`).

Returns NamedTuple `(t_grid, trajectory, obs, u)` matching the Python dict.
"""
function simulate_em(; params = PARAM_SET_A,
                       init_state = INIT_STATE_A,
                       exogenous  = EXOGENOUS_A,
                       u::Union{Nothing,AbstractVector} = nothing,
                       seed::Integer = 0)
    dt     = exogenous.dt
    T      = exogenous.T
    x0_var = exogenous.x0_var
    x0_mean = init_state.x_0

    A  = 1.0 - params.a * dt
    B  = params.b * dt
    sw = params.sigma_w * sqrt(dt)

    rng = MersenneTwister(seed)
    u_vec = u === nothing ? zeros(Float64, T) : Float64.(u)

    x = zeros(Float64, T)
    x[1] = x0_mean + sqrt(x0_var) * randn(rng)
    for k in 2:T
        x[k] = A * x[k-1] + B * u_vec[k-1] + sw * randn(rng)
    end
    obs = x .+ params.sigma_v .* randn(rng, T)

    return (
        t_grid     = collect(0:T-1) .* dt,
        trajectory = reshape(x, :, 1),
        obs        = obs,
        u          = u_vec,
    )
end


# ── Path 2: StochasticDiffEq.jl (production path) ───────────────────────────

"""
    simulate_diffeq(; params, init_state, exogenous, u=nothing, seed=0)

Same model, same dt, same EM integrator — but routed through
`StochasticDiffEq.jl`'s `EM()` solver. Used to confirm the Julia
ecosystem's path agrees with the hand-rolled one and to demonstrate the
production interface (charter §13: `sde_solver_diffrax.py` →
`StochasticDiffEq.jl`).

Note: StochasticDiffEq's `EM` consumes a `(du, u, p, t)` in-place drift
and a separate `(du, u, p, t)` diffusion (g) callback. The Wiener-noise
RNG is internal to StochasticDiffEq and not bit-comparable with the
hand-rolled MersenneTwister path; numerical agreement is statistical
(across seeds), not bit-for-bit.
"""
function simulate_diffeq(; params = PARAM_SET_A,
                           init_state = INIT_STATE_A,
                           exogenous  = EXOGENOUS_A,
                           u::Union{Nothing,AbstractVector} = nothing,
                           seed::Integer = 0)
    dt = exogenous.dt
    T  = exogenous.T
    x0_mean = init_state.x_0
    x0_var  = exogenous.x0_var
    u_vec   = u === nothing ? zeros(Float64, T) : Float64.(u)

    # In-place drift / diffusion. We bake u(t) as a step-wise function of t.
    function f!(du, x, p, t)
        # control schedule: piecewise-constant at floor(t/dt) bin
        k     = clamp(Int(floor(t / dt)) + 1, 1, T)
        u_t   = u_vec[k]
        du[1] = -params.a * x[1] + params.b * u_t
        return nothing
    end
    function g!(du, x, p, t)
        du[1] = params.sigma_w
        return nothing
    end

    rng_init = MersenneTwister(seed)
    x0 = [x0_mean + sqrt(x0_var) * randn(rng_init)]

    prob = SDEProblem(f!, g!, x0, (0.0, (T - 1) * dt), nothing)
    sol  = solve(prob, EM(); dt = dt, saveat = 0:dt:(T-1)*dt, seed = seed)

    traj = reduce(hcat, sol.u)'   # (T, 1)
    # Observations sampled with a fresh RNG so seed reuses the Wiener stream
    # without cross-contaminating the obs noise.
    rng_obs = MersenneTwister(seed + 1_000_003)
    obs = vec(traj[:, 1]) .+ params.sigma_v .* randn(rng_obs, T)

    return (
        t_grid     = collect(0:T-1) .* dt,
        trajectory = traj,
        obs        = obs,
        u          = u_vec,
    )
end

end # module Simulation
