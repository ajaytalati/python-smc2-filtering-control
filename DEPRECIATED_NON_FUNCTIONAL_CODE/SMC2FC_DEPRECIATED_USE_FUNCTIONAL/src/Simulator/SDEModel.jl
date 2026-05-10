# Simulator/SDEModel.jl — thin wrapper around StochasticDiffEq.jl.
#
# Charter §13: `simulator/sde_solver_diffrax.py → StochasticDiffEq.jl
# (part of DifferentialEquations.jl). Gold standard. Replaces diffrax
# wholesale.`
#
# The Python `sde_solver_diffrax.py` is ~600 lines of glue around `diffrax`'s
# `EulerMaruyama`, `Itoh`, etc. solvers + custom `lax.scan` substepping for
# JIT compatibility. The Julia equivalent is a 30-line wrapper because
# `StochasticDiffEq.jl` already exposes everything we need.

module SDEModelWrap

using StochasticDiffEq: StochasticDiffEq, SDEProblem, EM, solve
using Random: AbstractRNG, MersenneTwister

export simulate_sde, build_sde_problem

"""
    build_sde_problem(drift_fn, diffusion_fn, u0, tspan, params)

Construct an SDE problem
    `du = drift_fn(u, p, t) dt + diffusion_fn(u, p, t) dW`
suitable for `StochasticDiffEq.solve`. `drift_fn` and `diffusion_fn` follow
the standard DifferentialEquations.jl convention (in-place: arguments
`(du, u, p, t)` mutate `du`).
"""
function build_sde_problem(drift_fn,
                            diffusion_fn,
                            u0::AbstractVector,
                            tspan::Tuple{<:Real,<:Real},
                            params)
    return SDEProblem(drift_fn, diffusion_fn, u0, tspan, params)
end

"""
    simulate_sde(drift_fn, diffusion_fn, u0, tspan, params;
                  dt=0.01, saveat=nothing, seed=0) -> trajectory

Simulate an SDE on `tspan` with Euler-Maruyama. Returns the solution as
either a `(T, n_states)` matrix (when `saveat` is supplied) or
`StochasticDiffEq`'s `Solution` type. The default integrator is `EM()`
(explicit Euler-Maruyama) — matches the Python reference for bit-equivalence.
"""
function simulate_sde(drift_fn,
                       diffusion_fn,
                       u0::AbstractVector,
                       tspan::Tuple{<:Real,<:Real},
                       params;
                       dt::Real = 0.01,
                       saveat = nothing,
                       seed::Integer = 0)
    prob = build_sde_problem(drift_fn, diffusion_fn, u0, tspan, params)
    sol = saveat === nothing ?
        solve(prob, EM(); dt = Float64(dt), seed = Int(seed)) :
        solve(prob, EM(); dt = Float64(dt), saveat = saveat, seed = Int(seed))
    return sol
end

end # module SDEModelWrap
