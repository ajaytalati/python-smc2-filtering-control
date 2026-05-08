"""
    Simulator/SDEModel.jl

Thin wrapper around `StochasticDiffEq.jl` for forward SDE simulation.

Replaces the Python reference's ~600-line glue around `diffrax`
(`smc2fc/simulator/sde_solver_diffrax.py`) — the wrapper is small here
because `StochasticDiffEq.jl` already exposes the integrators and
sub-stepping the Python had to roll by hand for JIT compatibility.
"""
module SDEModelWrap

using StochasticDiffEq: StochasticDiffEq, SDEProblem, EM, solve
using Random: AbstractRNG, MersenneTwister

export simulate_sde, build_sde_problem

"""
    build_sde_problem(drift_fn, diffusion_fn, u0, tspan, params) -> SDEProblem

Construct an SDE problem `du = drift_fn(u, p, t) dt + diffusion_fn(u, p, t) dW`
suitable for `StochasticDiffEq.solve`.

# Arguments
- `drift_fn`: in-place drift, signature `(du, u, p, t)` writes into `du`.
- `diffusion_fn`: in-place diffusion, signature `(du, u, p, t)`.
- `u0::AbstractVector`: initial state.
- `tspan::Tuple{<:Real,<:Real}`: `(t0, t1)` time span.
- `params`: parameter container passed to `drift_fn` / `diffusion_fn`.

# Returns
- `SDEProblem`: the StochasticDiffEq.jl problem object.

# Notes
- The drift and diffusion follow the standard DifferentialEquations.jl
    convention. Mutation of `du` happens inside the solver and is not
    visible at the call site of `simulate_sde`.
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
                 dt=0.01, saveat=nothing, seed=0) -> Solution

Simulate an SDE on `tspan` with explicit Euler-Maruyama (`EM()`).

# Arguments
- `drift_fn`, `diffusion_fn`: see `build_sde_problem`.
- `u0::AbstractVector`: initial state.
- `tspan::Tuple{<:Real,<:Real}`: time span.
- `params`: solver parameter container.

# Keyword arguments
- `dt::Real = 0.01`: integration step.
- `saveat`: when set, restricts saved trajectory to these time points.
- `seed::Integer = 0`: PRNG seed.

# Returns
- `Solution`: StochasticDiffEq.jl solution. Treat it as a callable
    `t -> u(t)` or extract the state matrix via `Array(sol)`.

# Notes
- `EM()` is chosen to match the Python reference for bit-equivalence
    on the simple-SDE benchmarks.
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
