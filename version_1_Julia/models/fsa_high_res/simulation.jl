# FSA-v2 simulator + initial conditions + horizon defaults.
# Mirrors the constants in `version_1/models/fsa_high_res/control.py`.

module Simulation

using Random: AbstractRNG, MersenneTwister
import ..Dynamics: TRUTH_PARAMS, em_step_substepped

# ── Initial state + exogenous defaults (verbatim Python copy) ──────────────

const INIT_STATE = (B = 0.05, F = 0.30, A = 0.10)

const EXOGENOUS = (
    T_total      = 42.0,                 # days — canonical horizon
    dt_days      = 1.0 / 96.0,           # 15 min outer step
    n_substeps   = 4,
    F_max        = 0.40,
    Phi_max      = 3.0,
    Phi_default  = 1.0,
)


# ── Simulator: forward roll-out under a Φ schedule ─────────────────────────

"""
    simulate_em(; params, init_state, Phi_schedule, dt, n_substeps, seed)

Simulate one trajectory of the FSA-v2 SDE under the supplied Φ schedule.
Returns NamedTuple `(t_grid, B, F, A, Phi)`.

`Phi_schedule` is a `(n_steps,)` vector — the control input at each
outer step.
"""
function simulate_em(; params = TRUTH_PARAMS,
                       init_state = INIT_STATE,
                       Phi_schedule::AbstractVector{<:Real},
                       dt::Real        = EXOGENOUS.dt_days,
                       n_substeps::Int = EXOGENOUS.n_substeps,
                       seed::Integer   = 0)
    n_steps = length(Phi_schedule)
    rng = MersenneTwister(seed)

    B = zeros(Float64, n_steps); F = zeros(Float64, n_steps); A = zeros(Float64, n_steps)
    y = [init_state.B, init_state.F, init_state.A]

    for k in 1:n_steps
        noise = randn(rng, 3)
        y = em_step_substepped(y, params, noise, Phi_schedule[k], dt;
                                 n_substeps = n_substeps)
        B[k], F[k], A[k] = y[1], y[2], y[3]
    end

    return (
        t_grid = collect(0:n_steps-1) .* dt,
        B      = B, F = F, A = A,
        Phi    = Float64.(Phi_schedule),
    )
end

end # module Simulation
