# FSA v1.5 plant — purely functional, stateless.
#
# An immutable PlantState record + two pure functions:
#   plant_step(s, Φ_t, params, dt, key)        → (state, obs)
#   plant_rollout(s0, Φ_subdaily, params, dt, key0) → trajectory + obs vectors
#
# No mutable struct, no `!`-suffix, no MersenneTwister threaded as state,
# no history dict. The bench builds up a history by collecting returned
# values via foldl / accumulate (see bench_smc_full_mpc_fsa_gpu.jl).

module Plant

using StaticArrays
using StableRNGs
using Match

import ..Dynamics: drift, diffusion_state_dep
import ..Simulation: sample_obs_bfa, DEFAULT_PARAMS, INIT_STATE, DT_BIN_DAYS,
                     params_v15_to_v1_nt

export PlantState, plant_step, plant_rollout, init_plant_state


# ── Immutable state record ────────────────────────────────────────────────

"""
    PlantState

Immutable plant state. (B, F, A) carried as an SVector for stack
allocation; t_bin is the global bin counter (advances by 1 per
`plant_step`).
"""
struct PlantState
    bfa::SVector{3, Float64}
    t_bin::Int
end

"""
    init_plant_state(; init_state = INIT_STATE) -> PlantState

Build a fresh PlantState at the canonical init.
"""
function init_plant_state(; init_state = INIT_STATE)
    return PlantState(
        SVector{3, Float64}(Float64(init_state.B),
                            Float64(init_state.F),
                            Float64(init_state.A)),
        0,
    )
end


# ── Boundary reflection helper (pure) ─────────────────────────────────────
# Implemented via `@match` with guards — maps 1:1 to the Lean4 port's
# `match x with | x => if x < 0 then -x else if x > 1 then 2 - x else x`.

@inline _reflect_unit(x::Float64) = @match x begin
    x, if x < 0.0 end => -x
    x, if x > 1.0 end => 2.0 - x
    _                 => x
end


# ── Pure single-bin step ──────────────────────────────────────────────────

"""
    plant_step(s, Φ_t, params, dt, key) -> NamedTuple

One Euler-Maruyama step under control `Φ_t` from state `s`. Returns

    (state = PlantState(...), obs = (obs_B, obs_F, obs_A))

Same `(s, Φ_t, params, dt, key)` always returns the same result.
"""
function plant_step(s::PlantState,
                     Φ_t::Float64,
                     params::Dict{Symbol, Float64},
                     dt::Float64,
                     key::UInt64)
    rng = StableRNG(key)

    # v1.5 → v1 basis adapter: drift expects (kappa_B, kappa_F, ...).
    params_nt = params_v15_to_v1_nt(params)

    bfa = s.bfa
    d   = drift(bfa, params_nt, Φ_t)              # Vector{Float64}, length 3
    σ   = diffusion_state_dep(bfa, params_nt)     # Vector{Float64}, length 3
    ξ   = randn(rng, 3)                            # Vector{Float64}, length 3

    # SDE step (Float64 throughout — no fp32 inside the plant)
    y_pred = bfa .+ d .* dt .+ σ .* sqrt(dt) .* SVector{3, Float64}(ξ[1], ξ[2], ξ[3])

    # Boundary reflection (B ∈ [0, 1], F & A ≥ 0)
    y_next = SVector{3, Float64}(
        _reflect_unit(y_pred[1]),
        abs(y_pred[2]),
        abs(y_pred[3]),
    )

    # Obs from new state — independent key derived from input key
    obs = sample_obs_bfa(y_next, params, hash((key, :obs)))

    return (
        state = PlantState(y_next, s.t_bin + 1),
        obs   = obs,
    )
end


# ── Pure stride rollout ───────────────────────────────────────────────────

"""
    plant_rollout(s0, Φ_subdaily, params, dt, key0) -> NamedTuple

Apply `length(Φ_subdaily)` consecutive `plant_step`s from `s0`. Returns

    (final_state = PlantState(...),
     trajectory  = Matrix{Float64} of shape (stride_bins, 3),
     obs_B, obs_F, obs_A = Vector{Float32} of length stride_bins,
     Phi          = Vector{Float32} of length stride_bins)

The function is pure: same inputs always produce the same outputs.
"""
function plant_rollout(s0::PlantState,
                        Φ_subdaily::AbstractVector,
                        params::Dict{Symbol, Float64},
                        dt::Float64,
                        key0::UInt64)
    n     = length(Φ_subdaily)
    traj  = Matrix{Float64}(undef, n, 3)
    obs_B = Vector{Float64}(undef, n)
    obs_F = Vector{Float64}(undef, n)
    obs_A = Vector{Float64}(undef, n)

    s = s0
    @inbounds for k in 1:n
        sub_key = hash((key0, :step, k))
        nxt     = plant_step(s, Float64(Φ_subdaily[k]), params, dt, sub_key)
        s       = nxt.state
        traj[k, 1] = s.bfa[1]
        traj[k, 2] = s.bfa[2]
        traj[k, 3] = s.bfa[3]
        obs_B[k]   = nxt.obs.obs_B
        obs_F[k]   = nxt.obs.obs_F
        obs_A[k]   = nxt.obs.obs_A
    end

    return (
        final_state = s,
        trajectory  = traj,
        obs_B = Float32.(obs_B),
        obs_F = Float32.(obs_F),
        obs_A = Float32.(obs_A),
        Phi   = Float32.(Φ_subdaily),
    )
end

end # module Plant
