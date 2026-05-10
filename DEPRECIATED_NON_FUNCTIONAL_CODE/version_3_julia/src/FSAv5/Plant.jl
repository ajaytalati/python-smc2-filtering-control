# FSAv5/Plant.jl — deterministic Euler-Maruyama + mutable StepwisePlant.
#
# Maps line-by-line to:
#   • LEAN  spec : `FSA_model_dev/lean/Fsa/V5/Plant.lean` (em_step)
#   • Python    : `models/fsa_high_res/_plant.py:84-150` (_plant_em_step)
#                 `models/fsa_high_res/_plant.py:158-329`  (StepwisePlant)
#
# Per the LEAN4-first charter: the EM-step transcription separates the
# math (drift + state-dependent diffusion) from the JAX RNG; noise is
# taken as an explicit input so the function is purely deterministic
# and amenable to differential testing.
#
# StepwisePlant is the stateful wrapper used by the closed-loop bench
# drivers. It mutates `state` and `t_bin` between strides; per the
# charter §12.2 mutation is `!`-suffixed (`advance!`).

using Random: AbstractRNG, MersenneTwister, randn

# ── Clipping/floor epsilons (mirror `_plant.py:104-106`) ───────────────────

const PLANT_EPS_B = 1.0e-4
const PLANT_EPS_S = 1.0e-4
const PLANT_EPS_A = 1.0e-4

@inline _clamp(lo::Float64, hi::Float64, x::Float64) =
    x < lo ? lo : (x > hi ? hi : x)

# ── Single-bin deterministic Euler-Maruyama step ───────────────────────────

"""
    em_step(y::FSAv5State, phi::BimodalPhi, p::DynParams,
             sigma_diag::AbstractVector{Float64}, dt::Float64,
             noise::AbstractVector{Float64})::FSAv5State

One Euler-Maruyama bin update with state-dependent diffusion. Pure
function — noise is explicit input (no internal RNG).

  - `y` : current state.
  - `phi` : per-bin stimulus.
  - `p` : truth parameters.
  - `sigma_diag` : 6-vector of diffusion scales.
  - `dt` : bin width in days.
  - `noise` : 6-vector of standard-normal samples drawn upstream.

Returns next state with boundary handling (B, S clamped to [ε, 1-ε];
F, A, K_* floored at 0).

Mirrors `Fsa.V5.emStep` in `Plant.lean` and the body of
`_plant._plant_em_step` at `_plant.py:108-146`.
"""
@inline function em_step(y::FSAv5State, phi::BimodalPhi, p::DynParams,
                          sigma_diag::AbstractVector{Float64}, dt::Float64,
                          noise::AbstractVector{Float64})::FSAv5State
    @assert length(sigma_diag) == 6
    @assert length(noise) == 6
    d_y = dynamics_drift(y, p, phi)
    sqrt_dt = sqrt(dt)

    # State-dependent diffusion magnitudes (Jacobi for B,S; CIR for F,A,K_*).
    B_cl   = _clamp(PLANT_EPS_B, 1.0 - PLANT_EPS_B, y.B)
    S_cl   = _clamp(PLANT_EPS_S, 1.0 - PLANT_EPS_S, y.S)
    F_cl   = max(y.F,   0.0)
    A_cl   = max(y.A,   0.0)
    KFB_cl = max(y.KFB, 0.0)
    KFS_cl = max(y.KFS, 0.0)
    g_B   = sqrt(B_cl * (1.0 - B_cl))
    g_S   = sqrt(S_cl * (1.0 - S_cl))
    g_F   = sqrt(F_cl)
    g_A   = sqrt(A_cl + PLANT_EPS_A)
    g_KFB = sqrt(KFB_cl)
    g_KFS = sqrt(KFS_cl)

    yB_next   = y.B   + dt * d_y.B   + sigma_diag[1] * g_B   * sqrt_dt * noise[1]
    yS_next   = y.S   + dt * d_y.S   + sigma_diag[2] * g_S   * sqrt_dt * noise[2]
    yF_next   = y.F   + dt * d_y.F   + sigma_diag[3] * g_F   * sqrt_dt * noise[3]
    yA_next   = y.A   + dt * d_y.A   + sigma_diag[4] * g_A   * sqrt_dt * noise[4]
    yKFB_next = y.KFB + dt * d_y.KFB + sigma_diag[5] * g_KFB * sqrt_dt * noise[5]
    yKFS_next = y.KFS + dt * d_y.KFS + sigma_diag[6] * g_KFS * sqrt_dt * noise[6]

    return FSAv5State(
        _clamp(PLANT_EPS_B, 1.0 - PLANT_EPS_B, yB_next),
        _clamp(PLANT_EPS_S, 1.0 - PLANT_EPS_S, yS_next),
        max(yF_next,   0.0),
        max(yA_next,   0.0),
        max(yKFB_next, 0.0),
        max(yKFS_next, 0.0),
    )
end

# ── Mutable StepwisePlant ──────────────────────────────────────────────────
# Stateful wrapper for closed-loop bench drivers. Holds current state
# + bin counter + truth params; `advance!` takes a stride-worth of
# daily Φ values, expands to sub-daily, integrates, and returns
# stride-summary observation data.
#
# Per charter §12.2 mutation is opt-in via `!` suffix; the public API
# remains functional at module boundaries.

"""
    StepwisePlant

Mutable plant simulator for the FSA-v5 closed-loop MPC bench.

Fields:
  - `state::FSAv5State` : current 6D latent state.
  - `t_bin::Int`         : current global bin index (0-based; first
                           call to `advance!` starts at `t_bin=0`).
  - `params::DynParams`  : truth parameters.
  - `sigma_diag::Vector{Float64}` : 6-vector of diffusion scales.
  - `dt::Float64`        : bin width (days; default 1/96).
  - `seed_offset::Int`   : RNG seed offset for reproducibility.

Constructor:
    `StepwisePlant(; state, params, sigma_diag, dt=DT_BIN_DAYS, seed_offset=0)`
"""
mutable struct StepwisePlant
    state       :: FSAv5State
    t_bin       :: Int
    params      :: DynParams
    sigma_diag  :: Vector{Float64}
    dt          :: Float64
    seed_offset :: Int
end

function StepwisePlant(; state::FSAv5State,
                          params::DynParams,
                          sigma_diag::AbstractVector{Float64},
                          dt::Float64       = DT_BIN_DAYS,
                          seed_offset::Int  = 0)
    return StepwisePlant(state, 0, params, Vector{Float64}(sigma_diag), dt, seed_offset)
end

"""
    advance!(plant::StepwisePlant, stride_bins::Int,
              phi_daily::AbstractMatrix{Float64};
              rng::Union{Nothing,AbstractRNG} = nothing)::Matrix{Float64}

Advance the plant by `stride_bins` time-bins under a stride-worth of
daily Φ values. `phi_daily` is shape `(n_days, 2)` — column 1 is
Φ_B, column 2 is Φ_S — covering at least `stride_bins / BINS_PER_DAY`
days. Each daily value is expanded to its 96-bin Gamma envelope via
`expand_daily_phi_to_subdaily`.

Mutates `plant.state` and `plant.t_bin`. Returns the realised
trajectory of shape `(stride_bins, 6)` for downstream observation
sampling and bookkeeping. RNG defaults to a deterministic
`MersenneTwister` seeded from `seed_offset + t_bin`.

Mirrors `_plant.StepwisePlant.advance` at `_plant.py:204-336`. -
"""
function advance!(plant::StepwisePlant, stride_bins::Int,
                   phi_daily::AbstractMatrix{Float64};
                   rng::Union{Nothing,AbstractRNG} = nothing)::Matrix{Float64}
    @assert size(phi_daily, 2) == 2 "phi_daily must be (n_days, 2); got $(size(phi_daily))"
    n_days_in = size(phi_daily, 1)
    n_days_needed = ceil(Int, stride_bins / BINS_PER_DAY)
    if n_days_in < n_days_needed
        # Repeat the last day if caller didn't supply enough — mirrors
        # the Python plant's tile-out behaviour for partial slices.
        pad = repeat(phi_daily[end:end, :], n_days_needed - n_days_in, 1)
        phi_daily = vcat(phi_daily, pad)
    end

    # Expand each channel separately to sub-daily Φ.
    phi_B_sub = expand_daily_phi_to_subdaily(phi_daily[:, 1])
    phi_S_sub = expand_daily_phi_to_subdaily(phi_daily[:, 2])
    @assert length(phi_B_sub) >= stride_bins
    @assert length(phi_S_sub) >= stride_bins

    # Deterministic RNG for reproducibility (matches Python's
    # `jax.random.PRNGKey(seed_offset + t_bin)` pattern).
    rng_local = rng === nothing ?
                  MersenneTwister(plant.seed_offset + plant.t_bin) :
                  rng

    traj = Matrix{Float64}(undef, stride_bins, 6)
    y = plant.state
    @inbounds for k in 1:stride_bins
        phi_k = BimodalPhi(phi_B_sub[k], phi_S_sub[k])
        noise = randn(rng_local, 6)
        y = em_step(y, phi_k, plant.params, plant.sigma_diag, plant.dt, noise)
        traj[k, :] .= (y.B, y.S, y.F, y.A, y.KFB, y.KFS)
    end

    plant.state = y
    plant.t_bin += stride_bins
    return traj
end

export em_step, StepwisePlant, advance!
export PLANT_EPS_B, PLANT_EPS_S, PLANT_EPS_A
