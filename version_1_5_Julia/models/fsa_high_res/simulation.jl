# FSA v1.5 simulation — pure, stateless.
#
# Owns:
#   - BINS_PER_DAY (parsed from FSA_STEP_MINUTES env var at module load)
#   - DT_BIN_DAYS  = 1 / BINS_PER_DAY
#   - DEFAULT_PARAMS  — 14 dynamics + 3 obs-noise params, in v1.5 PARAMETRISATION
#   - INIT_STATE      — canonical (B=0.05, F=0.30, A=0.10)
#   - sample_obs_bfa  — pure Gaussian obs sampler keyed by UInt64
#   - params_v15_to_v1_nt  — adapter to v1's drift NamedTuple (basis rotation)
#
# Re-parametrisation (option B from FIM gate):
#   κ_B is replaced by B_inf = κ_B · τ_B (steady-state B at Φ=1, no A coupling).
#   κ_F is replaced by F_inf = κ_F · τ_F (steady-state F at Φ=1, no A coupling).
# The adapter `params_v15_to_v1_nt` converts back to v1's drift form so v1's
# `_dynamics.jl` can stay verbatim; only the parameter basis changes.
#
# No HR / sleep / stress / steps. No circadian C(t). No EXOGENOUS.
# No burst envelope. The plant feeds per-bin Φ directly to the SDE,
# and obs is one-shot Gaussian on each latent every bin (no gating).

module Simulation

using StaticArrays
using StableRNGs
using Match

import ..Dynamics: TRUTH_PARAMS

export BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE, PINNED_PARAMS
export sample_obs_bfa, params_v15_to_v1_nt, fill_pinned_nt


# ── Time-grid (resolves at module load) ────────────────────────────────────

const _STEP_MIN     = parse(Int, get(ENV, "FSA_STEP_MINUTES", "60"))
const BINS_PER_DAY  = (60 * 24) ÷ _STEP_MIN
const DT_BIN_DAYS   = 1.0 / BINS_PER_DAY


# ── Default parameters: 14 dynamics (from v1's Dynamics.TRUTH_PARAMS)
#    + 3 obs-noise (PINNED at 0.005 each per plan decision A) ──────────────

const DEFAULT_PARAMS = Dict{Symbol, Float64}(
    # Dynamics in v1.5 parametrisation. Truth values are computed from
    # v1's TRUTH_PARAMS so the plant trajectories are bit-equivalent at
    # truth — only the basis differs.
    :tau_B     => TRUTH_PARAMS.tau_B,                                # 42.0
    :tau_F     => TRUTH_PARAMS.tau_F,                                #  7.0
    :B_inf     => TRUTH_PARAMS.kappa_B * TRUTH_PARAMS.tau_B,         #  0.504
    :F_inf     => TRUTH_PARAMS.kappa_F * TRUTH_PARAMS.tau_F,         #  0.21
    :epsilon_A => TRUTH_PARAMS.epsilon_A,                             # 0.40
    :lambda_A  => TRUTH_PARAMS.lambda_A,                              # 1.00
    :mu_0      => TRUTH_PARAMS.mu_0,                                  # 0.02
    :mu_B      => TRUTH_PARAMS.mu_B,                                  # 0.30
    :mu_F      => TRUTH_PARAMS.mu_F,                                  # 0.10
    :mu_FF     => TRUTH_PARAMS.mu_FF,                                 # 0.40
    :eta       => TRUTH_PARAMS.eta,                                   # 0.20
    :sigma_B   => TRUTH_PARAMS.sigma_B,                               # 0.010
    :sigma_F   => TRUTH_PARAMS.sigma_F,                               # 0.012
    :sigma_A   => TRUTH_PARAMS.sigma_A,                               # 0.020

    # Observation noise — pinned. The filter does not estimate these;
    # they are constants of the experiment.
    :sigma_B_obs => 0.005,
    :sigma_F_obs => 0.005,
    :sigma_A_obs => 0.005,
)

const INIT_STATE = (B = 0.05, F = 0.30, A = 0.10)


# ── Pinned parameters (per FIM gate decision: option B + pin) ─────────────
# Four parameters are NOT estimated by the filter at v1.5's bench horizon:
#   τ_B  (chronic-B time constant)        — slow, can't be pinned in 14 d
#   η    (Stuart-Landau cubic damping)    — half of the (η, μ_FF) doublet
#   ε_A  (B-gain autonomic coupling)      — partner in the (τ_B, ε_A) doublet
#   μ_FF (Stuart-Landau F² curvature)     — partner in the (μ_F, μ_FF) shear
# All four are fixed at their truth values everywhere in the codebase.

const PINNED_PARAMS = Dict{Symbol, Float64}(
    :tau_B     => DEFAULT_PARAMS[:tau_B],
    :eta       => DEFAULT_PARAMS[:eta],
    :epsilon_A => DEFAULT_PARAMS[:epsilon_A],
    :mu_FF     => DEFAULT_PARAMS[:mu_FF],
)


# ── Adapter: v1.5 params (B_inf, F_inf, ...) → v1 drift NamedTuple ───────
# v1's `Dynamics.drift` expects a NamedTuple with `kappa_B`, `kappa_F`,
# `tau_B`, `tau_F`, etc. v1.5's parametrisation replaces (kappa_B, kappa_F)
# with (B_inf = kappa_B * tau_B, F_inf = kappa_F * tau_F). This adapter
# rotates the basis back at the call site. Pure: same input → same output.
#
# Generic on the value type so ForwardDiff can dual-number through it.

"""
    params_v15_to_v1_nt(p) -> NamedTuple

Convert a v1.5 parameter container to the NamedTuple v1's `drift()` and
`diffusion_state_dep()` expect. `p` may be a `Dict{Symbol, T}` (truth
values, plant) or a NamedTuple (filter / FIM call sites under
ForwardDiff). T is generic.

Conversion: `kappa_B = B_inf / tau_B`, `kappa_F = F_inf / tau_F`.

Implemented via `@match` on the input type — this maps 1:1 to the
Lean4 port's `match p with | .DictForm d => ... | .NamedTupleForm nt =>
...` over an `inductive ParamsForm` sum type.
"""
@inline params_v15_to_v1_nt(p) = @match p begin
    ::Dict => (
        tau_B    = p[:tau_B],
        tau_F    = p[:tau_F],
        kappa_B  = p[:B_inf] / p[:tau_B],
        kappa_F  = p[:F_inf] / p[:tau_F],
        epsilon_A = p[:epsilon_A],
        lambda_A  = p[:lambda_A],
        mu_0     = p[:mu_0],
        mu_B     = p[:mu_B],
        mu_F     = p[:mu_F],
        mu_FF    = p[:mu_FF],
        eta      = p[:eta],
        sigma_B  = p[:sigma_B],
        sigma_F  = p[:sigma_F],
        sigma_A  = p[:sigma_A],
    )
    ::NamedTuple => (
        tau_B    = p.tau_B,
        tau_F    = p.tau_F,
        kappa_B  = p.B_inf / p.tau_B,
        kappa_F  = p.F_inf / p.tau_F,
        epsilon_A = p.epsilon_A,
        lambda_A  = p.lambda_A,
        mu_0     = p.mu_0,
        mu_B     = p.mu_B,
        mu_F     = p.mu_F,
        mu_FF    = p.mu_FF,
        eta      = p.eta,
        sigma_B  = p.sigma_B,
        sigma_F  = p.sigma_F,
        sigma_A  = p.sigma_A,
    )
end


"""
    fill_pinned_nt(estimated::NamedTuple) -> NamedTuple

Take a NamedTuple with the 10 estimated v1.5 params (7 drift + 3
diffusion) and return a 14-field NamedTuple with the 4 pinned values
(`tau_B`, `eta`, `epsilon_A`, `mu_FF`) inserted from `PINNED_PARAMS`.
Generic on element type so ForwardDiff can dual-number through the
estimated entries.

Estimated NamedTuple must have fields:
  tau_F, B_inf, F_inf, lambda_A,
  mu_0, mu_B, mu_F, sigma_B, sigma_F, sigma_A
"""
@inline function fill_pinned_nt(estimated::NamedTuple)
    T = eltype(values(estimated))
    return (
        tau_B    = T(PINNED_PARAMS[:tau_B]),
        tau_F    = estimated.tau_F,
        B_inf    = estimated.B_inf,
        F_inf    = estimated.F_inf,
        epsilon_A = T(PINNED_PARAMS[:epsilon_A]),
        lambda_A  = estimated.lambda_A,
        mu_0     = estimated.mu_0,
        mu_B     = estimated.mu_B,
        mu_F     = estimated.mu_F,
        mu_FF    = T(PINNED_PARAMS[:mu_FF]),
        eta      = T(PINNED_PARAMS[:eta]),
        sigma_B  = estimated.sigma_B,
        sigma_F  = estimated.sigma_F,
        sigma_A  = estimated.sigma_A,
    )
end


# ── Pure obs sampler ──────────────────────────────────────────────────────

"""
    sample_obs_bfa(state::SVector{3}, params, key::UInt64)
        -> NamedTuple{(:obs_B, :obs_F, :obs_A), Tuple{Float64, Float64, Float64}}

Sample one Gaussian obs per latent. Pure: same `(state, key)` always
returns the same `(obs_B, obs_F, obs_A)`.
"""
function sample_obs_bfa(state::SVector{3, Float64},
                         params::Dict{Symbol, Float64},
                         key::UInt64)
    rng = StableRNG(key)
    σ_B = params[:sigma_B_obs]
    σ_F = params[:sigma_F_obs]
    σ_A = params[:sigma_A_obs]
    return (
        obs_B = state[1] + σ_B * randn(rng),
        obs_F = state[2] + σ_F * randn(rng),
        obs_A = state[3] + σ_A * randn(rng),
    )
end

end # module Simulation
