# FSA-v5 estimation — purely functional, stateless. Mirrors v1.5's
# `estimation.jl` but for the v5 surface:
#
#   - 37 estimated parameters (per tech guide §7.4, lines 904-927)
#   - 5-channel observation log-weight (HR / Sleep / Stress / Steps / VL)
#     with explicit per-channel gating (caller supplies the mask)
#   - `propagate_v5` advances a 6D particle cloud one bin via `em_step_v5`
#
# The channel-mean primitives (`hr_mean`, `sleep_prob`, etc.) and the
# core EM step (`em_step_v5`) are diff-tested against the Lean reference;
# the composition (sum of channel log-likelihoods, particle-cloud
# wrapper) is Julia-only and inherits correctness from those primitives.

module EstimationV5

using StaticArrays
using StableRNGs

import ..DynamicsV5: em_step_v5
import ..ObsV5:      hr_mean, sleep_prob, stress_mean,
                      steps_log_mean, volume_load_mean
import ..SimulationV5: TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5

export PARAM_NAMES_V5, PARAM_PRIOR_CONFIG_V5
export obs_log_weight_v5, propagate_v5


# ── 37 estimated parameter names ──────────────────────────────────────
# Per tech guide §7.4 (lines 904-927). 14 params are FROZEN
# (`SimulationV5.FROZEN_PARAMS_V5`); their values are merged in by the
# bench driver before calling `drift_v5` (per tech guide §9.5 — Bug 5
# prevention).

const PARAM_NAMES_V5 = [
    # Aerobic-capacity dynamics (3)
    :tau_B, :kappa_B, :epsilon_AB,
    # Strength-capacity dynamics (3)
    :tau_S, :kappa_S, :epsilon_AS,
    # Fatigue dynamics (2)
    :tau_F, :lambda_A,
    # Variable-dose K dynamics (1; tau_K and K^0 frozen)
    :mu_K,
    # Stuart-Landau drive (6)
    :mu_0, :mu_B, :mu_S, :mu_F, :mu_FF, :eta,
    # HR observation (5)
    :HR_base, :kappa_B_HR, :alpha_A_HR, :beta_C_HR, :sigma_HR,
    # Sleep Bernoulli (3)
    :k_C, :k_A, :c_tilde,
    # Stress observation (5)
    :S_base, :k_F, :k_A_S, :beta_C_S, :sigma_S_obs,
    # Steps observation (6)
    :mu_step0, :beta_B_st, :beta_F_st, :beta_A_st, :beta_C_st, :sigma_st,
    # Volume Load observation (3)
    :beta_S_VL, :beta_F_VL, :sigma_VL,
]

# Sanity check: the count must match tech guide §7.1's "37 estimated".
@assert length(PARAM_NAMES_V5) == 37 "PARAM_NAMES_V5 must list 37 parameters"


# ── Prior config — LogNormal centred on truth (tech guide §7) ─────────
# All 37 estimated parameters get LogNormal(log(truth), 0.30) priors
# unless they can be negative (none of these can: by construction the
# v5 parameters are non-negative). σ = 0.30 matches v1.5's prior width
# convention.

# Helper: look up the truth value for a parameter (it lives in either
# the dynamics dict or the obs-channel dict).
function _truth_for(k::Symbol)
    if haskey(TRUTH_PARAMS_V5, k)
        return TRUTH_PARAMS_V5[k]
    elseif haskey(DEFAULT_OBS_PARAMS_V5, k)
        return DEFAULT_OBS_PARAMS_V5[k]
    else
        error("no truth value for parameter $k")
    end
end

"""
    PARAM_PRIOR_CONFIG_V5 :: Vector{Tuple{Symbol, Symbol, Float64, Float64}}

`(name, kind, μ, σ)` per parameter. `kind ∈ (:LogNormal, :Normal)`.
For v5 every parameter uses `:LogNormal` since they are all strictly
positive in the tech guide's parametrisation. `μ` is `log(truth)`.

Centred on truth so a cold-start filter is correctly biased; tunable
later by widening σ if richer calibration data shifts the means.
"""
const PARAM_PRIOR_CONFIG_V5 = [
    (k, :LogNormal, log(_truth_for(k)), 0.30) for k in PARAM_NAMES_V5
]


# ── Observation log-weight (5 channels with explicit gating) ──────────

"""
    obs_log_weight_v5(particles, obs, gates, C, params) -> Vector{Float64}

For each particle row `(B, S, F, A, K_FB, K_FS)`, sum the per-channel
log-likelihood contributions across the 5 v5 observation channels,
honouring the per-channel `gates` (Bool flags from `align_obs_fn`-style
production usage).

Arguments:
  - `particles :: Matrix{Float64}` of shape `(M, 6)`.
  - `obs :: NamedTuple` with keys
       `(:obs_HR, :obs_sleep, :obs_S, :obs_steps, :obs_VL)`.
       `obs_sleep` is a `Bool`; the others are `Float64`.
       For `obs_steps` the convention is that the input is already in
       log-space (per tech guide §3.1: μ^st is a LOG mean; the channel
       is log-Gaussian).
  - `gates :: NamedTuple` with keys
       `(:hr_present, :sleep_present, :stress_present, :steps_present, :vl_present)`,
       all `Bool`. Channels with `false` contribute zero to the sum.
  - `C :: Float64` — circadian regressor at this bin.
  - `params :: Dict{Symbol, Float64}` — must contain dynamics +
       obs-channel parameters merged (Bug 5 prevention).

Returns a fresh `Vector{Float64}` of length M; does not mutate inputs.
"""
function obs_log_weight_v5(particles::AbstractMatrix{<:Real},
                            obs::NamedTuple,
                            gates::NamedTuple,
                            C::Real,
                            params::Dict{Symbol, Float64})
    M = size(particles, 1)
    out = Vector{Float64}(undef, M)

    # Pre-compute the constant log-normalisation terms for each
    # Gaussian channel (independent of particles).
    log2π = log(2π)
    σ_HR  = params[:sigma_HR]
    σ_S   = params[:sigma_S_obs]
    σ_st  = params[:sigma_st]
    σ_VL  = params[:sigma_VL]
    log_norm_HR = -0.5 * log2π - log(σ_HR)
    log_norm_S  = -0.5 * log2π - log(σ_S)
    log_norm_st = -0.5 * log2π - log(σ_st)
    log_norm_VL = -0.5 * log2π - log(σ_VL)
    inv2σHR² = 0.5 / (σ_HR * σ_HR)
    inv2σS²  = 0.5 / (σ_S  * σ_S)
    inv2σst² = 0.5 / (σ_st * σ_st)
    inv2σVL² = 0.5 / (σ_VL * σ_VL)

    @inbounds for m in 1:M
        y_m  = @view particles[m, :]
        lw_m = 0.0

        if gates.hr_present
            μ_HR = hr_mean(y_m, C, params)
            Δ    = obs.obs_HR - μ_HR
            lw_m += log_norm_HR - inv2σHR² * Δ * Δ
        end

        if gates.sleep_present
            p_s = sleep_prob(y_m, C, params)
            # Bernoulli: log p if obs_sleep else log(1-p).
            # Clamp away from 0 / 1 to avoid -Inf in degenerate cases.
            p_s_safe = clamp(p_s, 1e-12, 1.0 - 1e-12)
            lw_m += obs.obs_sleep ? log(p_s_safe) : log(1.0 - p_s_safe)
        end

        if gates.stress_present
            μ_S = stress_mean(y_m, C, params)
            Δ   = obs.obs_S - μ_S
            lw_m += log_norm_S - inv2σS² * Δ * Δ
        end

        if gates.steps_present
            μ_st = steps_log_mean(y_m, C, params)
            Δ    = obs.obs_steps - μ_st
            lw_m += log_norm_st - inv2σst² * Δ * Δ
        end

        if gates.vl_present
            μ_VL = volume_load_mean(y_m, params)
            Δ    = obs.obs_VL - μ_VL
            lw_m += log_norm_VL - inv2σVL² * Δ * Δ
        end

        out[m] = lw_m
    end

    return out
end


# ── Particle-cloud propagation (one bin EM step) ──────────────────────

"""
    propagate_v5(particles, phi, params, dt, key) -> Matrix{Float64}

Advance every particle by one EM bin step under bimodal control
`phi = (Phi_B, Phi_S)`. Returns a fresh `(M, 6)` matrix; does NOT
mutate `particles`.

`key :: UInt64` is the per-call RNG seed. Sub-keys are derived
deterministically as `hash((key, :prop, m))` so the same `(particles,
phi, params, dt, key)` always returns the same result.
"""
function propagate_v5(particles::AbstractMatrix{Float64},
                       phi::Tuple{<:Real, <:Real},
                       params::Dict{Symbol, Float64},
                       dt::Float64,
                       key::UInt64)
    M = size(particles, 1)
    out = Matrix{Float64}(undef, M, 6)

    sigma_diag = SVector{6, Float64}(
        params[:sigma_B], params[:sigma_S], params[:sigma_F],
        params[:sigma_A], params[:sigma_K], params[:sigma_K],
    )

    @inbounds for m in 1:M
        rng_m = StableRNG(hash((key, :prop, m)))
        y_m   = SVector{6, Float64}(
            particles[m, 1], particles[m, 2], particles[m, 3],
            particles[m, 4], particles[m, 5], particles[m, 6],
        )
        ξ = SVector{6, Float64}(
            randn(rng_m), randn(rng_m), randn(rng_m),
            randn(rng_m), randn(rng_m), randn(rng_m),
        )
        y_next = em_step_v5(y_m, phi, params, sigma_diag, dt, ξ)
        out[m, 1], out[m, 2], out[m, 3] = y_next[1], y_next[2], y_next[3]
        out[m, 4], out[m, 5], out[m, 6] = y_next[4], y_next[5], y_next[6]
    end
    return out
end

end # module EstimationV5
