# FSA v1.5 estimation — purely functional, stateless.
#
# This file gives the FILTER side just enough to:
#   - know the 14 dynamics-parameter prior config (PARAM_PRIOR_CONFIG)
#   - evaluate the obs log-weight at a given (particles, obs) pair
#   - propagate particles forward by one bin
#
# All functions are pure: same inputs → same outputs, no mutation.
#
# Note: `propagate` here uses the prior-predictive (no locally-guided
# proposal) because the obs is so informative (direct Gaussian on each
# latent every bin) that prior-predictive does not degenerate.

module Estimation

using StableRNGs

import ..Dynamics: drift, diffusion_state_dep
import ..Simulation: DEFAULT_PARAMS, params_v15_to_v1_nt

export PARAM_NAMES, PARAM_PRIOR_CONFIG, propagate, obs_log_weight


# ── 14 dynamics parameter names ───────────────────────────────────────────

"""
    PARAM_NAMES :: Vector{Symbol}

Parameter ordering used by the filter. The 3 obs-noise params
(σ_B_obs, σ_F_obs, σ_A_obs) are PINNED per the plan and are NOT here.
The 4 dynamics params (`τ_B`, `η`, `ε_A`, `μ_FF`) are also PINNED per
the FIM gate decision (option B + pin {τ_B, η, ε_A, μ_FF}) and are NOT
here.

Filter estimates: 7 drift + 3 diffusion = 10 params.
"""
const PARAM_NAMES = [
    :tau_F, :B_inf, :F_inf,
    :lambda_A,
    :mu_0, :mu_B, :mu_F,
    :sigma_B, :sigma_F, :sigma_A,
]


# ── Prior config — lognormal centred on truth where positivity required ───
# Each entry is (param_name, prior_kind, μ_unconstrained, σ_unconstrained).
# - prior_kind = :LogNormal means the parameter is exp(unconstrained); the
#   filter samples in unconstrained space and the model sees exp(z).
# - prior_kind = :Normal means the parameter is the unconstrained value
#   itself (used for parameters that can be negative or near-zero).

"""
    PARAM_PRIOR_CONFIG :: Vector{Tuple{Symbol, Symbol, Float64, Float64}}

`(name, kind, μ, σ)` per parameter. `kind ∈ (:LogNormal, :Normal)`. `μ`
is the prior mean in the unconstrained (= log for :LogNormal) space; `σ`
is the prior standard deviation in that space.

Centred on truth (so a cold-start filter is correctly biased) with
modest σ's that match v2's prior widths for the shared parameters.
"""
const PARAM_PRIOR_CONFIG = [
    (:tau_F,     :LogNormal, log(DEFAULT_PARAMS[:tau_F]),     0.30),
    (:B_inf,     :LogNormal, log(DEFAULT_PARAMS[:B_inf]),     0.30),
    (:F_inf,     :LogNormal, log(DEFAULT_PARAMS[:F_inf]),     0.30),
    (:lambda_A,  :LogNormal, log(DEFAULT_PARAMS[:lambda_A]),  0.30),
    (:mu_0,      :LogNormal, log(DEFAULT_PARAMS[:mu_0]),      0.30),
    (:mu_B,      :LogNormal, log(DEFAULT_PARAMS[:mu_B]),      0.30),
    (:mu_F,      :LogNormal, log(DEFAULT_PARAMS[:mu_F]),      0.30),
    (:sigma_B,   :LogNormal, log(DEFAULT_PARAMS[:sigma_B]),   0.30),
    (:sigma_F,   :LogNormal, log(DEFAULT_PARAMS[:sigma_F]),   0.30),
    (:sigma_A,   :LogNormal, log(DEFAULT_PARAMS[:sigma_A]),   0.30),
]


# ── Pure obs log-weight ───────────────────────────────────────────────────

"""
    obs_log_weight(particles, obs, params) -> Vector{Float64}

For each particle row `(B, F, A)`, return

    Σ_X  -0.5·((obs_X - X) / σ_X_obs)²  -  0.5·log(2π σ_X_obs²)

for X ∈ {B, F, A}. Pure: returns a fresh vector, does not mutate inputs.

`particles :: Matrix{Float64}` of shape (M, 3) (one row per particle).
`obs       :: NamedTuple{(:obs_B, :obs_F, :obs_A), ...}`
`params    :: Dict{Symbol, Float64}` — must contain σ_B_obs, σ_F_obs, σ_A_obs.
"""
function obs_log_weight(particles::AbstractMatrix{<:Real},
                         obs::NamedTuple,
                         params::Dict{Symbol, Float64})
    M = size(particles, 1)
    σ_B = params[:sigma_B_obs]
    σ_F = params[:sigma_F_obs]
    σ_A = params[:sigma_A_obs]

    # Const log-norm term (per-bin, per-channel) — same for every particle
    log_norm = -0.5 * log(2π) - log(σ_B) +
               -0.5 * log(2π) - log(σ_F) +
               -0.5 * log(2π) - log(σ_A)

    inv2σB² = 0.5 / (σ_B * σ_B)
    inv2σF² = 0.5 / (σ_F * σ_F)
    inv2σA² = 0.5 / (σ_A * σ_A)

    obs_B = obs.obs_B
    obs_F = obs.obs_F
    obs_A = obs.obs_A

    out = Vector{Float64}(undef, M)
    @inbounds for m in 1:M
        ΔB = obs_B - particles[m, 1]
        ΔF = obs_F - particles[m, 2]
        ΔA = obs_A - particles[m, 3]
        out[m] = log_norm - inv2σB² * ΔB^2 - inv2σF² * ΔF^2 - inv2σA² * ΔA^2
    end
    return out
end


# ── Pure propagate (prior-predictive) ─────────────────────────────────────

"""
    propagate(particles, Φ_t, params, dt, key) -> Matrix{Float64}

Advance every particle by ONE Euler-Maruyama step under control `Φ_t`.
Returns a NEW (M, 3) matrix; does NOT mutate `particles`.

`Φ_t :: Float64` — the per-bin control input.
`params :: Dict{Symbol, Float64}` — full param dict (dynamics + obs noise).
`dt :: Float64` — bin step in days.
`key :: UInt64` — RNG key; deterministic from this.
"""
function propagate(particles::AbstractMatrix{Float64},
                    Φ_t::Float64,
                    params::Dict{Symbol, Float64},
                    dt::Float64,
                    key::UInt64)
    M = size(particles, 1)
    out = Matrix{Float64}(undef, M, 3)

    # v1.5 → v1 basis adapter (B_inf, F_inf → kappa_B, kappa_F).
    params_nt = params_v15_to_v1_nt(params)
    sqrt_dt = sqrt(dt)

    @inbounds for m in 1:M
        rng_m = StableRNG(hash((key, :prop, m)))
        y     = [particles[m, 1], particles[m, 2], particles[m, 3]]
        d     = drift(y, params_nt, Φ_t)
        σ     = diffusion_state_dep(y, params_nt)
        ξ     = randn(rng_m, 3)
        y_pred = y .+ d .* dt .+ σ .* sqrt_dt .* ξ
        # Boundary reflection
        out[m, 1] = y_pred[1] < 0.0 ? -y_pred[1] :
                     (y_pred[1] > 1.0 ? 2.0 - y_pred[1] : y_pred[1])
        out[m, 2] = abs(y_pred[2])
        out[m, 3] = abs(y_pred[3])
    end
    return out
end

end # module Estimation
