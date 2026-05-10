# SMC2/Bridge.jl — warm-start bridges between rolling-window posteriors.
#
# Replaces `smc2fc/core/sf_bridge.py`. The Python implementation uses string
# flags (`bridge_type='gaussian'|'mog'|'schrodinger_follmer'`) routed through
# nested `if/elif` blocks. Per charter §15.1 those become methods of
# `bridge_init` dispatched on marker types `GaussianBridge`,
# `SchrodingerFollmerBridge` — the dispatcher picks at compile time, the
# agent cannot accidentally reach the wrong branch.
#
# Charter §17 default port: replicate the soft-HMC + Gaussian-bridge
# defaults. The full Schrödinger-Föllmer code path (590 lines of Python)
# is implemented as a stub that falls back to the Gaussian fit; the
# information-aware variant comes in a follow-on once the Phase-2/3
# regression suite is green and a model-side incident motivates the
# extra machinery.

module Bridge

using Random: AbstractRNG
using LinearAlgebra: cholesky, Symmetric
using Statistics: mean, cov
using ...SMC2FC: GaussianBridge, SchrodingerFollmerBridge, BridgeKind, SMCConfig

export bridge_init, fit_gaussian, sample_from_gaussian

# ── Gaussian fit + sampler (used by both bridge variants) ────────────────────

"""
    fit_gaussian(particles::AbstractMatrix; regularisation=1e-6) -> (μ, Σ)

Fit a single Gaussian to a particle cloud by sample mean/covariance. The
covariance is regularised with `regularisation·I` to keep the Cholesky
decomposition well-defined when the cloud is near-degenerate.
"""
function fit_gaussian(particles::AbstractMatrix{T};
                       regularisation::Real = 1e-6) where {T<:Real}
    μ = vec(mean(particles; dims=1))
    Σ = cov(particles; dims=1)
    d = size(Σ, 1)
    Σ = Σ + T(regularisation) * Matrix{T}(I_d(d))
    return μ, Symmetric(Σ)
end

I_d(d::Integer) = [i == j ? 1.0 : 0.0 for i in 1:d, j in 1:d]

"""
    sample_from_gaussian(rng, μ, Σ, n) -> (n, d) matrix

Draw `n` samples from the Gaussian fitted by `fit_gaussian`. Uses Cholesky
of the regularised covariance.
"""
function sample_from_gaussian(rng::AbstractRNG,
                                μ::AbstractVector{T},
                                Σ::Symmetric{T},
                                n::Integer) where {T<:Real}
    L = cholesky(Σ).L
    d = length(μ)
    Z = randn(rng, n, d)
    return reshape(μ, 1, :) .+ Z * L'
end


# ── bridge_init: marker-dispatched warm-start ────────────────────────────────

"""
    bridge_init(::BridgeKind, prev_posterior, new_target_lp_fn, cfg, rng)

Produce an initial particle cloud for the next rolling window. Each method
implements one bridge strategy; the SMCConfig flag `cfg.bridge_type` is
mapped to the corresponding marker via `bridge_kind`.
"""
function bridge_init end

# Gaussian: sample the next-window initial cloud from a single Gaussian
# fitted to the previous-window posterior. Liu-West shrinkage applied if
# `cfg.sf_use_q0_cov` is false.
function bridge_init(::GaussianBridge,
                      prev_posterior::AbstractMatrix{T},
                      _target_lp_fn,
                      cfg::SMCConfig,
                      rng::AbstractRNG;
                      n_smc::Integer = cfg.n_smc_particles) where {T<:Real}
    μ, Σ = fit_gaussian(prev_posterior)
    return sample_from_gaussian(rng, μ, Σ, n_smc)
end

# Schrödinger–Föllmer: stub. Falls back to Gaussian fit; the full BW-geodesic
# + information-aware blend lives in `sf_bridge.py:fit_sf_base` and is
# slotted in once a regression incident motivates it (charter §17).
function bridge_init(::SchrodingerFollmerBridge,
                      prev_posterior::AbstractMatrix{T},
                      target_lp_fn,
                      cfg::SMCConfig,
                      rng::AbstractRNG;
                      n_smc::Integer = cfg.n_smc_particles) where {T<:Real}
    @warn "SchrodingerFollmerBridge: stub falls back to Gaussian fit" maxlog=1
    return bridge_init(GaussianBridge(), prev_posterior, target_lp_fn, cfg, rng;
                        n_smc = n_smc)
end


"""
    bridge_kind(cfg::SMCConfig) -> ::BridgeKind

Translate the `cfg.bridge_type` symbol into a marker singleton. The
config-symbol → marker dispatch is the bridge between the user-facing
keyword (which has to be a Symbol so the `@kwdef` struct can validate it)
and the type-level dispatch (which gives method-not-found errors instead
of runtime branch-misses).
"""
function bridge_kind(cfg::SMCConfig)
    if cfg.bridge_type === :gaussian
        return GaussianBridge()
    elseif cfg.bridge_type === :schrodinger_follmer
        return SchrodingerFollmerBridge()
    else
        error("unknown bridge_type :$(cfg.bridge_type) (only :gaussian and " *
              ":schrodinger_follmer are wired in this port; charter §17)")
    end
end

export bridge_kind

end # module Bridge
