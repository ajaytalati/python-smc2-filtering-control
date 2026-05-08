"""
    SMC2/Bridge.jl

Warm-start bridges between rolling-window posteriors. Replaces
`smc2fc/core/sf_bridge.py`.

The Python reference uses string flags
(`bridge_type='gaussian' | 'mog' | 'schrodinger_follmer'`) routed through
nested if/elif. Here, those become methods of `bridge_init` dispatched
on marker types `GaussianBridge`, `SchrodingerFollmerBridge`. The
config-symbol → marker translation is done by `bridge_kind`.

The full Schrödinger-Föllmer code path (the BW-geodesic + information-
aware blend) is implemented as a stub that falls back to the Gaussian
fit. The full implementation will be slotted in once a regression
incident motivates it.
"""
module Bridge

using Random: AbstractRNG
using LinearAlgebra: cholesky, Symmetric
using Statistics: mean, cov
using ...SMC2FC_functional: GaussianBridge, SchrodingerFollmerBridge,
                            BridgeKind, SMCConfig

export bridge_init, fit_gaussian, sample_from_gaussian, bridge_kind

# ── Gaussian fit + sampler ──────────────────────────────────────────────────

"""
    fit_gaussian(particles; regularisation=1e-6) -> (μ, Σ)

Fit a single Gaussian to a particle cloud by sample mean/covariance.

# Arguments
- `particles::AbstractMatrix{T}`: `(n, d)` cloud — rows are samples.

# Keyword arguments
- `regularisation::Real = 1e-6`: added to `Σ`'s diagonal to keep the
    Cholesky decomposition well-defined for near-degenerate clouds.

# Returns
- `μ::Vector{T}`: sample mean, length `d`.
- `Σ::Symmetric{T}`: regularised sample covariance, `d × d`.
"""
function fit_gaussian(particles::AbstractMatrix{T};
                       regularisation::Real = 1e-6) where {T<:Real}
    μ = vec(mean(particles; dims=1))
    Σ = cov(particles; dims=1)
    d = size(Σ, 1)
    Σ = Σ + T(regularisation) * Matrix{T}(_eye(d))
    return μ, Symmetric(Σ)
end

# Local identity-matrix helper (avoids importing `LinearAlgebra.I` just for this).
_eye(d::Integer) = [i == j ? 1.0 : 0.0 for i in 1:d, j in 1:d]

"""
    sample_from_gaussian(rng, μ, Σ, n) -> Matrix

Draw `n` samples from the Gaussian fitted by `fit_gaussian`.

# Arguments
- `rng::AbstractRNG`: PRNG.
- `μ::AbstractVector{T}`: mean.
- `Σ::Symmetric{T}`: covariance (regularised).
- `n::Integer`: sample count.

# Returns
- `Matrix{T}`: `(n, d)` matrix of samples.
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


# ── bridge_init: marker-dispatched warm-start ───────────────────────────────

"""
    bridge_init(::BridgeKind, prev_posterior, target_lp_fn, cfg, rng;
                 n_smc=cfg.n_smc_particles) -> Matrix

Produce an initial particle cloud for the next rolling window. Each
method implements one bridge strategy.

# Arguments
- `::BridgeKind` (marker singleton): `GaussianBridge()` or
    `SchrodingerFollmerBridge()` — multiple-dispatched.
- `prev_posterior::AbstractMatrix{T}`: previous-window posterior cloud,
    `(n_smc, d_theta)`.
- `target_lp_fn`: log-density of the new-window target (used by the SF
    method; ignored by the Gaussian method).
- `cfg::SMCConfig`: hyperparameters.
- `rng::AbstractRNG`: PRNG.

# Keyword arguments
- `n_smc::Integer = cfg.n_smc_particles`: size of the returned cloud.

# Returns
- `Matrix{T}`: `(n_smc, d_theta)` next-window initial cloud.
"""
function bridge_init end

function bridge_init(::GaussianBridge,
                      prev_posterior::AbstractMatrix{T},
                      _target_lp_fn,
                      cfg::SMCConfig,
                      rng::AbstractRNG;
                      n_smc::Integer = cfg.n_smc_particles) where {T<:Real}
    μ, Σ = fit_gaussian(prev_posterior)
    return sample_from_gaussian(rng, μ, Σ, n_smc)
end

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
    bridge_kind(cfg) -> ::BridgeKind

Translate `cfg.bridge_type::Symbol` into a marker singleton.

# Arguments
- `cfg::SMCConfig`: holds the user-facing `bridge_type::Symbol` keyword.

# Returns
- A `BridgeKind` singleton (`GaussianBridge()` or
    `SchrodingerFollmerBridge()`).

# Throws
- `ErrorException`: on unknown `bridge_type`.
"""
function bridge_kind(cfg::SMCConfig)
    if cfg.bridge_type === :gaussian
        return GaussianBridge()
    elseif cfg.bridge_type === :schrodinger_follmer
        return SchrodingerFollmerBridge()
    else
        error("unknown bridge_type :$(cfg.bridge_type) (only :gaussian and " *
              ":schrodinger_follmer are wired in this port)")
    end
end

end # module Bridge
