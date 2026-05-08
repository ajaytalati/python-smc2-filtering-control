"""
    Filtering/Kernels.jl

Liu–West shrinkage moment match, Gaussian smoothing kernels, and ESS
computation. Port of `smc2fc/filtering/_gk_kernel.py`.

All functions are generic over `AbstractArray{T,N}` so the same code paths
run with `Array{Float64}` (CPU) and `CuArray{Float32}` (GPU).
The public API of this module is already purely functional — every
function takes immutable inputs and returns a freshly allocated array.
"""
module Kernels

using LinearAlgebra: norm
using Statistics: std
using LogExpFunctions: logsumexp
using ..SMC2FC_functional: AbstractBackend, CPUBackend, CUDABackend

export compute_ess
export silverman_bandwidth, log_kernel_matrix
export ess_bandwidth_factor
export smooth_resample_basic, smooth_resample_ess_scaled
export smooth_resample, smooth_resample_ess_scaled_lw

"""Bandwidth floor — matches `_gk_kernel.py:104`."""
const _MIN_BW = 1e-6

# ── ESS ─────────────────────────────────────────────────────────────────────

"""
    compute_ess(log_w) -> Float64

Effective sample size from un-normalised log-weights.

# Arguments
- `log_w::AbstractVector{T}`: log-weights, any length.

# Returns
- Scalar in `[1, length(log_w)]`. Computes
    `ESS = 1 / Σ_i w̃_i²` where `w̃_i = w_i / Σ_j w_j`.
"""
function compute_ess(log_w::AbstractVector{T}) where {T<:Real}
    log_w_norm = log_w .- logsumexp(log_w)
    log_sum_w2 = logsumexp(2.0 .* log_w_norm)
    return exp(-log_sum_w2)
end

# ── Silverman bandwidth + log-kernel matrix ─────────────────────────────────

"""
    silverman_bandwidth(particles, stochastic_idx, K, scale) -> Vector

Per-component Silverman-rule bandwidth for the Gaussian kernel.

# Arguments
- `particles::AbstractMatrix{T}`: `(K, n_states)` cloud.
- `stochastic_idx::AbstractVector{<:Integer}`: state components that
    carry noise.
- `K::Integer`: particle count.
- `scale::Real`: user-supplied bandwidth multiplier.

# Returns
- `Vector{T}`: length `n_states`. Stochastic components hold
    `Silverman·k·scale·σ_d`; deterministic components hold `1e6` so they
    do not contribute to the pairwise kernel.
"""
function silverman_bandwidth(particles::AbstractMatrix{T},
                              stochastic_idx::AbstractVector{<:Integer},
                              K::Integer,
                              scale::Real) where {T<:Real}
    n_s  = size(particles, 2)
    n_st = length(stochastic_idx)

    silverman_factor = (4.0 / (n_st + 2.0)) ^ (1.0 / (n_st + 4.0))
    k_factor         = float(K) ^ (-1.0 / (n_st + 4.0))
    factor           = silverman_factor * k_factor * scale

    sub  = particles[:, stochastic_idx]
    σ    = vec(std(sub; dims=1))
    h_st = max.(factor .* σ, T(_MIN_BW))

    h_full = fill!(similar(particles, T, n_s), T(1e6))
    h_full[stochastic_idx] = h_st
    return h_full
end

"""
    log_kernel_matrix(particles, stochastic_idx, h) -> Matrix

Pairwise log-Gaussian kernel on the stochastic subspace.

# Arguments
- `particles::AbstractMatrix{T}`: `(K, n_states)` cloud.
- `stochastic_idx::AbstractVector{<:Integer}`: stochastic component indices.
- `h::AbstractVector{T}`: per-component bandwidths from
    `silverman_bandwidth`.

# Returns
- `Matrix{T}`: `(K, K)` matrix whose entries are
    `L[i,j] = -½ Σ_d ((x_i^d - x_j^d) / h_d)²`.
"""
function log_kernel_matrix(particles::AbstractMatrix{T},
                            stochastic_idx::AbstractVector{<:Integer},
                            h::AbstractVector{T}) where {T<:Real}
    sub   = particles[:, stochastic_idx]
    h_sub = h[stochastic_idx]
    scaled = sub ./ reshape(h_sub, 1, :)
    diff   = reshape(scaled, size(scaled, 1), 1, size(scaled, 2)) .-
             reshape(scaled, 1, size(scaled, 1), size(scaled, 2))
    sq     = dropdims(sum(diff .* diff; dims=3); dims=3)
    return -0.5 .* sq
end

# ── ESS-scaled bandwidth factor ─────────────────────────────────────────────

"""
    ess_bandwidth_factor(log_w, K) -> Float64

Smooth ESS-based bandwidth scaling.

# Arguments
- `log_w::AbstractVector{T}`: per-particle log-weights.
- `K::Integer`: particle count.

# Returns
- Scalar in `[0, 1]`: `(1 - ESS/K)²`. Healthy clouds → 0 (no blending);
    degenerate clouds → 1 (full Silverman blending). Derivative vanishes
    at ESS = K so the function is smooth at the boundary.
"""
function ess_bandwidth_factor(log_w::AbstractVector{T}, K::Integer) where {T<:Real}
    ess = compute_ess(log_w)
    ess_frac = clamp(ess / float(K), 0.0, 1.0)
    return (1.0 - ess_frac) ^ 2
end

# ── Smooth resample variants ────────────────────────────────────────────────

"""
    smooth_resample_basic(particles, log_w, stochastic_idx, K, bandwidth_scale) -> Matrix

v0 baseline blend: kernel-weighted average over particles, NO Liu–West
correction. Cost O(K²). Diagnostic only — production path is the
systematic + Liu–West variant in `Bootstrap.jl`.
"""
function smooth_resample_basic(particles::AbstractMatrix{T},
                                log_w::AbstractVector{T},
                                stochastic_idx::AbstractVector{<:Integer},
                                K::Integer,
                                bandwidth_scale::Real) where {T<:Real}
    h     = silverman_bandwidth(particles, stochastic_idx, K, bandwidth_scale)
    L     = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = reshape(log_w, 1, :) .+ L
    log_A   = log_w_b .- logsumexp(log_w_b; dims=2)
    A       = exp.(log_A)
    return A * particles
end

"""
    smooth_resample_ess_scaled(particles, log_w, stochastic_idx, K, bandwidth_scale) -> Matrix

v1.2 blend: kernel + ESS-scaled bandwidth, NO Liu–West.
"""
function smooth_resample_ess_scaled(particles::AbstractMatrix{T},
                                     log_w::AbstractVector{T},
                                     stochastic_idx::AbstractVector{<:Integer},
                                     K::Integer,
                                     bandwidth_scale::Real) where {T<:Real}
    ess_factor      = ess_bandwidth_factor(log_w, K)
    effective_scale = bandwidth_scale * ess_factor
    return smooth_resample_basic(particles, log_w, stochastic_idx, K, effective_scale)
end

"""
    smooth_resample(particles, log_w, stochastic_idx, K, bandwidth_scale) -> Matrix

v2 blend: kernel + Liu–West shrinkage correction. Fixed Silverman
bandwidth.

# Returns
- `Matrix{T}`: `a · A·particles + (1-a) · μ_w` where `A` is the
    kernel-weight matrix, `a = sqrt(1 - h_norm²)`, and `μ_w` is the
    weighted mean.
"""
function smooth_resample(particles::AbstractMatrix{T},
                          log_w::AbstractVector{T},
                          stochastic_idx::AbstractVector{<:Integer},
                          K::Integer,
                          bandwidth_scale::Real) where {T<:Real}
    h     = silverman_bandwidth(particles, stochastic_idx, K, bandwidth_scale)
    L     = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = reshape(log_w, 1, :) .+ L
    log_A   = log_w_b .- logsumexp(log_w_b; dims=2)
    A       = exp.(log_A)
    blended = A * particles

    n_st = length(stochastic_idx)
    silverman_factor = (4.0 / (n_st + 2.0)) ^ (1.0 / (n_st + 4.0))
    k_factor         = float(K) ^ (-1.0 / (n_st + 4.0))
    h_norm           = silverman_factor * k_factor * bandwidth_scale
    a = sqrt(clamp(1.0 - h_norm^2, 0.0, 1.0))

    w_norm = exp.(log_w .- logsumexp(log_w))
    μ_w    = vec(sum(reshape(w_norm, :, 1) .* particles; dims=1))

    return a .* blended .+ (1.0 - a) .* reshape(μ_w, 1, :)
end

"""
    smooth_resample_ess_scaled_lw(particles, log_w, stochastic_idx, K, bandwidth_scale) -> Matrix

v1.2 + Liu–West: ESS-scaled bandwidth and Liu–West shrinkage applied with
the *effective* bandwidth (so both reduce to identity in healthy regimes).
"""
function smooth_resample_ess_scaled_lw(particles::AbstractMatrix{T},
                                        log_w::AbstractVector{T},
                                        stochastic_idx::AbstractVector{<:Integer},
                                        K::Integer,
                                        bandwidth_scale::Real) where {T<:Real}
    ess_factor      = ess_bandwidth_factor(log_w, K)
    effective_scale = bandwidth_scale * ess_factor

    h     = silverman_bandwidth(particles, stochastic_idx, K, effective_scale)
    L     = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = reshape(log_w, 1, :) .+ L
    log_A   = log_w_b .- logsumexp(log_w_b; dims=2)
    A       = exp.(log_A)
    blended = A * particles

    n_st = length(stochastic_idx)
    silverman_factor = (4.0 / (n_st + 2.0)) ^ (1.0 / (n_st + 4.0))
    k_factor         = float(K) ^ (-1.0 / (n_st + 4.0))
    h_norm           = silverman_factor * k_factor * effective_scale
    a = sqrt(clamp(1.0 - h_norm^2, 0.0, 1.0))

    w_norm = exp.(log_w .- logsumexp(log_w))
    μ_w    = vec(sum(reshape(w_norm, :, 1) .* particles; dims=1))

    return a .* blended .+ (1.0 - a) .* reshape(μ_w, 1, :)
end

end # module Kernels
