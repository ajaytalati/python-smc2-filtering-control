# Filtering/Kernels.jl — Liu-West shrinkage moment match + Gaussian smoothing
# kernels + ESS computation.
#
# Port of `smc2fc/filtering/_gk_kernel.py`. The Python file is 360 lines but
# most of those are docstrings and the JAX `jax.checkpoint` / `jax.lax` glue;
# the actual math is ~80 lines.
#
# All functions are generic over `AbstractArray{T,N}` so the same code paths
# run with `Array{Float64}` (CPU) and `CuArray{Float32}` (GPU). Per charter
# §15.1: write the function once against the abstract type, run on whichever
# backend the caller supplied.

module Kernels

using LinearAlgebra: norm
using Statistics: std
using LogExpFunctions: logsumexp
using ..SMC2FC: AbstractBackend, CPUBackend, CUDABackend

export compute_ess
export silverman_bandwidth, log_kernel_matrix
export ess_bandwidth_factor
export smooth_resample_basic, smooth_resample_ess_scaled
export smooth_resample, smooth_resample_ess_scaled_lw

# Bandwidth floor (matches `_gk_kernel.py:104`).
const _MIN_BW = 1e-6

# ── ESS ──────────────────────────────────────────────────────────────────────

"""
    compute_ess(log_w::AbstractVector) -> scalar

Effective Sample Size from un-normalised log-weights.

ESS = 1 / Σ_i w̃_i²   where w̃_i = w_i / Σ_j w_j

Returns a scalar in [1, K]. Direct port of
`smc2fc/filtering/_gk_kernel.py:compute_ess`.
"""
function compute_ess(log_w::AbstractVector{T}) where {T<:Real}
    log_w_norm = log_w .- logsumexp(log_w)
    log_sum_w2 = logsumexp(2.0 .* log_w_norm)
    return exp(-log_sum_w2)
end

# ── Silverman bandwidth + log-kernel matrix ──────────────────────────────────

"""
    silverman_bandwidth(particles, stochastic_idx, K, scale)

Per-component Silverman-rule bandwidth for the Gaussian kernel.
Deterministic state dimensions are assigned `h = 1e6` so they do not
contribute to the pairwise kernel.
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

    # Materialise the slice (don't use @view — std over a SubArray of CuArray
    # falls through to scalar indexing).
    sub  = particles[:, stochastic_idx]
    σ    = vec(std(sub; dims=1))                              # (n_st,)
    h_st = max.(factor .* σ, T(_MIN_BW))                      # (n_st,) on same device

    # Build h_full on the same device as `particles`. Vectorised indexed
    # assignment `h_full[stochastic_idx] = h_st` is GPU-supported.
    h_full = fill!(similar(particles, T, n_s), T(1e6))
    h_full[stochastic_idx] = h_st
    return h_full
end

"""
    log_kernel_matrix(particles, stochastic_idx, h)

Pairwise log-Gaussian kernel on the stochastic subspace:
`L_ij = -½ Σ_d ((x_i^d - x_j^d) / h_d)²`. Returns a `(K, K)` matrix.
"""
function log_kernel_matrix(particles::AbstractMatrix{T},
                            stochastic_idx::AbstractVector{<:Integer},
                            h::AbstractVector{T}) where {T<:Real}
    sub   = particles[:, stochastic_idx]                      # (K, n_st)
    h_sub = h[stochastic_idx]                                 # (n_st,)
    scaled = sub ./ reshape(h_sub, 1, :)                      # (K, n_st)
    # Pairwise squared distance via 3-axis broadcast — runs on CPU and CUDA.
    diff   = reshape(scaled, size(scaled, 1), 1, size(scaled, 2)) .-
             reshape(scaled, 1, size(scaled, 1), size(scaled, 2))    # (K, K, n_st)
    sq     = dropdims(sum(diff .* diff; dims=3); dims=3)             # (K, K)
    return -0.5 .* sq
end

# ── ESS-scaled bandwidth factor ──────────────────────────────────────────────

"""
    ess_bandwidth_factor(log_w, K) -> scalar in [0, 1]

Smooth ESS-based scaling:  factor = (1 - ESS/K)².
- factor → 0 as ESS → K (healthy cloud, no blending)
- factor → 1 as ESS → 1 (degenerate cloud, full Silverman blending)
- derivative vanishes at ESS = K (smooth at the boundary)

Charter rationale and the MCLMC-tuner-regression motivation are in
`smc2fc/filtering/_gk_kernel.py:203-235`.
"""
function ess_bandwidth_factor(log_w::AbstractVector{T}, K::Integer) where {T<:Real}
    ess = compute_ess(log_w)
    ess_frac = clamp(ess / float(K), 0.0, 1.0)
    return (1.0 - ess_frac) ^ 2
end

# ── Smooth resample variants ─────────────────────────────────────────────────

"""
    smooth_resample_basic(particles, log_w, stochastic_idx, K, bandwidth_scale)

v0 baseline blend: kernel-weighted average over particles, NO Liu-West
correction. Cost O(K²). Used as a diagnostic comparison; the production code
path is the systematic + Liu-West variant in `Bootstrap.jl`.
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
    smooth_resample_ess_scaled(particles, log_w, stochastic_idx, K, bandwidth_scale)

v1.2 blend: kernel + ESS-scaled bandwidth, NO Liu-West.
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
    smooth_resample(particles, log_w, stochastic_idx, K, bandwidth_scale)

v2 blend: kernel + Liu-West shrinkage correction. Fixed Silverman bandwidth.
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
    μ_w    = vec(sum(reshape(w_norm, :, 1) .* particles; dims=1))   # (n_s,)

    return a .* blended .+ (1.0 - a) .* reshape(μ_w, 1, :)
end

"""
    smooth_resample_ess_scaled_lw(particles, log_w, stochastic_idx, K, bandwidth_scale)

v1.2 + Liu-West: ESS-scaled bandwidth and Liu-West shrinkage applied with the
*effective* bandwidth (so both reduce to identity in healthy regimes).
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
