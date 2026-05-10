# Filtering/OT.jl — low-rank optimal-transport rescue with sigmoid blend.
#
# Replaces:
#   smc2fc/filtering/transport_kernel.py    (Nyström kernel factor)
#   smc2fc/filtering/sinkhorn.py            (low-rank Sinkhorn iterations)
#   smc2fc/filtering/project.py             (barycentric projection)
#   smc2fc/filtering/resample.py            (`ot_resample_lr` driver)
#
# The Python implementation hand-rolls Nyström + Sinkhorn + projection because
# JAX needs every operation to be JIT-compatible and fori_loop-friendly. The
# Julia port keeps the same algorithm — Nyström anchors, factored matvecs,
# fixed-iteration Sinkhorn — because OptimalTransport.jl's full Sinkhorn is
# O(N²) per iteration, while the bootstrap filter's hot path needs the
# low-rank O(N·r) variant. ~80 LOC, charter §13.
#
# All operations are AbstractArray-generic so the same code runs on CPU and
# CUDA backends.

module OT

using LogExpFunctions: logsumexp
using LinearAlgebra: norm
using Random: AbstractRNG
using ..Kernels: compute_ess

export compute_kernel_factor, factor_matvec, factor_matvec_batch
export sinkhorn_scalings, barycentric_projection
export ot_resample_lr, ot_blended_resample

# ── Nyström kernel factor (transport_kernel.py:compute_kernel_factor) ────────

"""
    compute_kernel_factor(x, anchor_idx, ε)

Nyström factor `K_NR ∈ R^{N×r}` such that `K_approx = K_NR * K_NR^T` ≈ K, the
full Gaussian kernel `K[i,j] = exp(-‖x_i - z_j‖² / ε)`. Column-normalised for
numerical stability. No matrix inverse — all operations are matmuls.
"""
function compute_kernel_factor(x::AbstractMatrix{T},
                                anchor_idx::AbstractVector{<:Integer},
                                ε::Real) where {T<:Real}
    anchors = x[anchor_idx, :]                                  # (r, d)
    # Pairwise squared distance via 3-axis broadcast — runs on CPU and CUDA.
    diff = reshape(x, size(x, 1), 1, size(x, 2)) .-
           reshape(anchors, 1, size(anchors, 1), size(anchors, 2))   # (N, r, d)
    sq   = dropdims(sum(diff .* diff; dims=3); dims=3)               # (N, r)
    K_NR = exp.(-sq ./ T(ε))
    col_norms = max.(vec(sum(K_NR; dims=1)), T(1e-30))               # (r,)
    return K_NR ./ reshape(col_norms, 1, :)
end

"""
    factor_matvec(v, K_NR) -> N-vector

`K_approx * v = K_NR * (K_NR' * v)` in O(N·r) instead of O(N²).
"""
factor_matvec(v::AbstractVector{T}, K_NR::AbstractMatrix{T}) where {T<:Real} =
    K_NR * (K_NR' * v)

"""
    factor_matvec_batch(V, K_NR) -> (N, d) matrix

`K_approx * V` for V shape (N, d). O(N·r·d).
"""
factor_matvec_batch(V::AbstractMatrix{T}, K_NR::AbstractMatrix{T}) where {T<:Real} =
    K_NR * (K_NR' * V)


# ── Sinkhorn scalings (sinkhorn.py:sinkhorn_scalings) ────────────────────────

"""
    sinkhorn_scalings(a, b, K_NR; n_iter=10) -> (u, v)

Fixed-iteration Sinkhorn in scaling form. Alternates row/column normalisation
of the factored kernel until `diag(u) * K_approx * diag(v)` has marginals (a, b).

`n_iter` is fixed (no convergence criterion) so that the differential test has
a deterministic graph and so the AD path through the resampler is well-defined.
"""
function sinkhorn_scalings(a::AbstractVector{T},
                            b::AbstractVector{T},
                            K_NR::AbstractMatrix{T};
                            n_iter::Integer = 10) where {T<:Real}
    # Build u, v on the same device as `a` (so a CuArray-resident `a` keeps
    # the whole iteration on GPU; otherwise BLAS gemv hits a mixed-device
    # ptr conversion error).
    u = fill!(similar(a, T), one(T))
    v = fill!(similar(b, T), one(T))
    for _ in 1:n_iter
        Kv    = factor_matvec(v, K_NR)
        u_new = a ./ max.(Kv, T(1e-30))
        Ku    = factor_matvec(u_new, K_NR)
        v_new = b ./ max.(Ku, T(1e-30))
        u, v  = u_new, v_new
    end
    return u, v
end


# ── Barycentric projection (project.py:barycentric_projection) ───────────────

"""
    barycentric_projection(u, v, x, K_NR) -> (N, d)

Convex-combination transport:
`new_x_i = (u_i · [K_approx (v ⊙ x)]_i) / (u_i · [K_approx v]_i)`.

Convex-combination guarantee: if all input rows lie in `[0,1]^d`, all output
rows lie in `[0,1]^d`. No logit/sigmoid round-trip needed.
"""
function barycentric_projection(u::AbstractVector{T},
                                 v::AbstractVector{T},
                                 x::AbstractMatrix{T},
                                 K_NR::AbstractMatrix{T}) where {T<:Real}
    denom = u .* factor_matvec(v, K_NR)
    denom = max.(denom, T(1e-30))
    vx    = reshape(v, :, 1) .* x
    numer = reshape(u, :, 1) .* factor_matvec_batch(vx, K_NR)
    return numer ./ reshape(denom, :, 1)
end


# ── Top-level resampler (resample.py:ot_resample_lr) ─────────────────────────

"""
    ot_resample_lr(particles, log_weights, rng, stochastic_indices;
                   ε=0.5, n_iter=10, rank=50) -> (N, n_states)

Differentiable OT resampling on the stochastic subspace. Deterministic state
components pass through unchanged.

Algorithm: extract stochastic states → pick `rank` random anchors → Nyström
factor → Sinkhorn scalings → barycentric projection → reassemble.
"""
function ot_resample_lr(particles::AbstractMatrix{T},
                         log_weights::AbstractVector{T},
                         rng::AbstractRNG,
                         stochastic_indices::AbstractVector{<:Integer};
                         ε::Real = 0.5,
                         n_iter::Integer = 10,
                         rank::Integer = 50) where {T<:Real}
    N, _    = size(particles)
    sto_idx = stochastic_indices

    a = fill(T(1.0) / N, N)
    b = exp.(log_weights .- logsumexp(log_weights))

    x = particles[:, sto_idx]                              # (N, d_s)

    rank_eff = min(rank, N)
    anchor_idx = randperm_n(rng, N, rank_eff)              # rank ≤ N, no replace

    K_NR = compute_kernel_factor(x, anchor_idx, ε)
    u, v = sinkhorn_scalings(a, b, K_NR; n_iter=n_iter)
    new_x = barycentric_projection(u, v, x, K_NR)

    out = copy(particles)
    @inbounds for (j, idx) in enumerate(sto_idx), i in 1:N
        out[i, idx] = new_x[i, j]
    end
    return out
end

# Tiny helper: choose `rank` distinct integers in [1, N].
function randperm_n(rng::AbstractRNG, N::Integer, rank::Integer)
    rank > N && error("rank ($rank) > N ($N)")
    return collect(view(randperm_internal(rng, N), 1:rank))
end

# Pure-Julia randperm to avoid pulling Random.randperm! through CUDA-array dispatch.
function randperm_internal(rng::AbstractRNG, N::Integer)
    p = collect(1:N)
    @inbounds for i in N:-1:2
        j = rand(rng, 1:i)
        p[i], p[j] = p[j], p[i]
    end
    return p
end


# ── Sigmoid-blended OT rescue (gk_dpf_v3_lite.py:198-218) ────────────────────

"""
    ot_blended_resample(systematic_resampled, ot_rescued, log_weights;
                        ot_max_weight, ot_threshold, ot_temperature) -> (N, n_states)

Sigmoid blend between systematic-resampled and OT-rescued clouds. Mirrors
`gk_dpf_v3_lite.py:209-212`:

    ot_weight = ot_max * sigmoid((ot_threshold - ESS) / ot_temperature)
    output    = (1 - ot_weight) * systematic + ot_weight * ot_rescued

When ESS is healthy (ESS ≫ ot_threshold), `ot_weight ≈ 0` and the output is
the cheap systematic-resampled cloud. When ESS collapses, `ot_weight → ot_max`
and the OT rescue smoothly takes over.
"""
function ot_blended_resample(systematic::AbstractMatrix{T},
                              ot_rescued::AbstractMatrix{T},
                              log_weights::AbstractVector{T};
                              ot_max_weight::Real,
                              ot_threshold::Real,
                              ot_temperature::Real) where {T<:Real}
    ess = compute_ess(log_weights)
    ot_weight = ot_max_weight * sigmoid((ot_threshold - ess) / ot_temperature)
    return (1.0 - ot_weight) .* systematic .+ ot_weight .* ot_rescued
end

@inline sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))

end # module OT
