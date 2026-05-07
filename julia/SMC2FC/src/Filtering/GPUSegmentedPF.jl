# Filtering/GPUSegmentedPF.jl — GPU-resident segmented PF primitives.
#
# Shared across all GPU PF models. Mirrors the framework-side path of
# `smc2fc/filtering/gk_dpf_v3_lite.py`:
#
#   - per-chain weight reductions (max, log Z, ESS, weighted mean,
#     normalised weights, cumulative sum)
#   - per-chain systematic resampling (deterministic offset for CRN)
#   - Liu-West shrinkage between segments
#   - OT sigmoid-blend rescue (Nyström + low-rank Sinkhorn) — uses the
#     framework's `OT.jl` primitives on CuArray inputs
#
# All kernels are KernelAbstractions / CUDA.jl based; particle/log_w/weights
# stay GPU-resident across the full window. CPU is only touched for:
#   - launching kernels
#   - reading the per-chain log-likelihood at the end of a window
#
# Models supply ONLY their per-particle propagate kernel.

module GPUSegmentedPF

using CUDA
using KernelAbstractions
using LogExpFunctions: logsumexp
using Random: AbstractRNG, MersenneTwister

using ..OT: compute_kernel_factor, sinkhorn_scalings, barycentric_projection,
            ot_resample_lr

export gpu_resample_liu_west_kernel!
export gpu_per_chain_stats_kernel!
export gpu_normalize_and_cumsum_kernel!
export gpu_ot_blend_chain_kernel!
export GPUSegmentedBuffers
export run_segmented_smc_step!


# ── Per-chain weight stats kernel (one thread per chain) ─────────────────
#
# For each chain m ∈ [1, M], compute:
#   log_max[m]  = max log_w over chain's K particles
#   log_z[m]    = log Σ exp(log_w − log_max)
#   ess[m]      = 1 / Σ w_normalised²
#   mu[m, d]    = Σ w · x_d  (weighted mean per state dim)
#
# Embarrassingly parallel across chains — M threads, each doing K-step
# serial reduction. M ≪ GPU SM count is fine; M·d_state mults dominate
# the reduction body so vectorised loads through the L1 cache are enough.

@kernel function gpu_per_chain_stats_kernel!(
    log_max,             # (M,) Float32
    log_z,               # (M,) Float32
    ess,                 # (M,) Float32
    mu_per_chain,        # (M, n_states) Float32
    log_w,               # (M·K,) Float32
    particles,           # (M·K, n_states) Float32
    M::Int, K::Int, n_states::Int,
)
    m = @index(Global, Linear)
    if m <= M
        base = (m - 1) * K

        # Pass 1: find max log_w in chain
        @inbounds lm = log_w[base + 1]
        @inbounds for k in 2:K
            v = log_w[base + k]
            if v > lm
                lm = v
            end
        end
        log_max[m] = lm

        # Pass 2: sum exp(log_w - lm)
        sume = 0f0
        @inbounds for k in 1:K
            sume += exp(log_w[base + k] - lm)
        end
        log_z[m] = log(sume)

        # Pass 3: ESS + weighted mean
        sum_sq = 0f0
        @inbounds for d in 1:n_states
            mu_per_chain[m, d] = 0f0
        end
        @inbounds for k in 1:K
            i = base + k
            w = exp(log_w[i] - lm) / sume
            sum_sq += w * w
            for d in 1:n_states
                mu_per_chain[m, d] += w * particles[i, d]
            end
        end
        ess[m] = 1f0 / max(sum_sq, 1f-30)
    end
end


# ── Normalised weights + per-chain cumulative sum kernel ─────────────────
#
# Computes the normalised weights and per-chain prefix sum. To stay simple
# we use one thread per chain doing a serial cumulative sum. Faster
# alternatives exist (Blelloch scan in shared memory) but K ≤ 1000 makes
# the serial scan ~1µs per chain — fine.

@kernel function gpu_normalize_and_cumsum_kernel!(
    weights,             # (M·K,) Float32 — output normalised
    cumsum_w,            # (M·K,) Float32 — output per-chain cumulative
    log_w,               # (M·K,) Float32
    log_max,             # (M,)
    log_z,               # (M,)
    M::Int, K::Int,
)
    m = @index(Global, Linear)
    if m <= M
        base = (m - 1) * K
        lm = log_max[m]
        lz = log_z[m]
        s = 0f0
        @inbounds for k in 1:K
            i = base + k
            w = exp(log_w[i] - lm - lz)
            weights[i] = w
            s += w
            cumsum_w[i] = s
        end
    end
end


# ── Per-chain systematic resample + Liu-West shrinkage (n_states-generic) ──
#
# Each thread handles one particle. Binary-searches the chain's cumsum
# for the systematic-resample target (deterministic 0.5/K offset for CRN),
# then writes  x_new = a · x_resampled + (1 − a) · μ_chain.

@kernel function gpu_resample_liu_west_kernel!(
    particles_old,       # (M·K, n_states)
    particles_new,       # (M·K, n_states)
    cumsum_w,            # (M·K,)
    mu_per_chain,        # (M, n_states)
    a_shrink::Float32,
    log_w_out,           # (M·K,) reset to 0
    K_per_chain::Int,
    n_states::Int,
)
    i_global = @index(Global, Linear)
    m_chain = ((i_global - 1) ÷ K_per_chain) + 1
    i_local = ((i_global - 1) % K_per_chain) + 1
    base = (m_chain - 1) * K_per_chain

    target_v = (Float32(i_local - 1) + 0.5f0) / Float32(K_per_chain)

    lo = 1
    hi = K_per_chain
    @inbounds while lo < hi
        mid = (lo + hi) ÷ 2
        if cumsum_w[base + mid] < target_v
            lo = mid + 1
        else
            hi = mid
        end
    end
    idx = lo

    @inbounds for d in 1:n_states
        x_resampled = particles_old[base + idx, d]
        mu = mu_per_chain[m_chain, d]
        particles_new[i_global, d] = a_shrink * x_resampled + (1f0 - a_shrink) * mu
    end
    @inbounds log_w_out[i_global] = 0f0
end


# ── OT projection per chain (GPU-resident) ───────────────────────────────
#
# Applies the OT sigmoid-blend rescue from `gk_dpf_v3_lite.py:198-218`
# entirely on the GPU. For each chain m:
#   1. Build Nyström kernel factor K_NR[m] from the chain's stochastic-
#      subspace particles + a fixed CRN anchor set (uploaded once).
#   2. Sinkhorn-scale n_iter iterations to get (u, v) → barycentric
#      projection x_proj[m].
#   3. Compute per-chain ESS-driven blend weight ot_w[m].
#   4. x_new[m] = (1 − ot_w) · x_old[m] + ot_w · x_proj[m].
#
# The Nyström + Sinkhorn + barycentric primitives come from `OT.jl` and are
# already AbstractMatrix-generic. We invoke them per-chain via slicing
# CuArray views, so ALL the BLAS-level work runs on device.

"""
    gpu_ot_blend_chain!(particles_gpu, log_w_gpu, ess_gpu,
                          anchor_idx_per_chain_gpu,
                          ot_max, ot_threshold, ot_temperature,
                          ε, n_iter, M, K, n_states, stochastic_indices)

GPU OT sigmoid-blend rescue. `particles_gpu` is `(M·K, n_states)` on device,
`anchor_idx_per_chain_gpu` is `(M, rank)` on device — pre-generated CPU-
side and uploaded once. Mutates `particles_gpu` in place.
"""
function gpu_ot_blend_chain!(particles_gpu::AbstractMatrix{Float32},
                              log_w_gpu::AbstractVector{Float32},
                              ess_gpu::AbstractVector{Float32},
                              anchor_idx_per_chain_cpu::AbstractMatrix{<:Integer},
                              ot_max::Real, ot_threshold::Real, ot_temperature::Real,
                              ε::Real, n_iter::Integer,
                              M::Int, K::Int, n_states::Int,
                              stochastic_indices::AbstractVector{<:Integer})
    ot_max_f32 = Float32(ot_max)
    if ot_max_f32 < 1f-6
        return
    end
    rank = size(anchor_idx_per_chain_cpu, 2)
    ess_cpu = Array(ess_gpu)         # (M,) — small, cheap

    # Per chain: do the OT projection on the chain's slice of particles_gpu
    # (CuArray views). Skips chains with high ESS for speed.
    @inbounds for m in 1:M
        ess_now = Float64(ess_cpu[m])
        ot_w = Float64(ot_max_f32) *
               (1.0 / (1.0 + exp((ess_now - Float64(ot_threshold)) /
                                  Float64(ot_temperature))))
        if ot_w < 1e-6
            continue
        end
        base = (m - 1) * K
        # Slice this chain's particles + log_w (still GPU-resident).
        chain_parts = view(particles_gpu, base+1:base+K, :)
        chain_log_w = view(log_w_gpu, base+1:base+K)

        # Build kernel factor using this chain's pre-generated anchor indices.
        anchor_idx = anchor_idx_per_chain_cpu[m, :]
        # x is the stochastic-subspace particles (we use all 3 indices).
        x_chain = chain_parts[:, stochastic_indices]    # CuArray
        try
            K_NR = compute_kernel_factor(x_chain, anchor_idx, ε)
            # Marginals
            a = CUDA.fill(Float32(1) / Float32(K), K)
            # b = exp(log_w − logsumexp) — compute on GPU
            log_w_arr = Array(chain_log_w)
            lm = maximum(log_w_arr)
            sume = sum(exp.(log_w_arr .- lm))
            b_cpu = Float32.(exp.(log_w_arr .- lm) ./ sume)
            b_gpu = CuArray(b_cpu)
            u, v = sinkhorn_scalings(a, b_gpu, K_NR; n_iter = n_iter)
            x_proj = barycentric_projection(u, v, x_chain, K_NR)
            # Blend on GPU (in place)
            ot_w_f = Float32(ot_w)
            for (j, idx) in enumerate(stochastic_indices)
                @views chain_parts[:, idx] .=
                    (1f0 - ot_w_f) .* chain_parts[:, idx] .+
                    ot_w_f .* x_proj[:, j]
            end
        catch err
            @warn "OT rescue failed for chain $m" exception=err
        end
    end
    return
end


# ── Pre-allocated buffers for the segmented driver ───────────────────────

mutable struct GPUSegmentedBuffers
    K_per_chain::Int
    M_max::Int
    n_states::Int
    particles_a::CuArray{Float32,2}    # (M·K, n_states)
    particles_b::CuArray{Float32,2}
    log_w::CuArray{Float32,1}
    weights::CuArray{Float32,1}
    cumsum_w::CuArray{Float32,1}
    log_max::CuArray{Float32,1}        # (M,) per-chain max log_w
    log_z::CuArray{Float32,1}          # (M,)
    ess::CuArray{Float32,1}            # (M,)
    mu_per_chain::CuArray{Float32,2}   # (M, n_states)
    # Pre-generated anchor indices for OT (M_max × rank), CPU side.
    anchor_idx_cpu::Matrix{Int}
    rank::Int
    resample_kernel::Any
    stats_kernel::Any
    norm_cumsum_kernel::Any
end


function GPUSegmentedBuffers(K_per_chain::Int, M_max::Int, n_states::Int;
                              ot_rank::Int = 50, anchor_seed::Int = 0)
    Ntot = K_per_chain * M_max
    rank_eff = min(ot_rank, K_per_chain)
    rng = MersenneTwister(anchor_seed)
    # Pre-generate anchor indices per chain — they are CRN-shared across
    # iterations of the SMC² outer loop (Common Random Numbers).
    anchor_idx_cpu = Matrix{Int}(undef, M_max, rank_eff)
    for m in 1:M_max
        perm = randperm_internal_cpu(rng, K_per_chain)
        anchor_idx_cpu[m, :] = perm[1:rank_eff]
    end
    return GPUSegmentedBuffers(
        K_per_chain, M_max, n_states,
        CUDA.zeros(Float32, Ntot, n_states),
        CUDA.zeros(Float32, Ntot, n_states),
        CUDA.zeros(Float32, Ntot),
        CUDA.zeros(Float32, Ntot),
        CUDA.zeros(Float32, Ntot),
        CUDA.zeros(Float32, M_max),
        CUDA.zeros(Float32, M_max),
        CUDA.zeros(Float32, M_max),
        CUDA.zeros(Float32, M_max, n_states),
        anchor_idx_cpu, rank_eff,
        gpu_resample_liu_west_kernel!(CUDABackend(), 256),
        gpu_per_chain_stats_kernel!(CUDABackend(), 64),
        gpu_normalize_and_cumsum_kernel!(CUDABackend(), 64),
    )
end


# Tiny helper used only at buffer construction (CPU side).
function randperm_internal_cpu(rng::AbstractRNG, N::Integer)
    p = collect(1:N)
    @inbounds for i in N:-1:2
        j = rand(rng, 1:i)
        p[i], p[j] = p[j], p[i]
    end
    return p
end


# ── End-of-segment SMC step (GPU-resident, no host sync until window end) ──
# Wraps: stats kernel → normalize+cumsum → OT blend (optional) → resample.

"""
    run_segmented_smc_step!(bufs, M, n_states, a_shrink;
                              ot_max, ot_threshold, ot_temperature,
                              ot_epsilon, ot_n_iter, use_ot)

Run end-of-segment housekeeping on the per-chain particle cloud:
1. per-chain weight stats (max, logZ, ESS, weighted mean) — GPU kernel
2. normalised weights + cumulative sum — GPU kernel
3. (optional) OT sigmoid-blend rescue — GPU primitives
4. systematic resample + Liu-West shrink — GPU kernel
"""
function run_segmented_smc_step!(bufs::GPUSegmentedBuffers, M::Int, a_shrink::Float32;
                                   ot_max::Real = 0.01,
                                   ot_threshold::Real = 0.05 * 1.0,
                                   ot_temperature::Real = 5.0,
                                   ot_epsilon::Real = 0.5,
                                   ot_n_iter::Integer = 10,
                                   use_ot::Bool = true)
    K = bufs.K_per_chain
    n_states = bufs.n_states
    Ntot = M * K

    # 1. Per-chain stats (GPU)
    bufs.stats_kernel(
        view(bufs.log_max, 1:M),
        view(bufs.log_z, 1:M),
        view(bufs.ess, 1:M),
        view(bufs.mu_per_chain, 1:M, :),
        view(bufs.log_w, 1:Ntot),
        view(bufs.particles_b, 1:Ntot, :),
        M, K, n_states;
        ndrange = M,
    )
    KernelAbstractions.synchronize(CUDABackend())

    # 2. Normalised weights + cumsum (GPU)
    bufs.norm_cumsum_kernel(
        view(bufs.weights, 1:Ntot),
        view(bufs.cumsum_w, 1:Ntot),
        view(bufs.log_w, 1:Ntot),
        view(bufs.log_max, 1:M),
        view(bufs.log_z, 1:M),
        M, K;
        ndrange = M,
    )
    KernelAbstractions.synchronize(CUDABackend())

    # 3. OT rescue (optional) — GPU-resident, per chain
    if use_ot
        gpu_ot_blend_chain!(view(bufs.particles_b, 1:Ntot, :),
                            view(bufs.log_w, 1:Ntot),
                            view(bufs.ess, 1:M),
                            view(bufs.anchor_idx_cpu, 1:M, :),
                            ot_max, ot_threshold, ot_temperature,
                            ot_epsilon, ot_n_iter,
                            M, K, n_states, [1, 2, 3])
    end

    # 4. Resample + Liu-West shrink (GPU, b → a)
    bufs.resample_kernel(
        view(bufs.particles_b, 1:Ntot, :),
        view(bufs.particles_a, 1:Ntot, :),
        view(bufs.cumsum_w, 1:Ntot),
        view(bufs.mu_per_chain, 1:M, :),
        a_shrink,
        view(bufs.log_w, 1:Ntot),
        K, n_states;
        ndrange = Ntot,
    )
    KernelAbstractions.synchronize(CUDABackend())
    return
end


end # module GPUSegmentedPF
