"""
    Filtering/Bootstrap.jl

Inner bootstrap particle filter producing the scalar log-likelihood
`L̂_N(θ_dyn)` consumed by the outer SMC². Port of the production path in
`smc2fc/filtering/gk_dpf_v3_lite.py` (the `compileonce` factory).

Functional-API contract:

- The public entry point `bootstrap_log_likelihood` takes immutable
    inputs and returns a scalar. No `!` is exposed in the public API.
- The pre-allocated buffer container `BootstrapWorkspace` is *itself*
    immutable — its inner array fields are mutable and may be written
    into during the call, but the workspace identity is preserved.
- The internal helpers `clip_to_bounds!`, `systematic_indices!`, and
    `liu_west_shrink!` retain their `!` suffix (they mutate the array
    arguments) but are module-private.

The mutable-struct `BootstrapBuffers` of the original port is replaced by
the immutable `BootstrapWorkspace` here — same memory layout, same
performance, cleaner semantics.
"""
module Bootstrap

using Random
using Random: AbstractRNG, default_rng, randn!
using LogExpFunctions: logsumexp
using LinearAlgebra: dot
using ..Kernels: compute_ess
using ..OT: ot_resample_lr, ot_blended_resample
using ...SMC2FC_functional: EstimationModel, n_params, SMCConfig, PriorType
using ...SMC2FC_functional: unconstrained_to_constrained,
                            log_prior_unconstrained, all_priors

export bootstrap_log_likelihood, BootstrapWorkspace

# ── Pre-allocated working buffers ───────────────────────────────────────────

"""
    BootstrapWorkspace{T,M,V}

Immutable container of pre-allocated mutable arrays for one
bootstrap-PF instance. One workspace per outer-particle thread; reused
for the duration of an SMC² run.

# Fields
- `K::Int`, `n_states::Int`: shape constants.
- `particles, new_particles, resampled, sys_lw, noise::M`: `(K, n_states)`
    state buffers.
- `log_w, log_w_pre, weights, cumsum_w, sys_offsets::V`: `(K,)` weight
    buffers.
- `indices::Vector{Int}`: `(K,)` resample indices (always plain `Vector`).

# Notes
- The struct itself is **immutable** — once constructed, you cannot
    rebind any field. The inner arrays remain mutable and are written
    into during `bootstrap_log_likelihood`. This is the canonical Julia
    "mutable inside, immutable outside" pattern.
- Use `BootstrapWorkspace{T}(K, n_states; backend=Array)` to construct.
"""
struct BootstrapWorkspace{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    K::Int
    n_states::Int
    particles::M
    new_particles::M
    resampled::M
    sys_lw::M
    log_w::V
    log_w_pre::V
    noise::M
    weights::V
    cumsum_w::V
    indices::Vector{Int}
    sys_offsets::V
end

"""
    BootstrapWorkspace{T}(K, n_states; backend=Array) -> BootstrapWorkspace

Allocate scratch space for one bootstrap-PF instance.

# Arguments
- `K::Integer`: particle count.
- `n_states::Integer`: state-space dimension.

# Keyword arguments
- `backend = Array`: array constructor — `Array` for CPU,
    `CUDA.CuArray` for GPU. Same struct shape on either side.

# Returns
- `BootstrapWorkspace{T,M,V}`: immutable container with inner arrays
    allocated on the chosen backend.
"""
function BootstrapWorkspace{T}(K::Integer, n_states::Integer;
                                backend = Array) where {T<:Real}
    sys_offsets_cpu = T[(i - 1) / T(K) for i in 1:K]
    sys_offsets     = backend{T}(undef, K)
    copyto!(sys_offsets, sys_offsets_cpu)
    M = typeof(backend{T}(undef, K, n_states))
    V = typeof(sys_offsets)
    return BootstrapWorkspace{T,M,V}(
        K, n_states,
        backend{T}(undef, K, n_states),
        backend{T}(undef, K, n_states),
        backend{T}(undef, K, n_states),
        backend{T}(undef, K, n_states),
        backend{T}(undef, K),
        backend{T}(undef, K),
        backend{T}(undef, K, n_states),
        backend{T}(undef, K),
        backend{T}(undef, K),
        Vector{Int}(undef, K),
        sys_offsets,
    )
end


# ── Internal mutating helpers (module-private) ─────────────────────────────

"""
    clip_to_bounds!(particles, state_bounds) -> particles

Apply per-component state bounds in place. Two methods: a plain-`Matrix`
fast path (scalar loop, ForwardDiff-friendly) and an `AbstractMatrix`
broadcast path that runs on `CuArray`.

# Arguments
- `particles`: matrix to clamp in place.
- `state_bounds::Vector{Tuple{Float64,Float64}}`: per-component `(lo, hi)`.

# Returns
- `particles` (same object, after mutation).
"""
function clip_to_bounds!(particles::Matrix{T},
                         state_bounds::Vector{Tuple{Float64,Float64}}) where {T<:Real}
    K_, n_s = size(particles)
    @inbounds for d in 1:n_s
        lo, hi = state_bounds[d]
        for i in 1:K_
            particles[i, d] = clamp(particles[i, d], T(lo), T(hi))
        end
    end
    return particles
end

function clip_to_bounds!(particles::AbstractMatrix{T},
                         state_bounds::Vector{Tuple{Float64,Float64}}) where {T<:Real}
    n_s = size(particles, 2)
    lo = T[T(state_bounds[d][1]) for d in 1:n_s]
    hi = T[T(state_bounds[d][2]) for d in 1:n_s]
    lo_dev = similar(particles, T, n_s); copyto!(lo_dev, lo)
    hi_dev = similar(particles, T, n_s); copyto!(hi_dev, hi)
    particles .= clamp.(particles, reshape(lo_dev, 1, :), reshape(hi_dev, 1, :))
    return particles
end

"""
    systematic_indices!(indices, cumsum_w, sys_offsets, rng) -> indices

Systematic-resample indices. Single uniform shift `u ∼ U(0, 1/K)`;
offsets `(i−1)/K + u` mapped through the cumulative weight distribution
via binary search.

# Arguments
- `indices::Vector{Int}`: output, mutated in place.
- `cumsum_w::AbstractVector{T}`: monotone non-decreasing CDF in `[0, 1]`.
- `sys_offsets::AbstractVector{T}`: pre-built offsets `(i−1)/K`.
- `rng::AbstractRNG`: PRNG.

# Returns
- `indices` (same object).
"""
function systematic_indices!(indices::Vector{Int},
                              cumsum_w::AbstractVector{T},
                              sys_offsets::AbstractVector{T},
                              rng::AbstractRNG) where {T<:Real}
    K = length(indices)
    u_shift = T(rand(rng)) / T(K)
    @inbounds for i in 1:K
        target = sys_offsets[i] + u_shift
        lo, hi = 1, K
        while lo < hi
            mid = (lo + hi) >>> 1
            if cumsum_w[mid] < target
                lo = mid + 1
            else
                hi = mid
            end
        end
        indices[i] = clamp(lo, 1, K)
    end
    return indices
end


# ── Main entry point ───────────────────────────────────────────────────────

"""
    bootstrap_log_likelihood(model, u, grid_obs, fixed_init_state, priors,
                              cfg, rng;
                              dt, t_steps, window_start_bin=0,
                              workspace=nothing) -> Real

Inner bootstrap particle filter. Returns
`log p(y_{1:T} | θ) + log p(θ)` on the unconstrained scale `u`, summed
over the rolling window. This is the log-density consumed by the outer
SMC² / HMC.

# Arguments
- `model::EstimationModel`: provides `propagate_fn`, `obs_log_weight_fn`,
    `diffusion_fn`, `shard_init_fn`, `state_bounds`, `stochastic_indices`.
- `u::AbstractVector{T}`: unconstrained parameter vector. `T` may be
    `Float64` (production) or `ForwardDiff.Dual{...}` (under AD).
- `grid_obs::Dict`: grid-aligned observations + exogenous channels.
- `fixed_init_state::AbstractVector{<:Real}`: externally-supplied initial
    latent state.
- `priors::Vector{<:PriorType}`: same length as `u`.
- `cfg::SMCConfig`: inherits `n_pf_particles`, `bandwidth_scale`, OT params.
- `rng::AbstractRNG`: PRNG; consumed deterministically.

# Keyword arguments
- `dt::Real`: integration step.
- `t_steps::Integer`: window length in bins.
- `window_start_bin::Integer = 0`: window framing offset.
- `workspace = nothing`: optional pre-allocated `BootstrapWorkspace`. If
    `nothing`, a fresh one is allocated. Reuse across calls is ~50% faster
    on small windows.

# Returns
- `Real`: log-density at `u` — `total_ll + log_prior(u)`. Element type
    matches `eltype(u)` so ForwardDiff `Dual`s flow through.

# Notes
- The function is the GPU-resident inner loop in the hybrid pipeline. It
    runs entirely on whichever backend the workspace's arrays live on
    (`Array` vs `CuArray`). The outer SMC² consumer transfers only the
    scalar return value across the PCIe bus.
- For AD: `T` is the element type of `u` — either `Float64` or a Dual
    type. `fixed_init_state` stays plain `Real`; assignment into the
    `T`-typed buffer auto-promotes via `convert(T, x)`. Random noise has
    zero gradient w.r.t. `u` by construction; `Float64` samples
    auto-promote to `T`.
"""
function bootstrap_log_likelihood(model::EstimationModel,
                                   u::AbstractVector{T},
                                   grid_obs::Dict,
                                   fixed_init_state::AbstractVector{<:Real},
                                   priors::Vector{<:PriorType},
                                   cfg::SMCConfig,
                                   rng::AbstractRNG;
                                   dt::Real,
                                   t_steps::Integer,
                                   window_start_bin::Integer = 0,
                                   workspace = nothing) where {T<:Real}

    K        = cfg.n_pf_particles
    n_s      = model.n_states
    bw_scale = cfg.bandwidth_scale
    sto_idx  = model.stochastic_indices
    n_st     = length(sto_idx)
    sqrt_dt  = sqrt(T(dt))

    ws = workspace === nothing ?
        BootstrapWorkspace{T}(K, n_s) :
        workspace::BootstrapWorkspace{T}

    # ── Setup: constrained params, σ_diag, init particles ────────────────────
    θ        = unconstrained_to_constrained(u, priors)
    params   = @view θ[1:n_params(model)]
    σ_diag   = model.diffusion_fn(params)
    exog     = Dict(k => grid_obs[k] for k in model.exogenous_keys)
    base     = model.shard_init_fn(window_start_bin, params, exog, fixed_init_state)

    use_batch = (model.propagate_batch_fn !== nothing) &&
                (model.obs_log_weight_batch_fn !== nothing)
    if use_batch && !(ws.particles isa Matrix)
        Random.randn!(rng, ws.noise)
        base_dev    = similar(ws.particles, T, n_s); copyto!(base_dev, T.(collect(base)))
        sigma_dev   = similar(ws.particles, T, n_s); copyto!(sigma_dev, T.(collect(σ_diag)))
        ws.particles .= reshape(base_dev, 1, :) .+
                         reshape(sigma_dev, 1, :) .* sqrt_dt .* ws.noise
    else
        @inbounds for i in 1:K, d in 1:n_s
            ws.particles[i, d] = base[d] + σ_diag[d] * sqrt_dt * randn(rng, Float64)
        end
    end
    clip_to_bounds!(ws.particles, model.state_bounds)
    fill!(ws.log_w, zero(T))

    ot_threshold = T(K * cfg.ot_ess_frac)
    ot_temp      = T(cfg.ot_temperature)
    ot_max       = T(cfg.ot_max_weight)
    ot_active    = ot_max >= 1e-6

    total_ll = zero(T)

    # ── Per-step PF scan ─────────────────────────────────────────────────────
    for k in 1:t_steps
        t_global = T((window_start_bin + k - 1) * dt)

        if use_batch
            Random.randn!(rng, ws.noise)
            new_parts, pred_lw = model.propagate_batch_fn(
                ws.particles, t_global, T(dt), params, grid_obs, k, σ_diag,
                ws.noise, rng,
            )
            copyto!(ws.new_particles, new_parts)
            obs_lw = model.obs_log_weight_batch_fn(ws.new_particles, grid_obs, k, params)
            ws.log_w_pre .= ws.log_w .+ pred_lw .+ obs_lw
        else
            @inbounds for d in 1:n_s, i in 1:K
                ws.noise[i, d] = randn(rng, Float64)
            end
            @inbounds for i in 1:K
                y_old = @view ws.particles[i, :]
                ξ     = @view ws.noise[i, :]
                x_new, pred_lw = model.propagate_fn(y_old, t_global, T(dt), params,
                                                     grid_obs, k, σ_diag, ξ, rng)
                obs_lw = model.obs_log_weight_fn(x_new, grid_obs, k, params)
                for d in 1:n_s
                    ws.new_particles[i, d] = x_new[d]
                end
                ws.log_w_pre[i] = ws.log_w[i] + pred_lw + obs_lw
            end
        end

        lik_inc = logsumexp(ws.log_w_pre) - logsumexp(ws.log_w)
        total_ll += lik_inc

        has_obs = haskey(grid_obs, :has_any_obs) ? grid_obs[:has_any_obs][k] : 1.0
        do_resample = has_obs > 0.5

        if do_resample
            log_norm     = logsumexp(ws.log_w_pre)
            ws.weights .= exp.(ws.log_w_pre .- log_norm)

            cumsum!(ws.cumsum_w, ws.weights)

            if ws.cumsum_w isa Matrix || ws.cumsum_w isa Vector
                systematic_indices!(ws.indices, ws.cumsum_w, ws.sys_offsets, rng)
            else
                cumsum_cpu  = Array(ws.cumsum_w)
                offsets_cpu = Array(ws.sys_offsets)
                systematic_indices!(ws.indices, cumsum_cpu, offsets_cpu, rng)
            end

            ws.resampled .= ws.new_particles[ws.indices, :]

            ess_frac    = clamp(compute_ess(ws.log_w_pre) / T(K), 0.0, 1.0)
            ess_factor  = (1.0 - ess_frac) ^ 2
            silverman_factor = (4.0 / (n_st + 2.0)) ^ (1.0 / (n_st + 4.0))
            k_factor         = float(K) ^ (-1.0 / (n_st + 4.0))
            h_norm           = silverman_factor * k_factor * bw_scale * ess_factor
            a = T(sqrt(clamp(1.0 - h_norm^2, 0.0, 1.0)))
            μ_w = reshape(ws.weights, 1, :) * ws.new_particles
            ws.sys_lw .= a .* ws.resampled .+ (T(1) - a) .* μ_w

            if ot_active
                ot_raw = ot_resample_lr(ws.new_particles, ws.log_w_pre, rng,
                                         collect(sto_idx);
                                         ε = cfg.ot_epsilon,
                                         n_iter = cfg.ot_n_iter,
                                         rank = cfg.ot_rank)
                clip_to_bounds!(ot_raw, model.state_bounds)
                blended = ot_blended_resample(ws.sys_lw, ot_raw, ws.log_w_pre;
                                               ot_max_weight = ot_max,
                                               ot_threshold  = ot_threshold,
                                               ot_temperature = ot_temp)
                copyto!(ws.particles, blended)
            else
                copyto!(ws.particles, ws.sys_lw)
            end
            fill!(ws.log_w, zero(T))
        else
            copyto!(ws.particles, ws.new_particles)
            copyto!(ws.log_w, ws.log_w_pre)
        end

        clip_to_bounds!(ws.particles, model.state_bounds)
    end

    total_ll += logsumexp(ws.log_w) - log(T(K))

    lp = log_prior_unconstrained(u, priors)
    return total_ll + lp
end

end # module Bootstrap
