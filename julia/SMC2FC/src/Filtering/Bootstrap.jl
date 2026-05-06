# Filtering/Bootstrap.jl — inner bootstrap particle filter producing the
# scalar log-likelihood L̂_N(θ_dyn) consumed by the outer SMC² in Phase 3.
#
# Port of the production path in `smc2fc/filtering/gk_dpf_v3_lite.py`
# (the `compileonce` factory). The Python file is 700 LOC; most of that is
# JAX scaffolding (jax.checkpoint, lax.scan, fp32 staging via tree_map, the
# closure-over-data recompile dance from charter §11.3). The Julia version
# is a plain `for` loop with pre-allocated buffers — no scaffolding survives
# the translation. Charter §13: "~150 LOC of Julia."
#
# Hot path discipline (charter §15.1):
#   - Pre-allocate `particles`, `log_w`, `noise`, etc. ONCE before the loop.
#   - Inner step uses in-place updates (`particles_next .= ...`).
#   - `@inbounds @simd` for the per-particle propagation.
#   - AbstractArray{T,N} signatures so Array (CPU) and CuArray (GPU) share
#     the same compiled code.

module Bootstrap

using Random: AbstractRNG, default_rng
using LogExpFunctions: logsumexp
using LinearAlgebra: dot
using ..Kernels: compute_ess
using ..OT: ot_resample_lr, ot_blended_resample
using ...SMC2FC: EstimationModel, n_params, SMCConfig, PriorType
using ...SMC2FC: unconstrained_to_constrained, log_prior_unconstrained, all_priors

export bootstrap_log_likelihood, BootstrapBuffers

# ── Pre-allocated working buffers ────────────────────────────────────────────
# One `BootstrapBuffers{T,A}` is allocated per outer-particle thread; it stays
# resident for the duration of an SMC² run. Charter §15.1 zero-allocation rule.

mutable struct BootstrapBuffers{T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    K::Int
    n_states::Int
    particles::M          # (K, n_states)
    new_particles::M      # (K, n_states)
    resampled::M          # (K, n_states)
    sys_lw::M             # (K, n_states)  Liu-West-corrected
    log_w::V              # (K,)
    log_w_pre::V          # (K,)
    noise::M              # (K, n_states)
    weights::V            # (K,)
    cumsum_w::V           # (K,)
    indices::Vector{Int}  # (K,)
    sys_offsets::V        # (K,)  precomputed (i-1)/K for systematic resampling
end

"""
    BootstrapBuffers{T}(K::Integer, n_states::Integer; backend=Array)

Allocate scratch space for one bootstrap-PF instance. `backend` is the
array constructor — `Array` for CPU, `CUDA.CuArray` for GPU. Same struct
shape on either side.
"""
function BootstrapBuffers{T}(K::Integer, n_states::Integer;
                              backend = Array) where {T<:Real}
    sys_offsets = backend{T}(undef, K)
    @inbounds for i in 1:K
        sys_offsets[i] = (i - 1) / T(K)
    end
    M = typeof(backend{T}(undef, K, n_states))
    V = typeof(sys_offsets)
    return BootstrapBuffers{T,M,V}(
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


# ── Helpers ──────────────────────────────────────────────────────────────────

"""
Apply per-component state bounds (from `model.state_bounds`) in place.
"""
function clip_to_bounds!(particles::AbstractMatrix{T},
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

"""
Systematic-resample indices. Single uniform shift `u ∼ U(0, 1/K)`; offsets
`(i-1)/K + u` mapped through the cumulative weight distribution.
"""
function systematic_indices!(indices::Vector{Int},
                              cumsum_w::AbstractVector{T},
                              sys_offsets::AbstractVector{T},
                              rng::AbstractRNG) where {T<:Real}
    K = length(indices)
    u_shift = T(rand(rng)) / T(K)
    @inbounds for i in 1:K
        target = sys_offsets[i] + u_shift
        # binary search; cumsum_w is monotonically non-decreasing in [0, 1].
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


# ── Liu-West shrinkage (one-shot, no kernel blend) ───────────────────────────
# Mirrors gk_dpf_v3_lite.py:181-191. Resampled particles are shrunk toward
# the weighted mean: `corrected = a * resampled + (1-a) * μ_w`, where
# `a = sqrt(1 - h_norm²)` and `h_norm` is the ESS-scaled Silverman factor.

function liu_west_shrink!(out::AbstractMatrix{T},
                           resampled::AbstractMatrix{T},
                           new_particles::AbstractMatrix{T},
                           weights::AbstractVector{T},
                           bandwidth_scale::Real,
                           ess_factor::Real,
                           K::Integer,
                           n_st::Integer) where {T<:Real}
    silverman_factor = (4.0 / (n_st + 2.0)) ^ (1.0 / (n_st + 4.0))
    k_factor         = float(K) ^ (-1.0 / (n_st + 4.0))
    h_norm           = silverman_factor * k_factor * bandwidth_scale * ess_factor
    a = sqrt(clamp(1.0 - h_norm^2, 0.0, 1.0))

    n_s = size(out, 2)
    @inbounds for d in 1:n_s
        μ = zero(T)
        for i in 1:K
            μ += weights[i] * new_particles[i, d]
        end
        for i in 1:K
            out[i, d] = T(a) * resampled[i, d] + T(1 - a) * μ
        end
    end
    return out
end


# ── Main entry point ─────────────────────────────────────────────────────────

"""
    bootstrap_log_likelihood(model, u, grid_obs, fixed_init_state, priors,
                              cfg, rng;
                              dt, t_steps, window_start_bin=0,
                              buffers=nothing)

Inner bootstrap particle filter. Returns
    log p(y_{1:T} | θ) + log p(θ)
on the unconstrained scale `u`, summed over the rolling window. This is the
log-density consumed by the outer SMC²/HMC.

Arguments mirror the Python `make_gk_dpf_v3_lite_log_density(...)(u)`:
  - `model::EstimationModel` provides `propagate_fn`, `obs_log_weight_fn`,
    `diffusion_fn`, `shard_init_fn`, `state_bounds`, `stochastic_indices`.
  - `u`: unconstrained parameter vector.
  - `grid_obs`: dict of grid-aligned observations + exogenous channels.
  - `fixed_init_state`: externally-supplied initial latent state.
  - `priors`: same length as u.
  - `cfg::SMCConfig`: inherits `n_pf_particles`, `bandwidth_scale`, OT params.
  - `rng`: PRNG; consumed deterministically.
  - `dt`, `t_steps`, `window_start_bin`: window framing (see Python:340-377).
  - `buffers`: optional pre-allocated `BootstrapBuffers`; if absent, allocated
    here. Reuse across calls is ~50 % faster on small windows (charter §15.1).

This function is the GPU-resident inner loop in the hybrid §14 pipeline. It
runs entirely on whichever backend `buffers.particles` lives on (Array vs
CuArray); the outer SMC² consumer transfers only the scalar return value
across the PCIe bus.
"""
function bootstrap_log_likelihood(model::EstimationModel,
                                   u::AbstractVector{T},
                                   grid_obs::Dict,
                                   fixed_init_state::AbstractVector{T},
                                   priors::Vector{<:PriorType},
                                   cfg::SMCConfig,
                                   rng::AbstractRNG;
                                   dt::Real,
                                   t_steps::Integer,
                                   window_start_bin::Integer = 0,
                                   buffers = nothing) where {T<:Real}

    K        = cfg.n_pf_particles
    n_s      = model.n_states
    bw_scale = cfg.bandwidth_scale
    sto_idx  = model.stochastic_indices
    n_st     = length(sto_idx)
    sqrt_dt  = sqrt(T(dt))

    bufs = buffers === nothing ?
        BootstrapBuffers{T}(K, n_s) :
        buffers::BootstrapBuffers{T}

    # ── Setup: constrained params, σ_diag, init particles ────────────────────
    θ        = unconstrained_to_constrained(u, priors)
    params   = @view θ[1:n_params(model)]
    σ_diag   = model.diffusion_fn(params)
    exog     = Dict(k => grid_obs[k] for k in model.exogenous_keys)
    base     = model.shard_init_fn(window_start_bin, params, exog, fixed_init_state)

    @inbounds for i in 1:K, d in 1:n_s
        bufs.particles[i, d] = base[d] + σ_diag[d] * sqrt_dt * randn(rng, T)
    end
    clip_to_bounds!(bufs.particles, model.state_bounds)
    fill!(bufs.log_w, zero(T))

    # OT-rescue thresholds
    ot_threshold = T(K * cfg.ot_ess_frac)
    ot_temp      = T(cfg.ot_temperature)
    ot_max       = T(cfg.ot_max_weight)
    ot_active    = ot_max >= 1e-6

    total_ll = zero(T)

    # ── Per-step PF scan ─────────────────────────────────────────────────────
    for k in 1:t_steps
        t_global = T((window_start_bin + k - 1) * dt)

        # propagate + log-weight update
        @inbounds for d in 1:n_s, i in 1:K
            bufs.noise[i, d] = randn(rng, T)
        end

        @inbounds for i in 1:K
            y_old = @view bufs.particles[i, :]
            ξ     = @view bufs.noise[i, :]
            x_new, pred_lw = model.propagate_fn(y_old, t_global, T(dt), params,
                                                 grid_obs, k, σ_diag, ξ, rng)
            obs_lw = model.obs_log_weight_fn(x_new, grid_obs, k, params)
            for d in 1:n_s
                bufs.new_particles[i, d] = x_new[d]
            end
            bufs.log_w_pre[i] = bufs.log_w[i] + pred_lw + obs_lw
        end

        # marginal-likelihood increment
        lik_inc = logsumexp(bufs.log_w_pre) - logsumexp(bufs.log_w)
        total_ll += lik_inc

        # Skip resampling if the step is unobserved (observation-conditional)
        has_obs = haskey(grid_obs, :has_any_obs) ? grid_obs[:has_any_obs][k] : 1.0
        do_resample = has_obs > 0.5

        if do_resample
            # Normalised weights
            log_norm = logsumexp(bufs.log_w_pre)
            @inbounds for i in 1:K
                bufs.weights[i] = exp(bufs.log_w_pre[i] - log_norm)
            end

            # cumsum (in-place)
            acc = zero(T)
            @inbounds for i in 1:K
                acc += bufs.weights[i]
                bufs.cumsum_w[i] = acc
            end

            # Systematic resample → indices → resampled cloud
            systematic_indices!(bufs.indices, bufs.cumsum_w, bufs.sys_offsets, rng)
            @inbounds for d in 1:n_s, i in 1:K
                bufs.resampled[i, d] = bufs.new_particles[bufs.indices[i], d]
            end

            # Liu-West shrinkage with ESS-scaled bandwidth
            ess_frac   = clamp(compute_ess(bufs.log_w_pre) / T(K), 0.0, 1.0)
            ess_factor = (1.0 - ess_frac) ^ 2
            liu_west_shrink!(bufs.sys_lw, bufs.resampled, bufs.new_particles,
                              bufs.weights, bw_scale, ess_factor, K, n_st)

            # Optional OT rescue (sigmoid blend)
            if ot_active
                ot_raw = ot_resample_lr(bufs.new_particles, bufs.log_w_pre, rng,
                                         collect(sto_idx);
                                         ε = cfg.ot_epsilon,
                                         n_iter = cfg.ot_n_iter,
                                         rank = cfg.ot_rank)
                clip_to_bounds!(ot_raw, model.state_bounds)
                blended = ot_blended_resample(bufs.sys_lw, ot_raw, bufs.log_w_pre;
                                               ot_max_weight = ot_max,
                                               ot_threshold  = ot_threshold,
                                               ot_temperature = ot_temp)
                copyto!(bufs.particles, blended)
            else
                copyto!(bufs.particles, bufs.sys_lw)
            end
            fill!(bufs.log_w, zero(T))
        else
            copyto!(bufs.particles, bufs.new_particles)
            copyto!(bufs.log_w, bufs.log_w_pre)
        end

        clip_to_bounds!(bufs.particles, model.state_bounds)
    end

    # Trailing logsumexp - log K (Python:231-232)
    total_ll += logsumexp(bufs.log_w) - log(T(K))

    # Add log-prior on u (Python:232)
    lp = log_prior_unconstrained(u, priors)
    return total_ll + lp
end

end # module Bootstrap
