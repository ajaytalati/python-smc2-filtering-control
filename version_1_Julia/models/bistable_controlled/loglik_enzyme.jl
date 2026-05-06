# Self-contained Enzyme-clean bootstrap PF log-likelihood for bistable.
#
# Why this exists: the framework's `SMC2FC.Bootstrap.bootstrap_log_likelihood`
# uses kwargs (`@NamedTuple` construction inside the diff'd code), in-place
# `BootstrapBuffers` mutation, and runtime dispatch on `Nothing` defaults.
# Enzyme's reverse-mode trips on the kwarg NamedTuple shadow construction
# before it even gets to the PF math. Making the framework function
# Enzyme-clean is genuinely a multi-hour refactor.
#
# This module provides a model-specific PF log-likelihood that's been
# stripped of every Enzyme-hostile pattern:
#   - positional args only (no kwargs → no NamedTuple shadows)
#   - allocates fresh buffers per call (Enzyme handles this; framework's
#     in-place mutation pattern doesn't survive shadow analysis)
#   - everything inlined, no runtime dispatch
#   - Float64 throughout, no `T_u` parameterisation
#   - random noise pre-generated in a Float64 array (zero-gradient
#     `Const` from Enzyme's perspective)
#
# Used by `tools/bench_smc_closed_loop_bistable.jl` for Phase 1 inference
# only — the framework's general path stays in service everywhere else.

module LoglikEnzyme

using Random: AbstractRNG, MersenneTwister
using LogExpFunctions: logsumexp

export bistable_pf_loglik, PFBuffers

const HALF_LOG_2PI = 0.9189385332046727

"""
    bistable_pf_loglik(u, obs, x_init, u_init, T_intervention, u_on_default,
                        dt, T_steps, K, noise_x, noise_u)

Bootstrap PF log-likelihood at unconstrained-space parameters `u` for the
bistable model. Returns log p(y | θ) — does NOT include log p(u).

Args:
  u             — 8-D unconstrained params [log α, log a, log σ_x,
                  log γ, log σ_u, log σ_obs, x_0, u_0]
  obs           — (T_steps,) observation sequence
  x_init        — scalar Float64 initial x (point estimate)
  u_init        — scalar Float64 initial u
  T_intervention, u_on_default — schedule for u_target (Phase 1 = 0
                  for all t, but the function is general)
  dt            — time step
  T_steps       — number of obs steps
  K             — PF particle count
  noise_x, noise_u — pre-generated `(K, T_steps + 1)` Float64 noise grids
                  (zero gradient w.r.t. u; Enzyme `Const`)

The function is Enzyme-clean: positional args, no kwargs, no closures
over locals, no runtime-typed dispatch.

Note (engineering finding): Enzyme reverse-mode on this PF turned out
~2.7× SLOWER than ForwardDiff with 8 active partials at K=400/T=144,
because ForwardDiff with d=8 fits its partials into one AVX-256 register
and amortises the SDE-step overhead across all partials in one SIMD pass,
while Enzyme's reverse pass walks ~1k allocations per call. The JAX
advantage was XLA kernel fusion, not reverse-mode AD itself. Module
kept for reference; production B3 path uses ForwardDiff.
"""
function bistable_pf_loglik end   # forward declaration for the docstring

# Pre-allocated scratch — passing this in via the `bufs` arg avoids
# ~8 fresh `Vector{T}` allocations per timestep.
mutable struct PFBuffers{T<:Real}
    x_p::Vector{T}; u_p::Vector{T}
    log_w::Vector{T}; log_w_pre::Vector{T}
    weights::Vector{T}; cumsum_w::Vector{T}
    x_p_new::Vector{T}; u_p_new::Vector{T}
end
PFBuffers{T}(K::Integer) where {T<:Real} = PFBuffers{T}(
    Vector{T}(undef, K), Vector{T}(undef, K),
    Vector{T}(undef, K), Vector{T}(undef, K),
    Vector{T}(undef, K), Vector{T}(undef, K),
    Vector{T}(undef, K), Vector{T}(undef, K),
)

function bistable_pf_loglik(
    u::AbstractVector{T},
    obs::AbstractVector{Float64},
    x_init::Float64, u_init::Float64,
    T_intervention::Float64, u_on_default::Float64,
    dt::Float64, T_steps::Int, K::Int,
    noise_x::AbstractMatrix{Float64},
    noise_u::AbstractMatrix{Float64},
    bufs::Union{PFBuffers{T},Nothing} = nothing,
) where {T<:Real}
    # Unpack unconstrained → constrained
    α      = exp(u[1])
    a_p    = exp(u[2])
    σ_x    = exp(u[3])
    γ      = exp(u[4])
    σ_u    = exp(u[5])
    σ_obs  = exp(u[6])
    # u[7], u[8] are the init priors (NormalPrior identity); not used here
    # because we pass x_init, u_init explicitly as the PF init point.

    sx_sd   = sqrt(2.0 * σ_x)
    su_sd   = sqrt(2.0 * σ_u)
    sqrt_dt = sqrt(dt)

    # Use pre-allocated buffers if supplied (faster under Enzyme), else
    # allocate fresh.
    if bufs === nothing
        x_p = Vector{T}(undef, K)
        u_p = Vector{T}(undef, K)
        log_w = Vector{T}(undef, K)
        log_w_pre = Vector{T}(undef, K)
        weights = Vector{T}(undef, K)
        cumsum_w = Vector{T}(undef, K)
    else
        x_p = bufs.x_p; u_p = bufs.u_p
        log_w = bufs.log_w; log_w_pre = bufs.log_w_pre
        weights = bufs.weights; cumsum_w = bufs.cumsum_w
    end

    # Initial particles: tight Gaussian around (x_init, u_init).
    for i in 1:K
        x_p[i] = x_init + sx_sd * sqrt_dt * noise_x[i, 1]
        u_p[i] = u_init + su_sd * sqrt_dt * noise_u[i, 1]
        log_w[i] = zero(T)
    end

    total_ll = zero(T)

    for k in 1:T_steps
        t = (k - 1) * dt
        u_tgt = t < T_intervention ? 0.0 : u_on_default

        # Propagate + obs weight
        for i in 1:K
            dx = α * x_p[i] * (a_p * a_p - x_p[i] * x_p[i]) + u_p[i]
            du = -γ * (u_p[i] - u_tgt)
            x_p[i] = x_p[i] + dt * dx + sx_sd * sqrt_dt * noise_x[i, k + 1]
            u_p[i] = u_p[i] + dt * du + su_sd * sqrt_dt * noise_u[i, k + 1]
            ν = obs[k] - x_p[i]
            obs_lw = -0.5 * (ν / σ_obs)^2 - log(σ_obs) - HALF_LOG_2PI
            log_w_pre[i] = log_w[i] + obs_lw
        end

        # Likelihood increment
        lik_inc = logsumexp(log_w_pre) - logsumexp(log_w)
        total_ll += lik_inc

        # Systematic resample (every step, no Liu-West shrinkage — match
        # the framework's gk_dpf_v3_lite path with bandwidth_scale = 0).
        log_norm = logsumexp(log_w_pre)
        for i in 1:K
            weights[i] = exp(log_w_pre[i] - log_norm)
        end
        # Cumulative sum
        acc = 0.0
        for i in 1:K
            acc += weights[i]
            cumsum_w[i] = acc
        end
        # Systematic resample with deterministic offsets (centred). No
        # `rng` inside the diff'd code → noise contributions are Const
        # for Enzyme. The integer indices are non-differentiable; gradient
        # flows through the gathered values.
        K_f = Float64(K)
        u_shift = 0.5 / K_f
        if bufs === nothing
            x_p_new = Vector{T}(undef, K)
            u_p_new = Vector{T}(undef, K)
        else
            x_p_new = bufs.x_p_new
            u_p_new = bufs.u_p_new
        end
        for i in 1:K
            target = (i - 1) / K_f + u_shift
            idx = K
            for j in 1:K
                if cumsum_w[j] ≥ target
                    idx = j
                    break
                end
            end
            x_p_new[i] = x_p[idx]
            u_p_new[i] = u_p[idx]
        end
        for i in 1:K
            x_p[i] = x_p_new[i]
            u_p[i] = u_p_new[i]
            log_w[i] = zero(T)
        end
    end

    # Final logsumexp - log K
    total_ll += logsumexp(log_w) - log(Float64(K))
    return total_ll
end

end # module LoglikEnzyme
