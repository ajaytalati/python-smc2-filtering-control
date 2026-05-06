# SMC2/TemperedSMC.jl — outer adaptive-tempered SMC² driver.
#
# Composes:
#   Tempering.solve_delta_for_ess           (adaptive δλ via bisection)
#   MassMatrix.estimate_mass_matrix          (per-level diagonal mass)
#   HMC.hmc_step_chain                       (per-particle rejuvenation)
#   Bridge.bridge_init                       (warm-start across windows)
#
# Port of `smc2fc/core/jax_native_smc.py:run_smc_window_native` and
# `run_smc_window_bridge_native`. The Python pipeline is wrapped in a
# `jax.jit` + `lax.while_loop` so the tempering decisions stay on-device;
# Julia uses an ordinary `while` loop and per-particle parallelism via
# `Threads.@threads` per charter §15.1.

module TemperedSMC

using Random: AbstractRNG, default_rng
using LogExpFunctions: logsumexp
using Statistics: mean, std
using ..Tempering: solve_delta_for_ess
using ..MassMatrix: estimate_mass_matrix
using ..HMC: hmc_step_chain
using ..Bridge: bridge_init, bridge_kind
using ..Sampling: sample_from_prior
using ...SMC2FC: SMCConfig, PriorType
using ...SMC2FC: log_prior_unconstrained

export run_smc_window, run_smc_window_bridge, TemperedSMCResult

# ── Result struct ────────────────────────────────────────────────────────────

struct TemperedSMCResult
    particles::Matrix{Float64}    # (n_smc, d)
    n_temp::Int                   # number of tempering steps taken
    elapsed::Float64              # wall time (s)
end


# ── Systematic resampling (CPU; tempered SMC stays on CPU per charter §14) ──

function systematic_resample_indices(rng::AbstractRNG,
                                      weights::AbstractVector{Float64},
                                      n::Integer)
    indices = Vector{Int}(undef, n)
    cumsum_w = cumsum(weights)
    u_shift  = rand(rng) / n
    @inbounds for i in 1:n
        target = (i - 1) / n + u_shift
        idx = searchsortedfirst(cumsum_w, target)
        indices[i] = clamp(idx, 1, n)
    end
    return indices
end


# ── One tempering step ───────────────────────────────────────────────────────

function _tempered_step!(particles::Matrix{Float64},
                          λ::Float64,
                          loglikelihood_fn,
                          logprior_fn,
                          cfg::SMCConfig,
                          max_λ_inc::Float64,
                          rng::AbstractRNG)
    n_smc = size(particles, 1)

    # 1) Per-particle log-likelihood (the only inner-loop cost — could be
    #    parallelised across outer particles via @threads).
    ll = Vector{Float64}(undef, n_smc)
    Threads.@threads for m in 1:n_smc
        ll[m] = loglikelihood_fn(@view particles[m, :])
    end

    # 2) Adaptive δλ via bisection (Tempering.solve_delta_for_ess)
    δλ_max  = min(1.0 - λ, max_λ_inc)
    δ       = solve_delta_for_ess(ll, cfg.target_ess_frac, δλ_max)
    next_λ  = λ + δ < 1.0 - 1e-6 ? λ + δ : 1.0
    Δλ      = next_λ - λ

    # 3) Reweight + systematic resample
    log_w   = Δλ .* ll
    log_wn  = log_w .- logsumexp(log_w)
    w       = exp.(log_wn)
    indices = systematic_resample_indices(rng, w, n_smc)
    resampled = particles[indices, :]

    # 4) Mass matrix (diag) from resampled cloud
    inv_mass = estimate_mass_matrix(resampled)

    # 5) HMC moves at temperature next_λ
    function tempered_lp(u)
        return logprior_fn(u) + next_λ * loglikelihood_fn(u)
    end

    # Pre-generate per-thread seeds OUTSIDE the @threads loop. Julia's
    # MersenneTwister is NOT thread-safe — calling `rand(parent_rng, ...)`
    # from multiple threads concurrently corrupts its internal state and
    # triggers `AssertionError: length(ints) == 501`. By drawing the
    # `n_smc` integer seeds serially before the parallel loop, each
    # worker thread gets its own independent `MersenneTwister(seed[m])`.
    seeds_per_particle = Vector{UInt64}(undef, n_smc)
    @inbounds for m in 1:n_smc
        seeds_per_particle[m] = abs(rand(rng, Int) % typemax(Int)) ⊻
                                  (UInt64(m) * 0x9E3779B97F4A7C15)
    end

    new_particles = Matrix{Float64}(undef, n_smc, size(particles, 2))
    Threads.@threads for m in 1:n_smc
        local_rng = Random.MersenneTwister(seeds_per_particle[m])
        new_particles[m, :] = hmc_step_chain(
            collect(@view resampled[m, :]),
            tempered_lp,
            cfg.num_mcmc_steps,
            cfg.hmc_step_size,
            inv_mass,
            cfg.hmc_num_leapfrog,
            local_rng;
            ad_backend = cfg.ad_backend,
            sampler    = cfg.sampler,
        )
    end
    copyto!(particles, new_particles)

    return next_λ
end

using Random


# ── Cold-start: prior → posterior via adaptive tempering ─────────────────────

"""
    run_smc_window(loglikelihood_fn, priors, cfg, rng) -> TemperedSMCResult

Cold-start outer SMC². Initial cloud sampled from the unconstrained-space
prior; tempered until λ → 1 with adaptive δλ. Each tempering level resamples,
re-estimates a diagonal mass, and runs `cfg.num_mcmc_steps` HMC moves per
particle.

This is the Julia analogue of `run_smc_window_native(..., bridge=False)`
in the Python framework. The hybrid CPU/GPU contract per charter §14 sits
*inside* `loglikelihood_fn`: the closure transfers θ to the GPU, runs the
inner PF (Phase 2), and returns a scalar back to CPU.
"""
function run_smc_window(loglikelihood_fn,
                         priors::Vector{<:PriorType},
                         cfg::SMCConfig,
                         rng::AbstractRNG)
    t0 = time()
    n_smc = cfg.n_smc_particles

    particles = sample_from_prior(n_smc, priors, rng)
    λ         = 0.0
    n_temp    = 0

    logprior_fn(u) = log_prior_unconstrained(u, priors)

    while λ < 1.0 - 1e-6
        λ = _tempered_step!(particles, λ, loglikelihood_fn, logprior_fn,
                             cfg, cfg.max_lambda_inc, rng)
        n_temp += 1
        # Safety cap so a degenerate likelihood landscape can't loop forever.
        n_temp > 200 && break
    end

    return TemperedSMCResult(particles, n_temp, time() - t0)
end


# ── Warm-start: previous posterior → next-window posterior ───────────────────

"""
    run_smc_window_bridge(loglikelihood_fn, prev_posterior, priors, cfg, rng)

Warm-start variant: initial cloud comes from `bridge_init` (Gaussian by
default; SchrodingerFollmerBridge stub falls back to Gaussian). The
tempering loop uses `cfg.max_lambda_inc_bridge` and `cfg.num_mcmc_steps_bridge`
which are typically larger / smaller than the cold-start values
respectively (see Python config docstring lines 16-30).
"""
function run_smc_window_bridge(loglikelihood_fn,
                                 prev_posterior::AbstractMatrix{Float64},
                                 priors::Vector{<:PriorType},
                                 cfg::SMCConfig,
                                 rng::AbstractRNG)
    t0 = time()
    n_smc = cfg.n_smc_particles

    particles = bridge_init(bridge_kind(cfg), prev_posterior, loglikelihood_fn, cfg, rng;
                             n_smc = n_smc)
    λ         = 0.0
    n_temp    = 0

    logprior_fn(u) = log_prior_unconstrained(u, priors)

    while λ < 1.0 - 1e-6
        λ = _tempered_step!(particles, λ, loglikelihood_fn, logprior_fn,
                             cfg, cfg.max_lambda_inc_bridge, rng)
        n_temp += 1
        n_temp > 200 && break
    end

    return TemperedSMCResult(particles, n_temp, time() - t0)
end

end # module TemperedSMC
