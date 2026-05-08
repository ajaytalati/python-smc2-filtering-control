"""
    SMC2/TemperedSMC.jl

Outer adaptive-tempered SMC² driver. Composes:

- `Tempering.solve_delta_for_ess` — adaptive δλ via bisection.
- `MassMatrix.estimate_mass_matrix` — per-level diagonal mass matrix.
- `HMC.hmc_step_chain` — per-particle rejuvenation.
- `Bridge.bridge_init` — warm-start across windows.

Port of `smc2fc/core/jax_native_smc.py:run_smc_window_native` and
`run_smc_window_bridge_native`. The Python pipeline wraps the loop in
`jax.jit` + `lax.while_loop` so tempering decisions stay on-device; here
we use a plain `while` loop with `Threads.@threads` parallelism.

Functional API contract:

- `run_smc_window` and `run_smc_window_bridge` take immutable inputs
    (priors list, config, RNG) and return a fresh `TemperedSMCResult`.
- The internal `_tempered_step!` is module-private and mutates a
    `particles::Matrix` buffer for performance, but this is never visible
    at the public API.
"""
module TemperedSMC

using Random
using Random: AbstractRNG, default_rng
using LogExpFunctions: logsumexp
using Statistics: mean, std
using ..Tempering: solve_delta_for_ess
using ..MassMatrix: estimate_mass_matrix
using ..HMC: hmc_step_chain, chees_adapt_L
using ..Bridge: bridge_init, bridge_kind
using ..Sampling: sample_from_prior
using ...SMC2FC_functional: SMCConfig, PriorType
using ...SMC2FC_functional: log_prior_unconstrained

export run_smc_window, run_smc_window_bridge, TemperedSMCResult

# ── Result struct ───────────────────────────────────────────────────────────

"""
    TemperedSMCResult

Result of one rolling-window outer SMC² run.

# Fields
- `particles::Matrix{Float64}`: `(n_smc, d_theta)` posterior cloud.
- `n_temp::Int`: number of tempering levels traversed.
- `elapsed::Float64`: wall-clock time in seconds.
"""
struct TemperedSMCResult
    particles::Matrix{Float64}
    n_temp::Int
    elapsed::Float64
end


# ── Systematic resample ─────────────────────────────────────────────────────

"""
    systematic_resample_indices(rng, weights, n) -> Vector{Int}

Systematic resampling indices on the CPU. Returns `n` indices into a
particle cloud whose row weights are `weights` (assumed to sum to 1).

# Arguments
- `rng::AbstractRNG`: PRNG.
- `weights::AbstractVector{Float64}`: normalised weights.
- `n::Integer`: output length.

# Returns
- `Vector{Int}`: resample indices in `[1, n]`.
"""
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


# ── One tempering step (module-private, mutates particles) ──────────────────

"""
    _tempered_step!(particles, λ, loglik_fn, logprior_fn, cfg, max_λ_inc, rng) -> Float64

Run one tempering level. Mutates `particles` in place. Module-private —
not exported.

# Returns
- `Float64`: the new tempering level `next_λ ∈ (λ, 1.0]`.

# Notes
- Uses `Threads.@threads` for the per-particle log-likelihood and HMC
    moves. Per-thread `MersenneTwister`s are seeded outside the parallel
    loop because the parent `MersenneTwister` is not thread-safe.
"""
function _tempered_step!(particles::Matrix{Float64},
                          λ::Float64,
                          loglikelihood_fn,
                          logprior_fn,
                          cfg::SMCConfig,
                          max_λ_inc::Float64,
                          rng::AbstractRNG)
    n_smc = size(particles, 1)

    ll = Vector{Float64}(undef, n_smc)
    Threads.@threads for m in 1:n_smc
        ll[m] = loglikelihood_fn(@view particles[m, :])
    end

    δλ_max  = min(1.0 - λ, max_λ_inc)
    δ       = solve_delta_for_ess(ll, cfg.target_ess_frac, δλ_max)
    next_λ  = λ + δ < 1.0 - 1e-6 ? λ + δ : 1.0
    Δλ      = next_λ - λ

    log_w   = Δλ .* ll
    log_wn  = log_w .- logsumexp(log_w)
    w       = exp.(log_wn)
    indices = systematic_resample_indices(rng, w, n_smc)
    resampled = particles[indices, :]

    inv_mass = estimate_mass_matrix(resampled)

    function tempered_lp(u)
        return logprior_fn(u) + next_λ * loglikelihood_fn(u)
    end

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


# ── Cold-start: prior → posterior via adaptive tempering ────────────────────

"""
    run_smc_window(loglikelihood_fn, priors, cfg, rng) -> TemperedSMCResult

Cold-start outer SMC². Initial cloud is sampled from the unconstrained-
space prior; the cloud is tempered until λ → 1 with adaptive δλ.

# Arguments
- `loglikelihood_fn`: `u -> Float64` likelihood at θ-particle `u`.
    Closures the inner PF (Phase 2 GPU code) and returns a scalar.
- `priors::Vector{<:PriorType}`: prior list, length `d_theta`.
- `cfg::SMCConfig`: outer-SMC, HMC, mass-matrix, AD hyperparameters.
- `rng::AbstractRNG`: PRNG.

# Returns
- `TemperedSMCResult`: posterior cloud, `n_temp`, `elapsed`.

# Notes
- Each tempering level resamples, re-estimates a diagonal mass matrix,
    and runs `cfg.num_mcmc_steps` HMC / MALA / NUTS moves per particle.
- Safety cap of 200 tempering levels — a degenerate likelihood landscape
    could otherwise loop forever.
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
        n_temp > 200 && break
    end

    return TemperedSMCResult(particles, n_temp, time() - t0)
end


# ── Warm-start: previous posterior → next-window posterior ──────────────────

"""
    run_smc_window_bridge(loglikelihood_fn, prev_posterior, priors, cfg, rng)
        -> TemperedSMCResult

Warm-start variant. Initial cloud is produced by
`Bridge.bridge_init(bridge_kind(cfg), prev_posterior, ...)`; the
tempering loop uses `cfg.max_lambda_inc_bridge` and
`cfg.num_mcmc_steps_bridge`.

# Arguments
- `loglikelihood_fn`, `priors`, `cfg`, `rng`: as in `run_smc_window`.
- `prev_posterior::AbstractMatrix{Float64}`: `(n_smc, d_theta)` cloud
    from the previous rolling window.

# Returns
- `TemperedSMCResult`.
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
