"""
    Control/TemperedSMC.jl

Outer SMC² loop for control-as-inference. Re-uses Phase 3.3's outer-step
machinery with a cost-as-likelihood substitution: the outer SMC² treats
the policy parameter θ as a posterior under

    p(θ | success) ∝ p(θ) · exp(-β · J(θ))

where J(θ) is the model's `cost_fn`. The tempering parameter β goes from
0 (prior) to β_max (full cost) over adaptive levels. This is the
control-as-inference duality (Toussaint 2009; Levine 2018; Kappen 2005)
— same outer kernel, different inner target.

Functional API contract:

- `run_tempered_smc_loop` takes immutable inputs (`spec`, `cfg`, `rng`)
    and returns a fresh `ControlResult` struct.
- The internal helpers do mutate the tempering buffer for performance,
    but no `!` is exposed to callers.
"""
module ControlLoop

using Random: AbstractRNG
using Statistics: mean, std
using LogExpFunctions: logsumexp
using ..Spec: ControlSpec
using ..Calibration: calibrate_beta_max
using ...SMC2FC_functional: SMCConfig, NormalPrior, PriorType
using ...SMC2FC_functional: log_prior_unconstrained
using ...SMC2FC_functional.Tempering: solve_delta_for_ess
using ...SMC2FC_functional.MassMatrix: estimate_mass_matrix
using ...SMC2FC_functional.HMC: hmc_step_chain

export run_tempered_smc_loop, ControlResult

# ── Result struct ───────────────────────────────────────────────────────────

"""
    ControlResult

Result of one control-side tempered SMC² run.

# Fields
- `particles::Matrix{Float64}`: `(n_smc, theta_dim)` posterior cloud.
- `cost_per_particle::Vector{Float64}`: final cost evaluated on each row.
- `mean_schedule::Vector{Float64}`: schedule decoded from the posterior-
    mean θ.
- `n_temp::Int`, `elapsed::Float64`: tempering levels and wall-time.
- `β_max::Float64`, `prior_cost_mean::Float64`, `prior_cost_std::Float64`:
    auto-calibration outputs.
"""
struct ControlResult
    particles::Matrix{Float64}
    cost_per_particle::Vector{Float64}
    mean_schedule::Vector{Float64}
    n_temp::Int
    elapsed::Float64
    β_max::Float64
    prior_cost_mean::Float64
    prior_cost_std::Float64
end


# ── Internal helpers ────────────────────────────────────────────────────────

"""
    _systematic_resample_indices(rng, weights, n) -> Vector{Int}

Module-private CPU systematic-resampling indices.
"""
function _systematic_resample_indices(rng::AbstractRNG,
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


"""
    run_tempered_smc_loop(spec, cfg, rng; calib_n=256, target_nats=8.0)
        -> ControlResult

Outer adaptive-tempered SMC² for control.

# Arguments
- `spec::ControlSpec`: the control task — cost function, schedule decoder,
    prior mean / σ.
- `cfg::SMCConfig`: outer-SMC, HMC, mass-matrix hyperparameters.
- `rng::AbstractRNG`: PRNG.

# Keyword arguments
- `calib_n::Integer = 256`: prior-draws used by `calibrate_beta_max`.
- `target_nats::Real = 8.0`: target tempering nats (≈ 16 effective levels).

# Returns
- `ControlResult`: posterior cloud + cost-per-particle + decoded
    mean schedule + bookkeeping fields.

# Notes
- Steps:
    1. Auto-calibrate β_max via `Calibration.calibrate_beta_max`.
    2. Sample initial cloud from `N(prior_mean, σ_prior² · I)`.
    3. Adaptively temper from `β = 0` to `β = β_max` with HMC
       rejuvenation per level.
    4. Decode posterior-mean θ through `spec.schedule_from_theta`.
- The tempering bookkeeping is on β (not λ ∈ [0, 1]), so the adaptive
    bisection hunts for δβ via ESS but the loop variable is `β_curr`.
"""
function run_tempered_smc_loop(spec::ControlSpec,
                                 cfg::SMCConfig,
                                 rng::AbstractRNG;
                                 calib_n::Integer = 256,
                                 target_nats::Real = 8.0)
    t0 = time()

    μ_vec = isempty(spec.prior_mean) ?
        zeros(spec.theta_dim) :
        spec.prior_mean
    β_max, c_mean, c_std = calibrate_beta_max(
        spec.cost_fn;
        theta_dim   = spec.theta_dim,
        sigma_prior = spec.sigma_prior,
        prior_mean  = μ_vec,
        n_samples   = calib_n,
        target_nats = target_nats,
    )

    n_smc     = cfg.n_smc_particles
    particles = reshape(μ_vec, 1, :) .+
                spec.sigma_prior .* randn(rng, n_smc, spec.theta_dim)

    cost_per_particle = Vector{Float64}(undef, n_smc)
    β_curr = 0.0
    n_temp = 0

    function loglik_at(u)
        return -spec.cost_fn(u)
    end

    function logprior(u)
        δ  = u .- μ_vec
        return -0.5 * sum(abs2, δ ./ spec.sigma_prior) -
               spec.theta_dim * log(spec.sigma_prior)
    end

    while β_curr < β_max - 1e-6
        for m in 1:n_smc
            cost_per_particle[m] = spec.cost_fn(@view particles[m, :])
        end
        ll = -cost_per_particle

        δβ_max = min(β_max - β_curr, β_max * cfg.max_lambda_inc)
        δβ     = solve_delta_for_ess(ll, cfg.target_ess_frac, δβ_max)
        next_β = β_curr + δβ < β_max - 1e-6 ? β_curr + δβ : β_max
        Δβ     = next_β - β_curr

        log_w   = Δβ .* ll
        log_wn  = log_w .- logsumexp(log_w)
        w       = exp.(log_wn)
        idx     = _systematic_resample_indices(rng, w, n_smc)
        resampled = particles[idx, :]

        inv_mass = estimate_mass_matrix(resampled)

        function tempered_lp(u)
            return logprior(u) + next_β * loglik_at(u)
        end

        new_particles = Matrix{Float64}(undef, n_smc, spec.theta_dim)
        for m in 1:n_smc
            new_particles[m, :] = hmc_step_chain(
                collect(@view resampled[m, :]),
                tempered_lp,
                cfg.num_mcmc_steps,
                cfg.hmc_step_size,
                inv_mass,
                cfg.hmc_num_leapfrog,
                rng,
            )
        end
        particles = new_particles
        β_curr    = next_β
        n_temp   += 1
        n_temp > 200 && break
    end

    θ_post_mean = vec(mean(particles; dims=1))
    mean_sched  = spec.schedule_from_theta(θ_post_mean)

    for m in 1:n_smc
        cost_per_particle[m] = spec.cost_fn(@view particles[m, :])
    end

    return ControlResult(
        particles, cost_per_particle, mean_sched,
        n_temp, time() - t0,
        β_max, c_mean, c_std,
    )
end

end # module ControlLoop
