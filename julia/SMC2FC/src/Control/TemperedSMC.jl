# Control/TemperedSMC.jl — outer SMC² loop for control-as-inference.
#
# Charter §15.5: this module re-uses Phase 3.3's `_tempered_step!` with the
# cost-as-likelihood substitution. The outer SMC² treats the policy
# parameter θ as a posterior under
#
#     p(θ | success) ∝ p(θ) · exp(-β · J(θ))
#
# where J(θ) is the model's `cost_fn`. The "tempering" parameter β goes
# from 0 (prior) to β_max (full cost) over adaptive levels.
#
# This is the control-as-inference duality (Toussaint 2009; Levine 2018;
# Kappen 2005) — same outer kernel, different inner target.

module ControlLoop

using Random: AbstractRNG
using Statistics: mean, std
using LogExpFunctions: logsumexp
using ..Spec: ControlSpec
using ..Calibration: calibrate_beta_max
using ...SMC2FC: SMCConfig, NormalPrior, PriorType
using ...SMC2FC: log_prior_unconstrained
using ...SMC2FC.Tempering: solve_delta_for_ess
using ...SMC2FC.MassMatrix: estimate_mass_matrix
using ...SMC2FC.HMC: hmc_step_chain

export run_tempered_smc_loop, ControlResult

# ── Result struct ────────────────────────────────────────────────────────────

struct ControlResult
    particles::Matrix{Float64}      # (n_smc, theta_dim)
    cost_per_particle::Vector{Float64}
    mean_schedule::Vector{Float64}  # decoded from posterior mean θ
    n_temp::Int                      # tempering levels taken
    elapsed::Float64                 # wall time (s)
    β_max::Float64
    prior_cost_mean::Float64
    prior_cost_std::Float64
end


# ── Systematic resample (CPU; control side stays on CPU like outer SMC²) ────
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
    run_tempered_smc_loop(spec::ControlSpec, cfg::SMCConfig, rng::AbstractRNG;
                          calib_n=256, target_nats=8.0)
        -> ControlResult

Outer adaptive-tempered SMC² for control. Returns the posterior particles,
their costs, and the decoded mean schedule.

Steps:
  1. Auto-calibrate β_max via `calibrate_beta_max`.
  2. Sample initial cloud from `N(prior_mean, σ_prior² · I)`.
  3. Adaptively temper from β=0 to β=β_max with HMC rejuvenation per level.
  4. Decode the posterior-mean θ through `spec.schedule_from_theta`.
"""
function run_tempered_smc_loop(spec::ControlSpec,
                                 cfg::SMCConfig,
                                 rng::AbstractRNG;
                                 calib_n::Integer = 256,
                                 target_nats::Real = 8.0)
    t0 = time()

    # ── Step 1: auto-calibrate β_max from prior-cloud cost spread ───────────
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

    # ── Step 2: initial cloud from broad Gaussian prior ─────────────────────
    n_smc     = cfg.n_smc_particles
    particles = reshape(μ_vec, 1, :) .+
                spec.sigma_prior .* randn(rng, n_smc, spec.theta_dim)

    # ── Step 3: outer tempered SMC² loop ────────────────────────────────────
    # Cost-as-likelihood: ll(u) = -β · cost(u). We re-implement the outer
    # step here (rather than calling Phase 3's `_tempered_step!`) because
    # the tempering parameter is β ∈ [0, β_max], NOT λ ∈ [0, 1] — the
    # adaptive-bisection still hunts for δβ via ESS, but the bookkeeping
    # is on β rather than λ.
    cost_per_particle = Vector{Float64}(undef, n_smc)
    β_curr = 0.0
    n_temp = 0

    function loglik_at(u)
        return -spec.cost_fn(u)
    end

    function logprior(u)
        # Gaussian prior in the unconstrained search space.
        δ  = u .- μ_vec
        return -0.5 * sum(abs2, δ ./ spec.sigma_prior) -
               spec.theta_dim * log(spec.sigma_prior)
    end

    while β_curr < β_max - 1e-6
        # Per-particle log-likelihood = -cost
        for m in 1:n_smc
            cost_per_particle[m] = spec.cost_fn(@view particles[m, :])
        end
        ll = -cost_per_particle

        # Bisection on δβ.
        δβ_max = min(β_max - β_curr, β_max * cfg.max_lambda_inc)
        δβ     = solve_delta_for_ess(ll, cfg.target_ess_frac, δβ_max)
        next_β = β_curr + δβ < β_max - 1e-6 ? β_curr + δβ : β_max
        Δβ     = next_β - β_curr

        # Reweight + systematic resample.
        log_w   = Δβ .* ll
        log_wn  = log_w .- logsumexp(log_w)
        w       = exp.(log_wn)
        idx     = _systematic_resample_indices(rng, w, n_smc)
        resampled = particles[idx, :]

        # Mass matrix from resampled cloud.
        inv_mass = estimate_mass_matrix(resampled)

        # HMC moves at temperature next_β.
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

    # ── Step 4: decode posterior-mean θ → schedule ──────────────────────────
    θ_post_mean = vec(mean(particles; dims=1))
    mean_sched  = spec.schedule_from_theta(θ_post_mean)

    # Final cost evaluation per particle.
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
