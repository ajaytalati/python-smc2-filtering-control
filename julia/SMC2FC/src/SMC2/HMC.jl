# SMC2/HMC.jl — wraps AdvancedHMC.jl for the per-tempering-level HMC moves.
#
# The Python implementation builds the BlackJAX HMC kernel ONCE at module
# load (charter §11.4: NUTS-inside-vmap-inside-while-loop suffers 60-85×
# warp-divergence cost; we deliberately use HMC, not NUTS). The Julia port
# hands that contract to AdvancedHMC.jl per charter §13.
#
# AdvancedHMC.jl runs on the CPU. Per charter §14 the outer SMC² rejuvenation
# stays on the CPU's branchy deep-cache cores; the Phase 2 PF likelihood is
# the GPU-resident piece. Only the scalar log-likelihood crosses PCIe.

module HMC

using Random: AbstractRNG
using AdvancedHMC: AdvancedHMC, Hamiltonian, DiagEuclideanMetric,
                   Leapfrog, HMCKernel, Trajectory,
                   EndPointTS, FixedNSteps,
                   MultinomialTS, GeneralisedNoUTurn,
                   NoAdaptation, sample
using LogDensityProblems
using LogDensityProblemsAD: ADgradient
# `using ForwardDiff` triggers the LogDensityProblemsAD ↔ ForwardDiff
# package extension, which is what makes `ADgradient(:ForwardDiff, target)`
# resolve. Without this import the call falls back to the no-op stub method.
using ForwardDiff
# Same trigger for Enzyme — importing makes `ADgradient(:Enzyme, target)`
# dispatch to LogDensityProblemsADEnzymeExt. Enzyme reverse-mode is the
# Phase 6 follow-up backend that gets ~1 PF eval per gradient (matches
# JAX's reverse-mode), vs ForwardDiff's (1 + d_θ) PF evals per gradient.
using Enzyme

export hmc_step_chain, build_target

# ── LogDensityProblems wrapper around an arbitrary callable ──────────────────
# AdvancedHMC consumes targets via the LogDensityProblems interface. This
# wrapper turns any `u -> scalar` callable into a typed target with a
# declared dimension; `ADgradient` adds ∂/∂u via ForwardDiff (or Enzyme).

struct CallableTarget{F}
    f::F
    d::Int
end
LogDensityProblems.logdensity(t::CallableTarget, u) = t.f(u)
LogDensityProblems.dimension(t::CallableTarget)     = t.d
LogDensityProblems.capabilities(::Type{<:CallableTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

"""
    build_target(lp_fn, d; ad_backend=:ForwardDiff) -> ADgradient-wrapped target

Build a LogDensityProblems target. `lp_fn(u)` is a scalar callable, `d` is
the problem dimension. `ad_backend` is one of:

  - `:ForwardDiff` (default): robust dual-number forward-mode. Cost per
    gradient = (1 + d) PF evals — good for d ≲ 8.

  - `:Enzyme`: reverse-mode AD. Cost per gradient = ~1 PF eval (matches
    JAX's reverse-mode). Charter §13's named production backend, used
    here for the SMC²-MPC closed-loop work where d_θ ≥ 8 and the
    ForwardDiff overhead becomes prohibitive. We pass
    `set_runtime_activity(Reverse)` so Enzyme handles closure-captured
    constants without static-activity errors (small perf cost vs the
    pure static path; see Enzyme.jl FAQ on runtime activity).
"""
function build_target(lp_fn, d::Integer; ad_backend::Symbol = :ForwardDiff)
    target = CallableTarget(lp_fn, Int(d))
    if ad_backend === :Enzyme
        # `set_runtime_activity(Reverse)` defers activity analysis to
        # runtime — needed when the log-density closes over Float64
        # constants (TRUTH params, obs vectors) that Enzyme can't
        # statically prove are non-active.
        return ADgradient(:Enzyme, target;
                           mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    else
        return ADgradient(ad_backend, target)
    end
end


# ── HMC step chain (port of jax_native_smc.py:_hmc_step_chain) ───────────────

"""
    hmc_step_chain(initial_position, lp_fn, num_steps, step_size,
                    inv_mass_diag, num_leapfrog, rng;
                    ad_backend=:ForwardDiff, sampler=:HMC,
                    nuts_max_depth=10) -> Vector{Float64}

Apply `num_steps` MCMC moves to a single particle's `initial_position` under
the temperature-tempered log-density `lp_fn(u)`. Returns the final position.

`sampler` selects the kernel:

  - `:HMC`  — fixed-step HMC with `num_leapfrog` leapfrog steps per move
              and `EndPointTS` trajectory (the v1 default). Cost per
              move = `num_leapfrog` gradient evals + 1 MH accept.

  - `:NUTS` — No-U-Turn Sampler with multinomial trajectory selection +
              generalised U-turn termination, max tree depth
              `nuts_max_depth`. Cost varies — typically 10–100 grads/move
              (each tree-doubling = 2× the gradient calls).

  - `:MALA` — Metropolis-adjusted Langevin Algorithm. Cost per move =
              **1 gradient eval** + 1 MH accept. The proposal is
              `u' = u + ε²/2 · M⁻¹ · ∇log p(u) + ε · M⁻¹/² · ξ`
              with `M = diag(1 / inv_mass_diag)`. `num_leapfrog` is
              unused. Closest cheap analogue of MCLMC for our use case;
              roughly matches Python's MCLMC per-step cost (~1 grad).

  - `:AutoMALA` — MALA with Robbins-Monro step-size adaptation toward
              the optimal MALA acceptance rate of ~0.574 (Roberts &
              Rosenthal 2001). Same per-step cost as :MALA but no
              hand-tuning of `hmc_step_size` — the kernel auto-shrinks
              ε when acceptance is low and grows it when acceptance is
              high. Inspired by `Pigeons.jl`'s `AutoMALA` explorer
              (which we couldn't use directly — its newer versions
              conflict with our AdvancedHMC/Distributions deps and
              the v0.1.1 that does resolve doesn't ship AutoMALA).

Inv_mass_diag is the diagonal inverse mass matrix — full-mass kernels
collapse to zero acceptance by λ ≈ 0.3 on the PF likelihood landscape.
"""
function hmc_step_chain(initial_position::AbstractVector{Float64},
                         lp_fn,
                         num_steps::Integer,
                         step_size::Real,
                         inv_mass_diag::AbstractVector{Float64},
                         num_leapfrog::Integer,
                         rng::AbstractRNG;
                         ad_backend::Symbol = :ForwardDiff,
                         sampler::Symbol = :HMC,
                         nuts_max_depth::Integer = 10)
    d           = length(initial_position)
    target_grad = build_target(lp_fn, d; ad_backend = ad_backend)

    if sampler === :MALA
        return mala_step_chain(initial_position, target_grad, num_steps,
                                Float64(step_size), inv_mass_diag, rng)
    elseif sampler === :AutoMALA
        return automala_step_chain(initial_position, target_grad, num_steps,
                                     Float64(step_size), inv_mass_diag, rng)
    end

    metric      = DiagEuclideanMetric(inv_mass_diag)
    hamiltonian = Hamiltonian(metric, target_grad)
    integrator  = Leapfrog(Float64(step_size))

    kernel = if sampler === :NUTS
        HMCKernel(Trajectory{MultinomialTS}(integrator,
            GeneralisedNoUTurn(Int(nuts_max_depth), 1000.0)))
    else
        HMCKernel(Trajectory{EndPointTS}(integrator,
            FixedNSteps(Int(num_leapfrog))))
    end

    adaptor = NoAdaptation()
    samples, _stats = sample(rng, hamiltonian, kernel,
                              collect(initial_position), Int(num_steps), adaptor;
                              progress = false, verbose = false)
    return samples[end]
end


"""
    mala_step_chain(u0, target_grad, num_steps, step_size, inv_mass_diag, rng)
        -> Vector{Float64}

Metropolis-adjusted Langevin Algorithm. Each step:

    grad   = ∇log p(u)
    drift  = (ε²/2) · inv_mass_diag .* grad
    noise  = ε · sqrt.(inv_mass_diag) .* randn(d)
    u_new  = u + drift + noise

with a standard MH accept that uses the asymmetric proposal density
(reverse-direction drift + log-density ratio).

1 gradient evaluation per step, 0 leapfrog inner loop. Equivalent in
cost to MCLMC's "Langevin step" but with the MH correction so the
chain is exact. If MH acceptance drops, decrease `step_size`.
"""
function mala_step_chain(u0::AbstractVector{Float64},
                          target_grad,
                          num_steps::Integer,
                          step_size::Float64,
                          inv_mass_diag::AbstractVector{Float64},
                          rng::AbstractRNG)
    d   = length(u0)
    ε   = step_size
    ε2  = ε * ε
    u   = collect(u0)
    M⁻¹  = inv_mass_diag
    sM⁻¹ = sqrt.(inv_mass_diag)

    # Initial grad + log-density
    val_u, grad_u = LogDensityProblems.logdensity_and_gradient(target_grad, u)

    @inbounds for _ in 1:Int(num_steps)
        # Proposal: u' = u + (ε²/2) M⁻¹ ∇log p(u) + ε M⁻¹/² ξ
        ξ      = randn(rng, d)
        drift  = (ε2 / 2) .* M⁻¹ .* grad_u
        noise  = ε .* sM⁻¹ .* ξ
        u_new  = u .+ drift .+ noise

        # Evaluate target at proposal
        val_new, grad_new = LogDensityProblems.logdensity_and_gradient(target_grad, u_new)

        # Reverse-direction drift for the proposal density
        drift_rev = (ε2 / 2) .* M⁻¹ .* grad_new
        # Forward proposal density: log q(u'|u)
        # q(u'|u) = N(u'; u + drift, ε² M⁻¹) — diagonal cov
        # log q = -∑ (u' - u - drift_i)² / (2 ε² M⁻¹_i) - ∑ ½ log(2π ε² M⁻¹_i)
        # Reverse:
        # log q(u|u') = -∑ (u - u' - drift_rev_i)² / (2 ε² M⁻¹_i) - same const
        # Acceptance ratio: log p(u') - log p(u) + log q(u|u') - log q(u'|u)
        diff_fwd = u_new .- u .- drift
        diff_rev = u    .- u_new .- drift_rev
        log_q_fwd = -sum(@. diff_fwd^2 / (2 * ε2 * M⁻¹))
        log_q_rev = -sum(@. diff_rev^2 / (2 * ε2 * M⁻¹))
        log_α     = (val_new - val_u) + (log_q_rev - log_q_fwd)

        if log(rand(rng)) < log_α
            u       = u_new
            val_u   = val_new
            grad_u  = grad_new
        end
    end
    return u
end

"""
    automala_step_chain(u0, target_grad, num_steps, step_size_init,
                         inv_mass_diag, rng;
                         target_accept=0.574, adapt_rate=0.05) -> Vector{Float64}

MALA with Robbins-Monro step-size adaptation. Each step does one MALA
move and then nudges `log(ε)` based on the local accept/reject decision:

    log(ε) ← log(ε) + adapt_rate · (accept_indicator - target_accept)

where `accept_indicator ∈ {0, 1}` is whether the MH proposal was
accepted at this step. This is the standard Roberts-Rosenthal Robbins-
Monro adaptation. After `num_steps` moves the step size has settled
near the value that produces ~target_accept (= 0.574 for MALA).

`step_size_init` only seeds the adaptation; the final ε is independent
of it (modulo a transient).

Returns the final position. The adapted ε is discarded — each call
to `automala_step_chain` re-adapts. For longer chains this is
suboptimal; for our SMC² use case with `num_mcmc_steps ≈ 40` it
converges fast enough.
"""
function automala_step_chain(u0::AbstractVector{Float64},
                              target_grad,
                              num_steps::Integer,
                              step_size_init::Float64,
                              inv_mass_diag::AbstractVector{Float64},
                              rng::AbstractRNG;
                              target_accept::Float64 = 0.574,
                              adapt_rate::Float64    = 0.05)
    d   = length(u0)
    log_ε = log(step_size_init)
    u   = collect(u0)
    M⁻¹  = inv_mass_diag
    sM⁻¹ = sqrt.(inv_mass_diag)

    val_u, grad_u = LogDensityProblems.logdensity_and_gradient(target_grad, u)

    @inbounds for _ in 1:Int(num_steps)
        ε   = exp(log_ε)
        ε2  = ε * ε
        ξ      = randn(rng, d)
        drift  = (ε2 / 2) .* M⁻¹ .* grad_u
        noise  = ε .* sM⁻¹ .* ξ
        u_new  = u .+ drift .+ noise

        val_new, grad_new = LogDensityProblems.logdensity_and_gradient(target_grad, u_new)

        drift_rev = (ε2 / 2) .* M⁻¹ .* grad_new
        diff_fwd  = u_new .- u .- drift
        diff_rev  = u    .- u_new .- drift_rev
        log_q_fwd = -sum(@. diff_fwd^2 / (2 * ε2 * M⁻¹))
        log_q_rev = -sum(@. diff_rev^2 / (2 * ε2 * M⁻¹))
        log_α     = (val_new - val_u) + (log_q_rev - log_q_fwd)

        accepted = log(rand(rng)) < log_α
        if accepted
            u       = u_new
            val_u   = val_new
            grad_u  = grad_new
        end

        # Robbins-Monro: log(ε) ← log(ε) + adapt_rate · (1{accept} - target).
        log_ε += adapt_rate * ((accepted ? 1.0 : 0.0) - target_accept)
        # Safety clamp so a string of rejects doesn't drive ε to 0.
        log_ε = clamp(log_ε, -10.0, 5.0)
    end
    return u
end

"""
    chees_adapt_L(particles_subset, lp_fn, ε, inv_mass_diag,
                   L_candidates, n_steps, rng;
                   ad_backend = :ForwardDiff) -> best_L

ChEES adaptation (Hoffman, Radul & Sountsov, 2021): pick the static
HMC trajectory length L that maximises **expected squared jump distance
per gradient eval**:

    ESJD(L) = (1/N_subset) Σ_i ‖ θ_i^new − θ_i^old ‖²
    score(L) = ESJD(L) / (L · ε)

Sweep `L_candidates`, run an `n_steps`-length static-L HMC chain on each
subset particle for each candidate, score, return argmax. The chosen
L is then used as the **fixed** leapfrog count for the main rejuvenation
moves on all `n_smc` particles. Fixed L → no branch divergence (the
GPU/SIMD friendly property NUTS lacks).
"""
function chees_adapt_L(particles_subset::AbstractMatrix{Float64},
                        lp_fn,
                        ε::Float64,
                        inv_mass_diag::AbstractVector{Float64},
                        L_candidates::AbstractVector{<:Integer},
                        n_steps::Integer,
                        rng::AbstractRNG;
                        ad_backend::Symbol = :ForwardDiff)
    N_sub = size(particles_subset, 1)
    d     = size(particles_subset, 2)
    target_grad = build_target(lp_fn, d; ad_backend = ad_backend)

    metric      = DiagEuclideanMetric(inv_mass_diag)
    hamiltonian = Hamiltonian(metric, target_grad)
    integrator  = Leapfrog(ε)

    best_L     = first(L_candidates)
    best_score = -Inf
    for L in L_candidates
        kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(Int(L))))
        sqd_total = 0.0
        for i in 1:N_sub
            θ0 = collect(@view particles_subset[i, :])
            samples, _ = sample(rng, hamiltonian, kernel, θ0, Int(n_steps),
                                  NoAdaptation(); progress = false, verbose = false)
            θ1 = samples[end]
            sqd_total += sum(abs2, θ1 .- θ0)
        end
        esjd  = sqd_total / N_sub
        score = esjd / (L * ε)
        if score > best_score
            best_score = score
            best_L     = Int(L)
        end
    end
    return best_L
end

end # module HMC
