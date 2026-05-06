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
              and `EndPointTS` trajectory (the v1 default).
  - `:NUTS` — No-U-Turn Sampler with multinomial trajectory selection +
              generalised U-turn termination, max tree depth `nuts_max_depth`.
              `num_leapfrog` is unused under NUTS; the trajectory length
              adapts per move based on the U-turn check.

`inv_mass_diag` is the diagonal inverse mass matrix — full-mass HMC
collapses to zero acceptance by λ ≈ 0.3 on the PF likelihood landscape
(see Python `mass_matrix.py` rationale).
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
    metric      = DiagEuclideanMetric(inv_mass_diag)
    hamiltonian = Hamiltonian(metric, target_grad)
    integrator  = Leapfrog(Float64(step_size))

    kernel = if sampler === :NUTS
        # Multinomial trajectory + generalised U-turn termination = the
        # NUTS variant Hoffman & Gelman 2014 + Betancourt 2017 describe.
        # Δ_max=1000 disables divergence-cap early stop.
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

end # module HMC
