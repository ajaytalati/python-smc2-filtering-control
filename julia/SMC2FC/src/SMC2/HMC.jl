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
                   Leapfrog, HMCKernel, Trajectory, EndPointTS, FixedNSteps,
                   NoAdaptation, sample
using LogDensityProblems
using LogDensityProblemsAD: ADgradient
# `using ForwardDiff` triggers the LogDensityProblemsAD ↔ ForwardDiff
# package extension, which is what makes `ADgradient(:ForwardDiff, target)`
# resolve. Without this import the call falls back to the no-op stub method.
using ForwardDiff

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
the problem dimension. `ad_backend` is one of `:ForwardDiff` (default;
robust, good for d ≲ 50) or `:Enzyme` (faster on the 37-D θ_dyn posterior
per charter §13).
"""
function build_target(lp_fn, d::Integer; ad_backend::Symbol = :ForwardDiff)
    target = CallableTarget(lp_fn, Int(d))
    return ADgradient(ad_backend, target)
end


# ── HMC step chain (port of jax_native_smc.py:_hmc_step_chain) ───────────────

"""
    hmc_step_chain(initial_position, lp_fn, num_steps, step_size,
                    inv_mass_diag, num_leapfrog, rng;
                    ad_backend=:ForwardDiff) -> Vector{Float64}

Apply `num_steps` HMC moves to a single particle's `initial_position` under
the temperature-tempered log-density `lp_fn(u)`. Returns the final position.

`inv_mass_diag` is the *diagonal* of the inverse mass matrix — full-mass
HMC collapses to zero acceptance by λ ≈ 0.3 on the PF likelihood landscape
(see Python `mass_matrix.py` rationale).
"""
function hmc_step_chain(initial_position::AbstractVector{Float64},
                         lp_fn,
                         num_steps::Integer,
                         step_size::Real,
                         inv_mass_diag::AbstractVector{Float64},
                         num_leapfrog::Integer,
                         rng::AbstractRNG;
                         ad_backend::Symbol = :ForwardDiff)
    d           = length(initial_position)
    target_grad = build_target(lp_fn, d; ad_backend = ad_backend)
    metric      = DiagEuclideanMetric(inv_mass_diag)
    hamiltonian = Hamiltonian(metric, target_grad)
    integrator  = Leapfrog(Float64(step_size))
    kernel      = HMCKernel(Trajectory{EndPointTS}(integrator,
                                                    FixedNSteps(Int(num_leapfrog))))
    adaptor     = NoAdaptation()

    # AdvancedHMC's `sample` returns (samples_vec, stats_vec). Take the last
    # sample as the new particle position.
    samples, _stats = sample(rng, hamiltonian, kernel,
                              collect(initial_position), Int(num_steps), adaptor;
                              progress = false, verbose = false)
    return samples[end]
end

end # module HMC
