"""
    SMC2/HMC.jl

Wraps `AdvancedHMC.jl` for the per-tempering-level HMC moves and provides
MALA, AutoMALA, and ChEES variants. Direct port of the existing
`julia/SMC2FC/src/SMC2/HMC.jl` with the public functions documented in
Google style.

The Python reference (`smc2fc/core/jax_native_smc.py`) builds the BlackJAX
HMC kernel once at module load and then calls `_hmc_step_chain`. Julia's
analogue is `AdvancedHMC.jl`. This module deliberately uses HMC, not NUTS,
because NUTS-inside-vmap-inside-while-loop suffers a 60–85× warp-divergence
cost on GPU; here AdvancedHMC runs CPU-side and only the scalar
log-likelihood crosses PCIe.

The public API of every function is functional: it takes immutable inputs
and returns the new sampled vector.
"""
module HMC

using Random: AbstractRNG
using AdvancedHMC: AdvancedHMC, Hamiltonian, DiagEuclideanMetric,
                   Leapfrog, HMCKernel, Trajectory,
                   EndPointTS, FixedNSteps,
                   MultinomialTS, GeneralisedNoUTurn,
                   NoAdaptation, sample
using LogDensityProblems
using LogDensityProblemsAD: ADgradient
using ForwardDiff
using Enzyme

export hmc_step_chain, build_target

# ── LogDensityProblems wrapper ──────────────────────────────────────────────

"""
    CallableTarget{F}

Internal: turns any `u -> scalar` callable into a typed
`LogDensityProblems` target with declared dimension.

# Fields
- `f::F`: the callable.
- `d::Int`: the problem dimension.
"""
struct CallableTarget{F}
    f::F
    d::Int
end
LogDensityProblems.logdensity(t::CallableTarget, u) = t.f(u)
LogDensityProblems.dimension(t::CallableTarget)     = t.d
LogDensityProblems.capabilities(::Type{<:CallableTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

"""
    build_target(lp_fn, d; ad_backend=:ForwardDiff) -> ADgradient

Build an AD-augmented `LogDensityProblems` target.

# Arguments
- `lp_fn`: `u -> scalar` callable.
- `d::Integer`: problem dimension.

# Keyword arguments
- `ad_backend::Symbol = :ForwardDiff`: `:ForwardDiff` (forward-mode duals,
    cost `(1+d)` PF evals/grad) or `:Enzyme` (reverse-mode, ~1 PF
    eval/grad). For `:Enzyme`, runtime activity is set on `Reverse` so
    closure-captured constants don't trip static analysis.

# Returns
- `ADgradient`-wrapped target consumable by `AdvancedHMC.sample`.
"""
function build_target(lp_fn, d::Integer; ad_backend::Symbol = :ForwardDiff)
    target = CallableTarget(lp_fn, Int(d))
    if ad_backend === :Enzyme
        return ADgradient(:Enzyme, target;
                           mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    else
        return ADgradient(ad_backend, target)
    end
end


# ── HMC step chain ──────────────────────────────────────────────────────────

"""
    hmc_step_chain(initial_position, lp_fn, num_steps, step_size,
                    inv_mass_diag, num_leapfrog, rng;
                    ad_backend=:ForwardDiff, sampler=:HMC,
                    nuts_max_depth=10) -> Vector{Float64}

Apply `num_steps` MCMC moves to `initial_position` under the tempered
log-density `lp_fn(u)`.

# Arguments
- `initial_position::AbstractVector{Float64}`: chain start.
- `lp_fn`: `u -> scalar` log-density (already temperature-scaled).
- `num_steps::Integer`: number of MCMC moves.
- `step_size::Real`: leapfrog/MALA step size ε.
- `inv_mass_diag::AbstractVector{Float64}`: diagonal inverse mass matrix.
- `num_leapfrog::Integer`: leapfrog count for `:HMC` (ignored otherwise).
- `rng::AbstractRNG`: PRNG.

# Keyword arguments
- `ad_backend::Symbol = :ForwardDiff`: `:ForwardDiff | :Enzyme`.
- `sampler::Symbol = :HMC`: `:HMC | :NUTS | :MALA | :AutoMALA`. NUTS uses
    multinomial trajectories with generalised U-turn termination.
- `nuts_max_depth::Integer = 10`: max tree depth for NUTS.

# Returns
- `Vector{Float64}`: final position.

# Notes
- Diagonal mass-matrix only — full-mass kernels collapse acceptance to
    zero by λ ≈ 0.3 on the PF likelihood landscape.
- For MALA / AutoMALA, dispatches to the dedicated chain functions
    below.
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

Metropolis-adjusted Langevin Algorithm.

# Arguments
- `u0::AbstractVector{Float64}`: chain start.
- `target_grad`: `ADgradient` wrapper from `build_target`.
- `num_steps::Integer`: number of MALA moves.
- `step_size::Float64`: ε.
- `inv_mass_diag::AbstractVector{Float64}`: diagonal inverse mass.
- `rng::AbstractRNG`: PRNG.

# Returns
- `Vector{Float64}`: final position.

# Notes
- Each step does **one gradient evaluation** + 1 MH accept. Closest cheap
    analogue of MCLMC's "Langevin step" but with the MH correction so the
    chain is exact.
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

    val_u, grad_u = LogDensityProblems.logdensity_and_gradient(target_grad, u)

    @inbounds for _ in 1:Int(num_steps)
        ξ      = randn(rng, d)
        drift  = (ε2 / 2) .* M⁻¹ .* grad_u
        noise  = ε .* sM⁻¹ .* ξ
        u_new  = u .+ drift .+ noise

        val_new, grad_new = LogDensityProblems.logdensity_and_gradient(target_grad, u_new)

        drift_rev = (ε2 / 2) .* M⁻¹ .* grad_new
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
                         target_accept=0.574, adapt_rate=0.05)
        -> Vector{Float64}

MALA with Robbins–Monro step-size adaptation toward `target_accept`
(0.574 is the optimal MALA acceptance, Roberts & Rosenthal 2001).

# Arguments / keyword arguments
- See `mala_step_chain`. Additionally:
- `target_accept::Float64 = 0.574`.
- `adapt_rate::Float64 = 0.05`.

# Returns
- `Vector{Float64}`: final position. The adapted ε is discarded — each
    call re-adapts.

# Notes
- After `num_steps` moves, ε settles near the value producing the
    target acceptance. `step_size_init` only seeds the adaptation;
    final ε is independent (modulo a transient).
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

        log_ε += adapt_rate * ((accepted ? 1.0 : 0.0) - target_accept)
        log_ε = clamp(log_ε, -10.0, 5.0)
    end
    return u
end

"""
    chees_adapt_L(particles_subset, lp_fn, ε, inv_mass_diag,
                   L_candidates, n_steps, rng;
                   ad_backend=:ForwardDiff) -> Int

ChEES adaptation (Hoffman, Radul & Sountsov, 2021): pick the static
HMC trajectory length L that maximises ESJD-per-gradient.

# Arguments
- `particles_subset::AbstractMatrix{Float64}`: `(N_sub, d)` subset of
    the cloud used for the L-sweep.
- `lp_fn`, `ε::Float64`, `inv_mass_diag`, `n_steps::Integer`, `rng`: as
    in `hmc_step_chain`.
- `L_candidates::AbstractVector{<:Integer}`: candidate trajectory lengths.

# Keyword arguments
- `ad_backend::Symbol = :ForwardDiff`.

# Returns
- `Int`: best L by `score(L) = ESJD(L) / (L · ε)`.

# Notes
- Fixed L → no branch divergence — the property NUTS lacks on GPU.
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
