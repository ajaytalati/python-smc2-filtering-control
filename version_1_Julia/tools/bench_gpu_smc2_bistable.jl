# GPU end-to-end SMC²-on-bistable bench — option (C) deliverable.
#
# Composes:
#   - The fused KernelAbstractions PF kernel (Phase 6 follow-up #2.5)
#   - GPU log-density wrapper + finite-difference gradient (CRN noise)
#   - AdvancedHMC.jl with the GPU log-density as a LogDensityProblems target
#   - ChEES static-L adaptation (Hoffman et al. 2021): pick L that
#     maximises ESJD / (L·ε) per tempering level
#   - A small adaptive-tempered SMC² outer loop, run sequentially on the GPU
#     (one chain at a time — the GPU is the shared resource)
#
# This is a **demonstration** that the full GPU pipeline composes
# end-to-end, NOT a speed claim — at d=8 / T=144 the per-move cost
# (~320 ms = 8 leapfrog × 1 grad × 17 PF evals × ~2.4 ms each) is
# comparable to the CPU AutoMALA path we have. The GPU win only
# appears at larger d_θ (FSA-v5's 37-D posterior) or larger K (the
# fused kernel sustains 2.45e10 particle-steps/sec at K=1M).
#
# Run from version_1_Julia/:
#   julia --threads auto --project=. tools/bench_gpu_smc2_bistable.jl

using Random, Statistics, Printf
using LogDensityProblems
using AdvancedHMC
using LogExpFunctions: logsumexp
using CUDA

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "models", "bistable_controlled", "gpu_pf.jl"))
using .BistableGPUPF: BistableGPUTarget, gpu_log_density, gpu_log_density_with_grad
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))
using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A, simulate_em

# ── Build target ────────────────────────────────────────────────────────────
println("="^70)
println("GPU end-to-end SMC² on bistable — Phase 1 inference demo")
println("="^70)
data = simulate_em(seed = 7); n1 = 144
obs = Float64.(data.obs[1:n1])

const K_PF = 100_000
target_gpu = BistableGPUTarget(
    K = K_PF, T_steps = n1, dt = EXOGENOUS_A.dt,
    T_intervention = 24.0, u_on = 0.0,
    x_init = INIT_STATE_A.x_0, u_init = INIT_STATE_A.u_0,
    obs_seq_cpu = obs, noise_seed = 0,
)
println("GPU target: K=$K_PF fp32 particles, T=$n1 obs")

# ── Prior + LogDensityProblems target ──────────────────────────────────────
const PRIOR_SIGMAS  = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]   # 6 lognormal + 2 normal
const PRIOR_MEANS   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        INIT_STATE_A.x_0, INIT_STATE_A.u_0]

logprior(u) = -0.5 * sum(@. ((u - PRIOR_MEANS) / PRIOR_SIGMAS)^2)
logprior_grad(u) = -(u .- PRIOR_MEANS) ./ (PRIOR_SIGMAS.^2)

struct GPUDensityTarget
    target_gpu::BistableGPUTarget
end
LogDensityProblems.dimension(::GPUDensityTarget) = 8
LogDensityProblems.capabilities(::Type{<:GPUDensityTarget}) =
    LogDensityProblems.LogDensityOrder{1}()

function _ll_at_lambda(t::GPUDensityTarget, u, λ)
    val_ll = gpu_log_density(t.target_gpu, u)
    return logprior(u) + λ * val_ll
end
function _ll_grad_at_lambda(t::GPUDensityTarget, u, λ)
    val, g = gpu_log_density_with_grad(t.target_gpu, u)
    return logprior(u) + λ * val, logprior_grad(u) .+ λ .* g
end

# Tempered LogDensityProblems target — λ baked into the closure for HMC use.
struct TemperedTarget
    base::GPUDensityTarget
    λ::Float64
end
LogDensityProblems.dimension(::TemperedTarget) = 8
LogDensityProblems.capabilities(::Type{<:TemperedTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(t::TemperedTarget, u) = _ll_at_lambda(t.base, u, t.λ)
LogDensityProblems.logdensity_and_gradient(t::TemperedTarget, u) =
    _ll_grad_at_lambda(t.base, u, t.λ)

base_target = GPUDensityTarget(target_gpu)

# ── ChEES L adaptation ─────────────────────────────────────────────────────
function chees_pick_L(t_temp::TemperedTarget, particles_subset::AbstractMatrix{Float64};
                       ε::Float64, L_candidates::Vector{Int} = [2, 4, 8, 16, 32],
                       n_steps::Int = 5, rng = MersenneTwister(0))
    metric      = AdvancedHMC.DiagEuclideanMetric(ones(8))
    hamiltonian = AdvancedHMC.Hamiltonian(metric, t_temp)
    integrator  = AdvancedHMC.Leapfrog(ε)
    best_L      = first(L_candidates)
    best_score  = -Inf
    for L in L_candidates
        kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
            integrator, AdvancedHMC.FixedNSteps(L)))
        sqd = 0.0
        for i in 1:size(particles_subset, 1)
            θ0 = collect(@view particles_subset[i, :])
            samples, _ = AdvancedHMC.sample(rng, hamiltonian, kernel, θ0, n_steps,
                                              AdvancedHMC.NoAdaptation();
                                              progress = false, verbose = false)
            sqd += sum(abs2, samples[end] .- θ0)
        end
        esjd  = sqd / size(particles_subset, 1)
        score = esjd / (L * ε)
        if score > best_score
            best_score = score; best_L = L
        end
    end
    return best_L, best_score
end

# ── Outer SMC² loop (small demo: n_smc=16, ESS-adaptive λ) ────────────────
const N_SMC = 16
const TARGET_ESS_FRAC = 0.5
const MAX_LAMBDA_INC  = 0.20
const NUM_MCMC        = 3
const HMC_STEP_SIZE   = 0.025
const CHEES_CANDS     = [2, 4, 8, 16]

println("\nOuter SMC²: n_smc=$N_SMC, target_ess_frac=$TARGET_ESS_FRAC, max_λ_inc=$MAX_LAMBDA_INC")
println("Per-level: ChEES adapts L, then $NUM_MCMC HMC moves per particle.")

# Initial cloud from prior
rng = MersenneTwister(2026)
particles = Matrix{Float64}(undef, N_SMC, 8)
for m in 1:N_SMC, j in 1:8
    particles[m, j] = PRIOR_MEANS[j] + PRIOR_SIGMAS[j] * randn(rng)
end

t_start = time()

# Top-level `while` loops in Julia have soft scope: writes to outer
# variables get reinterpreted as locals, breaking the loop. The cleanest
# fix is to put the SMC² loop in a function. Below, `run_outer_smc!`
# mutates `particles` in place and returns the final tempering count.
function run_outer_smc!(particles::Matrix{Float64})
    λ_var = 0.0
    n_temp = 0
    while λ_var < 1.0 - 1e-6
    # Per-particle log-likelihood at current cloud
    ll = Vector{Float64}(undef, N_SMC)
    for m in 1:N_SMC
        ll[m] = gpu_log_density(target_gpu, @view particles[m, :])
    end
    # Adaptive δλ
    δ_max = min(1.0 - λ, MAX_LAMBDA_INC)
    # Bisection
    lo, hi = 0.0, δ_max
    target_ess = TARGET_ESS_FRAC * N_SMC
    function ess_at(δ)
        log_w = δ .* ll
        log_w_n = log_w .- logsumexp(log_w)
        return exp(-logsumexp(2.0 .* log_w_n))
    end
    if ess_at(δ_max) >= target_ess
        δ = δ_max
    else
        for _ in 1:30
            mid = 0.5 * (lo + hi)
            if ess_at(mid) > target_ess; lo = mid; else; hi = mid; end
        end
        δ = lo
    end
    next_λ = (λ + δ < 1.0 - 1e-6) ? λ + δ : 1.0
    Δλ = next_λ - λ

    # Reweight + systematic resample
    log_w = Δλ .* ll
    w = exp.(log_w .- logsumexp(log_w))
    cumsum_w = cumsum(w)
    indices = Vector{Int}(undef, N_SMC)
    u_shift = rand(rng) / N_SMC
    for i in 1:N_SMC
        target = (i - 1) / N_SMC + u_shift
        indices[i] = clamp(searchsortedfirst(cumsum_w, target), 1, N_SMC)
    end
    resampled = particles[indices, :]

    # ChEES adapt L on a 4-particle subset
    t_temp = TemperedTarget(base_target, next_λ)
    chees_subset = resampled[1:min(4, N_SMC), :]
    best_L, best_score = chees_pick_L(t_temp, chees_subset;
                                         ε = HMC_STEP_SIZE,
                                         L_candidates = CHEES_CANDS,
                                         n_steps = 3, rng = MersenneTwister(11 + n_temp))
    println(@sprintf("[%2d] λ %.3f → %.3f  (Δλ=%.3f)  ChEES L=%d  (score=%.2g)",
                      n_temp + 1, λ, next_λ, Δλ, best_L, best_score))

    # HMC moves at temperature next_λ — sequential on GPU
    metric = AdvancedHMC.DiagEuclideanMetric(ones(8))
    hamiltonian = AdvancedHMC.Hamiltonian(metric, t_temp)
    integrator = AdvancedHMC.Leapfrog(HMC_STEP_SIZE)
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
        integrator, AdvancedHMC.FixedNSteps(best_L)))
    for m in 1:N_SMC
        samples, _ = AdvancedHMC.sample(MersenneTwister(101 * n_temp + m),
                                          hamiltonian, kernel,
                                          collect(@view resampled[m, :]),
                                          NUM_MCMC, AdvancedHMC.NoAdaptation();
                                          progress = false, verbose = false)
        particles[m, :] = samples[end]
    end

    λ = next_λ
    n_temp += 1
    n_temp > 30 && break
end

t_total = time() - t_start
println(@sprintf("\nSMC² done: %d tempering levels, %.1f s wall time", n_temp, t_total))

# Posterior summary
particles_constr = similar(particles)
for m in 1:N_SMC
    particles_constr[m, 1:6] = exp.(particles[m, 1:6])
    particles_constr[m, 7:8] = particles[m, 7:8]
end
emp_mean = vec(mean(particles_constr; dims = 1))
truth = [PARAM_SET_A.alpha, PARAM_SET_A.a, PARAM_SET_A.sigma_x,
         PARAM_SET_A.gamma, PARAM_SET_A.sigma_u, PARAM_SET_A.sigma_obs,
         INIT_STATE_A.x_0, INIT_STATE_A.u_0]
println("\nPosterior means vs truth:")
for (i, name) in enumerate([:α, :a, :σ_x, :γ, :σ_u, :σ_obs, :x_0, :u_0])
    rel_err = abs(emp_mean[i] - truth[i]) / max(abs(truth[i]), 1e-3)
    println(@sprintf("  %-6s truth=%.4f  mean=%.4f  rel.err=%.1f%%",
                      string(name), truth[i], emp_mean[i], 100 * rel_err))
end
