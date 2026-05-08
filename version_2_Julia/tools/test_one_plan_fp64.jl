#!/usr/bin/env julia
# Single-plan diagnostic: run ONE controller plan over the full T=14 d
# horizon with the fp64-promoted drift, init state at the canonical
# (B=0.05, F=0.30, A=0.10), TRUTH params. Reports the decoded per-bin
# Φ schedule sampled at every other day.
#
# Hypothesis being tested: fp32 cancellation in the v2 G1-reparametrised
# drift was distorting the cost surface and displacing the global optimum
# from recovery→overload to flat-low. With fp64 substep arithmetic the
# recovery→overload pattern should re-emerge.

ENV["FSA_STEP_MINUTES"] = "15"

using CUDA, Random, Statistics, Printf, LinearAlgebra
using SMC2FC: run_tempered_smc_gpu

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn
using .FSAHighRes.Dynamics: TRUTH_PARAMS

# ── Config (light — fp64 substep is ~64× slower than fp32 on consumer 5090,
# so we drop n_smc / n_inner / num_mcmc to keep wall time under a few minutes
# while still letting the controller find a clear posterior-mean schedule) ──
const D            = 8
const N_SMC        = 64
const N_INNER      = 16
const N_ANCHORS    = 8
const M_MAX        = N_SMC * (1 + 2 * N_ANCHORS)
const SIGMA_PRIOR  = 1.5
const TARGET_NATS  = 8.0
const HMC_STEP     = 0.2
const HMC_LEAP     = 8
const NUM_MCMC     = 3
const MAX_LEVELS   = 15

const T_DAYS       = 14
const BINS_PER_DAY = 96
const N_STEPS      = T_DAYS * BINS_PER_DAY
const DT           = 1.0 / BINS_PER_DAY
const PHI_MAX      = 3.0
const PHI_DEFAULT  = 1.0


function decode_per_bin_phi(theta::Vector{Float64}, n_steps::Int, n_anchors::Int,
                            dt::Float64, phi_max::Float64, c_Phi::Float64)
    T_total = n_steps * dt
    t_grid = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length = n_anchors))
    σ_rbf = T_total / n_anchors
    out = zeros(Float64, n_steps)
    for k in 1:n_steps
        raw = c_Phi
        for j in 1:n_anchors
            v = exp(-0.5 * ((t_grid[k] - anchors[j]) / σ_rbf)^2)
            raw += theta[j] * v
        end
        out[k] = phi_max / (1.0 + exp(-raw))
    end
    return out
end


target = FSAControlGPUTarget(
    n_inner = N_INNER, M_max = M_MAX, n_steps = N_STEPS,
    n_anchors = N_ANCHORS, n_substeps = 4, dt = DT,
    F_max = 0.40, Phi_max = PHI_MAX, Phi_default = PHI_DEFAULT,
    lam_F = 1.0,
    sigma_prior = SIGMA_PRIOR,
    params = TRUTH_PARAMS,
    init_state = [0.05, 0.30, 0.10],
    noise_seed = 42,
)

@info "GPU device: $(CUDA.name(CUDA.device()))"
@info "Cost: J(θ) = -∫A dt + λ_F · ∫max(F-F_max, 0)² dt   (Eq 37)"
@info "Drift inside substep promoted to fp64. State / accumulators stay fp32."
@info "Running ONE plan at T=14 d, init state (B=0.05, F=0.30, A=0.10)..."

t0 = time()
log_density_fn = make_log_density_fn(target)
rng = MersenneTwister(123)
posterior, n_temp, β_max = run_tempered_smc_gpu(
    log_density_fn, M_MAX, N_SMC, D,
    0.0, SIGMA_PRIOR, rng;
    target_nats = TARGET_NATS,
    target_ess_frac = 0.5,
    max_lambda_inc = 0.20,
    max_temp_levels = MAX_LEVELS,
    num_mcmc_steps = NUM_MCMC,
    hmc_step_size = HMC_STEP,
    hmc_num_leapfrog = HMC_LEAP,
    chees_L_candidates = [16, 32, 64, 128, 256],
    h_fd = 1e-4, calib_n = 64,
    verbose = false,
)
t = time() - t0

theta_post_mean = vec(mean(posterior; dims = 1))
c_Phi = log((PHI_DEFAULT / PHI_MAX) / (1.0 - PHI_DEFAULT / PHI_MAX))
phi_per_bin = decode_per_bin_phi(theta_post_mean, N_STEPS, N_ANCHORS, DT, PHI_MAX, c_Phi)

println("="^72)
@printf("Result: %d tempering levels, β_max = %.3f, %.1fs wall\n",
        n_temp, β_max, t)
println("-"^72)
@printf("Posterior mean θ = %s\n",
        "[" * join([@sprintf("%+.3f", v) for v in theta_post_mean], ", ") * "]")
println()
@printf("Decoded Φ schedule sampled at days 0, 1, 2, ..., 14:\n")
for d in 0:T_DAYS
    k = clamp(d * BINS_PER_DAY + 1, 1, N_STEPS)
    @printf("  day %2d  Φ = %.3f\n", d, phi_per_bin[k])
end
println()
@printf("Φ summary:  min = %.3f   mean = %.3f   max = %.3f\n",
        minimum(phi_per_bin), mean(phi_per_bin), maximum(phi_per_bin))
println()
println("Pattern check:")
println("  spec  expected: low Φ (~0.2) days 1-5, ramping to ~1.2 by day 14")
println("  v2 fp32 (prior): flat Φ ~0.20 across all days")
println("  this run (fp64 substep): see above")
