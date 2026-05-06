# Phase 6 — Per-window cost benchmark.
#
# Charter §15.7 step 23:
#   "Benchmark per-window cost on CPU and GPU against the JAX-native Python
#    baseline of ~1 second/window. Record the numbers in the package README;
#    this is the artefact that justifies the port to anyone reading it later."
#
# Run with:  julia --project=.. benchmarks/run_phase6_bench.jl
#
# What this measures:
#   1. Inner bootstrap PF cost per call (Phase 2), CPU only at this stage —
#      the GPU-end-to-end PF needs the AD-friendly buffer rewrite (Phase 6
#      follow-up; tracked in README).
#   2. Outer SMC² rejuvenation cost per tempering level (Phase 3), CPU.
#   3. Phase 2 kernel ops on CPU vs GPU, to confirm the GPU path saturates
#      under the actual workloads we ship.

using SMC2FC
using BenchmarkTools
using Random
using Statistics: mean
using LinearAlgebra
using CUDA

# ── Bench 1: Phase 2 kernel ops (CPU vs GPU) ─────────────────────────────────
println("="^70)
println("Bench 1 — Phase 2 kernel ops (CPU vs GPU)")
println("="^70)

K, n_st = 400, 6     # operating-point per CLAUDE.md (N_PF_PARTICLES=400)
particles_cpu = randn(K, n_st)
log_w_cpu     = randn(K)
sto_idx       = collect(1:n_st)

t_ess_cpu  = @belapsed compute_ess($log_w_cpu)
t_band_cpu = @belapsed silverman_bandwidth($particles_cpu, $sto_idx, $K, 1.0)
t_lkm_cpu  = @belapsed log_kernel_matrix($particles_cpu, $sto_idx,
                                           silverman_bandwidth($particles_cpu, $sto_idx, $K, 1.0))

println("  CPU compute_ess(K=$K)            : ", round(t_ess_cpu * 1e6, digits=1),  " µs")
println("  CPU silverman_bandwidth(K=$K)    : ", round(t_band_cpu * 1e6, digits=1), " µs")
println("  CPU log_kernel_matrix(K=$K, n_st=$n_st): ", round(t_lkm_cpu * 1e3, digits=2), " ms")

if CUDA.functional()
    particles_gpu = CuArray(particles_cpu)
    log_w_gpu     = CuArray(log_w_cpu)
    # Warm up to avoid measuring CUDA's first-call latency.
    compute_ess(log_w_gpu); CUDA.synchronize()
    silverman_bandwidth(particles_gpu, sto_idx, K, 1.0); CUDA.synchronize()
    h_gpu = silverman_bandwidth(particles_gpu, sto_idx, K, 1.0)
    log_kernel_matrix(particles_gpu, sto_idx, h_gpu); CUDA.synchronize()

    t_ess_gpu  = @belapsed (compute_ess($log_w_gpu); CUDA.synchronize())
    t_band_gpu = @belapsed (silverman_bandwidth($particles_gpu, $sto_idx, $K, 1.0); CUDA.synchronize())
    t_lkm_gpu  = @belapsed (log_kernel_matrix($particles_gpu, $sto_idx, $h_gpu); CUDA.synchronize())

    println("  GPU compute_ess(K=$K)            : ", round(t_ess_gpu * 1e6, digits=1),  " µs")
    println("  GPU silverman_bandwidth(K=$K)    : ", round(t_band_gpu * 1e6, digits=1), " µs")
    println("  GPU log_kernel_matrix(K=$K, n_st=$n_st): ", round(t_lkm_gpu * 1e3, digits=2), " ms")
else
    println("  (CUDA not functional — GPU rows skipped)")
end

# ── Bench 2: Phase 2 bootstrap PF, AR(1) model, K=400 / T=50 ────────────────
println()
println("="^70)
println("Bench 2 — Phase 2 bootstrap PF (per call, K=400 / T=50, CPU)")
println("="^70)

a_truth, b_truth, ρ_truth = 0.85, 0.4, 0.3
T_obs = 50
rng_data = MersenneTwister(0)
y = zeros(T_obs); x = zeros(T_obs)
x[1] = randn(rng_data); y[1] = x[1] + ρ_truth * randn(rng_data)
for k in 2:T_obs
    x[k] = a_truth * x[k-1] + b_truth * randn(rng_data)
    y[k] = x[k]              + ρ_truth * randn(rng_data)
end

function _propagate(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng_)
    a_, b_, ρ_ = params[1], params[2], params[3]
    return [a_ * y_old[1] + b_ * ξ[1]], 0.0
end
_diffusion(p) = [p[2]]
function _obs_log_weight(x_new, grid_obs, k, params)
    ρ_ = params[3]
    ν  = grid_obs[:y][k] - x_new[1]
    return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
end
_shard_init(t_off, p, exog, init) = init
_align_obs(args...) = Dict()

model = EstimationModel(
    name = "AR1_Bench", version = "phase6",
    n_states = 1, n_stochastic = 1, stochastic_indices = [1],
    state_bounds = [(-50.0, 50.0)],
    param_priors = [(:a, NormalPrior(0.0, 1.0)),
                    (:b, LogNormalPrior(0.0, 1.0)),
                    (:ρ, LogNormalPrior(0.0, 1.0))],
    init_state_priors = Tuple{Symbol,PriorType}[],
    frozen_params = Dict{Symbol,Float64}(),
    propagate_fn = _propagate, diffusion_fn = _diffusion,
    obs_log_weight_fn = _obs_log_weight,
    align_obs_fn = _align_obs, shard_init_fn = _shard_init,
)

priors_b = all_priors(model)
u_truth = [a_truth, log(b_truth), log(ρ_truth)]
cfg_inner = SMCConfig(n_pf_particles = 400, bandwidth_scale = 0.0, ot_max_weight = 0.0)
grid_obs_b = Dict(:y => y)
fixed_init_b = [0.0]
bench_rng = MersenneTwister(11)

# Warm-up
bootstrap_log_likelihood(model, u_truth, grid_obs_b, fixed_init_b, priors_b,
                          cfg_inner, bench_rng;
                          dt=1.0, t_steps=T_obs, window_start_bin=0)

t_pf = @belapsed begin
    bootstrap_log_likelihood($model, $u_truth, $grid_obs_b, $fixed_init_b, $priors_b,
                              $cfg_inner, MersenneTwister(11);
                              dt=1.0, t_steps=$T_obs, window_start_bin=0)
end
println("  bootstrap_log_likelihood (K=400, T=50): ", round(t_pf * 1e3, digits=2), " ms")

# ── Bench 3: outer SMC² window time on a Gaussian target (Phase 3) ──────────
println()
println("="^70)
println("Bench 3 — Outer SMC² per window (K=128 SMC, d=4, Gaussian target, CPU)")
println("="^70)

d = 4
μ_t = [1.5, -0.7, 2.1, 0.3]
loglik(u) = -0.5 * sum(abs2, u .- μ_t) / 0.5^2
priors_g = PriorType[NormalPrior(0.0, 1.0) for _ in 1:d]
cfg_g = SMCConfig(n_smc_particles = 128, target_ess_frac = 0.5,
                   num_mcmc_steps = 8, max_lambda_inc = 0.1,
                   hmc_step_size = 0.15, hmc_num_leapfrog = 8)

# Warm-up
run_smc_window(loglik, priors_g, cfg_g, MersenneTwister(0))

t_outer = @belapsed run_smc_window($loglik, $priors_g, $cfg_g, MersenneTwister(0))
println("  outer SMC² window (full run, prior → posterior): ",
        round(t_outer, digits=2), " s")

println()
println("Reference baseline (Python JAX-native, RTX 5090, FSA-v5 production cfg):")
println("  per-window cost ≈ 1 s   (charter §15.7 step 23)")
println()
println("Note: the AR(1) bench above runs the inner bootstrap PF in pure Julia;")
println("it does NOT yet exercise the GPU-resident PF path because the AD-")
println("compatible buffer rewrite is the Phase 6 follow-up (tracked in README).")
