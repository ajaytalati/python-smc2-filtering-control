# GPU saturation demo on the bistable model.
#
# Runs the framework's bootstrap PF at TRUTH params on the full 432-step
# trajectory, with K=100,000 fp32 particles resident on `CUDA.CuArray`.
# This exercises the Phase 6 follow-up #2 path (`propagate_batch_fn` +
# `obs_log_weight_batch_fn` on `Float32` CuArrays) end-to-end.
#
# Per charter §14, the inner PF is the GPU-resident piece; the outer SMC²
# stays on CPU. This bench is INNER-only — no HMC, no AD, no outer SMC². It
# answers "does the GPU saturate at fp32 with K=100k?".
#
# To watch the GPU during the run, in another terminal:
#   nvidia-smi dmon -s u -c 30
#
# Run from version_1_Julia/:
#   julia --threads auto --project=. tools/bench_gpu_bistable.jl

using Random
using Statistics: mean
using Printf
using CUDA
using BenchmarkTools

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))

using SMC2FC
using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A,
                            simulate_em, build_estimation_model

if !CUDA.functional()
    println("CUDA not functional — aborting.")
    exit(1)
end

# Print device info
dev = CUDA.device()
println("="^70)
println("GPU bench — bistable bootstrap PF, full trajectory, fp32")
println("="^70)
println("GPU: ", CUDA.name(dev))
println("CUDA driver: ", CUDA.driver_version())
println("CUDA runtime: ", CUDA.runtime_version())
free, total = CUDA.Mem.info()
println(@sprintf("VRAM: %.1f / %.1f GiB free", free / 2^30, total / 2^30))
println()

# ── Data: full 432-step trajectory at truth params ────────────────────────
T_steps = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))
data = simulate_em(seed = 7)

# Build EstimationModel + truth params in unconstrained space.
model  = build_estimation_model()
priors = SMC2FC.all_priors(model)
u_truth = [
    log(PARAM_SET_A.alpha),
    log(PARAM_SET_A.a),
    log(PARAM_SET_A.sigma_x),
    log(PARAM_SET_A.gamma),
    log(PARAM_SET_A.sigma_u),
    log(PARAM_SET_A.sigma_obs),
    INIT_STATE_A.x_0,                      # NormalPrior identity
    INIT_STATE_A.u_0,
]

grid_obs = Dict{Symbol,Any}(
    :obs_value      => Float64.(data.obs),
    :obs_present    => ones(Float64, T_steps),
    :has_any_obs    => ones(Float64, T_steps),
    :T_intervention => EXOGENOUS_A.T_intervention,
    :u_on           => EXOGENOUS_A.u_on,
)
fixed_init = [INIT_STATE_A.x_0, INIT_STATE_A.u_0]

# ── CPU baseline (Float64, small K) ─────────────────────────────────────────
cfg_cpu = SMCConfig(n_pf_particles = 5_000,
                     bandwidth_scale = 0.0,
                     ot_max_weight = 0.0)
println("CPU Float64 baseline: K=$(cfg_cpu.n_pf_particles), T=$T_steps")
ll_cpu = bootstrap_log_likelihood(model, u_truth, grid_obs, fixed_init,
                                    priors, cfg_cpu, MersenneTwister(11);
                                    dt = EXOGENOUS_A.dt, t_steps = T_steps,
                                    window_start_bin = 0)
t_cpu = @elapsed bootstrap_log_likelihood(model, u_truth, grid_obs, fixed_init,
                                            priors, cfg_cpu, MersenneTwister(11);
                                            dt = EXOGENOUS_A.dt, t_steps = T_steps,
                                            window_start_bin = 0)
println(@sprintf("  log p(y|θ) + log p(u): %.3f", ll_cpu))
println(@sprintf("  CPU wall time: %.2f s", t_cpu))
println()

# ── GPU run (Float32, huge K) ─────────────────────────────────────────────
# K = 1,000,000 fp32 particles to amortise CUDA kernel-launch overhead.
# At T=432 obs steps and ~10 small kernels per step (drift + diffusion +
# gather + Liu-West blend + cumsum + ...) the per-step compute is sub-
# millisecond at K=100k → launch latency dominates. Pushing to K=1M makes
# the per-step compute 10× larger so kernels actually saturate the SMs.
const K_GPU = 1_000_000
println("GPU Float32 saturation: K=$K_GPU, T=$T_steps")

cfg_gpu = SMCConfig(n_pf_particles = K_GPU,
                     bandwidth_scale = 0.0,
                     ot_max_weight = 0.0)

bufs_gpu = BootstrapBuffers{Float32}(K_GPU, model.n_states; backend = CuArray)
println("GPU buffers allocated: $(K_GPU)×$(model.n_states) Float32 CuArray")
free_after, total_post = CUDA.Mem.info()
println(@sprintf("VRAM after alloc: %.2f / %.1f GiB used",
                  (total_post - free_after) / 2^30, total_post / 2^30))

# Warm up (compilation + kernel cache).
println("\nWarm-up call ...")
CUDA.synchronize()
ll_gpu_warm = bootstrap_log_likelihood(model, Float32.(u_truth),
                                         grid_obs, Float32.(fixed_init),
                                         priors, cfg_gpu, MersenneTwister(11);
                                         dt = EXOGENOUS_A.dt, t_steps = T_steps,
                                         window_start_bin = 0,
                                         buffers = bufs_gpu)
CUDA.synchronize()
println(@sprintf("  warm-up log p(y|θ) + log p(u): %.3f", ll_gpu_warm))

# Timed run
println("\nTimed run ...")
t_gpu = @elapsed begin
    bootstrap_log_likelihood(model, Float32.(u_truth),
                              grid_obs, Float32.(fixed_init),
                              priors, cfg_gpu, MersenneTwister(13);
                              dt = EXOGENOUS_A.dt, t_steps = T_steps,
                              window_start_bin = 0,
                              buffers = bufs_gpu)
    CUDA.synchronize()
end
println(@sprintf("  GPU wall time: %.2f s", t_gpu))

free_run, total_run = CUDA.Mem.info()
println(@sprintf("VRAM peak during run: ~%.2f GiB",
                  (total_run - free_run) / 2^30))

# ── Throughput summary ─────────────────────────────────────────────────────
particle_steps_cpu = cfg_cpu.n_pf_particles * T_steps
particle_steps_gpu = K_GPU                  * T_steps
println()
println("="^70)
println("Throughput (particle-step ops per second)")
println("="^70)
println(@sprintf("  CPU 5k @ Float64 : %.2e ops/s   (%.2f s for %.2e ops)",
                  particle_steps_cpu / t_cpu, t_cpu, particle_steps_cpu))
println(@sprintf("  GPU 100k @ Float32: %.2e ops/s   (%.2f s for %.2e ops)",
                  particle_steps_gpu / t_gpu, t_gpu, particle_steps_gpu))
println(@sprintf("  GPU/CPU speedup at this scale: %.1f×",
                  (particle_steps_gpu / t_gpu) / (particle_steps_cpu / t_cpu)))
println()
println("To verify GPU saturation, run during the timed call:")
println("  nvidia-smi dmon -s u -c 30")
println("  nvtop")
