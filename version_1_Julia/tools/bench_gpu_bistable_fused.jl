# Phase 6 follow-up #2.5 — fused KernelAbstractions kernel for true GPU
# saturation on the bistable bootstrap PF.
#
# Diagnosis recap (from `bench_gpu_bistable.jl`):
#   The framework's `bootstrap_log_likelihood` decomposes each PF step
#   into ~10 small CUDA kernel launches (randn fill, drift, diffusion,
#   obs weight, log-w accumulate, cumsum, gather, Liu-West shrink, ...).
#   At T = 432 steps that's ~4,320 launches per PF call. Each launch
#   has ~5–10 µs latency on CUDA. With K = 1M fp32 particles each kernel
#   runs in microseconds, so launch latency completely dominates over
#   compute and `nvidia-smi dmon` reports sm % ≈ 0.
#
# Fix: collapse the WHOLE per-particle PF trajectory into a single
# `KernelAbstractions.@kernel`. One thread = one particle. Each thread
# runs the full T-step loop inside the kernel body — no per-step launches,
# no host-side scheduling. Total launches per call: 1 (plus a tiny
# logsumexp at the end).
#
# Trade-off: this kernel runs WITHOUT inner resampling — pure bootstrap
# log-likelihood estimator. For short horizons (T ≤ ~500) and large K
# (≥ 10⁵) the estimator's variance is acceptable. For longer horizons
# you'd switch back to the framework's multi-launch path (it does
# Liu-West + resampling at every observed step).
#
# This bench is the proof-of-concept that the fused-kernel pattern
# saturates the SMs. The framework integration (per-step resample fused
# in via batched scan) is a separate follow-up.
#
# Run from version_1_Julia/:
#   julia --threads auto --project=. tools/bench_gpu_bistable_fused.jl
#
# Watch the GPU during the timed call (separate terminal):
#   nvidia-smi dmon -s u -c 30 -d 1

using Random
using Statistics: mean
using Printf
using CUDA
using KernelAbstractions
using BenchmarkTools
using LogExpFunctions: logsumexp

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))

using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A, simulate_em

if !CUDA.functional()
    println("CUDA not functional — aborting.")
    exit(1)
end

# ── The fused per-particle PF kernel ───────────────────────────────────────
# Each thread index `i` runs particle i's full T-step trajectory:
#   - Initial state (init + σ √dt · ξ_init)
#   - For k in 1:T_steps:
#       drift_x = α x (a² − x²) + u
#       drift_u = -γ (u − u_target(t))
#       x_new = x + dt drift_x + sx_sd √dt · ξ_x[k]
#       u_new = u + dt drift_u + su_sd √dt · ξ_u[k]
#       log_w += -½ ((y_obs[k] − x_new)/σ_obs)² − log σ_obs − ½ log 2π
#   - Write log_w to log_w_out[i].
#
# All inputs are scalar params + 1-D arrays (obs sequence, noise grids,
# intervention threshold). The compute per thread is ~T_steps · 20 ops.
# At K = 1M threads × T = 432 × 20 ≈ 9 G fp32 ops per call, well within
# the 5090's per-launch sweet spot.

@kernel function bootstrap_pf_kernel!(
    log_w_out,         # (K,)               output log-weights
    x_init, u_init,                          # scalars
    α, a_param, γ,
    sx_sd, su_sd, sigma_obs,
    dt, sqrt_dt,
    T_i_threshold, u_on,                     # schedule scalars
    obs_seq,           # (T,)                observation sequence
    noise_x,           # (K, T+1)            x-channel noise per particle, per step
    noise_u,           # (K, T+1)            u-channel noise
    T_steps::Int,
)
    i = @index(Global, Linear)

    # Initial state: tight Gaussian around init
    x = x_init + sx_sd * sqrt_dt * noise_x[i, 1]
    u = u_init + su_sd * sqrt_dt * noise_u[i, 1]

    log_w = zero(eltype(log_w_out))
    half_log_2pi = oftype(log_w, 0.9189385332046727)

    @inbounds for k in 1:T_steps
        # Schedule (piecewise constant; each thread evaluates same value)
        t = (k - 1) * dt
        u_tgt = t < T_i_threshold ? zero(t) : u_on

        # Drift
        drift_x = α * x * (a_param * a_param - x * x) + u
        drift_u = -γ * (u - u_tgt)

        # Euler-Maruyama step
        x = x + dt * drift_x + sx_sd * sqrt_dt * noise_x[i, k + 1]
        u = u + dt * drift_u + su_sd * sqrt_dt * noise_u[i, k + 1]

        # Gaussian obs log-weight on x channel
        ν = obs_seq[k] - x
        log_w += -oftype(log_w, 0.5) * (ν / sigma_obs)^2 -
                  log(sigma_obs) - half_log_2pi
    end

    log_w_out[i] = log_w
end

# ── Driver ─────────────────────────────────────────────────────────────────

const K_GPU = 1_000_000   # 1 million particles — exercises the full SM array
const T_steps = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))

println("="^70)
println("Phase 6 follow-up #2.5 — fused KernelAbstractions PF on bistable")
println("="^70)
println("K = $K_GPU, T = $T_steps, fp32, single fused kernel per call")
println()

# Truth params + simulate data
params = PARAM_SET_A
α    = Float32(params.alpha)
a_p  = Float32(params.a)
γ    = Float32(params.gamma)
sx_sd = Float32(sqrt(2.0 * params.sigma_x))
su_sd = Float32(sqrt(2.0 * params.sigma_u))
sigma_obs = Float32(params.sigma_obs)
dt   = Float32(EXOGENOUS_A.dt)
sqrt_dt = sqrt(dt)
T_i_threshold = Float32(EXOGENOUS_A.T_intervention)
u_on = Float32(EXOGENOUS_A.u_on)
x_init = Float32(INIT_STATE_A.x_0)
u_init = Float32(INIT_STATE_A.u_0)

println("--- truth parameters ---")
println(@sprintf("  α = %.3f, a = %.3f, γ = %.3f", α, a_p, γ))
println(@sprintf("  σ_x = %.3f, σ_u = %.3f, σ_obs = %.3f",
                  params.sigma_x, params.sigma_u, params.sigma_obs))
println()

data = simulate_em(seed = 7)
obs_cpu = Float32.(data.obs)
println(@sprintf("--- simulated data ready: %d obs in [%.2f, %.2f] ---",
                  length(obs_cpu), minimum(obs_cpu), maximum(obs_cpu)))
println()

# Allocate device buffers
println("--- allocating GPU buffers ---")
free_pre, total = CUDA.Mem.info()
println(@sprintf("  VRAM before: %.2f / %.1f GiB free", free_pre / 2^30, total / 2^30))

obs_gpu     = CuArray(obs_cpu)
noise_x_gpu = CuArray(randn(Float32, K_GPU, T_steps + 1))
noise_u_gpu = CuArray(randn(Float32, K_GPU, T_steps + 1))
log_w_gpu   = CUDA.zeros(Float32, K_GPU)

free_post, _ = CUDA.Mem.info()
alloc_gib    = (free_pre - free_post) / 2^30
println(@sprintf("  Allocated: %.2f GiB (noise grids dominate)", alloc_gib))
println()

# Build the kernel for the CUDA backend
backend = CUDABackend()
kernel  = bootstrap_pf_kernel!(backend, 256)   # 256 threads/block

# Warm up (JIT compile + cache)
println("--- warm-up (JIT compile) ---")
kernel(log_w_gpu, x_init, u_init, α, a_p, γ,
        sx_sd, su_sd, sigma_obs, dt, sqrt_dt,
        T_i_threshold, u_on, obs_gpu, noise_x_gpu, noise_u_gpu, T_steps;
        ndrange = K_GPU)
KernelAbstractions.synchronize(backend)
ll_warm = logsumexp(Array(log_w_gpu)) - log(Float32(K_GPU))
println(@sprintf("  warm-up log p(y|truth): %.4f", ll_warm))
println()

# Single-call timed run
println("--- timed run (single call) ---")
t_gpu = @elapsed begin
    kernel(log_w_gpu, x_init, u_init, α, a_p, γ,
            sx_sd, su_sd, sigma_obs, dt, sqrt_dt,
            T_i_threshold, u_on, obs_gpu, noise_x_gpu, noise_u_gpu, T_steps;
            ndrange = K_GPU)
    KernelAbstractions.synchronize(backend)
end
println(@sprintf("  GPU wall time: %.4f s (single call)", t_gpu))

# Sustained timed run — call the kernel N_REPEATS times back-to-back so the
# total wall time is long enough for nvidia-smi's 1-sec sampling to catch
# the active period. Otherwise the 25 ms single-call run completes between
# nvidia-smi samples and the dashboard reports sm % near 0 (false negative).
const N_REPEATS = 5_000
println("\n--- sustained timed run ($N_REPEATS back-to-back calls) ---")
println("    Watch sm % in `nvidia-smi dmon -s u -c 30 -d 1` during this run.")
t_sustained = @elapsed begin
    for _ in 1:N_REPEATS
        kernel(log_w_gpu, x_init, u_init, α, a_p, γ,
                sx_sd, su_sd, sigma_obs, dt, sqrt_dt,
                T_i_threshold, u_on, obs_gpu, noise_x_gpu, noise_u_gpu, T_steps;
                ndrange = K_GPU)
    end
    KernelAbstractions.synchronize(backend)
end
println(@sprintf("  Sustained: %d calls in %.2f s = %.1f ms/call",
                  N_REPEATS, t_sustained, t_sustained * 1000 / N_REPEATS))

ll_gpu = logsumexp(Array(log_w_gpu)) - log(Float32(K_GPU))
println(@sprintf("  log p(y|truth) = %.4f  (logsumexp - log K)", ll_gpu))
println()

# CPU baseline for sanity (small K)
const K_CPU = 5_000
println("--- CPU baseline for comparison (Float64, K = $K_CPU) ---")
function cpu_bootstrap_at_truth()
    rng = MersenneTwister(11)
    log_w = zeros(Float64, K_CPU)
    sx_sd64 = sqrt(2.0 * params.sigma_x)
    su_sd64 = sqrt(2.0 * params.sigma_u)
    half_log_2pi = 0.9189385332046727
    @inbounds for i in 1:K_CPU
        x = INIT_STATE_A.x_0 + sx_sd64 * sqrt(EXOGENOUS_A.dt) * randn(rng)
        u = INIT_STATE_A.u_0 + su_sd64 * sqrt(EXOGENOUS_A.dt) * randn(rng)
        s = 0.0
        for k in 1:T_steps
            t = (k - 1) * EXOGENOUS_A.dt
            u_tgt = t < EXOGENOUS_A.T_intervention ? 0.0 : EXOGENOUS_A.u_on
            x += EXOGENOUS_A.dt * (params.alpha * x * (params.a^2 - x^2) + u) +
                  sx_sd64 * sqrt(EXOGENOUS_A.dt) * randn(rng)
            u += EXOGENOUS_A.dt * (-params.gamma * (u - u_tgt)) +
                  su_sd64 * sqrt(EXOGENOUS_A.dt) * randn(rng)
            ν = data.obs[k] - x
            s += -0.5 * (ν / params.sigma_obs)^2 -
                  log(params.sigma_obs) - half_log_2pi
        end
        log_w[i] = s
    end
    return logsumexp(log_w) - log(K_CPU)
end

t_cpu = @elapsed ll_cpu = cpu_bootstrap_at_truth()
println(@sprintf("  CPU wall time: %.4f s", t_cpu))
println(@sprintf("  CPU log p(y|truth) = %.4f", ll_cpu))
println()

# ── Throughput summary ─────────────────────────────────────────────────────
particle_steps_gpu = K_GPU * T_steps
particle_steps_cpu = K_CPU * T_steps
println("="^70)
println("Throughput")
println("="^70)
println(@sprintf("  CPU 5k @ Float64    : %.2e particle-steps/s   (%.3f s)",
                  particle_steps_cpu / t_cpu, t_cpu))
println(@sprintf("  GPU 1M @ Float32    : %.2e particle-steps/s   (%.3f s)",
                  particle_steps_gpu / t_gpu, t_gpu))
println(@sprintf("  GPU/CPU speedup     : %.0fx",
                  (particle_steps_gpu / t_gpu) / (particle_steps_cpu / t_cpu)))
println()

# Estimator-quality check: GPU and CPU should agree to within PF MC noise.
println(@sprintf("  GPU log p(y|truth)   : %.4f", ll_gpu))
println(@sprintf("  CPU log p(y|truth)   : %.4f", ll_cpu))
println(@sprintf("  Δ (GPU − CPU)        : %.4f", ll_gpu - ll_cpu))
println()

println("To verify SM saturation, in another terminal:")
println("  nvidia-smi dmon -s u -c 30 -d 1")
println("  nvtop")
println()
println("With this fused kernel, expect sm% > 80 % during the ~1 second")
println("active window. The launch-bound regime is gone because the")
println("ENTIRE 432-step trajectory runs inside one kernel call.")
