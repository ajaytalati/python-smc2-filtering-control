#!/usr/bin/env julia
# GPU profiling for the Julia FSA bench. Reports for each kernel:
#   - kernel wall time (median over warm calls)
#   - thread count (ndrange)
#   - achieved utilisation vs RTX 5090 fp32 peak (104.8 TFLOPS @ boost)
#   - GPU memory used / available
#   - Per-call host-sync count (CPU↔GPU round-trips)
#
# Two profile modes:
#   --filter    : profile gpu_log_density_batched (filter PF kernel)
#   --controller: profile gpu_cost_log_density_batched (controller cost kernel)
#   --both      : default — runs both
#
# Usage:
#   julia --project=. tools/profile_gpu.jl [--filter|--controller|--both] [--verbose]

ENV["FSA_STEP_MINUTES"] = "15"

using CUDA
using Statistics
using Printf
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUPF: FSAGPUTargetBatched, gpu_log_density_batched,
                          update_window_obs!
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched
using .FSAHighRes.Estimation: get_init_theta, build_estimation_model,
                                PARAM_NAMES, PARAM_PRIOR_CONFIG, align_obs_fn
using .FSAHighRes.Dynamics: TRUTH_PARAMS
using .FSAHighRes.Plant: StepwisePlant, advance!
using SMC2FC: LogNormalPrior, NormalPrior, constrained_to_unconstrained


# ── 5090 spec ────────────────────────────────────────────────────────────
const FP32_PEAK_TFLOPS = 104.8       # RTX 5090 fp32 peak (with boost)


function bytes_GB(b)
    return Float64(b) / 1e9
end

function print_gpu_state(label::AbstractString)
    dev = CUDA.device()
    free, total = CUDA.Mem.info()
    used = total - free
    util = util_pct = ""
    try
        out = read(`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`, String)
        util = strip(out)
    catch
        util = "?"
    end
    @printf("  [%s] GPU mem: %.2f / %.2f GB used (%.1f%%);  util: %s%%\n",
            label, bytes_GB(used), bytes_GB(total),
            100.0 * Float64(used) / Float64(total), util)
end


# ── Filter PF profile ────────────────────────────────────────────────────

function profile_filter()
    println("="^72)
    println("PROFILE: filter PF kernel (gpu_log_density_batched)")
    println("="^72)
    print_gpu_state("at start")

    BINS_PER_DAY = 96
    WINDOW_BINS  = BINS_PER_DAY
    N_smc = 32
    K_per_chain = 400
    d = 30
    M_max = N_smc * (1 + 2 * d)

    @printf("Config: N_smc=%d, K_per_chain=%d, M_max=%d (FD-batched), T_steps=%d\n",
            N_smc, K_per_chain, M_max, WINDOW_BINS)

    target = FSAGPUTargetBatched(K_per_chain=K_per_chain, M_max=M_max,
                                  T_steps=WINDOW_BINS, dt=1/96, R=4)
    print_gpu_state("after target build")

    # Synthetic obs window
    rng = MersenneTwister(0)
    obs_data = Dict{Symbol,Any}(
        :obs_HR     => Dict(:t_idx => collect(Int32, 0:WINDOW_BINS-1),
                            :obs_value => Float32.(60.0 .+ 2.0 .* randn(rng, WINDOW_BINS))),
        :obs_stress => Dict(:t_idx => collect(Int32, 0:WINDOW_BINS-1),
                            :obs_value => Float32.(30.0 .+ 4.0 .* randn(rng, WINDOW_BINS))),
        :obs_steps  => Dict(:t_idx => collect(Int32, 0:WINDOW_BINS-1),
                            :obs_value => Float32.(exp.(5.5 .+ 0.5 .* randn(rng, WINDOW_BINS)))),
        :obs_sleep  => Dict(:t_idx => collect(Int32, 0:WINDOW_BINS-1),
                            :sleep_label => Int32.(rand(rng, 0:1, WINDOW_BINS))),
        :Phi        => Dict(:Phi_value => Float32.(ones(WINDOW_BINS))),
        :C          => Dict(:C_value   => Float32.(cos.(2π .* (0:WINDOW_BINS-1) ./ BINS_PER_DAY))),
    )
    grid_obs = align_obs_fn(obs_data, WINDOW_BINS, 1/96)
    update_window_obs!(target, grid_obs; B_init=0.05, F_init=0.30, A_init=0.10)

    # Build M_max × d theta matrix.
    em = build_estimation_model()
    init_theta = Float64.(get_init_theta())
    priors = [last(p) for p in em.param_priors]
    u0 = Float64.(constrained_to_unconstrained(init_theta, priors))
    U = repeat(reshape(u0, 1, :), M_max, 1)

    # Warm up.
    @info "warming up..."
    _ = gpu_log_density_batched(target, U)
    CUDA.synchronize()

    # Time many calls.
    n_repeat = 5
    @info "timing $(n_repeat) calls..."
    times = Float64[]
    for _ in 1:n_repeat
        CUDA.synchronize()
        t0 = time_ns()
        _ = gpu_log_density_batched(target, U)
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end

    t_med = median(times)
    n_threads = M_max * K_per_chain
    @printf("\nResults:\n")
    @printf("  median time per call : %.4f s\n", t_med)
    @printf("  threads in flight    : %d (%d chains × %d state particles)\n",
            n_threads, M_max, K_per_chain)
    # Each thread: T_steps × (drift ~30 ops + 3-channel fusion ~80 ops + chol ~30 ops + obs ll ~40 ops)
    flops_per_thread = WINDOW_BINS * (30 * 4 + 80 + 30 + 40)
    flops = Float64(n_threads) * Float64(flops_per_thread)
    tflops_eff = flops / t_med / 1e12
    @printf("  approx FLOPs / call  : %.2e\n", flops)
    @printf("  effective TFLOPS     : %.2f / %.1f peak  =  %.1f%% util\n",
            tflops_eff, FP32_PEAK_TFLOPS,
            100 * tflops_eff / FP32_PEAK_TFLOPS)
    print_gpu_state("after timing")
    @printf("  host syncs per call  : 1 (logsumexp on CPU)\n")
    println()
    return t_med
end


# ── Controller cost profile ─────────────────────────────────────────────

function profile_controller()
    println("="^72)
    println("PROFILE: controller cost kernel (gpu_cost_log_density_batched)")
    println("="^72)
    print_gpu_state("at start")

    n_inner = 32
    n_smc = 1024
    n_anchors = 8
    n_steps = 14 * 96       # full T=14d horizon at h=15min
    M_max = n_smc * (1 + 2 * n_anchors)

    @printf("Config: n_smc=%d, n_inner=%d, n_anchors=%d, M_max=%d, n_steps=%d\n",
            n_smc, n_inner, n_anchors, M_max, n_steps)

    target = FSAControlGPUTarget(
        n_inner=n_inner, M_max=M_max, n_steps=n_steps, n_anchors=n_anchors,
        n_substeps=4, dt=1/96,
        F_max=0.40, Phi_max=3.0, Phi_default=1.0,
        params=TRUTH_PARAMS, init_state=[0.05, 0.30, 0.10],
    )
    print_gpu_state("after target build")

    U = randn(M_max, n_anchors)

    @info "warming up..."
    _ = gpu_cost_log_density_batched(target, U)
    CUDA.synchronize()

    n_repeat = 3
    @info "timing $(n_repeat) calls..."
    times = Float64[]
    for _ in 1:n_repeat
        CUDA.synchronize()
        t0 = time_ns()
        _ = gpu_cost_log_density_batched(target, U)
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end

    t_med = median(times)
    n_threads = M_max * n_inner
    @printf("\nResults:\n")
    @printf("  median time per call : %.4f s\n", t_med)
    @printf("  threads in flight    : %d (%d chains × %d trials)\n",
            n_threads, M_max, n_inner)
    # Each thread: n_steps × (RBF decode ~10 ops × n_anchors + drift ~30 ops × n_substeps + diffusion ~12 + cost accum ~6)
    flops_per_thread = n_steps * (10 * n_anchors + 30 * 4 + 12 + 6)
    flops = Float64(n_threads) * Float64(flops_per_thread)
    tflops_eff = flops / t_med / 1e12
    @printf("  approx FLOPs / call  : %.2e\n", flops)
    @printf("  effective TFLOPS     : %.2f / %.1f peak  =  %.1f%% util\n",
            tflops_eff, FP32_PEAK_TFLOPS,
            100 * tflops_eff / FP32_PEAK_TFLOPS)
    print_gpu_state("after timing")
    @printf("  host syncs per call  : 1 (per-chain sum on CPU)\n")
    println()
    return t_med
end


function main(argv::Vector{String} = String[])
    mode = "both"
    if length(argv) > 0
        if argv[1] == "--filter"
            mode = "filter"
        elseif argv[1] == "--controller"
            mode = "controller"
        elseif argv[1] == "--both"
            mode = "both"
        end
    end

    println("\nGPU device: $(CUDA.name(CUDA.device()))")
    free, total = CUDA.Mem.info()
    @printf("Total memory: %.2f GB, free: %.2f GB\n\n", bytes_GB(total), bytes_GB(free))

    if mode == "filter" || mode == "both"
        profile_filter()
    end
    if mode == "controller" || mode == "both"
        profile_controller()
    end

    println("\nNotes:")
    println("  - 'effective TFLOPS' is approximate (counts only the dominant fp ops in the inner loop).")
    println("  - 'util %' compares to RTX 5090's 104.8 TFLOPS fp32 peak. Saturating to >50% requires:")
    println("      • enough threads in flight (>= ~1M for the 5090's 21504 cores × 50× oversubscription),")
    println("      • compute-bound (not memory-bound) inner loop,")
    println("      • no host syncs blocking the stream.")
    println("  - To increase utilisation further: bump n_inner (controller) or K_per_chain (filter)")
    println("    until GPU memory is filled — currently we use ~4-8 GB of 32 GB.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
