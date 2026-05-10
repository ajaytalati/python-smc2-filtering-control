#!/usr/bin/env julia
# GPU profiling for the FSA v1.5 Julia bench. Direct port of
# `version_2_Julia/tools/profile_gpu.jl` adapted for v1.5's smaller
# API surface:
#
#   - FSAGPUTarget          (10 params, 3-channel direct-Gaussian obs)
#   - FSAv1ControlGPUTarget (single Φ, RBF schedule, no F-violation barrier)
#   - BINS_PER_DAY = 24 by default (60-min step, vs v2's 15-min)
#
# Two profile modes:
#   --filter     : profile gpu_log_density (filter PF kernel)
#   --controller : profile gpu_cost_log_density_batched (controller cost kernel)
#   --both       : default — runs both
#
# Reports for each kernel:
#   - kernel wall time (median over warm calls)
#   - threads in flight (chains × inner-loop length)
#   - approximate FLOPs / call (hand-counted dominant fp ops)
#   - effective TFLOPS vs RTX 5090 fp32 peak (104.8)
#   - GPU memory used / available
#   - host syncs per call (CUDA.synchronize round-trips)
#
# Usage:
#   julia --project=. tools/profile_gpu.jl [--filter|--controller|--both] [--verbose]

ENV["FSA_STEP_MINUTES"] = get(ENV, "FSA_STEP_MINUTES", "60")

using CUDA
using Statistics
using Printf
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS,
                                INIT_STATE, params_v15_to_v1_nt
using .FSAHighRes.Estimation: PARAM_NAMES
using .FSAHighRes.GPUPF: FSAGPUTarget, gpu_log_density
using .FSAHighRes.GPUControl: FSAv1ControlGPUTarget, gpu_cost_log_density_batched


# ── 5090 spec ────────────────────────────────────────────────────────────
const FP32_PEAK_TFLOPS = 104.8       # boost-clock fp32 peak per NVIDIA spec


bytes_GB(b) = Float64(b) / 1e9

function print_gpu_state(label::AbstractString)
    free, total = CUDA.Mem.info()
    used = total - free
    util = "?"
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

function profile_filter(; n_smc::Int = 32, K_per_chain::Int = 200,
                          n_repeat::Int = 5, verbose::Bool = false)
    println("="^72)
    println("PROFILE: filter PF kernel (gpu_log_density)")
    println("="^72)
    print_gpu_state("at start")

    WINDOW_BINS = BINS_PER_DAY    # 1-day window, matches the bench
    @printf("Config: n_smc=%d, K_per_chain=%d, T_steps=%d (BINS_PER_DAY=%d)\n",
            n_smc, K_per_chain, WINDOW_BINS, BINS_PER_DAY)

    target = FSAGPUTarget(K_per_chain = K_per_chain,
                           M_max       = n_smc,
                           T_steps     = WINDOW_BINS,
                           R           = 4,
                           dt          = DT_BIN_DAYS)
    print_gpu_state("after target build")

    # Synthetic 1-day obs window — direct Gaussian on each latent at
    # truth + small noise. Matches v1.5's obs schema (3 channels).
    rng = MersenneTwister(0)
    grid_obs = (
        Phi_seq = Float32.(ones(WINDOW_BINS)),
        obs_B   = Float32.(INIT_STATE.B .+ 0.005 .* randn(rng, WINDOW_BINS)),
        obs_F   = Float32.(INIT_STATE.F .+ 0.005 .* randn(rng, WINDOW_BINS)),
        obs_A   = Float32.(INIT_STATE.A .+ 0.005 .* randn(rng, WINDOW_BINS)),
        B_init  = Float64(INIT_STATE.B),
        F_init  = Float64(INIT_STATE.F),
        A_init  = Float64(INIT_STATE.A),
    )

    # θ batch at the prior mean (= log truth, since all 10 priors are LogNormal).
    truth_constrained = [Float64(DEFAULT_PARAMS[n]) for n in PARAM_NAMES]
    u_one = log.(truth_constrained)
    U = Matrix{Float64}(repeat(reshape(u_one, 1, :), n_smc, 1))

    # Warm up.
    @info "warming up..."
    _ = gpu_log_density(target, U, grid_obs, UInt64(0))
    CUDA.synchronize()
    print_gpu_state("after warmup")

    # Time many calls.
    @info "timing $(n_repeat) calls..."
    times = Float64[]
    for _ in 1:n_repeat
        CUDA.synchronize()
        t0 = time_ns()
        _ = gpu_log_density(target, U, grid_obs, UInt64(0))
        CUDA.synchronize()
        push!(times, (time_ns() - t0) / 1e9)
    end

    t_med = median(times)
    n_threads = n_smc * K_per_chain
    # Hand-counted dominant ops per (chain × particle) per bin —
    # mirrors the JAX profiler's count: drift ~30 ops × 4 substeps +
    # 3-channel diagonal-Gaussian log-pdf ~30 ops + reflection ~10 ops.
    flops_per_thread = WINDOW_BINS * (30 * 4 + 30 + 10)
    flops = Float64(n_threads) * Float64(flops_per_thread)
    tflops_eff = flops / t_med / 1e12

    @printf("\nResults:\n")
    @printf("  median time per call : %.4f s\n", t_med)
    @printf("  threads in flight    : %d (%d chains × %d state particles)\n",
            n_threads, n_smc, K_per_chain)
    @printf("  approx FLOPs / call  : %.2e\n", flops)
    @printf("  effective TFLOPS     : %.2f / %.1f peak  =  %.1f%% util\n",
            tflops_eff, FP32_PEAK_TFLOPS,
            100 * tflops_eff / FP32_PEAK_TFLOPS)
    print_gpu_state("after timing")
    @printf("  host syncs per call  : 1 (CUDA.synchronize)\n")
    if verbose
        @printf("  raw times (s)        : %s\n",
                join([round(t, digits=4) for t in times], ", "))
    end
    println()
    return t_med
end


# ── Controller cost profile ─────────────────────────────────────────────

function profile_controller(; n_smc::Int = 256, n_inner::Int = 64,
                              n_anchors::Int = 8, T_days::Float64 = 14.0,
                              n_repeat::Int = 3, verbose::Bool = false)
    println("="^72)
    println("PROFILE: controller cost kernel (gpu_cost_log_density_batched)")
    println("="^72)
    print_gpu_state("at start")

    n_steps = round(Int, T_days * BINS_PER_DAY)
    @printf("Config: n_smc=%d, n_inner=%d, n_anchors=%d, n_steps=%d (T=%.1fd × %d bins/day)\n",
            n_smc, n_inner, n_anchors, n_steps, T_days, BINS_PER_DAY)

    params_v1_nt = params_v15_to_v1_nt(DEFAULT_PARAMS)
    target = FSAv1ControlGPUTarget(
        n_inner    = n_inner,
        M_max      = n_smc,
        n_steps    = n_steps,
        n_anchors  = n_anchors,
        n_substeps = 4,
        dt         = DT_BIN_DAYS,
        F_max      = 0.40,
        Phi_max    = 3.0,
        Phi_default = 1.0,
        params     = params_v1_nt,
        init_state = [INIT_STATE.B, INIT_STATE.F, INIT_STATE.A],
    )
    print_gpu_state("after target build")

    rng = MersenneTwister(0)
    U = randn(rng, n_smc, n_anchors)

    @info "warming up..."
    _ = gpu_cost_log_density_batched(target, U)
    CUDA.synchronize()
    print_gpu_state("after warmup")

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
    n_threads = n_smc * n_inner
    # Same hand-counted formula as v2's profile_gpu.jl: RBF decode
    # (~10 ops × n_anchors) + drift (~30 × n_substeps=4) + diffusion
    # (~12) + cost accumulator (~6) — per bin per thread.
    flops_per_thread = n_steps * (10 * n_anchors + 30 * 4 + 12 + 6)
    flops = Float64(n_threads) * Float64(flops_per_thread)
    tflops_eff = flops / t_med / 1e12

    @printf("\nResults:\n")
    @printf("  median time per call : %.4f s\n", t_med)
    @printf("  threads in flight    : %d (%d chains × %d CRN trials)\n",
            n_threads, n_smc, n_inner)
    @printf("  approx FLOPs / call  : %.2e\n", flops)
    @printf("  effective TFLOPS     : %.2f / %.1f peak  =  %.1f%% util\n",
            tflops_eff, FP32_PEAK_TFLOPS,
            100 * tflops_eff / FP32_PEAK_TFLOPS)
    print_gpu_state("after timing")
    @printf("  host syncs per call  : 1 (per-chain sum on CPU)\n")
    if verbose
        @printf("  raw times (s)        : %s\n",
                join([round(t, digits=4) for t in times], ", "))
    end
    println()
    return t_med
end


# ── Main ────────────────────────────────────────────────────────────────

function _parse_args(argv::Vector{String})
    mode = "both"
    verbose = false
    for a in argv
        if a == "--filter"
            mode = "filter"
        elseif a == "--controller"
            mode = "controller"
        elseif a == "--both"
            mode = "both"
        elseif a == "--verbose"
            verbose = true
        elseif startswith(a, "--")
            error("Unknown flag: $a")
        end
    end
    return (mode = mode, verbose = verbose)
end


function main(argv::Vector{String} = String[])
    args = _parse_args(argv)

    println("\nGPU device: $(CUDA.name(CUDA.device()))")
    free, total = CUDA.Mem.info()
    @printf("Total memory: %.2f GB, free: %.2f GB\n\n", bytes_GB(total), bytes_GB(free))

    if args.mode == "filter" || args.mode == "both"
        profile_filter(verbose = args.verbose)
    end
    if args.mode == "controller" || args.mode == "both"
        profile_controller(verbose = args.verbose)
    end

    println("Notes:")
    println("  - 'effective TFLOPS' is approximate (counts only the dominant fp ops")
    println("    in the inner loop). Useful for relative comparison across stacks at")
    println("    matched config; not a substitute for nsys / nv-nsight tracing.")
    println("  - 'util %' compares to RTX 5090's 104.8 TFLOPS fp32 peak. Saturating")
    println("    to >50% requires:")
    println("      • enough threads in flight (>= ~1M for 5090's 21504 cores)")
    println("      • compute-bound (not memory-bound) inner loop")
    println("      • no host syncs blocking the stream")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
