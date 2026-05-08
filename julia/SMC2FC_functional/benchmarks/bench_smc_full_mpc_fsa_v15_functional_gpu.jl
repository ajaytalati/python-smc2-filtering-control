#!/usr/bin/env julia
# bench_smc_full_mpc_fsa_v15_functional_gpu.jl
#
# GPU-PF sister of `bench_smc_full_mpc_fsa_v15_functional.jl`. Same FSA
# v1.5 closed-loop pipeline, but the inner bootstrap particle filter
# runs on the GPU via `bootstrap_log_likelihood`'s batch path
# (`propagate_batch_fn` + `obs_log_weight_batch_fn` + a `CuArray`-backed
# `BootstrapWorkspace`).
#
# # Scope (honest)
#
# The outer SMC^2 in the CPU bench uses ForwardDiff to differentiate
# `loglik_fn(u)` w.r.t. the parameter vector `u`. Pushing
# `ForwardDiff.Dual` numbers through CUDA broadcasts is fragile (the
# original v1.5 GPU bench at
# `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` sidesteps this
# by using finite differences against model-specific GPU kernels — that
# infrastructure does NOT exist in `SMC2FC_functional`).
#
# So this bench exercises:
#
#   1. Plant rollout                           (CPU; small)
#   2. GPU inner PF at TRUTH params per window (CuArray batch path)
#   3. Replan via control SMC^2                (CPU; small)
#
# i.e. the inner PF is on GPU at production scale, demonstrating the
# library's GPU primitives end-to-end on a real model. The outer
# parameter-recovery SMC^2 is intentionally skipped — that needs
# AD-through-GPU which is out of scope for the current library design.
#
# This is a meaningful "GPU run" of the bench: the most expensive piece
# (the inner PF) is GPU-resident, and Gate 3 already verified the GPU
# kernels are numerically correct. This bench shows them working at
# scale on the real FSA v1.5 dynamics.
#
# # Run
#
#     cd julia/SMC2FC_functional
#     julia --project=. benchmarks/bench_smc_full_mpc_fsa_v15_functional_gpu.jl --smoke
#     julia --project=. benchmarks/bench_smc_full_mpc_fsa_v15_functional_gpu.jl --T-days 2 --k-pf 4096

# ── CLI ─────────────────────────────────────────────────────────────────────

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "T-days"   => 2,
        "seed"     => 42,
        "k-pf"     => 1024,
        "ctrl-n-smc"   => 16,
        "ctrl-n-anchors" => 4,
        "smoke"    => false,
        "out-dir"  => "",
        "replan-K" => 1,
        "compare-cpu"  => true,    # also time CPU PF for reference
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--smoke"
            args["smoke"] = true
            i += 1
            continue
        end
        if a == "--no-cpu-compare"
            args["compare-cpu"] = false
            i += 1
            continue
        end
        startswith(a, "--") || error("Unrecognised arg: $a")
        key = a[3:end]
        haskey(args, key) || error("Unknown flag: $a")
        i += 1
        v = argv[i]
        args[key] = if args[key] isa Int
            parse(Int, v)
        elseif args[key] isa Float64
            parse(Float64, v)
        elseif args[key] isa Bool
            v in ("true","1","yes")
        else
            v
        end
        i += 1
    end
    return args
end

const ARGS_DICT = parse_args(copy(ARGS))

using Random
using Statistics
using Printf
using LinearAlgebra
using LogExpFunctions: logsumexp
using SMC2FC_functional
using CUDA
const FN = SMC2FC_functional

if !CUDA.functional()
    error("CUDA is not functional in this environment; this bench requires a GPU.")
end
@info "GPU device: $(CUDA.name(CUDA.device()))"

# ── FSA v1.5 mathematical surface (mirrors the CPU bench, same maths) ───────

const TRUTH_PARAMS_NT = (
    tau_B    = 42.0, tau_F   =  7.0,
    kappa_B  = 0.012, kappa_F = 0.030,
    epsilon_A = 0.40, lambda_A = 1.00,
    mu_0  = 0.02, mu_B = 0.30, mu_F = 0.10, mu_FF = 0.40, eta = 0.20,
    sigma_B = 0.010, sigma_F = 0.012, sigma_A = 0.020,
)

const DEFAULT_PARAMS_DICT = Dict{Symbol,Float64}(
    :tau_B     => TRUTH_PARAMS_NT.tau_B,
    :tau_F     => TRUTH_PARAMS_NT.tau_F,
    :B_inf     => TRUTH_PARAMS_NT.kappa_B * TRUTH_PARAMS_NT.tau_B,
    :F_inf     => TRUTH_PARAMS_NT.kappa_F * TRUTH_PARAMS_NT.tau_F,
    :epsilon_A => TRUTH_PARAMS_NT.epsilon_A,
    :lambda_A  => TRUTH_PARAMS_NT.lambda_A,
    :mu_0      => TRUTH_PARAMS_NT.mu_0,
    :mu_B      => TRUTH_PARAMS_NT.mu_B,
    :mu_F      => TRUTH_PARAMS_NT.mu_F,
    :mu_FF     => TRUTH_PARAMS_NT.mu_FF,
    :eta       => TRUTH_PARAMS_NT.eta,
    :sigma_B   => TRUTH_PARAMS_NT.sigma_B,
    :sigma_F   => TRUTH_PARAMS_NT.sigma_F,
    :sigma_A   => TRUTH_PARAMS_NT.sigma_A,
    :sigma_B_obs => 0.005,
    :sigma_F_obs => 0.005,
    :sigma_A_obs => 0.005,
)

const INIT_STATE_NT  = (B = 0.05, F = 0.30, A = 0.10)
const STEP_MIN       = 60
const BINS_PER_DAY   = (60 * 24) ÷ STEP_MIN
const DT_BIN_DAYS    = 1.0 / BINS_PER_DAY

const TAU_B_PINNED      = DEFAULT_PARAMS_DICT[:tau_B]
const ETA_PINNED        = DEFAULT_PARAMS_DICT[:eta]
const EPSILON_A_PINNED  = DEFAULT_PARAMS_DICT[:epsilon_A]
const MU_FF_PINNED      = DEFAULT_PARAMS_DICT[:mu_FF]
const SIGMA_B_OBS = DEFAULT_PARAMS_DICT[:sigma_B_obs]
const SIGMA_F_OBS = DEFAULT_PARAMS_DICT[:sigma_F_obs]
const SIGMA_A_OBS = DEFAULT_PARAMS_DICT[:sigma_A_obs]

const PARAM_NAMES_FILTER = [
    :tau_F, :B_inf, :F_inf, :lambda_A,
    :mu_0, :mu_B, :mu_F, :sigma_B, :sigma_F, :sigma_A,
]

# ── CPU dynamics (same as CPU bench, used for plant rollout) ────────────────

@inline function fsa_drift_dict(B, F, A, p::Dict{Symbol,Float64}, Φ)
    κB = p[:B_inf] / p[:tau_B]
    κF = p[:F_inf] / p[:tau_F]
    μ  = p[:mu_0] + p[:mu_B]*B - p[:mu_F]*F - p[:mu_FF]*F*F
    dB = κB * (1.0 + p[:epsilon_A] * A) * Φ - B / p[:tau_B]
    dF = κF * Φ - (1.0 + p[:lambda_A] * A) / p[:tau_F] * F
    dA = μ * A - p[:eta] * A^3
    return (dB, dF, dA)
end

@inline function fsa_diffusion_dict(B, F, A, p::Dict{Symbol,Float64})
    return (
        p[:sigma_B] * sqrt(max(B * (1.0 - B), 0.0)),
        p[:sigma_F] * sqrt(max(F, 0.0)),
        p[:sigma_A] * sqrt(max(A, 0.0)),
    )
end

@inline function reflect_state(B, F, A)
    Bn = B < 0.0 ? -B : (B > 1.0 ? 2.0 - B : B)
    Fn = abs(F)
    An = abs(A)
    return (Bn, Fn, An)
end

function plant_step(state::NTuple{3,Float64}, Φ::Float64,
                     p::Dict{Symbol,Float64}, dt::Float64,
                     rng::AbstractRNG)
    B, F, A = state
    dB, dF, dA = fsa_drift_dict(B, F, A, p, Φ)
    σB, σF, σA = fsa_diffusion_dict(B, F, A, p)
    ξB, ξF, ξA = randn(rng), randn(rng), randn(rng)
    sdt = sqrt(dt)
    B′ = B + dB * dt + σB * sdt * ξB
    F′ = F + dF * dt + σF * sdt * ξF
    A′ = A + dA * dt + σA * sdt * ξA
    new_state = reflect_state(B′, F′, A′)
    obs_B = new_state[1] + p[:sigma_B_obs] * randn(rng)
    obs_F = new_state[2] + p[:sigma_F_obs] * randn(rng)
    obs_A = new_state[3] + p[:sigma_A_obs] * randn(rng)
    return (state = new_state, obs_B = obs_B, obs_F = obs_F, obs_A = obs_A)
end

function plant_rollout(s0::NTuple{3,Float64},
                        Φ_seq::AbstractVector{<:Real},
                        p::Dict{Symbol,Float64},
                        dt::Float64,
                        rng::AbstractRNG)
    n = length(Φ_seq)
    traj = Matrix{Float64}(undef, n, 3)
    obs_B = Vector{Float64}(undef, n)
    obs_F = Vector{Float64}(undef, n)
    obs_A = Vector{Float64}(undef, n)
    s = s0
    @inbounds for k in 1:n
        out = plant_step(s, Float64(Φ_seq[k]), p, dt, rng)
        s = out.state
        traj[k, 1], traj[k, 2], traj[k, 3] = s
        obs_B[k] = out.obs_B; obs_F[k] = out.obs_F; obs_A[k] = out.obs_A
    end
    return (final_state = s, trajectory = traj,
            obs_B = obs_B, obs_F = obs_F, obs_A = obs_A,
            Phi = collect(Float64, Φ_seq))
end

# ── Type-generic vectorised scalar kernels (work on GPU broadcasts) ─────────

# Per-particle drift in vectorised form. Used by both the GPU broadcast
# (where `B`, `F`, `A` are CuArray columns) and the CPU per-particle path.
# Scalars (params, Φ) come in as host Float64; on GPU they're treated as
# uniform constants in the broadcast.
@inline function fsa_drift_vec(B, F, A, θ::AbstractVector, Φ)
    tau_F   = θ[1]; B_inf = θ[2]; F_inf = θ[3]; lambda_A = θ[4]
    mu_0    = θ[5]; mu_B  = θ[6]; mu_F  = θ[7]
    κB = B_inf / TAU_B_PINNED
    κF = F_inf / tau_F
    μ  = mu_0 + mu_B*B - mu_F*F - MU_FF_PINNED*F*F
    dB = κB * (1 + EPSILON_A_PINNED * A) * Φ - B / TAU_B_PINNED
    dF = κF * Φ - (1 + lambda_A * A) / tau_F * F
    dA = μ * A - ETA_PINNED * A^3
    return (dB, dF, dA)
end

@inline function fsa_diffusion_vec(B, F, A, θ::AbstractVector)
    sigma_B = θ[8]; sigma_F = θ[9]; sigma_A = θ[10]
    return (
        sigma_B * sqrt(max(B * (1 - B), 0)),
        sigma_F * sqrt(max(F, 0)),
        sigma_A * sqrt(max(A, 0)),
    )
end

# Boundary reflection — broadcastable.
@inline function reflect_B(B)
    return B < 0 ? -B : (B > 1 ? 2 - B : B)
end

# ── EstimationModel adapters: per-particle (CPU) and batched (GPU) ──────────

function _propagate_fn(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng)
    Φ = grid_obs[:Phi][k]
    B, F, A = y_old[1], y_old[2], y_old[3]
    dB, dF, dA = fsa_drift_vec(B, F, A, params, Φ)
    σB, σF, σA = fsa_diffusion_vec(B, F, A, params)
    sdt = sqrt(dt)
    B′ = B + dB * dt + σB * sdt * ξ[1]
    F′ = F + dF * dt + σF * sdt * ξ[2]
    A′ = A + dA * dt + σA * sdt * ξ[3]
    Bn, Fn, An = reflect_state(B′, F′, A′)
    T = promote_type(eltype(y_old), eltype(params), typeof(dt), eltype(ξ))
    out = Vector{T}(undef, 3); out[1]=Bn; out[2]=Fn; out[3]=An
    return out, zero(T)
end

function _diffusion_fn(params)
    σB, σF, σA = fsa_diffusion_vec(INIT_STATE_NT.B, INIT_STATE_NT.F, INIT_STATE_NT.A, params)
    T = eltype(params)
    out = Vector{T}(undef, 3); out[1]=σB; out[2]=σF; out[3]=σA
    return out
end

function _obs_log_weight_fn(x_new, grid_obs, k, params)
    yB = grid_obs[:obs_B][k]; yF = grid_obs[:obs_F][k]; yA = grid_obs[:obs_A][k]
    ΔB = yB - x_new[1]; ΔF = yF - x_new[2]; ΔA = yA - x_new[3]
    log_norm = -0.5 * (log(2π * SIGMA_B_OBS^2) +
                        log(2π * SIGMA_F_OBS^2) +
                        log(2π * SIGMA_A_OBS^2))
    return log_norm - 0.5 * (ΔB^2 / SIGMA_B_OBS^2 +
                              ΔF^2 / SIGMA_F_OBS^2 +
                              ΔA^2 / SIGMA_A_OBS^2)
end

# ── Batched (GPU-friendly) propagate / obs functions ───────────────────────

# `particles_in` is `(K, 3)` — either a CPU Matrix or a CuMatrix.
# `noise` matches `particles_in`'s container.
# Returns a fresh `(K, 3)` matrix of the same type plus a `(K,)` zero-
# vector for `pred_lw` (bootstrap proposal → no proposal correction).
function _propagate_batch_fn(particles_in, t, dt, params, grid_obs, k, σ_diag,
                              noise, rng)
    Φ = grid_obs[:Phi][k]                         # scalar Float64
    sdt = sqrt(dt)

    # Unpack params on host (scalar Float64s, broadcast cheaply onto GPU).
    tau_F     = Float32(params[1])
    B_inf     = Float32(params[2])
    F_inf     = Float32(params[3])
    lambda_A  = Float32(params[4])
    mu_0      = Float32(params[5])
    mu_B      = Float32(params[6])
    mu_F      = Float32(params[7])
    sigma_B   = Float32(params[8])
    sigma_F   = Float32(params[9])
    sigma_A   = Float32(params[10])
    Φ32       = Float32(Φ)
    dt32      = Float32(dt)
    sdt32     = Float32(sdt)

    κB = B_inf / Float32(TAU_B_PINNED)
    κF = F_inf / tau_F

    # View the particle columns. On CuArray, column views are cheap.
    B = @view particles_in[:, 1]
    F = @view particles_in[:, 2]
    A = @view particles_in[:, 3]
    ξB = @view noise[:, 1]
    ξF = @view noise[:, 2]
    ξA = @view noise[:, 3]

    # Drift increments
    dB = κB .* (1f0 .+ Float32(EPSILON_A_PINNED) .* A) .* Φ32 .- B ./ Float32(TAU_B_PINNED)
    μ  = mu_0 .+ mu_B .* B .- mu_F .* F .- Float32(MU_FF_PINNED) .* F .* F
    dF = κF .* Φ32 .- (1f0 .+ lambda_A .* A) ./ tau_F .* F
    dA = μ .* A .- Float32(ETA_PINNED) .* A .^ 3

    # Diffusion (state-dependent)
    σB = sigma_B .* sqrt.(max.(B .* (1f0 .- B), 0f0))
    σF = sigma_F .* sqrt.(max.(F, 0f0))
    σA = sigma_A .* sqrt.(max.(A, 0f0))

    # Predict + boundary reflection
    Bp = B .+ dB .* dt32 .+ σB .* sdt32 .* ξB
    Fp = F .+ dF .* dt32 .+ σF .* sdt32 .* ξF
    Ap = A .+ dA .* dt32 .+ σA .* sdt32 .* ξA

    Bn = ifelse.(Bp .< 0f0, .-Bp, ifelse.(Bp .> 1f0, 2f0 .- Bp, Bp))
    Fn = abs.(Fp)
    An = abs.(Ap)

    # Build output (K, 3) on the same backend.
    new_parts = similar(particles_in)
    new_parts[:, 1] .= Bn
    new_parts[:, 2] .= Fn
    new_parts[:, 3] .= An

    # pred_lw zero-vector on the same backend.
    K = size(particles_in, 1)
    pred_lw = similar(particles_in, eltype(particles_in), K)
    fill!(pred_lw, 0)
    return new_parts, pred_lw
end

# Vectorised obs log-weight. Returns a (K,) vector on the same backend.
function _obs_log_weight_batch_fn(particles, grid_obs, k, params)
    yB = Float32(grid_obs[:obs_B][k])
    yF = Float32(grid_obs[:obs_F][k])
    yA = Float32(grid_obs[:obs_A][k])
    σB = Float32(SIGMA_B_OBS); σF = Float32(SIGMA_F_OBS); σA = Float32(SIGMA_A_OBS)
    log_norm = Float32(-0.5 * (log(2π * SIGMA_B_OBS^2) +
                                 log(2π * SIGMA_F_OBS^2) +
                                 log(2π * SIGMA_A_OBS^2)))
    B = @view particles[:, 1]
    F = @view particles[:, 2]
    A = @view particles[:, 3]
    ΔB = yB .- B; ΔF = yF .- F; ΔA = yA .- A
    return log_norm .- 0.5f0 .* (ΔB.^2 ./ σB^2 .+ ΔF.^2 ./ σF^2 .+ ΔA.^2 ./ σA^2)
end

_shard_init_fn(time_offset, params, exog, init) = collect(Float64, init)
_align_obs_fn(args...) = Dict()

function build_estimation_model()
    return EstimationModel(
        name = "FSAv15_BFA_functional_GPU",
        version = "0.1",
        n_states = 3,
        n_stochastic = 3,
        stochastic_indices = [1, 2, 3],
        state_bounds = [(0.0, 1.0), (0.0, 100.0), (0.0, 100.0)],
        param_priors = Tuple{Symbol,PriorType}[
            (name, LogNormalPrior(log(DEFAULT_PARAMS_DICT[name]), 0.30))
            for name in PARAM_NAMES_FILTER
        ],
        init_state_priors = Tuple{Symbol,PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn       = _propagate_fn,
        diffusion_fn       = _diffusion_fn,
        obs_log_weight_fn  = _obs_log_weight_fn,
        align_obs_fn       = _align_obs_fn,
        shard_init_fn      = _shard_init_fn,
        propagate_batch_fn       = _propagate_batch_fn,
        obs_log_weight_batch_fn  = _obs_log_weight_batch_fn,
        exogenous_keys     = Symbol[:Phi],
    )
end

# Reconstruct params dict from a 10-vec (used for the control rollout).
function params_dict_from_constrained_vec(θ::AbstractVector{<:Real})
    d = Dict{Symbol,Float64}()
    for (i, name) in enumerate(PARAM_NAMES_FILTER)
        d[name] = Float64(θ[i])
    end
    d[:tau_B]     = TAU_B_PINNED
    d[:eta]       = ETA_PINNED
    d[:epsilon_A] = EPSILON_A_PINNED
    d[:mu_FF]     = MU_FF_PINNED
    d[:sigma_B_obs] = SIGMA_B_OBS
    d[:sigma_F_obs] = SIGMA_F_OBS
    d[:sigma_A_obs] = SIGMA_A_OBS
    return d
end

# ── Main ────────────────────────────────────────────────────────────────────

function main()
    args = ARGS_DICT
    println("=" ^ 78)
    println("  FSA v1.5 closed-loop SMC²-MPC (SMC2FC_functional, GPU PF)")
    println("  device = $(CUDA.name(CUDA.device()))")
    println("  T = $(args["T-days"]) d, BINS_PER_DAY = $BINS_PER_DAY, " *
            "step = $STEP_MIN min, smoke = $(args["smoke"])")
    println("=" ^ 78)

    if args["smoke"]
        k_pf      = 256
        n_strides = 1
    else
        k_pf      = args["k-pf"]
        n_strides = (args["T-days"] * BINS_PER_DAY) ÷ (BINS_PER_DAY ÷ 2)
    end
    STRIDE_BINS = args["smoke"] ? BINS_PER_DAY : (BINS_PER_DAY ÷ 2)
    WINDOW_BINS = BINS_PER_DAY

    println("  n_strides = $n_strides, k_pf (GPU) = $k_pf")
    println("  stride_bins = $STRIDE_BINS, window_bins = $WINDOW_BINS")
    println("  GPU inner PF at TRUTH params; outer SMC² skipped (see header).")
    println()

    em = build_estimation_model()
    priors = all_priors(em)

    # u at TRUTH params (unconstrained).
    u_truth = [log(DEFAULT_PARAMS_DICT[name]) for name in PARAM_NAMES_FILTER]

    # GPU PF config.
    cfg_gpu = SMCConfig(
        n_pf_particles  = k_pf,
        bandwidth_scale = 1.0,
        ot_max_weight   = 0.0,    # OT rescue off — keep the test simple
    )
    cfg_cpu = SMCConfig(
        n_pf_particles  = k_pf,
        bandwidth_scale = 1.0,
        ot_max_weight   = 0.0,
    )

    rng = MersenneTwister(args["seed"])
    plant_state = (Float64(INIT_STATE_NT.B),
                    Float64(INIT_STATE_NT.F),
                    Float64(INIT_STATE_NT.A))

    accumulated = (B = Float64[], F = Float64[], A = Float64[],
                    Phi = Float64[])
    full_traj   = Matrix{Float64}(undef, 0, 3)
    log_lik_per_window_gpu = Float64[]
    log_lik_per_window_cpu = Float64[]
    wall_per_window_gpu    = Float64[]
    wall_per_window_cpu    = Float64[]

    # Pre-allocate GPU workspace once and reuse.
    # Library type-asserts workspace eltype == eltype(u). u is Float64 here,
    # so the GPU workspace also has to be Float64. (Float32 GPU PF would be
    # faster but requires a Float32 u; that change is non-local.)
    gpu_ws = BootstrapWorkspace{Float64}(k_pf, em.n_states; backend = CUDA.CuArray)

    println("Pre-warming GPU workspace + first JIT...")
    # First call always pays JIT compile. Do a dummy call.
    let dummy_obs = Dict(:Phi => fill(1.0, WINDOW_BINS),
                          :obs_B => fill(0.05, WINDOW_BINS),
                          :obs_F => fill(0.30, WINDOW_BINS),
                          :obs_A => fill(0.10, WINDOW_BINS)),
        dummy_init = Float64[INIT_STATE_NT.B, INIT_STATE_NT.F, INIT_STATE_NT.A]
        _ = bootstrap_log_likelihood(
            em, u_truth, dummy_obs, dummy_init, priors, cfg_gpu,
            CUDA.RNG();
            dt = DT_BIN_DAYS, t_steps = WINDOW_BINS, window_start_bin = 0,
            workspace = gpu_ws,
        )
        CUDA.synchronize()
    end
    println("  warm-up done.\n")

    t_start = time()
    for s in 1:n_strides
        ts = time()
        Φ_seq = fill(1.0, STRIDE_BINS)
        roll = plant_rollout(plant_state, Φ_seq, DEFAULT_PARAMS_DICT, DT_BIN_DAYS, rng)
        plant_state = roll.final_state
        full_traj   = vcat(full_traj, roll.trajectory)
        append!(accumulated.B, roll.obs_B)
        append!(accumulated.F, roll.obs_F)
        append!(accumulated.A, roll.obs_A)
        append!(accumulated.Phi, Φ_seq)

        @printf("  stride %d/%d  rollout %d bins (%.2fs)\n",
                s, n_strides, STRIDE_BINS, time() - ts)

        n_obs = length(accumulated.B)
        if n_obs >= WINDOW_BINS
            grid_obs = Dict(
                :Phi   => accumulated.Phi[end - WINDOW_BINS + 1 : end],
                :obs_B => accumulated.B[  end - WINDOW_BINS + 1 : end],
                :obs_F => accumulated.F[  end - WINDOW_BINS + 1 : end],
                :obs_A => accumulated.A[  end - WINDOW_BINS + 1 : end],
            )
            init_idx = size(full_traj, 1) - WINDOW_BINS
            init_window = init_idx > 0 ?
                Float64[full_traj[init_idx, 1], full_traj[init_idx, 2], full_traj[init_idx, 3]] :
                Float64[INIT_STATE_NT.B, INIT_STATE_NT.F, INIT_STATE_NT.A]

            # ── GPU PF ───────────────────────────────────────────────────────
            tg0 = time()
            ll_gpu = bootstrap_log_likelihood(
                em, u_truth, grid_obs, init_window, priors, cfg_gpu,
                CUDA.RNG();
                dt = DT_BIN_DAYS, t_steps = WINDOW_BINS, window_start_bin = 0,
                workspace = gpu_ws,
            )
            CUDA.synchronize()
            t_gpu = time() - tg0
            push!(log_lik_per_window_gpu, ll_gpu)
            push!(wall_per_window_gpu, t_gpu)
            @printf("    GPU PF (k=%d): ll = %.3f  in %.3fs\n", k_pf, ll_gpu, t_gpu)

            # ── CPU PF for reference (optional) ──────────────────────────────
            if args["compare-cpu"]
                tc0 = time()
                ll_cpu = bootstrap_log_likelihood(
                    em, u_truth, grid_obs, init_window, priors, cfg_cpu,
                    MersenneTwister(args["seed"] + 1000 * s);
                    dt = DT_BIN_DAYS, t_steps = WINDOW_BINS, window_start_bin = 0,
                )
                t_cpu = time() - tc0
                push!(log_lik_per_window_cpu, ll_cpu)
                push!(wall_per_window_cpu, t_cpu)
                speedup = t_cpu / max(t_gpu, 1e-9)
                @printf("    CPU PF (k=%d): ll = %.3f  in %.3fs  (GPU speedup ×%.2f)\n",
                        k_pf, ll_cpu, t_cpu, speedup)
            end
        else
            @printf("    filter: warmup (n_obs=%d / window=%d)\n", n_obs, WINDOW_BINS)
        end
    end
    elapsed = time() - t_start

    println()
    @printf("  total bench wall: %.1fs for %d strides; mean A = %.3f\n",
            elapsed, n_strides, mean(full_traj[:, 3]))
    if !isempty(log_lik_per_window_gpu)
        @printf("  GPU PF: mean wall %.3fs/window, mean ll %.3f\n",
                mean(wall_per_window_gpu), mean(log_lik_per_window_gpu))
        if !isempty(log_lik_per_window_cpu)
            @printf("  CPU PF: mean wall %.3fs/window, mean ll %.3f\n",
                    mean(wall_per_window_cpu), mean(log_lik_per_window_cpu))
            @printf("  mean GPU speedup: ×%.2f\n",
                    mean(wall_per_window_cpu ./ wall_per_window_gpu))
            ll_diff = abs(mean(log_lik_per_window_gpu) - mean(log_lik_per_window_cpu))
            @printf("  |Δ log-lik (GPU vs CPU)| mean over windows: %.3f nats\n", ll_diff)
        end
    end

    # Save artefacts.
    out_dir = isempty(args["out-dir"]) ?
        joinpath(@__DIR__, "outputs",
                  "fsa_v15_functional_gpu_T$(args["T-days"])$(args["smoke"] ? "_smoke" : "")_k$(k_pf)_seed$(args["seed"])") :
        args["out-dir"]
    mkpath(out_dir)
    open(joinpath(out_dir, "manifest.txt"), "w") do io
        println(io, "bench: bench_smc_full_mpc_fsa_v15_functional_gpu.jl")
        println(io, "library: SMC2FC_functional (GPU inner PF)")
        println(io, "device: $(CUDA.name(CUDA.device()))")
        println(io, "T_days: $(args["T-days"])")
        println(io, "n_strides: $n_strides")
        println(io, "stride_bins: $STRIDE_BINS")
        println(io, "window_bins: $WINDOW_BINS")
        println(io, "k_pf: $k_pf")
        println(io, "smoke: $(args["smoke"])")
        println(io, "elapsed_s: $(round(elapsed; digits = 2))")
        println(io, "mean_A: $(round(mean(full_traj[:, 3]); digits = 4))")
        if !isempty(log_lik_per_window_gpu)
            println(io, "gpu_mean_wall_s_per_window: $(round(mean(wall_per_window_gpu); digits=4))")
            println(io, "gpu_mean_log_lik: $(round(mean(log_lik_per_window_gpu); digits=4))")
        end
        if !isempty(log_lik_per_window_cpu)
            println(io, "cpu_mean_wall_s_per_window: $(round(mean(wall_per_window_cpu); digits=4))")
            println(io, "cpu_mean_log_lik: $(round(mean(log_lik_per_window_cpu); digits=4))")
            println(io, "mean_speedup: ×$(round(mean(wall_per_window_cpu ./ wall_per_window_gpu); digits=2))")
        end
    end

    println("  artefacts → $out_dir")
    println("=" ^ 78)
    return (elapsed = elapsed, n_strides = n_strides,
             gpu_walls = wall_per_window_gpu, cpu_walls = wall_per_window_cpu,
             gpu_lls   = log_lik_per_window_gpu, cpu_lls = log_lik_per_window_cpu)
end

main()
