#!/usr/bin/env julia
# bench_smc_full_mpc_fsa_v15_functional.jl
#
# Sister of `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py`,
# using the new `SMC2FC_functional` library (CPU-side) instead of the
# JAX framework or the existing Julia GPU port.
#
# Goal: a real-application smoke test confirming that the new
# functional library can drive a closed-loop SMC²-MPC pipeline end-to-end
# on the FSA v1.5 model.
#
# # No-edit policy
#
# This file does NOT modify any pre-existing repo file. The FSA v1.5
# mathematical surface (drift, diffusion, plant step, default params,
# prior config) is re-implemented locally inside this script — the
# originals live in `version_1_5_Julia/models/fsa_high_res/`
# (`_dynamics.jl`, `simulation.jl`, `_plant.jl`, `estimation.jl`). Each
# block below cites the file it mirrors.
#
# # Type-genericity
#
# The HMC moves inside the outer SMC² use ForwardDiff to differentiate
# `loglik_fn(u)` w.r.t. `u`. So every adapter that touches `params`
# (which is a view into the constrained-θ vector and may carry Dual
# numbers under AD) must keep its element type generic — no
# `Float64(...)` casts on params, no `Dict{Symbol,Float64}` snapshots
# of params. Pinned constants and observation noise scales (which never
# carry Duals) stay `Float64`.
#
# # Run
#
#     cd julia/SMC2FC_functional
#     julia --project=. benchmarks/bench_smc_full_mpc_fsa_v15_functional.jl --smoke
#     julia --project=. benchmarks/bench_smc_full_mpc_fsa_v15_functional.jl --T-days 2

# ── CLI ─────────────────────────────────────────────────────────────────────

function parse_args(argv::Vector{String})
    args = Dict{String,Any}(
        "T-days"   => 2,
        "seed"     => 42,
        "n-smc"    => 8,
        "k-pf"     => 32,
        "ctrl-n-smc"   => 16,
        "ctrl-n-anchors" => 4,
        "smoke"    => false,
        "out-dir"  => "",
        "replan-K" => 1,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--smoke"
            args["smoke"] = true
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
const FN = SMC2FC_functional

# ── FSA v1.5 mathematical surface (mirrors version_1_5_Julia/models/fsa_high_res/) ─

# Mirrors _dynamics.jl:TRUTH_PARAMS.
const TRUTH_PARAMS_NT = (
    tau_B    = 42.0, tau_F   =  7.0,
    kappa_B  = 0.012, kappa_F = 0.030,
    epsilon_A = 0.40, lambda_A = 1.00,
    mu_0  = 0.02, mu_B = 0.30, mu_F = 0.10, mu_FF = 0.40, eta = 0.20,
    sigma_B = 0.010, sigma_F = 0.012, sigma_A = 0.020,
)

# Mirrors simulation.jl:DEFAULT_PARAMS — v1.5 parametrisation
# (B_inf = κ_B · τ_B, F_inf = κ_F · τ_F).
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

# Pinned constants (FIM gate decision: τ_B, η, ε_A, μ_FF held at truth).
const TAU_B_PINNED      = DEFAULT_PARAMS_DICT[:tau_B]
const ETA_PINNED        = DEFAULT_PARAMS_DICT[:eta]
const EPSILON_A_PINNED  = DEFAULT_PARAMS_DICT[:epsilon_A]
const MU_FF_PINNED      = DEFAULT_PARAMS_DICT[:mu_FF]

# Observation noise scales (always Float64 — never carry Duals).
const SIGMA_B_OBS = DEFAULT_PARAMS_DICT[:sigma_B_obs]
const SIGMA_F_OBS = DEFAULT_PARAMS_DICT[:sigma_F_obs]
const SIGMA_A_OBS = DEFAULT_PARAMS_DICT[:sigma_A_obs]

# Filter-side parameter ordering (10 estimated params, mirrors estimation.jl:PARAM_NAMES).
const PARAM_NAMES_FILTER = [
    :tau_F, :B_inf, :F_inf, :lambda_A,
    :mu_0, :mu_B, :mu_F, :sigma_B, :sigma_F, :sigma_A,
]

# ── Type-generic dynamics on a 10-vector of estimated params ────────────────
# These mirror _dynamics.jl:drift / diffusion_state_dep but accept the
# estimated-param vector directly (rather than a Dict) so they propagate
# ForwardDiff Duals correctly.

# Param-vector layout per PARAM_NAMES_FILTER:
#   1 tau_F    2 B_inf    3 F_inf    4 lambda_A
#   5 mu_0     6 mu_B     7 mu_F     8 sigma_B   9 sigma_F  10 sigma_A
@inline function fsa_drift_vec(B::Real, F::Real, A::Real,
                                 θ::AbstractVector, Φ::Real)
    tau_F   = θ[1]; B_inf = θ[2]; F_inf = θ[3]; lambda_A = θ[4]
    mu_0    = θ[5]; mu_B  = θ[6]; mu_F  = θ[7]
    κB = B_inf / TAU_B_PINNED
    κF = F_inf / tau_F
    μ  = mu_0 + mu_B*B - mu_F*F - MU_FF_PINNED*F*F
    dB = κB * (1.0 + EPSILON_A_PINNED * A) * Φ - B / TAU_B_PINNED
    dF = κF * Φ - (1.0 + lambda_A * A) / tau_F * F
    dA = μ * A - ETA_PINNED * A^3
    return (dB, dF, dA)
end

@inline function fsa_diffusion_vec(B::Real, F::Real, A::Real, θ::AbstractVector)
    sigma_B = θ[8]; sigma_F = θ[9]; sigma_A = θ[10]
    return (
        sigma_B * sqrt(max(B * (1.0 - B), 0.0)),
        sigma_F * sqrt(max(F, 0.0)),
        sigma_A * sqrt(max(A, 0.0)),
    )
end

# Float64-Dict-based drift / diffusion (used only by the plant rollout, never
# under AD). Mirrors _dynamics.jl:drift / diffusion_state_dep.
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

# Mirrors _plant.jl:_reflect_unit + the boundary handling in plant_step.
@inline function reflect_state(B, F, A)
    Bn = B < 0.0 ? -B : (B > 1.0 ? 2.0 - B : B)
    Fn = abs(F)
    An = abs(A)
    return (Bn, Fn, An)
end

# ── Plant rollout (Float64 only — used to produce ground truth obs) ────────

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

# ── EstimationModel adapter (the bridge to SMC2FC_functional) ───────────────
#
# Adapters are TYPE-GENERIC in `params`. The `params` argument arrives from
# `bootstrap_log_likelihood` as `view(θ[1:n_params])` where θ has eltype
# matching `u` — Float64 in production, ForwardDiff.Dual{...} when the
# outer-SMC² HMC kernel is computing a gradient via ForwardDiff. None of
# the helpers below cast `params[i]` to Float64.

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
    # Build an output vector with the right element type for AD-friendliness.
    # `promote_type(eltype(y_old), eltype(params))` covers both pure-Float64
    # and Dual paths.
    T = promote_type(eltype(y_old), eltype(params), typeof(dt), eltype(ξ))
    out = Vector{T}(undef, 3)
    out[1] = Bn; out[2] = Fn; out[3] = An
    return out, zero(T)
end

function _diffusion_fn(params)
    σB, σF, σA = fsa_diffusion_vec(INIT_STATE_NT.B, INIT_STATE_NT.F, INIT_STATE_NT.A, params)
    T = eltype(params)
    out = Vector{T}(undef, 3)
    out[1] = σB; out[2] = σF; out[3] = σA
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

_shard_init_fn(time_offset, params, exog, init) = collect(Float64, init)
_align_obs_fn(args...) = Dict()

function build_estimation_model()
    return EstimationModel(
        name = "FSAv15_BFA_functional",
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
        exogenous_keys     = Symbol[:Phi],
    )
end

# Reconstruct a Float64 params dict from a 10-vector — used ONLY for the
# CONTROL side (which evaluates a fixed-θ cost rollout, no AD on params).
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

# ── ControlSpec: simple cost = -mean(A) + λ_F · soft F-cap penalty ──────────

function build_control_spec(; T_total_bins::Int, dt::Float64,
                              params::Dict{Symbol,Float64},
                              init_state::NTuple{3,Float64},
                              n_anchors::Int = 4,
                              n_inner::Int = 16,
                              F_max::Float64 = 0.40,
                              lam_F::Float64 = 1.0,
                              sigma_prior::Float64 = 1.5,
                              seed::Int = 0)

    rbf = RBFBasis(T_total_bins, dt, n_anchors; output = SigmoidOutput())
    Φ_designm = design_matrix(rbf)
    Φ_max = 3.0

    grids = build_crn_noise_grids(; n_inner = n_inner,
                                    n_steps = T_total_bins,
                                    n_channels = 3,
                                    seed = seed)
    wiener  = grids[:wiener]

    function schedule_from_theta_fn(θ::AbstractVector)
        s_unit = schedule_from_theta(rbf, θ; Φ = Φ_designm)
        return Φ_max .* s_unit
    end

    function cost_fn(θ::AbstractVector)
        Φ_seq = schedule_from_theta_fn(θ)
        cost_acc = 0.0
        @inbounds for r in 1:n_inner
            B, F, A = init_state
            mean_A = 0.0
            f_pen  = 0.0
            for k in 1:T_total_bins
                Φ = Φ_seq[k]
                dB, dF, dA = fsa_drift_dict(B, F, A, params, Φ)
                σB, σF, σA = fsa_diffusion_dict(B, F, A, params)
                ξB, ξF, ξA = wiener[r, k, 1], wiener[r, k, 2], wiener[r, k, 3]
                sdt = sqrt(dt)
                B  = B + dB * dt + σB * sdt * ξB
                F  = F + dF * dt + σF * sdt * ξF
                A  = A + dA * dt + σA * sdt * ξA
                B, F, A = reflect_state(B, F, A)
                mean_A += A
                f_pen  += max(0.0, F - F_max) ^ 2
            end
            mean_A /= T_total_bins
            f_pen  /= T_total_bins
            cost_acc += (-mean_A + lam_F * f_pen)
        end
        return cost_acc / n_inner
    end

    return ControlSpec(
        name = "FSAv15_RBF_functional",
        version = "0.1",
        dt = dt,
        n_steps = T_total_bins,
        n_substeps = 1,
        initial_state = collect(Float64, init_state),
        truth_params  = Dict{Symbol,Float64}(),
        theta_dim     = n_anchors,
        sigma_prior   = sigma_prior,
        prior_mean    = zeros(n_anchors),
        cost_fn       = cost_fn,
        schedule_from_theta = schedule_from_theta_fn,
    )
end

# ── Bench main ──────────────────────────────────────────────────────────────

function main()
    args = ARGS_DICT
    println("=" ^ 76)
    println("  FSA v1.5 closed-loop SMC²-MPC (SMC2FC_functional)")
    println("  T = $(args["T-days"]) d, BINS_PER_DAY = $BINS_PER_DAY, " *
            "step = $STEP_MIN min, smoke = $(args["smoke"])")
    println("=" ^ 76)

    if args["smoke"]
        n_smc      = 4
        k_pf       = 16
        ctrl_n_smc = 8
        n_strides  = 1
    else
        n_smc      = args["n-smc"]
        k_pf       = args["k-pf"]
        ctrl_n_smc = args["ctrl-n-smc"]
        n_strides   = (args["T-days"] * BINS_PER_DAY) ÷ (BINS_PER_DAY ÷ 2)
    end
    STRIDE_BINS = args["smoke"] ? BINS_PER_DAY : (BINS_PER_DAY ÷ 2)
    WINDOW_BINS = BINS_PER_DAY

    println("  n_strides = $n_strides, n_smc = $n_smc, k_pf = $k_pf, " *
            "ctrl_n_smc = $ctrl_n_smc")
    println("  stride_bins = $STRIDE_BINS, window_bins = $WINDOW_BINS")
    println()

    em = build_estimation_model()
    priors = all_priors(em)

    smc_cfg = SMCConfig(
        n_smc_particles = n_smc,
        target_ess_frac = 0.5,
        max_lambda_inc  = 0.25,
        num_mcmc_steps  = 2,
        hmc_step_size   = 0.05,
        hmc_num_leapfrog = 4,
        n_pf_particles   = k_pf,
        bandwidth_scale  = 1.0,
        ot_max_weight    = 0.0,
        bridge_type      = :gaussian,
    )

    ctrl_cfg = SMCConfig(
        n_smc_particles = ctrl_n_smc,
        target_ess_frac = 0.5,
        max_lambda_inc  = 0.25,
        num_mcmc_steps  = 2,
        hmc_step_size   = 0.1,
        hmc_num_leapfrog = 4,
        n_pf_particles   = k_pf,
        bandwidth_scale  = 1.0,
        ot_max_weight    = 0.0,
    )

    rng = MersenneTwister(args["seed"])
    plant_state = (Float64(INIT_STATE_NT.B),
                    Float64(INIT_STATE_NT.F),
                    Float64(INIT_STATE_NT.A))
    daily_phi   = fill(1.0, max(args["T-days"], 1))

    accumulated = (B = Float64[], F = Float64[], A = Float64[],
                    Phi = Float64[])
    full_traj   = Matrix{Float64}(undef, 0, 3)
    prev_particles = nothing
    posterior_samples = Dict{Int,Matrix{Float64}}()
    replan_log = NamedTuple[]

    t_start = time()
    for s in 1:n_strides
        ts = time()
        day_in_plan = ((s - 1) * STRIDE_BINS) ÷ BINS_PER_DAY + 1
        Φ_today = daily_phi[min(day_in_plan, length(daily_phi))]
        Φ_seq = fill(Φ_today, STRIDE_BINS)

        roll = plant_rollout(plant_state, Φ_seq, DEFAULT_PARAMS_DICT, DT_BIN_DAYS, rng)
        plant_state = roll.final_state
        full_traj   = vcat(full_traj, roll.trajectory)
        append!(accumulated.B, roll.obs_B)
        append!(accumulated.F, roll.obs_F)
        append!(accumulated.A, roll.obs_A)
        append!(accumulated.Phi, Φ_seq)

        @printf("  stride %d/%d  Φ=%.2f  rollout %d bins (%.1fs)\n",
                s, n_strides, Φ_today, STRIDE_BINS, time() - ts)

        n_obs = length(accumulated.B)
        if n_obs >= WINDOW_BINS
            grid_obs = Dict(
                :Phi   => Float64.(accumulated.Phi[end - WINDOW_BINS + 1 : end]),
                :obs_B => Float64.(accumulated.B[  end - WINDOW_BINS + 1 : end]),
                :obs_F => Float64.(accumulated.F[  end - WINDOW_BINS + 1 : end]),
                :obs_A => Float64.(accumulated.A[  end - WINDOW_BINS + 1 : end]),
            )
            init_idx = size(full_traj, 1) - WINDOW_BINS
            init_window = init_idx > 0 ?
                Float64[full_traj[init_idx, 1], full_traj[init_idx, 2], full_traj[init_idx, 3]] :
                Float64[INIT_STATE_NT.B, INIT_STATE_NT.F, INIT_STATE_NT.A]

            stride_seed = args["seed"] + 1000 * s
            # Type-GENERIC closure: do not collect to Float64.
            function loglik_fn(u::AbstractVector)
                pf_rng = MersenneTwister(stride_seed)
                return bootstrap_log_likelihood(em, u, grid_obs, init_window,
                                                 priors, smc_cfg, pf_rng;
                                                 dt = DT_BIN_DAYS,
                                                 t_steps = WINDOW_BINS,
                                                 window_start_bin = 0)
            end

            tf0 = time()
            res = if prev_particles === nothing
                run_smc_window(loglik_fn, priors, smc_cfg, MersenneTwister(stride_seed))
            else
                run_smc_window_bridge(loglik_fn, prev_particles, priors,
                                       smc_cfg, MersenneTwister(stride_seed))
            end
            prev_particles = res.particles
            posterior_samples[s] = copy(res.particles)
            @printf("    filter: %d temp levels in %.1fs\n",
                    res.n_temp, time() - tf0)
        else
            @printf("    filter: warmup (n_obs=%d / window=%d)\n", n_obs, WINDOW_BINS)
        end

        if !args["smoke"] && prev_particles !== nothing &&
           s > 0 && (s % args["replan-K"] == 0)
            u_mean = vec(mean(prev_particles; dims = 1))
            θ_mean = unconstrained_to_constrained(u_mean, priors)
            params_post = params_dict_from_constrained_vec(θ_mean)

            t_remaining_days = max(args["T-days"] - day_in_plan + 1, 1)
            T_total_bins     = t_remaining_days * BINS_PER_DAY
            cspec = build_control_spec(
                T_total_bins = T_total_bins,
                dt = DT_BIN_DAYS,
                params = params_post,
                init_state = plant_state,
                n_anchors = args["ctrl-n-anchors"],
                n_inner   = args["smoke"] ? 4 : 8,
                seed      = args["seed"] + 1000 * s,
            )
            tc0 = time()
            cres = run_tempered_smc_loop(cspec, ctrl_cfg,
                                          MersenneTwister(args["seed"] + 1000*s);
                                          calib_n = 32, target_nats = 4.0)
            Φ_plan = cres.mean_schedule
            n_plan_bins = length(Φ_plan)
            n_plan_days = max(n_plan_bins ÷ BINS_PER_DAY, 1)
            new_daily = Float64[]
            for d in 1:n_plan_days
                lo = (d - 1) * BINS_PER_DAY + 1
                hi = min(d * BINS_PER_DAY, n_plan_bins)
                push!(new_daily, mean(Φ_plan[lo:hi]))
            end
            day_now   = day_in_plan
            for di in 1:min(n_plan_days, length(daily_phi) - day_now + 1)
                daily_phi[day_now + di - 1] = new_daily[di]
            end
            push!(replan_log, (stride = s, mean_phi = mean(new_daily),
                                n_temp = cres.n_temp,
                                elapsed = time() - tc0))
            @printf("    replan: new daily Φ̄ = %.3f over next %d d (n_temp=%d, %.1fs)\n",
                    mean(new_daily), n_plan_days, cres.n_temp, time() - tc0)
        end
    end
    elapsed = time() - t_start

    println()
    @printf("  total: %.1fs for %d strides; mean A = %.3f\n",
            elapsed, n_strides, mean(full_traj[:, 3]))

    out_dir = isempty(args["out-dir"]) ?
        joinpath(@__DIR__, "outputs",
                  "fsa_v15_functional_T$(args["T-days"])$(args["smoke"] ? "_smoke" : "")_seed$(args["seed"])") :
        args["out-dir"]
    mkpath(out_dir)

    open(joinpath(out_dir, "manifest.txt"), "w") do io
        println(io, "bench: bench_smc_full_mpc_fsa_v15_functional.jl")
        println(io, "library: SMC2FC_functional (CPU-side)")
        println(io, "T_days: $(args["T-days"])")
        println(io, "n_strides: $n_strides")
        println(io, "stride_bins: $STRIDE_BINS")
        println(io, "window_bins: $WINDOW_BINS")
        println(io, "n_smc: $n_smc")
        println(io, "k_pf: $k_pf")
        println(io, "smoke: $(args["smoke"])")
        println(io, "elapsed_s: $(round(elapsed; digits = 2))")
        println(io, "mean_A: $(round(mean(full_traj[:, 3]); digits = 4))")
        println(io, "replans: $(length(replan_log))")
        for r in replan_log
            println(io, "  stride=$(r.stride)  mean_phi=$(round(r.mean_phi; digits=3))  n_temp=$(r.n_temp)")
        end
    end

    open(joinpath(out_dir, "trajectory.csv"), "w") do io
        println(io, "bin,B,F,A")
        for k in 1:size(full_traj, 1)
            @printf(io, "%d,%.6f,%.6f,%.6f\n",
                    k, full_traj[k, 1], full_traj[k, 2], full_traj[k, 3])
        end
    end

    println("  artefacts → $out_dir")
    println("=" ^ 76)
    return (elapsed = elapsed, n_strides = n_strides,
             mean_A = mean(full_traj[:, 3]),
             posterior_samples = posterior_samples,
             replan_log = replan_log)
end

main()
