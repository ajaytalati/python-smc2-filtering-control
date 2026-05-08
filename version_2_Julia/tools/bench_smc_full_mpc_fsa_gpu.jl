#!/usr/bin/env julia
# Closed-loop SMC²-MPC bench for FSA-v2 — GPU parallel-chains path.
#
# Per-window: outer SMC² with tempered + parallel-HMC kernel from gpu_pf.jl.
# Plant: stride-wise StepwisePlant under Φ=Phi_default=1.0 (canonical Banister).
# Filter only — no controller call (the param-trace plot is a filter artifact;
# the controller doesn't enter unless we want closed-loop control). The
# Python E5 bench does both filter + controller; this Julia bench focuses on
# reproducing the FILTER posterior plot.
#
# Output: data.jld2 + 30-panel param-trace PNG matching Python's format.

using Dates


function _parse_args(argv::Vector{String})
    defaults = Dict{String,Any}(
        "T-days"        => 14,
        "step-minutes"  => 15,        # matches Python + spec h=15min
        "replan-K"      => 1,         # replan every stride (per Algorithm 3)
        "N-smc"         => 32,
        "K-per-chain"   => 400,
        "num-mcmc"      => 5,
        "hmc-step-size" => 0.025,
        "hmc-leapfrog"  => 8,
        "max-lambda-inc" => 0.10,
        "target-ess-frac" => 0.5,
        "max-temp-levels" => 30,
        "seed"          => 42,
        "output-dir"    => "",
        # Open-loop mode: do ONE initial plan at INIT_STATE+TRUTH_PARAMS,
        # apply it throughout the bench, do not replan. Mirrors the ground-
        # truth driver tools/test_max_A_with_F_barrier.jl. Set "true" to
        # enable; default "false" runs the closed-loop replanning bench.
        "open-loop"     => "false",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            key = a[3:end]
            if !haskey(defaults, key)
                error("Unknown flag: $a")
            end
            i += 1
            v = argv[i]
            if defaults[key] isa Int
                defaults[key] = parse(Int, v)
            elseif defaults[key] isa Float64
                defaults[key] = parse(Float64, v)
            else
                defaults[key] = v
            end
            i += 1
        else
            error("Unrecognized arg: $a")
        end
    end
    return defaults
end


const ARGS_DICT = _parse_args(copy(ARGS))
ENV["FSA_STEP_MINUTES"] = string(ARGS_DICT["step-minutes"])

@info "loading model + framework (FSA_STEP_MINUTES=$(ENV["FSA_STEP_MINUTES"]))..."

using Random, Statistics, Printf, LinearAlgebra
using LogExpFunctions: logsumexp
using JLD2
using JSON3
using CUDA

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.PhiBurst: BINS_PER_DAY
using .FSAHighRes.Plant: StepwisePlant, advance!, advance_subdaily!
using .FSAHighRes.Estimation: build_estimation_model, align_obs_fn,
                               PARAM_NAMES, PARAM_PRIOR_CONFIG, get_init_theta
using .FSAHighRes.GPUPF: FSAGPUTargetBatched, gpu_log_density_batched,
                          gpu_grads_parallel_chains, parallel_hmc_one_move!,
                          update_window_obs!
using .FSAHighRes.Control: build_control
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn
using SMC2FC: run_tempered_smc_gpu
using .FSAHighRes.Dynamics: TRUTH_PARAMS
using SMC2FC: LogNormalPrior, NormalPrior


# ── Build prior mean/sigma in unconstrained space ────────────────────────

function _prior_unconstrained_mean_sigma()
    n = length(PARAM_PRIOR_CONFIG)
    means  = zeros(Float64, n)
    sigmas = zeros(Float64, n)
    for (i, (_, p)) in enumerate(PARAM_PRIOR_CONFIG)
        if p isa LogNormalPrior
            means[i]  = p.μ
            sigmas[i] = p.σ
        elseif p isa NormalPrior
            means[i]  = p.μ
            sigmas[i] = p.σ
        else
            means[i]  = 0.0
            sigmas[i] = 1.0
        end
    end
    return means, sigmas
end


# ── Outer SMC² loop (tempered + parallel HMC) ────────────────────────────

function run_outer_smc_gpu(target::FSAGPUTargetBatched,
                            n_smc::Int, max_lambda_inc::Float64,
                            target_ess_frac::Float64, num_mcmc::Int,
                            ε::Float64, L::Int, max_levels::Int,
                            prior_means::Vector{Float64},
                            prior_sigmas::Vector{Float64},
                            rng::AbstractRNG;
                            init_particles::Union{Nothing,Matrix{Float64}} = nothing)
    d = length(prior_means)
    if init_particles === nothing
        # Cold-start from prior.
        U = Matrix{Float64}(undef, n_smc, d)
        for m in 1:n_smc, j in 1:d
            U[m, j] = prior_means[j] + prior_sigmas[j] * randn(rng)
        end
    else
        U = copy(init_particles)
    end

    λ = 0.0
    n_temp = 0
    L_used = L

    while λ < 1.0 - 1e-6
        ll_data = gpu_log_density_batched(target, U)

        δ_max = min(1.0 - λ, max_lambda_inc)
        target_ess = target_ess_frac * n_smc
        function ess_at(δ)
            log_w = δ .* ll_data
            log_w_n = log_w .- logsumexp(log_w)
            return exp(-logsumexp(2.0 .* log_w_n))
        end
        δ = if ess_at(δ_max) >= target_ess
            δ_max
        else
            lo, hi = 0.0, δ_max
            for _ in 1:30
                mid = 0.5 * (lo + hi)
                if ess_at(mid) > target_ess; lo = mid; else; hi = mid; end
            end
            lo
        end
        next_λ = (λ + δ < 1.0 - 1e-6) ? λ + δ : 1.0
        Δλ = next_λ - λ

        log_w = Δλ .* ll_data
        w = exp.(log_w .- logsumexp(log_w))
        cumsum_w = cumsum(w)
        indices = Vector{Int}(undef, n_smc)
        u_shift = rand(rng) / n_smc
        for i in 1:n_smc
            t = (i - 1) / n_smc + u_shift
            indices[i] = clamp(searchsortedfirst(cumsum_w, t), 1, n_smc)
        end
        U_resampled = U[indices, :]

        n_acc_total = 0
        for _ in 1:num_mcmc
            n_acc_total += parallel_hmc_one_move!(U_resampled, target, ε, L_used,
                                                   prior_means, prior_sigmas, rng)
        end
        accept_frac = n_acc_total / (num_mcmc * n_smc)
        copyto!(U, U_resampled)
        n_temp += 1

        @info @sprintf("  [%2d] λ %.3f → %.3f  (Δλ=%.3f)  L=%d  accept=%.0f%%",
                       n_temp, λ, next_λ, Δλ, L_used, 100 * accept_frac)
        λ = next_λ
        n_temp >= max_levels && break
    end
    return U, n_temp
end


# ── Convert U_unc (n_smc, 30) to constrained particle cloud ──────────────

const LOGNORMAL_MASK = Bool[
    true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,
    false, true,  true,  false, true,
    true,  true,  false,
    false, true,  true,  false, true,
    false, true,  true,  true,  false, true,
]

function constrain_cloud(U::Matrix{Float64})
    n_smc, d = size(U)
    out = Matrix{Float64}(undef, n_smc, d)
    @inbounds for m in 1:n_smc, j in 1:d
        out[m, j] = LOGNORMAL_MASK[j] ? exp(clamp(U[m, j], -20.0, 20.0)) : U[m, j]
    end
    return out
end


# ── Helper: build window obs from the plant's accumulated history ────────
# We accumulate a rolling-window grid for each stride.

function _build_window_grid_obs(plant::StepwisePlant, t0_bin::Int, T_window::Int, dt::Real)
    # Slice the plant's history channels into [t0_bin, t0_bin + T_window).
    # Plant history: trajectory, Phi_value, C_value, obs_sleep_label
    #               (per-channel HR/stress/steps t_idx + value lists).
    traj = vcat(plant.history[:trajectory]...)
    phi  = vcat(plant.history[:Phi_value]...)
    cval = vcat(plant.history[:C_value]...)
    sleep_lab = vcat(plant.history[:obs_sleep_label]...)
    hr_idx = Int.(vcat(plant.history[:obs_HR_t_idx]...))
    hr_val = vcat(plant.history[:obs_HR_value]...)
    s_idx = Int.(vcat(plant.history[:obs_stress_t_idx]...))
    s_val = vcat(plant.history[:obs_stress_value]...)
    st_idx = Int.(vcat(plant.history[:obs_steps_t_idx]...))
    st_val = vcat(plant.history[:obs_steps_value]...)

    # Build per-bin window indices (0-based global → 0-based local).
    obs_HR_t = Int32[]; obs_HR_v = Float32[]
    obs_S_t  = Int32[]; obs_S_v  = Float32[]
    obs_St_t = Int32[]; obs_St_v = Float32[]
    for (gi, v) in zip(hr_idx, hr_val)
        if t0_bin <= gi < t0_bin + T_window
            push!(obs_HR_t, Int32(gi - t0_bin)); push!(obs_HR_v, Float32(v))
        end
    end
    for (gi, v) in zip(s_idx, s_val)
        if t0_bin <= gi < t0_bin + T_window
            push!(obs_S_t, Int32(gi - t0_bin)); push!(obs_S_v, Float32(v))
        end
    end
    for (gi, v) in zip(st_idx, st_val)
        if t0_bin <= gi < t0_bin + T_window
            push!(obs_St_t, Int32(gi - t0_bin)); push!(obs_St_v, Float32(v))
        end
    end
    sleep_window_label = sleep_lab[t0_bin + 1 : t0_bin + T_window]

    obs_data = Dict{Symbol,Any}(
        :obs_HR     => Dict(:t_idx => obs_HR_t, :obs_value => obs_HR_v),
        :obs_stress => Dict(:t_idx => obs_S_t,  :obs_value => obs_S_v),
        :obs_steps  => Dict(:t_idx => obs_St_t, :obs_value => obs_St_v),
        :obs_sleep  => Dict(:t_idx => collect(Int32, 0:T_window-1),
                             :sleep_label => Int32.(sleep_window_label)),
        :Phi        => Dict(:Phi_value => Float32.(phi[t0_bin + 1 : t0_bin + T_window])),
        :C          => Dict(:C_value   => Float32.(cval[t0_bin + 1 : t0_bin + T_window])),
    )
    grid_obs = align_obs_fn(obs_data, T_window, dt)
    # Initial state for this window = state at t0_bin (read from traj).
    init_state = t0_bin == 0 ?
        [0.05, 0.30, 0.10] :
        Float64.(traj[t0_bin, :])
    return grid_obs, init_state
end


function main(args::Dict{String,Any})
    BINS_PER_DAY_LOCAL = (60 * 24) ÷ args["step-minutes"]
    WINDOW_BINS = BINS_PER_DAY_LOCAL
    STRIDE_BINS = WINDOW_BINS ÷ 2
    DT_DAYS     = 1.0 / BINS_PER_DAY_LOCAL
    T_total_bins = args["T-days"] * BINS_PER_DAY_LOCAL
    n_strides   = (T_total_bins - WINDOW_BINS) ÷ STRIDE_BINS + 1

    @info "config: T=$(args["T-days"])d, step=$(args["step-minutes"])min, " *
          "BINS_PER_DAY=$BINS_PER_DAY_LOCAL, WINDOW=$WINDOW_BINS, STRIDE=$STRIDE_BINS, " *
          "n_strides=$n_strides"

    out_dir = isempty(args["output-dir"]) ?
        joinpath(REPO_ROOT, "outputs", "fsa_high_res", "g4_runs",
                  "T$(args["T-days"])d_replanK$(args["replan-K"])_h$(args["step-minutes"])min_no_infoaware") :
        args["output-dir"]
    mkpath(out_dir)

    @info "GPU device: $(CUDA.name(CUDA.device()))"

    # ── Two plants: MPC (driven by closed-loop controller) and baseline (Φ=1.0) ──
    @info "building MPC and baseline plants..."
    mpc_plant = StepwisePlant(seed_offset=args["seed"], dt=DT_DAYS)
    base_plant = StepwisePlant(seed_offset=args["seed"], dt=DT_DAYS)
    @info "  mpc plant init state: B=$(round(mpc_plant.state[1], digits=3)), " *
          "F=$(round(mpc_plant.state[2], digits=3)), A=$(round(mpc_plant.state[3], digits=3))"

    # ── Build GPU PF target (sized for parallel HMC) ─────────────────────
    n_smc = args["N-smc"]
    K_per_chain = args["K-per-chain"]
    d = 30
    M_max = n_smc * (1 + 2 * d)    # parallel HMC requires this many chains in one launch
    @info "building GPU target: N_SMC=$n_smc, K_per_chain=$K_per_chain, M_max=$M_max"
    target = FSAGPUTargetBatched(K_per_chain=K_per_chain, M_max=M_max,
                                  T_steps=WINDOW_BINS, dt=DT_DAYS, noise_seed=0)

    # ── Prior in unconstrained space ─────────────────────────────────────
    prior_means, prior_sigmas = _prior_unconstrained_mean_sigma()
    @info "prior means (first 5): $(round.(prior_means[1:5], digits=3))"

    # ── Closed-loop MPC: filter → posterior → replan → apply Φ → next stride ──
    posterior_particles = zeros(Float32, n_strides, n_smc, d)
    elapsed_per_window = zeros(Float64, n_strides)
    n_temp_per_window = zeros(Int, n_strides)
    daily_phi_plan_per_stride = zeros(Float32, n_strides)
    replan_strides = Int[]
    n_replans = 0
    replan_K = args["replan-K"]

    rng = MersenneTwister(args["seed"])
    init_particles = nothing

    # Controller config — match Python manifest exactly.
    ctrl_n_smc      = 1024
    ctrl_n_inner    = 128
    ctrl_n_anchors  = 8
    ctrl_target_ess_frac = 0.5
    ctrl_num_mcmc   = 10        # Python: 10
    ctrl_max_lambda_inc = 0.20
    ctrl_hmc_step   = 0.2       # Python: 0.2 (4× mine)
    ctrl_hmc_leap   = 16        # Python: 16 (2× mine) — 8× longer HMC trajectory
    ctrl_max_levels = 30        # Python: 30
    ctrl_target_nats = 8.0
    ctrl_sigma_prior = 1.5
    ctrl_M_max      = ctrl_n_smc * (1 + 2 * ctrl_n_anchors)   # 1024 * 17 = 17408 chains

    # State extraction (Algorithm 3 step 2): take the importance-weighted
    # mean of the inner-PF particle cloud at the end of the current window.
    # Each filter chain m carries its own θ_dyn sample; its K_per_chain state
    # particles, weighted by the filter log-weights, give p(x_T | θ_dyn^{(m)}, y).
    # Averaging the per-chain weighted means across M chains marginalises θ_dyn.
    function _extract_xhat(target_filter, M_smc::Int)
        K = target_filter.K_per_chain
        Ntot = M_smc * K
        parts_cpu = Array(view(target_filter.bufs.particles_b, 1:Ntot, :))
        logw_cpu  = Array(view(target_filter.bufs.log_w, 1:Ntot))
        x_hat = zeros(Float64, 3)
        @inbounds for m in 1:M_smc
            base = (m - 1) * K
            col = logw_cpu[base + 1 : base + K]
            lm = maximum(col)
            isfinite(lm) || continue
            sume = sum(exp.(Float64.(col) .- Float64(lm)))
            sume > 0 || continue
            for k in 1:K
                w_norm = exp(Float64(col[k]) - Float64(lm)) / sume / Float64(M_smc)
                x_hat[1] += w_norm * Float64(parts_cpu[base + k, 1])
                x_hat[2] += w_norm * Float64(parts_cpu[base + k, 2])
                x_hat[3] += w_norm * Float64(parts_cpu[base + k, 3])
            end
        end
        return x_hat
    end

    # Helper: build NamedTuple of constrained-space dynamics params from
    # the current posterior mean (or truth at stride 1 cold-start).
    function _params_nt_from_posterior(post_constrained::Matrix{Float32})
        post_mean = vec(mean(post_constrained; dims=1))
        # PARAM_NAMES order: tau_B, tau_F, kappa_B, kappa_F, epsilon_A, lambda_A,
        # mu_0, mu_B, mu_F, mu_FF, eta, ...
        return (
            tau_B    = Float64(post_mean[1]),
            tau_F    = Float64(post_mean[2]),
            kappa_B  = Float64(post_mean[3]),
            kappa_F  = Float64(post_mean[4]),
            epsilon_A = Float64(post_mean[5]),
            lambda_A  = Float64(post_mean[6]),
            mu_0     = Float64(post_mean[7]),
            mu_B     = Float64(post_mean[8]),
            mu_F     = Float64(post_mean[9]),
            mu_FF    = Float64(post_mean[10]),
            eta      = Float64(post_mean[11]),
            sigma_B  = 0.010,
            sigma_F  = 0.012,
            sigma_A  = 0.020,
        )
    end

    # Plan-state: per-bin Φ for the remaining horizon.
    # Initialise to the BASELINE schedule (Φ=Phi_default=1.0) over the entire
    # horizon. This mirrors Python's bench_smc_full_mpc_fsa.py:184
    #   daily_phi_plan = np.full(T_total_days, daily_phi_baseline)
    # which keeps the plant on the canonical-Banister default for the first
    # K strides until the first replan can run.  Without this, Julia replans
    # at stride 1 from the cold INIT_STATE and applies the recovery-start of
    # the resulting plan, which permanently keeps the plant in the recovery
    # regime — every subsequent replan is from a still-recovering state and
    # picks the same recovery-start.  Cumulative applied Φ collapses to the
    # flat-low [0.10, 0.30] band documented in writeup §2.
    current_phi_per_bin = fill(Float32(1.0), T_total_bins)
    current_plan_offset_bin = 0
    last_xhat = [Float64(INIT_STATE.B), Float64(INIT_STATE.F), Float64(INIT_STATE.A)]

    is_open_loop = lowercase(args["open-loop"]) in ("true", "1", "yes")

    if is_open_loop
        @info "running OPEN-LOOP bench: one initial plan at INIT_STATE+TRUTH_PARAMS, applied for $(n_strides) strides (no replan). Filter still runs but does not feed the controller."
    else
        @info "running closed-loop MPC SMC² ($(n_strides) windows, replan every K=$(replan_K))..."
    end
    t_start = time()

    # ── Open-loop initial plan ───────────────────────────────────────────
    # Mirror tools/test_max_A_with_F_barrier.jl's structure: ONE plan up
    # front from canonical INIT_STATE + TRUTH_PARAMS, applied unmodified
    # for the whole bench. The current_phi_per_bin buffer is overwritten
    # before the loop body so stride 1 already gets the planned Φ instead
    # of baseline.
    if is_open_loop
        params_init = (
            tau_B    = Float64(TRUTH_PARAMS.tau_B),
            tau_F    = Float64(TRUTH_PARAMS.tau_F),
            kappa_B  = Float64(TRUTH_PARAMS.kappa_B),
            kappa_F  = Float64(TRUTH_PARAMS.kappa_F),
            epsilon_A = Float64(TRUTH_PARAMS.epsilon_A),
            lambda_A  = Float64(TRUTH_PARAMS.lambda_A),
            mu_0     = Float64(TRUTH_PARAMS.mu_0),
            mu_B     = Float64(TRUTH_PARAMS.mu_B),
            mu_F     = Float64(TRUTH_PARAMS.mu_F),
            mu_FF    = Float64(TRUTH_PARAMS.mu_FF),
            eta      = Float64(TRUTH_PARAMS.eta),
            sigma_B  = Float64(TRUTH_PARAMS.sigma_B),
            sigma_F  = Float64(TRUTH_PARAMS.sigma_F),
            sigma_A  = Float64(TRUTH_PARAMS.sigma_A),
        )
        ctrl_target_init = FSAControlGPUTarget(
            n_inner   = ctrl_n_inner,
            M_max     = ctrl_M_max,
            n_steps   = T_total_bins,                   # full bench duration
            n_anchors = ctrl_n_anchors,
            n_substeps = max(1, BINS_PER_DAY_LOCAL ÷ 24),
            dt        = DT_DAYS,
            F_max     = 0.40,
            Phi_max   = 3.0,
            Phi_default = 1.0,
            lam_F     = 1.0,
            sigma_prior = ctrl_sigma_prior,
            params    = params_init,
            init_state = Float32[INIT_STATE.B, INIT_STATE.F, INIT_STATE.A],
            noise_seed = args["seed"],
        )
        log_density_init = make_log_density_fn(ctrl_target_init)
        rng_init = MersenneTwister(args["seed"])
        U_init, n_temp_init, _ = run_tempered_smc_gpu(
            log_density_init, ctrl_M_max, ctrl_n_smc, ctrl_n_anchors,
            0.0, ctrl_sigma_prior, rng_init;
            target_nats        = ctrl_target_nats,
            target_ess_frac    = ctrl_target_ess_frac,
            max_lambda_inc     = ctrl_max_lambda_inc,
            max_temp_levels    = ctrl_max_levels,
            num_mcmc_steps     = ctrl_num_mcmc,
            hmc_step_size      = ctrl_hmc_step,
            hmc_num_leapfrog   = ctrl_hmc_leap,
            chees_L_candidates = [16, 32, 64, 128, 256],
            h_fd               = 1e-4,
            calib_n            = 64,
            verbose            = false,
        )
        theta_post = vec(mean(U_init; dims=1))
        T_total_plan = T_total_bins * DT_DAYS
        t_grid_local = collect(0:T_total_bins-1) .* DT_DAYS
        anchors_local = collect(range(0.0, T_total_plan; length=ctrl_n_anchors))
        σ_rbf = T_total_plan / ctrl_n_anchors
        Phi_plan = zeros(Float32, T_total_bins)
        for k in 1:T_total_bins
            raw = Float64(ctrl_target_init.c_Phi)
            for j in 1:ctrl_n_anchors
                v = exp(-0.5 * ((t_grid_local[k] - anchors_local[j]) / σ_rbf)^2)
                raw += theta_post[j] * v
            end
            Phi_plan[k] = Float32(3.0 / (1.0 + exp(-raw)))
        end
        current_phi_per_bin = Phi_plan
        current_plan_offset_bin = 0
        n_replans = 1
        push!(replan_strides, 0)
        q = length(Phi_plan)
        @info @sprintf("  [open-loop initial plan] %d levels, Phi mean=%.3f  shape=[%.2f→%.2f→%.2f→%.2f→%.2f]",
                        n_temp_init, mean(Phi_plan),
                        Phi_plan[1], Phi_plan[max(1, q÷4)], Phi_plan[max(1, q÷2)],
                        Phi_plan[max(1, (3*q)÷4)], Phi_plan[end])
    end

    for s in 1:n_strides
        t_window = time()

        # ── Slice next stride's Φ from current plan ────────────────────
        if s == 1
            # First stride: WINDOW_BINS to seed the first window.
            advance_bins = WINDOW_BINS
        else
            advance_bins = STRIDE_BINS
        end
        slice_end = current_plan_offset_bin + advance_bins
        if slice_end > length(current_phi_per_bin)
            # Pad with last value if plan ran out (shouldn't happen).
            pad = slice_end - length(current_phi_per_bin)
            current_phi_per_bin = vcat(current_phi_per_bin,
                                        fill(current_phi_per_bin[end], pad))
        end
        phi_stride = current_phi_per_bin[current_plan_offset_bin + 1 : slice_end]
        current_plan_offset_bin += advance_bins

        # The bench used to aggregate the controller's per-bin Φ to a
        # single daily Φ and re-expand it through the burst envelope. That
        # threw away the controller's actual schedule (which is smooth
        # per-bin Φ via the sigmoid-RBF decoder) and replaced it with a
        # different signal: a daily-averaged Φ refracted through a Gamma-
        # shaped morning-burst envelope. Plant ↔ controller mismatch.
        # The ground-truth driver tools/test_max_A_with_F_barrier.jl uses
        # `advance_subdaily!` to apply the controller's per-bin Φ DIRECTLY
        # — same signal the cost kernel evaluates against. Doing the same
        # here.  Stash the daily mean only for diagnostic plotting.
        daily_phi_plan_per_stride[s] = mean(phi_stride)

        # ── Advance MPC plant under controller's exact per-bin Φ ───────
        advance_subdaily!(mpc_plant, Float32.(phi_stride))
        # ── Advance baseline plant under constant Φ=1.0 (smooth) ───────
        advance_subdaily!(base_plant, fill(Float32(1.0), advance_bins))
        # No envelope_mode kwarg / --plant switch — both modes were
        # equivalent for the controller-plant matching question (writeup
        # §2.10), the actual fix was to drop the burst-envelope refraction
        # entirely and call advance_subdaily!.

        # ── Filter on MPC plant's window obs ────────────────────────────
        t0_bin = (s - 1) * STRIDE_BINS
        grid_obs, init_state = _build_window_grid_obs(mpc_plant, t0_bin, WINDOW_BINS, DT_DAYS)
        update_window_obs!(target, grid_obs;
                           B_init = init_state[1],
                           F_init = init_state[2],
                           A_init = init_state[3])

        U_post, n_temp = run_outer_smc_gpu(
            target, n_smc, args["max-lambda-inc"], args["target-ess-frac"],
            args["num-mcmc"], args["hmc-step-size"], args["hmc-leapfrog"],
            args["max-temp-levels"], prior_means, prior_sigmas, rng;
            init_particles = init_particles,
        )
        posterior_particles[s, :, :] = Float32.(constrain_cloud(U_post))
        init_particles = copy(U_post)

        # Extract smoothed state x̂_n (Algorithm 3 step 2): importance-weighted
        # mean over the inner-PF cloud at end-of-window. Used as init_state for
        # the controller at the NEXT replan boundary.
        last_xhat = _extract_xhat(target, n_smc)

        # ── End-of-stride replan (mirrors Python's `if (s+1) % K == 0`) ────
        # Skipped entirely in --open-loop mode: the bench applies the single
        # initial plan throughout (matches the ground-truth driver
        # tools/test_max_A_with_F_barrier.jl).
        if !is_open_loop && s % replan_K == 0
            # Stride 1 cold-start uses TRUTH_PARAMS (no posterior yet);
            # subsequent replans use the filter's posterior-mean params.
            params_nt = if posterior_particles[s, 1, 1] == 0f0
                # No posterior yet (filter row still zeros) — cold start.
                (
                    tau_B    = Float64(TRUTH_PARAMS.tau_B),
                    tau_F    = Float64(TRUTH_PARAMS.tau_F),
                    kappa_B  = Float64(TRUTH_PARAMS.kappa_B),
                    kappa_F  = Float64(TRUTH_PARAMS.kappa_F),
                    epsilon_A = Float64(TRUTH_PARAMS.epsilon_A),
                    lambda_A  = Float64(TRUTH_PARAMS.lambda_A),
                    mu_0     = Float64(TRUTH_PARAMS.mu_0),
                    mu_B     = Float64(TRUTH_PARAMS.mu_B),
                    mu_F     = Float64(TRUTH_PARAMS.mu_F),
                    mu_FF    = Float64(TRUTH_PARAMS.mu_FF),
                    eta      = Float64(TRUTH_PARAMS.eta),
                    sigma_B  = Float64(TRUTH_PARAMS.sigma_B),
                    sigma_F  = Float64(TRUTH_PARAMS.sigma_F),
                    sigma_A  = Float64(TRUTH_PARAMS.sigma_A),
                )
            else
                _params_nt_from_posterior(posterior_particles[s, :, :])
            end

            # ──────────────────────────────────────────────────────────
            # Two distinct concepts, conflated in the original bench:
            #
            #   bench_duration_days  = how long the CLOSED-LOOP runs.
            #                          (= args["T-days"], the experiment.)
            #   planning_horizon_days = how far AHEAD the controller looks
            #                          at each replan. Held FIXED for the
            #                          whole bench. Set to T_total_days here
            #                          (one chronic-B time constant) so each
            #                          plan has enough lookahead for the
            #                          optimum to recover the U-shape.
            #
            # Original (incorrect) MPC formulation: plan the SHRINKING
            # remaining-bench horizon. At late strides this gave the
            # controller only ~4 days of lookahead, which collapsed the
            # cost optimum from "overload, recover, overload" (U-shape) to
            # "just recover" — there was no longer enough time left in the
            # planning window to commit to a final overload phase. The
            # cumulative applied schedule was therefore stuck in the
            # recovery band [0.10, 0.30] across every late stride.
            #
            # Correct receding-horizon formulation: plan a fixed-size
            # window AHEAD at every replan, regardless of how much bench
            # is left. Python's bench_smc_full_mpc_fsa.py:389 hard-codes
            # this via `plan_horizon_days=T_total_days`. The plan extends
            # past the bench end at late strides, but only the first
            # stride is committed before the next replan, so that's fine.
            # ──────────────────────────────────────────────────────────
            planning_horizon_days = Float64(args["T-days"])
            remaining_bins = args["T-days"] * BINS_PER_DAY_LOCAL - mpc_plant.t_bin
            remaining_days = remaining_bins / BINS_PER_DAY_LOCAL

            if remaining_days >= 0.5
                ctrl_n_steps = Int(round(planning_horizon_days / DT_DAYS))
                ctrl_target = FSAControlGPUTarget(
                    n_inner   = ctrl_n_inner,
                    M_max     = ctrl_M_max,
                    n_steps   = ctrl_n_steps,
                    n_anchors = ctrl_n_anchors,
                    n_substeps = max(1, BINS_PER_DAY_LOCAL ÷ 24),
                    dt        = DT_DAYS,
                    F_max     = 0.40,
                    Phi_max   = 3.0,
                    Phi_default = 1.0,
                    lam_F     = 1.0,
                    sigma_prior = ctrl_sigma_prior,
                    params    = params_nt,
                    init_state = Float32.(copy(last_xhat)),
                    noise_seed = args["seed"] + 100 + s,
                )

                t_plan = time()
                log_density_fn = make_log_density_fn(ctrl_target)
                rng_ctrl = MersenneTwister(args["seed"] + 100 + s)

                U_ctrl, ctrl_n_temp, _ = run_tempered_smc_gpu(
                    log_density_fn, ctrl_M_max, ctrl_n_smc, ctrl_n_anchors,
                    0.0, ctrl_sigma_prior, rng_ctrl;
                    target_nats        = ctrl_target_nats,
                    target_ess_frac    = ctrl_target_ess_frac,
                    max_lambda_inc     = ctrl_max_lambda_inc,
                    max_temp_levels    = ctrl_max_levels,
                    num_mcmc_steps     = ctrl_num_mcmc,
                    hmc_step_size      = ctrl_hmc_step,
                    hmc_num_leapfrog   = ctrl_hmc_leap,
                    chees_L_candidates = [16, 32, 64, 128, 256],
                    h_fd               = 1e-4,
                    calib_n            = 64,
                    verbose            = false,
                )

                theta_post_mean = vec(mean(U_ctrl; dims=1))
                T_total = ctrl_n_steps * DT_DAYS
                t_grid_local = collect(0:ctrl_n_steps-1) .* DT_DAYS
                anchors_local = collect(range(0.0, T_total; length=ctrl_n_anchors))
                σ_rbf = T_total / ctrl_n_anchors
                Phi_arr_local = zeros(Float32, ctrl_n_steps)
                for k in 1:ctrl_n_steps
                    raw = Float64(ctrl_target.c_Phi)
                    for j in 1:ctrl_n_anchors
                        v = exp(-0.5 * ((t_grid_local[k] - anchors_local[j]) / σ_rbf)^2)
                        raw += theta_post_mean[j] * v
                    end
                    Phi_arr_local[k] = Float32(3.0 / (1.0 + exp(-raw)))
                end

                elapsed_plan = time() - t_plan
                current_phi_per_bin = Phi_arr_local
                current_plan_offset_bin = 0
                n_replans += 1
                push!(replan_strides, s)
                q = length(Phi_arr_local)
                phi_samples = [Phi_arr_local[1],
                               Phi_arr_local[max(1, q÷4)],
                               Phi_arr_local[max(1, q÷2)],
                               Phi_arr_local[max(1, (3*q)÷4)],
                               Phi_arr_local[end]]
                @info @sprintf("  [end-of-stride %2d replan] GPU %d levels, %.1fs, Phi mean=%.3f  shape=[%.2f→%.2f→%.2f→%.2f→%.2f]",
                                s, ctrl_n_temp, elapsed_plan,
                                mean(current_phi_per_bin),
                                phi_samples[1], phi_samples[2], phi_samples[3],
                                phi_samples[4], phi_samples[5])
            end
        end

        elapsed_per_window[s] = time() - t_window
        n_temp_per_window[s]  = n_temp
        @info @sprintf("[stride %2d/%2d] %.1fs  %d filter levels  Φ=%.3f  x̂=(%.3f, %.3f, %.3f)",
                       s, n_strides, elapsed_per_window[s], n_temp,
                       daily_phi_plan_per_stride[s],
                       last_xhat[1], last_xhat[2], last_xhat[3])
    end
    t_total = time() - t_start
    @info @sprintf("Total wall time: %.1f s  (%d replans)", t_total, n_replans)

    # ── Save data.jld2 with the schema the plotter expects ───────────────
    truth = Float64[
        TRUTH_PARAMS.tau_B, TRUTH_PARAMS.tau_F, TRUTH_PARAMS.kappa_B,
        TRUTH_PARAMS.kappa_F, TRUTH_PARAMS.epsilon_A, TRUTH_PARAMS.lambda_A,
        TRUTH_PARAMS.mu_0, TRUTH_PARAMS.mu_B, TRUTH_PARAMS.mu_F,
        TRUTH_PARAMS.mu_FF, TRUTH_PARAMS.eta,
        62.0, 12.0, 3.0, -2.5, 2.0,
        3.0, 2.0, 0.5,
        30.0, 20.0, 8.0, -4.0, 4.0,
        5.5, 0.8, 0.5, 0.3, -0.8, 0.5,
    ]
    # MPC and baseline plant trajectories.
    traj_mpc  = vcat(mpc_plant.history[:trajectory]...)
    traj_base = vcat(base_plant.history[:trajectory]...)
    phi_mpc   = vcat(mpc_plant.history[:Phi_value]...)
    phi_base  = vcat(base_plant.history[:Phi_value]...)
    F_max = 0.40

    data_path = joinpath(out_dir, "data.jld2")
    JLD2.jldopen(data_path, "w") do f
        f["posterior_particles"] = posterior_particles
        f["truth_params"]        = truth
        f["param_names"]         = String.(PARAM_NAMES)
        f["n_strides"]           = n_strides
        f["elapsed_per_window_s"] = elapsed_per_window
        f["n_temp_per_window"]    = n_temp_per_window
        f["trajectory_mpc"]      = Float32.(traj_mpc)
        f["trajectory_baseline"] = Float32.(traj_base)
        f["Phi_per_bin_mpc"]     = Float32.(phi_mpc)
        f["Phi_per_bin_baseline"] = Float32.(phi_base)
        f["daily_phi_per_stride"] = Float32.(daily_phi_plan_per_stride)
        f["replan_strides"]       = Int32.(replan_strides)
        f["n_replans"]           = n_replans
        f["BINS_PER_DAY"]        = BINS_PER_DAY_LOCAL
        f["STRIDE_BINS"]         = STRIDE_BINS
        f["dt_days"]             = DT_DAYS
        f["F_max"]               = F_max
    end
    @info "wrote $(data_path)"

    # ── Auto-generate diagnostic plots ──────────────────────────────────
    # State traces (4-panel: B, F, A, applied Φ) and the 30-panel parameter
    # trace plot. Equivalent to running tools/plot_state_traces.jl and
    # tools/plot_param_traces.jl on the just-written data.jld2 — done
    # in-process so a finished bench produces ready-to-view plots.
    try
        let plotters_dir = @__DIR__
            include(joinpath(plotters_dir, "plot_state_traces.jl"))
            include(joinpath(plotters_dir, "plot_param_traces.jl"))
            # `include` inside main() puts new methods in a newer world age
            # than the calling frame, so direct calls hit MethodError. Wrap
            # via invokelatest to defer dispatch to the latest world.
            data_dict = Base.invokelatest(Main.load_run_data, data_path)

            traces_path = joinpath(out_dir, "E5_full_mpc_T$(args["T-days"])d_traces.png")
            Base.invokelatest(Main.plot_state_traces, data_dict; out_path=traces_path)
            @info "wrote $traces_path"

            params_path = joinpath(out_dir, "E5_full_mpc_T$(args["T-days"])d_param_traces.png")
            Base.invokelatest(Main.plot_param_traces, data_dict; out_path=params_path,
                              T_total_days=Float64(args["T-days"]),
                              step_minutes=args["step-minutes"],
                              stride_bins=STRIDE_BINS)
            @info "wrote $params_path"
        end
    catch e
        @warn "plot generation failed (data.jld2 still saved)" exception=(e, catch_backtrace())
    end

    # ── Manifest ─────────────────────────────────────────────────────────
    manifest = Dict(
        "schema_version"   => "1.0-julia",
        "T_total_days"     => args["T-days"],
        "step_minutes"     => args["step-minutes"],
        "BINS_PER_DAY"     => BINS_PER_DAY_LOCAL,
        "WINDOW_BINS"      => WINDOW_BINS,
        "STRIDE_BINS"      => STRIDE_BINS,
        "n_strides"        => n_strides,
        "smc_cfg"          => Dict(
            "n_smc_particles"  => n_smc,
            "K_per_chain"      => K_per_chain,
            "target_ess_frac"  => args["target-ess-frac"],
            "max_lambda_inc"   => args["max-lambda-inc"],
            "num_mcmc_steps"   => args["num-mcmc"],
            "hmc_step_size"    => args["hmc-step-size"],
            "hmc_num_leapfrog" => args["hmc-leapfrog"],
        ),
        "seed"             => args["seed"],
        "compute_s"        => t_total,
        "device"           => string(CUDA.name(CUDA.device())),
        "fp32_inner_loop"  => true,
        "fp64_outer_loop"  => true,
        "filter_only"      => true,
        "controller_used"  => false,
    )
    open(joinpath(out_dir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    # ── experiment_run.md ────────────────────────────────────────────────
    open(joinpath(out_dir, "experiment_run.md"), "w") do io
        println(io, "# FSA-v2 Julia GPU bench — full SMC² rolling-window filter")
        println(io)
        println(io, "- **Mode:** filter only (Φ=1.0 from plant; no controller)")
        println(io, "- **Timestamp:** $(Dates.now())")
        println(io, "- **Device:** $(CUDA.name(CUDA.device()))")
        println(io, "- **Wall time:** $(round(t_total, digits=1)) s ($(round(t_total/60, digits=1)) min)")
        println(io, "- **T_total_days:** $(args["T-days"]), step=$(args["step-minutes"]) min")
        println(io, "- **BINS_PER_DAY:** $BINS_PER_DAY_LOCAL, WINDOW=$WINDOW_BINS, STRIDE=$STRIDE_BINS")
        println(io, "- **n_strides:** $n_strides")
        println(io, "- **N_SMC:** $n_smc, K_per_chain=$K_per_chain")
        println(io, "- **HMC:** step_size=$(args["hmc-step-size"]), num_leapfrog=$(args["hmc-leapfrog"]), num_mcmc=$(args["num-mcmc"])")
        println(io, "- **Cold-start:** prior at first window; warm-start (carry posterior forward) for subsequent windows.")
        println(io)
        println(io, "## Implementation")
        println(io)
        println(io, "Locally-guided (Pitt-Shephard) PF on GPU. Per-particle, per-bin:")
        println(io, "1. G1-reparametrized prior predictive (mean + state-dep cov)")
        println(io, "2. Sequential scalar Kalman fusion across 3 Gaussian channels (HR / stress / log_steps)")
        println(io, "3. Cholesky-3 sample from fused N(μ_fused, P_fused)")
        println(io, "4. Predictive log-marginal accumulation + Bernoulli sleep ll")
        println(io)
        println(io, "All inner loops in fp32. Outer SMC² log-weights / ESS / posterior cloud in fp64.")
        println(io, "Parallel-chains HMC: M·(1+2d) chains in one kernel launch per leapfrog step.")
    end

    return data_path
end


if abspath(PROGRAM_FILE) == @__FILE__
    data_path = main(ARGS_DICT)
    @info "data saved: $data_path"
end
