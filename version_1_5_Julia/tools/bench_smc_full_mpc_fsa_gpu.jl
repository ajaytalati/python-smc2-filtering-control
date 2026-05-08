#!/usr/bin/env julia
# FSA v1.5 closed-loop SMC²-MPC bench — purely functional foldl over strides.
#
# Per stride:
#   1. Slice next stride's Φ from the current plan.
#   2. plant_rollout(state, Φ_slice, params, dt, key) → trajectory + obs.
#   3. Build window grid_obs from the accumulated obs history.
#   4. Run outer SMC² on the filter (tempered + parallel HMC).
#   5. Maybe replan: framework's run_tempered_smc_gpu on the controller's RBF θ.
#
# All ingredients are pure. The bench accumulator IS the history; mutation
# only happens (a) inside KernelAbstractions kernels (private to gpu_pf.jl)
# and (b) at the single I/O write at the end.
#
# Modes:
#   --open-loop false   default — closed-loop with replanning every K strides.
#   --open-loop true    one up-front plan from INIT_STATE+TRUTH_PARAMS, no replan.

using Dates


function _parse_args(argv::Vector{String})
    defaults = Dict{String,Any}(
        # ── Bench / filter ──
        "T-days"          => 14,
        "step-minutes"    => 60,
        "replan-K"        => 2,
        "N-smc"           => 32,
        "K-per-chain"     => 200,
        "num-mcmc"        => 3,
        "hmc-step-size"   => 0.05,
        "hmc-leapfrog"    => 4,
        "max-lambda-inc"  => 0.20,
        "target-ess-frac" => 0.5,
        "max-temp-levels" => 30,
        "seed"            => 42,
        "output-dir"      => "",
        "open-loop"       => "false",
        # ── Controller (the SMC²-MPC plan-finder, was hardcoded) ──
        "ctrl-n-smc"      => 256,         # outer SMC² particles (θ_ctrl)
        "ctrl-n-inner"    => 64,          # MC trials per cost evaluation
        "ctrl-num-mcmc"   => 8,           # ChEES-HMC moves per tempering level
        "ctrl-hmc-step"   => 0.2,         # base leapfrog step size
        "ctrl-hmc-leap"   => 16,          # base leapfrog trajectory length
        "ctrl-chees-max"  => 256,         # ChEES picker upper bound; list = [16, 32, 64, ..., max]
        "ctrl-max-levels" => 25,          # tempering bisection cap
        "ctrl-target-nats" => 8.0,
        "ctrl-sigma-prior" => 1.5,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("Unrecognized arg: $a")
        key = a[3:end]
        haskey(defaults, key) || error("Unknown flag: $a")
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
using StaticArrays
using StableRNGs
using Plots                # for the auto-generated param-traces plot

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: PlantState, plant_rollout, init_plant_state
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE,
                              PINNED_PARAMS, params_v15_to_v1_nt, fill_pinned_nt
using .FSAHighRes.Estimation: PARAM_NAMES, PARAM_PRIOR_CONFIG
using .FSAHighRes.GPUPF: FSAGPUTarget, gpu_log_density, gpu_grads, parallel_hmc_one_move
using .FSAHighRes.GPUControl: FSAv1ControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn
using SMC2FC: run_tempered_smc_gpu


# ── Pure outer SMC² — tempered + parallel HMC ─────────────────────────────
#
# Mirrors v2's run_outer_smc_gpu but written in functional form:
#   - input particles `U_init` are an immutable Matrix{Float64}
#   - returns a NEW (U_post, n_temp) tuple; `U_init` is not mutated
#   - RNG via explicit `key::UInt64`

function run_outer_smc(target::FSAGPUTarget,
                       grid_obs::NamedTuple,
                       n_smc::Int,
                       cfg::NamedTuple,
                       U_init::Union{Nothing, Matrix{Float64}},
                       prior_means::Vector{Float64},
                       prior_sigmas::Vector{Float64},
                       key::UInt64)
    d = length(prior_means)
    rng = StableRNG(key)
    U = if U_init === nothing
        # Cold-start from prior
        let buf = Matrix{Float64}(undef, n_smc, d)
            for m in 1:n_smc, j in 1:d
                buf[m, j] = prior_means[j] + prior_sigmas[j] * randn(rng)
            end
            buf
        end
    else
        copy(U_init)
    end

    λ = 0.0
    n_temp = 0
    while λ < 1.0 - 1e-6
        ll_data = gpu_log_density(target, U, grid_obs, hash((key, :ll, n_temp)))

        δ_max = min(1.0 - λ, cfg.max_lambda_inc)
        target_ess = cfg.target_ess_frac * n_smc
        ess_at(δ) = begin
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
                if ess_at(mid) > target_ess
                    lo = mid
                else
                    hi = mid
                end
            end
            lo
        end
        next_λ = (λ + δ < 1.0 - 1e-6) ? λ + δ : 1.0
        Δλ = next_λ - λ

        # Systematic resample
        log_w = Δλ .* ll_data
        w = exp.(log_w .- logsumexp(log_w))
        cumsum_w = cumsum(w)
        u_shift = rand(rng) / n_smc
        indices = Vector{Int}(undef, n_smc)
        for i in 1:n_smc
            t = (i - 1) / n_smc + u_shift
            indices[i] = clamp(searchsortedfirst(cumsum_w, t), 1, n_smc)
        end
        U_resampled = U[indices, :]

        # `cfg.num_mcmc` HMC moves — pure, returns new U each time
        U_curr = U_resampled
        n_acc_total = 0
        for k in 1:cfg.num_mcmc
            sub_key = hash((key, :hmc, n_temp, k))
            out = parallel_hmc_one_move(U_curr, target, grid_obs,
                                          cfg.hmc_step, cfg.hmc_leap,
                                          prior_means, prior_sigmas, sub_key;
                                          h_fd = cfg.h_fd)
            U_curr = out.U_new
            n_acc_total += out.n_acc
        end
        accept_frac = n_acc_total / max(1, cfg.num_mcmc * n_smc)

        U = U_curr
        n_temp += 1
        @info @sprintf("    [%2d] λ %.3f → %.3f (Δλ=%.3f) accept=%.0f%%",
                       n_temp, λ, next_λ, Δλ, 100 * accept_frac)
        λ = next_λ
        n_temp >= cfg.max_levels && break
    end

    # Final gpu_log_density call on the posterior chains so that
    # `bufs.mu_per_chain[1:n_smc]` reflects the M=n_smc posterior particles
    # (not the M=n_smc·(1+2d) FD batch from the last HMC FD call).
    # extract_xhat reads from bufs after this call.
    _ = gpu_log_density(target, U, grid_obs, hash((key, :final)))

    return (U_post = U, n_temp = n_temp)
end


# ── Posterior → constrained v1.5 NamedTuple → v1-form NamedTuple ─────────

function posterior_mean_v15(U_post::AbstractMatrix{Float64})
    # PARAM_NAMES order, all LogNormal — exp(mean of unconstrained).
    u_mean = vec(mean(U_post; dims = 1))
    return (
        tau_F    = exp(u_mean[1]),
        B_inf    = exp(u_mean[2]),
        F_inf    = exp(u_mean[3]),
        lambda_A  = exp(u_mean[4]),
        mu_0     = exp(u_mean[5]),
        mu_B     = exp(u_mean[6]),
        mu_F     = exp(u_mean[7]),
        sigma_B  = exp(u_mean[8]),
        sigma_F  = exp(u_mean[9]),
        sigma_A  = exp(u_mean[10]),
    )
end


# ── Pure controller plan call ────────────────────────────────────────────
# Builds an FSAControlGPUTarget with v1-form params and runs the framework's
# run_tempered_smc_gpu on the RBF θ. Decodes posterior-mean θ → per-bin Φ.

function controller_plan(params_v1::NamedTuple,
                          init_state::SVector{3, Float64},
                          T_total_bins::Int,
                          n_substeps::Int,
                          dt::Float64,
                          ctrl_cfg::NamedTuple,
                          key::UInt64)
    ctrl_target = FSAv1ControlGPUTarget(
        n_inner    = ctrl_cfg.n_inner,
        M_max      = ctrl_cfg.M_max,
        n_steps    = T_total_bins,
        n_anchors  = ctrl_cfg.n_anchors,
        n_substeps = n_substeps,
        dt         = dt,
        F_max      = 0.40,
        Phi_max    = 3.0,
        Phi_default = 1.0,
        lam_F      = 1.0,
        sigma_prior = ctrl_cfg.sigma_prior,
        params     = params_v1,
        init_state = Float32[init_state[1], init_state[2], init_state[3]],
        noise_seed = Int(key & typemax(Int32)),
    )
    log_density_fn = make_log_density_fn(ctrl_target)
    rng_ctrl = StableRNG(key)
    U_ctrl, n_temp_ctrl, _ = run_tempered_smc_gpu(
        log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, ctrl_cfg.n_anchors,
        0.0, ctrl_cfg.sigma_prior, rng_ctrl;
        target_nats        = ctrl_cfg.target_nats,
        target_ess_frac    = ctrl_cfg.target_ess_frac,
        max_lambda_inc     = ctrl_cfg.max_lambda_inc,
        max_temp_levels    = ctrl_cfg.max_levels,
        num_mcmc_steps     = ctrl_cfg.num_mcmc,
        hmc_step_size      = ctrl_cfg.hmc_step,
        hmc_num_leapfrog   = ctrl_cfg.hmc_leap,
        chees_L_candidates = ctrl_cfg.chees_L_candidates,
        h_fd               = 1e-4,
        calib_n            = 64,
        verbose            = false,
    )
    θ_post = vec(mean(U_ctrl; dims = 1))
    # Decode RBF θ → per-bin Φ
    T_total = T_total_bins * dt
    t_grid  = collect(0:T_total_bins-1) .* dt
    anchors = collect(range(0.0, T_total; length = ctrl_cfg.n_anchors))
    σ_rbf   = T_total / ctrl_cfg.n_anchors
    Phi_plan = zeros(Float32, T_total_bins)
    c_Phi   = Float64(ctrl_target.c_Phi)
    for k in 1:T_total_bins
        raw = c_Phi
        for j in 1:ctrl_cfg.n_anchors
            raw += θ_post[j] * exp(-0.5 * ((t_grid[k] - anchors[j]) / σ_rbf)^2)
        end
        Phi_plan[k] = Float32(3.0 / (1.0 + exp(-raw)))
    end
    return (Phi_plan = Phi_plan, n_temp_ctrl = n_temp_ctrl, theta = θ_post)
end


# ── State extraction from filter posterior ──────────────────────────────
# Importance-weighted mean of the inner-PF cloud at end-of-window. The
# stats kernel populates `bufs.mu_per_chain[1:M, :]` (M, 3) with weighted
# means per chain. We marginalise across M with uniform weights since
# each filter chain represents one θ posterior particle.

function extract_xhat(target::FSAGPUTarget, M::Int)
    mu_cpu = Array(view(target.bufs.mu_per_chain, 1:M, :))
    return SVector{3, Float64}(mean(mu_cpu[:, 1]),
                                mean(mu_cpu[:, 2]),
                                mean(mu_cpu[:, 3]))
end


# ── Build the per-stride window grid_obs from accumulated obs history ───

function window_grid_obs(obs_acc, traj_acc, t0_bin::Int, T_window::Int)
    obs_B = view(obs_acc.B, t0_bin + 1 : t0_bin + T_window)
    obs_F = view(obs_acc.F, t0_bin + 1 : t0_bin + T_window)
    obs_A = view(obs_acc.A, t0_bin + 1 : t0_bin + T_window)
    init  = t0_bin == 0 ?
            (Float64(INIT_STATE.B), Float64(INIT_STATE.F), Float64(INIT_STATE.A)) :
            (traj_acc[t0_bin, 1], traj_acc[t0_bin, 2], traj_acc[t0_bin, 3])
    Phi_seq = view(obs_acc.Phi, t0_bin + 1 : t0_bin + T_window)
    return (
        Phi_seq = collect(Float32, Phi_seq),
        obs_B   = collect(Float32, obs_B),
        obs_F   = collect(Float32, obs_F),
        obs_A   = collect(Float32, obs_A),
        B_init  = init[1], F_init = init[2], A_init = init[3],
    )
end


# ── Main bench ───────────────────────────────────────────────────────────

function main(args::Dict{String,Any})
    bins_per_day = (60 * 24) ÷ args["step-minutes"]
    window_bins  = bins_per_day
    stride_bins  = window_bins ÷ 2
    dt_days      = 1.0 / bins_per_day
    T_total_bins = args["T-days"] * bins_per_day
    n_strides    = (T_total_bins - window_bins) ÷ stride_bins + 1
    H_plan_bins  = T_total_bins                       # FIXED, per §2.11

    is_open_loop = lowercase(args["open-loop"]) in ("true", "1", "yes")

    @info "config: T=$(args["T-days"])d step=$(args["step-minutes"])min " *
          "BINS_PER_DAY=$bins_per_day WINDOW=$window_bins STRIDE=$stride_bins " *
          "n_strides=$n_strides  open_loop=$is_open_loop"

    out_dir = isempty(args["output-dir"]) ?
              joinpath(REPO_ROOT, "outputs", "fsa_high_res", "g4_runs",
                        "T$(args["T-days"])d_$(is_open_loop ? "OL" : "CL_K$(args["replan-K"])")_h$(args["step-minutes"])min") :
              args["output-dir"]
    mkpath(out_dir)

    @info "GPU device: $(CUDA.name(CUDA.device()))"

    # ── Build filter target ──
    n_smc = args["N-smc"]
    K_per_chain = args["K-per-chain"]
    d = length(PARAM_NAMES)
    M_max = n_smc * (1 + 2 * d)
    @info "filter target: N_SMC=$n_smc K_per_chain=$K_per_chain d=$d M_max=$M_max"
    target = FSAGPUTarget(
        K_per_chain = K_per_chain, M_max = M_max,
        T_steps = window_bins, R = 4, dt = dt_days,
        noise_seed = 0,
    )

    # ── Filter prior (unconstrained, v1.5 PARAM_PRIOR_CONFIG) ──
    prior_means  = Float64[m for (_, _, m, _) in PARAM_PRIOR_CONFIG]
    prior_sigmas = Float64[s for (_, _, _, s) in PARAM_PRIOR_CONFIG]
    @assert length(prior_means)  == d
    @assert length(prior_sigmas) == d

    filter_cfg = (
        max_lambda_inc  = args["max-lambda-inc"],
        target_ess_frac = args["target-ess-frac"],
        num_mcmc        = args["num-mcmc"],
        hmc_step        = args["hmc-step-size"],
        hmc_leap        = args["hmc-leapfrog"],
        max_levels      = args["max-temp-levels"],
        h_fd            = 1e-3,
    )

    # ── Controller config — all knobs CLI-exposed ──
    ctrl_n_anchors = 8
    ctrl_n_smc     = args["ctrl-n-smc"]
    ctrl_n_inner   = args["ctrl-n-inner"]
    ctrl_M_max     = ctrl_n_smc * (1 + 2 * ctrl_n_anchors)
    # ChEES-HMC candidate list: powers of 2 from 16 up to ctrl-chees-max.
    chees_max  = args["ctrl-chees-max"]
    chees_list = [16]
    while chees_list[end] * 2 <= chees_max
        push!(chees_list, chees_list[end] * 2)
    end
    ctrl_cfg = (
        n_smc       = ctrl_n_smc,
        n_inner     = ctrl_n_inner,
        n_anchors   = ctrl_n_anchors,
        M_max       = ctrl_M_max,
        sigma_prior = args["ctrl-sigma-prior"],
        target_nats = args["ctrl-target-nats"],
        target_ess_frac = 0.5,
        max_lambda_inc  = 0.20,
        max_levels      = args["ctrl-max-levels"],
        num_mcmc        = args["ctrl-num-mcmc"],
        hmc_step        = args["ctrl-hmc-step"],
        hmc_leap        = args["ctrl-hmc-leap"],
        chees_L_candidates = chees_list,
    )
    @info "controller cfg: n_smc=$ctrl_n_smc n_inner=$ctrl_n_inner num_mcmc=$(ctrl_cfg.num_mcmc) hmc_leap=$(ctrl_cfg.hmc_leap) chees=$chees_list max_levels=$(ctrl_cfg.max_levels)"

    # ── Pre-roll the BASELINE plant (Φ=1) for the full bench ──
    # The baseline plant produces obs-of-truth that we'll compare against
    # the MPC-plant. It also drives the FILTER (we filter on the
    # baseline obs to recover params, while the MPC plant uses the
    # controller's plan.). For v1.5 we filter on the MPC plant's obs
    # which is what the closed-loop bench actually does.

    base_seed = UInt64(args["seed"])
    base_plant_state = init_plant_state()
    base_phi  = fill(Float32(1.0), T_total_bins)
    base_out  = plant_rollout(base_plant_state, base_phi, DEFAULT_PARAMS, dt_days,
                                 hash((base_seed, :base_plant)))

    # ── Open-loop initial plan ──
    initial_plan = if is_open_loop
        @info "OPEN-LOOP: building one initial plan from INIT_STATE+TRUTH..."
        params_truth_v1 = params_v15_to_v1_nt(fill_pinned_nt((
            tau_F   = DEFAULT_PARAMS[:tau_F],
            B_inf   = DEFAULT_PARAMS[:B_inf],
            F_inf   = DEFAULT_PARAMS[:F_inf],
            lambda_A = DEFAULT_PARAMS[:lambda_A],
            mu_0    = DEFAULT_PARAMS[:mu_0],
            mu_B    = DEFAULT_PARAMS[:mu_B],
            mu_F    = DEFAULT_PARAMS[:mu_F],
            sigma_B = DEFAULT_PARAMS[:sigma_B],
            sigma_F = DEFAULT_PARAMS[:sigma_F],
            sigma_A = DEFAULT_PARAMS[:sigma_A],
        )))
        s0 = SVector{3,Float64}(Float64(INIT_STATE.B),
                                  Float64(INIT_STATE.F),
                                  Float64(INIT_STATE.A))
        out = controller_plan(params_truth_v1, s0, H_plan_bins, max(1, bins_per_day ÷ 24),
                               dt_days, ctrl_cfg, hash((base_seed, :ctrl_init)))
        @info @sprintf("  open-loop plan ready: Φ̄=%.3f  Φ_max=%.3f  Φ_min=%.3f  n_temp=%d",
                       mean(out.Phi_plan), maximum(out.Phi_plan), minimum(out.Phi_plan),
                       out.n_temp_ctrl)
        out.Phi_plan
    else
        fill(Float32(1.0), T_total_bins)
    end

    # ── Closed-loop foldl over strides ──
    # `all_filter_posts` is a per-stride list of unconstrained posterior
    # particle clouds (one (n_smc, n_params) Matrix per stride; Nothing
    # for warmup strides). Keeps the param-traces plot's input cheap to
    # build at end-of-bench.
    init_acc = (
        plant_state    = init_plant_state(),
        plan_phi       = initial_plan,
        plan_offset    = 0,
        filter_post    = nothing,
        all_filter_posts = Vector{Union{Nothing,Matrix{Float64}}}(),
        traj_history   = Matrix{Float64}(undef, 0, 3),
        obs_history    = (B = Float32[], F = Float32[], A = Float32[],
                          Phi = Float32[]),
        per_stride_log = NamedTuple[],
        last_xhat      = SVector{3, Float64}(INIT_STATE.B, INIT_STATE.F, INIT_STATE.A),
    )

    function stride_step(acc, stride_idx::Int)
        t_start = time()

        # 1. Slice next stride's Φ
        slice_end = acc.plan_offset + stride_bins
        @assert slice_end <= length(acc.plan_phi)
        phi = acc.plan_phi[acc.plan_offset + 1 : slice_end]

        # 2. Plant rollout
        p = plant_rollout(acc.plant_state, phi, DEFAULT_PARAMS, dt_days,
                           hash((base_seed, :mpc_plant, stride_idx)))

        # 3. Append obs to bench history
        new_obs = (
            B   = vcat(acc.obs_history.B, p.obs_B),
            F   = vcat(acc.obs_history.F, p.obs_F),
            A   = vcat(acc.obs_history.A, p.obs_A),
            Phi = vcat(acc.obs_history.Phi, p.Phi),
        )
        new_traj = vcat(acc.traj_history, p.trajectory)

        # 4. Filter window — start at end-of-history minus window_bins
        #    (the most recent window_bins of obs).
        hist_end = size(new_traj, 1)
        if hist_end < window_bins
            # Not enough obs yet — skip filter for this stride.
            new_filter_post = acc.filter_post
            new_xhat        = SVector{3,Float64}(p.final_state.bfa[1],
                                                   p.final_state.bfa[2],
                                                   p.final_state.bfa[3])
            n_temp_filter   = 0
            @info @sprintf("[stride %2d/%2d] %.1fs  warming up (hist=%d/%d) Φ̄=%.3f x̂=(%.3f, %.3f, %.3f)",
                           stride_idx, n_strides, time() - t_start,
                           hist_end, window_bins, mean(phi),
                           new_xhat[1], new_xhat[2], new_xhat[3])
        else
            t0 = hist_end - window_bins
            grid_obs = window_grid_obs(new_obs, new_traj, t0, window_bins)
            filter_out = run_outer_smc(target, grid_obs, n_smc, filter_cfg,
                                         acc.filter_post, prior_means, prior_sigmas,
                                         hash((base_seed, :filter, stride_idx)))
            new_filter_post = filter_out.U_post
            n_temp_filter   = filter_out.n_temp
            new_xhat        = extract_xhat(target, n_smc)

            @info @sprintf("[stride %2d/%2d] %.1fs  %d filter levels  Φ̄=%.3f x̂=(%.3f, %.3f, %.3f)",
                           stride_idx, n_strides, time() - t_start,
                           n_temp_filter, mean(phi),
                           new_xhat[1], new_xhat[2], new_xhat[3])
        end

        # 5. Maybe replan (closed-loop only)
        new_plan, new_offset = if !is_open_loop &&
                                  stride_idx % args["replan-K"] == 0 &&
                                  new_filter_post !== nothing
            t_plan = time()
            params_post_v15 = posterior_mean_v15(new_filter_post)
            params_post_v1  = params_v15_to_v1_nt(fill_pinned_nt(params_post_v15))
            ctrl_out = controller_plan(params_post_v1, new_xhat, H_plan_bins,
                                          max(1, bins_per_day ÷ 24), dt_days,
                                          ctrl_cfg,
                                          hash((base_seed, :ctrl, stride_idx)))
            @info @sprintf("  [replan @ stride %2d] %.1fs  Φ̄_plan=%.3f  shape=[%.2f→%.2f→%.2f]  ctrl_n_temp=%d",
                           stride_idx, time() - t_plan, mean(ctrl_out.Phi_plan),
                           ctrl_out.Phi_plan[1],
                           ctrl_out.Phi_plan[length(ctrl_out.Phi_plan) ÷ 2],
                           ctrl_out.Phi_plan[end],
                           ctrl_out.n_temp_ctrl)
            (ctrl_out.Phi_plan, 0)
        else
            (acc.plan_phi, slice_end)
        end

        # Snapshot the latest filter posterior into the per-stride store.
        # Stride that ran the filter contributes a Matrix; warmup strides
        # contribute `nothing` (caller handles the mask).
        post_entry = hist_end < window_bins ? nothing : new_filter_post
        new_all_filter_posts = vcat(acc.all_filter_posts,
                                      Union{Nothing,Matrix{Float64}}[post_entry])

        return (
            plant_state    = p.final_state,
            plan_phi       = new_plan,
            plan_offset    = new_offset,
            filter_post    = new_filter_post,
            all_filter_posts = new_all_filter_posts,
            traj_history   = new_traj,
            obs_history    = new_obs,
            per_stride_log = vcat(acc.per_stride_log, [(stride = stride_idx,
                                                          n_temp = n_temp_filter,
                                                          phi_mean = mean(phi))]),
            last_xhat      = new_xhat,
        )
    end

    t_total_start = time()
    final = foldl(stride_step, 1:n_strides; init = init_acc)
    t_total = time() - t_total_start

    @info @sprintf("Total wall time: %.1f s", t_total)

    # ── Build per-day Φ for plotting ──
    daily_phi_plan_per_stride = Float32[]
    log = final.per_stride_log
    for entry in log
        push!(daily_phi_plan_per_stride, Float32(entry.phi_mean))
    end

    # ── Save data.jld2 in the schema plot_state_traces.jl reads ──
    # Truncate baseline to MPC length: the foldl runs n_strides × stride_bins
    # bins, which can be < T_total_bins. plot_state_traces.jl assumes equal
    # lengths between MPC and baseline.
    n_mpc_bins = size(final.traj_history, 1)
    data_path = joinpath(out_dir, "data.jld2")
    # ── Per-stride posterior particles + mask, in CONSTRAINED space.
    # All 10 v1.5 priors are LogNormal, so constrained = exp(unconstrained).
    n_params_est = length(PARAM_NAMES)
    posterior_particles_arr = zeros(Float64, n_strides, n_smc, n_params_est)
    posterior_window_mask   = falses(n_strides)
    for (s, U) in enumerate(final.all_filter_posts)
        if U !== nothing
            posterior_particles_arr[s, 1:size(U, 1), :] = exp.(Float64.(U))
            posterior_window_mask[s] = true
        end
    end

    JLD2.jldopen(data_path, "w") do f
        f["trajectory_mpc"]       = Float32.(final.traj_history)
        f["trajectory_baseline"]  = Float32.(base_out.trajectory[1:n_mpc_bins, :])
        f["Phi_per_bin_mpc"]      = final.obs_history.Phi[1:n_mpc_bins]
        f["Phi_per_bin_baseline"] = base_phi[1:n_mpc_bins]
        f["daily_phi_per_stride"] = daily_phi_plan_per_stride
        f["BINS_PER_DAY"]         = bins_per_day
        f["STRIDE_BINS"]          = stride_bins
        f["WINDOW_BINS"]          = window_bins
        f["dt_days"]              = dt_days
        f["F_max"]                = 0.40
        if final.filter_post !== nothing
            f["filter_post_unc"]  = Float64.(final.filter_post)
        end
        f["posterior_particles"]    = posterior_particles_arr
        f["posterior_window_mask"]  = posterior_window_mask
        f["param_names"]          = String.(PARAM_NAMES)
        f["truth_params_dict"]    = Dict{String,Float64}(string(k) => v
                                                          for (k, v) in DEFAULT_PARAMS)
        f["seed"]                 = args["seed"]
        f["wall_seconds"]         = t_total
    end
    @info "wrote $data_path"

    # ── Manifest ──
    manifest = Dict(
        "schema_version"   => "1.0-v15",
        "T_total_days"     => args["T-days"],
        "step_minutes"     => args["step-minutes"],
        "BINS_PER_DAY"     => bins_per_day,
        "WINDOW_BINS"      => window_bins,
        "STRIDE_BINS"      => stride_bins,
        "n_strides"        => n_strides,
        "open_loop"        => is_open_loop,
        "replan_K"         => args["replan-K"],
        "smc_cfg"          => Dict(
            "N_smc"           => n_smc,
            "K_per_chain"     => K_per_chain,
            "num_mcmc"        => args["num-mcmc"],
            "hmc_step_size"   => args["hmc-step-size"],
            "hmc_num_leapfrog" => args["hmc-leapfrog"],
        ),
        "seed"             => args["seed"],
        "wall_seconds"     => t_total,
        "device"           => string(CUDA.name(CUDA.device())),
        "pinned_dynamics"  => Dict(string(k) => v for (k, v) in PINNED_PARAMS),
        "estimated_params" => String.(PARAM_NAMES),
    )
    open(joinpath(out_dir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    # ── experiment_run.md ──
    open(joinpath(out_dir, "experiment_run.md"), "w") do io
        println(io, "# FSA v1.5 closed-loop bench")
        println(io)
        println(io, "- **Mode:** $(is_open_loop ? "open-loop" : "closed-loop")")
        println(io, "- **Timestamp:** $(Dates.now())")
        println(io, "- **Wall time:** $(round(t_total, digits=1)) s")
        println(io, "- **T_total_days:** $(args["T-days"]), step=$(args["step-minutes"]) min")
        println(io, "- **BINS_PER_DAY:** $bins_per_day, WINDOW=$window_bins, STRIDE=$stride_bins")
        println(io, "- **n_strides:** $n_strides")
        println(io, "- **N_SMC:** $n_smc, K_per_chain=$K_per_chain")
        println(io, "- **Pinned at truth:** $(join(string.(keys(PINNED_PARAMS)), ", "))")
        println(io, "- **Filter estimates:** $(length(PARAM_NAMES)) ($(join(string.(PARAM_NAMES), ", ")))")
        println(io, "- **Final x̂:** B=$(round(final.last_xhat[1], digits=4)) " *
                     "F=$(round(final.last_xhat[2], digits=4)) " *
                     "A=$(round(final.last_xhat[3], digits=4))")
    end
    @info "wrote manifest.json + experiment_run.md"

    # ── Auto-plot via v2's plot_state_traces.jl (read-only reuse) ──
    try
        plot_path = joinpath(REPO_ROOT, "..", "version_2_Julia", "tools",
                              "plot_state_traces.jl")
        if isfile(plot_path)
            include(plot_path)
            data_dict = Base.invokelatest(Main.load_run_data, data_path)
            traces_path = joinpath(out_dir, "v15_T$(args["T-days"])d_traces.png")
            Base.invokelatest(Main.plot_state_traces, data_dict; out_path=traces_path)
            @info "wrote $traces_path"
        else
            @warn "plot_state_traces.jl not found at $plot_path; skipping auto-plot"
        end
    catch e
        @warn "plot generation failed (data.jld2 still saved)" exception=(e, catch_backtrace())
    end

    # ── Posterior parameter-traces plot (mirrors version_2_Python_JAX/
    #    tools/plot_param_traces.py: per-param 5/95 quantile band +
    #    median + truth red dashed horizontal line). ──
    if any(posterior_window_mask)
        try
            param_path = joinpath(out_dir, "v15_T$(args["T-days"])d_param_traces.png")
            _plot_param_traces_v15(
                posterior_particles_arr,
                posterior_window_mask,
                String.(PARAM_NAMES),
                Dict(string(k) => Float64(v) for (k, v) in DEFAULT_PARAMS),
                n_strides, stride_bins, window_bins, bins_per_day,
                args["T-days"], args["step-minutes"];
                out_path = param_path,
                mean_A_mpc  = mean(final.traj_history[:, 3]),
                mean_A_base = mean(base_out.trajectory[1:n_mpc_bins, 3]),
            )
            @info "wrote $param_path"
        catch e
            @warn "param-traces plot failed (data.jld2 still saved)" exception=(e, catch_backtrace())
        end
    end

    return final
end

# ── Param-traces plot helper ─────────────────────────────────────────────
"""
    _plot_param_traces_v15(posterior, mask, param_names, truth, n_strides,
                            stride_bins, window_bins, bins_per_day,
                            T_days, step_minutes; out_path, mean_A_mpc,
                            mean_A_base)

Per-parameter posterior trace across rolling windows. One panel per
estimated param; 5-95% quantile band + median + truth horizontal line.
Layout mirrors `version_2_Python_JAX/tools/plot_param_traces.py`.
"""
function _plot_param_traces_v15(
        posterior::Array{Float64,3},
        mask::AbstractVector{Bool},
        param_names::Vector{String},
        truth::Dict{String,Float64},
        n_strides::Int, stride_bins::Int, window_bins::Int,
        bins_per_day::Int, T_days, step_minutes;
        out_path::AbstractString,
        mean_A_mpc::Float64, mean_A_base::Float64)
    end_t_days = ((collect(0:n_strides-1) .* stride_bins) .+ window_bins) ./ bins_per_day
    end_t_days = end_t_days[mask]
    valid = posterior[mask, :, :]                    # (n_valid, n_smc, n_params)
    n_valid, n_smc, n_params = size(valid)
    q05 = [quantile(vec(valid[s, :, p]), 0.05)
           for s in 1:n_valid, p in 1:n_params]
    q50 = [quantile(vec(valid[s, :, p]), 0.50)
           for s in 1:n_valid, p in 1:n_params]
    q95 = [quantile(vec(valid[s, :, p]), 0.95)
           for s in 1:n_valid, p in 1:n_params]

    n_cols = 5
    n_rows = (n_params + n_cols - 1) ÷ n_cols
    panels = Plots.Plot[]
    for i in 1:n_params
        name = param_names[i]
        p = plot(end_t_days, q50[:, i];
                  ribbon = (q50[:, i] .- q05[:, i], q95[:, i] .- q50[:, i]),
                  fillalpha = 0.30, color = :steelblue, lw = 1.4,
                  label = "median", title = name, titlefontsize = 9,
                  legend = false, grid = true, gridalpha = 0.3,
                  tickfontsize = 7,
                  xlabel = i > n_params - n_cols ? "end of window (days)" : "",
                  xguidefontsize = 7)
        if haskey(truth, name)
            hline!(p, [truth[name]]; color = :red, ls = :dash, lw = 1.0,
                    label = "truth")
        end
        push!(panels, p)
    end
    for _ in n_params+1:n_rows*n_cols
        push!(panels, plot(framestyle = :none, ticks = false, legend = false))
    end
    fig = plot(panels...; layout = (n_rows, n_cols),
                size = (n_cols * 320, n_rows * 220), dpi = 120,
                plot_title = "FSA-v1.5 posterior parameter traces — " *
                             "T=$(T_days)d, h=$(step_minutes)min, " *
                             "n_strides=$n_strides, " *
                             "mean A $(round(mean_A_mpc, digits=3)) " *
                             "vs baseline $(round(mean_A_base, digits=3))",
                plot_titlefontsize = 10)
    savefig(fig, out_path)
end

main(ARGS_DICT)
