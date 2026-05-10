# bench/bench_controller.jl  (v5)
#
# Controller-side wrapper used by `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`.
# Builds an `FSAv5ControlGPUTarget` from the filter's posterior-mean
# parameters, calls the framework's `run_tempered_smc_gpu` with
# θ-dim = 2·n_anchors (bimodal schedule), and decodes the posterior-
# mean RBF coefficient vector into per-bin (Φ_B, Φ_S) plans.
#
# Public function:
#   - controller_plan_v5(params_v5, init_state, T_total_bins, n_substeps,
#                          dt, ctrl_cfg, key; collect_diagnostics=false)
#       -> (Phi_B_plan, Phi_S_plan, n_temp_ctrl, theta, diagnostics)
#
# Mirrors `tools/bench/bench_controller.jl` (v1.5) byte-for-byte except
# for the four structural differences mandated by the v5 surface:
#   1. `params_v5` is a Dict (28 dynamics + 22 obs-channel + frozen),
#      not a NamedTuple in v1's basis.
#   2. `init_state` is `SVector{6, Float64}`.
#   3. Decoder operates on `2·n_anchors` coefficients (first half = B
#      channel, second half = S channel) and produces TWO Φ vectors.
#   4. Framework's `n_anchors` argument is `2 · ctrl_cfg.n_anchors`
#      (the actual θ-dim seen by the SMC²; the n_anchors here is the
#      per-channel count).
#
# Dependencies (loaded by the calling bench script before this file
# is `include`d): `Statistics.mean`, `StableRNGs.StableRNG`,
# `StaticArrays.SVector`, the v5 controller GPU kernel
# (`FSAv5ControlGPUTarget`, `make_log_density_fn_v5`), and the
# framework's `run_tempered_smc_gpu`.

function controller_plan_v5(params_v5::Dict{Symbol, Float64},
                              init_state::SVector{6, Float64},
                              T_total_bins::Int,
                              n_substeps::Int,
                              dt::Float64,
                              ctrl_cfg::NamedTuple,
                              key::UInt64;
                              collect_diagnostics::Bool = false)
    n_anchors = ctrl_cfg.n_anchors
    theta_dim = 2 * n_anchors      # bimodal schedule: B + S channels

    ctrl_target = FSAv5ControlGPUTarget(
        n_inner    = ctrl_cfg.n_inner,
        M_max      = ctrl_cfg.M_max,
        n_steps    = T_total_bins,
        n_anchors  = n_anchors,
        n_substeps = n_substeps,
        dt         = dt,
        F_max      = 0.40,
        Phi_max    = 3.0,
        Phi_default = ctrl_cfg.phi_default,
        lam_Phi    = ctrl_cfg.lam_phi,
        lam_F      = ctrl_cfg.lam_f,
        lam_chance = ctrl_cfg.lam_chance,
        A_thr      = 0.05,
        beta_chance = 50.0,
        scale_chance = 0.10,
        sigma_prior = ctrl_cfg.sigma_prior,
        params     = params_v5,
        init_state = Float64[init_state[1], init_state[2], init_state[3],
                              init_state[4], init_state[5], init_state[6]],
        noise_seed = Int(key & typemax(Int32)),
    )
    log_density_fn = make_log_density_fn_v5(ctrl_target)
    rng_ctrl = StableRNG(key)
    smc_kwargs = (
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
    diagnostics = NamedTuple[]
    U_ctrl, n_temp_ctrl = if collect_diagnostics
        out = run_tempered_smc_gpu(
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, theta_dim,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
            collect_diagnostics = true,
        )
        diagnostics = out[4]
        (out[1], out[2])
    else
        out = run_tempered_smc_gpu(
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, theta_dim,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
        )
        (out[1], out[2])
    end
    θ_post = vec(mean(U_ctrl; dims = 1))
    @assert length(θ_post) == theta_dim

    # ── Decode RBF θ → per-bin (Φ_B, Φ_S) ─────────────────────────────
    # Same Gaussian RBF basis the kernel uses; the design matrix is
    # rebuilt CPU-side here so the bench can write Phi traces without
    # an extra GPU transfer. Mirrors v1.5's decoder loop but doubles
    # the inner work (B-channel coefs: θ[1:n_anchors];
    # S-channel coefs: θ[n_anchors+1:2·n_anchors]).
    T_total = T_total_bins * dt
    t_grid  = collect(0:T_total_bins-1) .* dt
    anchors = collect(range(0.0, T_total; length = n_anchors))
    σ_rbf   = T_total / n_anchors
    Phi_B_plan = zeros(Float32, T_total_bins)
    Phi_S_plan = zeros(Float32, T_total_bins)
    c_Phi   = Float64(ctrl_target.c_Phi)
    for k in 1:T_total_bins
        raw_B = c_Phi
        raw_S = c_Phi
        for j in 1:n_anchors
            basis_kj = exp(-0.5 * ((t_grid[k] - anchors[j]) / σ_rbf)^2)
            raw_B += θ_post[j]              * basis_kj
            raw_S += θ_post[n_anchors + j]  * basis_kj
        end
        Phi_B_plan[k] = Float32(3.0 / (1.0 + exp(-raw_B)))
        Phi_S_plan[k] = Float32(3.0 / (1.0 + exp(-raw_S)))
    end
    return (Phi_B_plan = Phi_B_plan, Phi_S_plan = Phi_S_plan,
            n_temp_ctrl = n_temp_ctrl, theta = θ_post,
            diagnostics = diagnostics)
end
