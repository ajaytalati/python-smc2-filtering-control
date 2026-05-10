# bench/bench_controller.jl
#
# Controller-side wrapper used by the bench
# `tools/bench_smc_full_mpc_fsa_gpu.jl`. Builds an `FSAv1ControlGPUTarget`
# from posterior-mean v1-form params, calls the framework's
# `run_tempered_smc_gpu` (with optional per-tempering-level diagnostics
# collection), and decodes the posterior-mean RBF coefficient vector
# into a per-bin Φ schedule.
#
# Public function (verbatim from the prior in-line definition):
#   - controller_plan(params_v1, init_state, T_total_bins, n_substeps, dt,
#                      ctrl_cfg, key; collect_diagnostics=false)
#       -> (Phi_plan, n_temp_ctrl, theta, diagnostics)
#
# Extracted 2026-05-09 as Phase 1d of the bench refactor described in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
# Verbatim move; no signature or behaviour change.
#
# Dependencies (all loaded by the calling bench script before this file
# is `include`d): `Statistics.mean`, `StableRNGs.StableRNG`,
# `StaticArrays.SVector`, the FSA controller GPU kernel
# (`FSAv1ControlGPUTarget`, `make_log_density_fn`), and the framework's
# `run_tempered_smc_gpu`.


# ── Pure controller plan call ────────────────────────────────────────────
# Builds an FSAControlGPUTarget with v1-form params and runs the framework's
# run_tempered_smc_gpu on the RBF θ. Decodes posterior-mean θ → per-bin Φ.

function controller_plan(params_v1::NamedTuple,
                          init_state::SVector{3, Float64},
                          T_total_bins::Int,
                          n_substeps::Int,
                          dt::Float64,
                          ctrl_cfg::NamedTuple,
                          key::UInt64;
                          collect_diagnostics::Bool = false)
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
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, ctrl_cfg.n_anchors,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
            collect_diagnostics = true,
        )
        diagnostics = out[4]
        (out[1], out[2])
    else
        out = run_tempered_smc_gpu(
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, ctrl_cfg.n_anchors,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
        )
        (out[1], out[2])
    end
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
    return (Phi_plan = Phi_plan, n_temp_ctrl = n_temp_ctrl, theta = θ_post,
            diagnostics = diagnostics)
end
