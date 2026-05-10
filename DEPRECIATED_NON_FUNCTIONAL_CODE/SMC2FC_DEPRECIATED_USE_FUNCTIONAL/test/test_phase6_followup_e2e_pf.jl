@testset "Phase 6 follow-up #1 integration — PF inside HMC inside SMC²" begin
    # End-to-end with the bootstrap PF as the inner likelihood (instead of the
    # Kalman shortcut used in `test_phase6_e2e.jl`):
    #
    #   simulate AR(1) data
    #   → outer SMC² over (a, b, ρ)
    #     → at each tempering level, AdvancedHMC.jl's HMC kernel takes
    #       gradient through `bootstrap_log_likelihood`
    #       → ForwardDiff traces the PF inner loop with Dual-typed buffers
    #
    # This is the path the Python SMC²-MPC bench uses for production runs;
    # the Kalman shortcut in v1 was the placeholder while the AD-compatible
    # PF was a documented follow-up. With Phase 6 follow-up #1 landed, the
    # full stack composes.
    #
    # NOTE on PF noise + HMC: the gradient through a stochastic PF is itself
    # a stochastic estimator, so HMC's leapfrog acceptance rate degrades vs
    # a deterministic likelihood. We compensate by:
    #   (a) hashing `primal(u)` for the per-call seed so a single ForwardDiff
    #       evaluation uses ONE realisation of the PF noise (primal + Dual
    #       partials computed at the same noise),
    #   (b) using a small leapfrog step,
    #   (c) using K=600 PF particles to keep the per-call MC noise tame,
    #   (d) accepting that the recovery tolerance is wider than the Kalman
    #       version. Both are diagnostically useful: Kalman tests the outer
    #       machinery; PF tests the AD composition end-to-end.

    using Random
    using Statistics: mean

    a_truth, b_truth, ρ_truth = 0.85, 0.4, 0.3
    T_obs = 25
    rng_data = MersenneTwister(2026_05_06)
    x = zeros(T_obs); y = zeros(T_obs)
    x[1] = randn(rng_data); y[1] = x[1] + ρ_truth * randn(rng_data)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng_data)
        y[k] = x[k]              + ρ_truth * randn(rng_data)
    end

    function _propagate(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng_)
        a_, b_, ρ_ = params[1], params[2], params[3]
        return [a_ * y_old[1] + b_ * ξ[1]], 0.0
    end
    _diffusion(p) = [p[2]]
    function _obs_log_weight(x_new, grid_obs, k, params)
        ρ_ = params[3]
        ν  = grid_obs[:y][k] - x_new[1]
        return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
    end
    _shard_init(t_off, p, exog, init) = init
    _align_obs(args...) = Dict()

    model = EstimationModel(
        name = "AR1_PF_E2E", version = "phase6_followup_integration",
        n_states = 1, n_stochastic = 1, stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = [(:a, NormalPrior(0.0, 1.0)),
                        (:b, LogNormalPrior(0.0, 1.0)),
                        (:ρ, LogNormalPrior(0.0, 1.0))],
        init_state_priors = Tuple{Symbol,PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn = _propagate, diffusion_fn = _diffusion,
        obs_log_weight_fn = _obs_log_weight,
        align_obs_fn = _align_obs, shard_init_fn = _shard_init,
    )

    priors  = all_priors(model)
    grid_obs = Dict(:y => y)
    fixed_init = [0.0]

    cfg_inner = SMCConfig(n_pf_particles = 600,
                           bandwidth_scale = 0.0,
                           ot_max_weight = 0.0)

    # log-likelihood the outer SMC² consumes — runs the AD-compatible PF
    using ForwardDiff: value
    function loglik_via_pf(u)
        primal_u = value.(u)
        seed     = abs(hash(primal_u)) % typemax(UInt32)
        target   = bootstrap_log_likelihood(
            model, collect(u), grid_obs, fixed_init, priors, cfg_inner,
            MersenneTwister(seed);
            dt = 1.0, t_steps = T_obs, window_start_bin = 0,
        )
        # bootstrap_log_likelihood returns log p(y|θ) + log p(u); the outer
        # SMC²'s `tempered_lp` adds log p(u) again, so we subtract here.
        return target - SMC2FC.log_prior_unconstrained(u, priors)
    end

    cfg_outer = SMCConfig(
        n_smc_particles  = 24,         # small cloud — integration smoke
        target_ess_frac  = 0.5,
        num_mcmc_steps   = 2,           # 2 HMC moves per level (PF-noise budget)
        max_lambda_inc   = 0.2,
        hmc_step_size    = 0.02,        # small step due to PF gradient noise
        hmc_num_leapfrog = 4,
    )

    rng_outer = MersenneTwister(11)
    t0 = time()
    result = run_smc_window(loglik_via_pf, priors, cfg_outer, rng_outer)
    elapsed_total = time() - t0

    @info "Phase 6 follow-up #1 integration (PF inside HMC inside SMC²)" n_temp=result.n_temp elapsed=elapsed_total

    @test result.n_temp ≥ 1
    @test size(result.particles) == (cfg_outer.n_smc_particles, 3)
    @test all(isfinite, result.particles)

    a_post   = mean(result.particles[:, 1])
    logb_post = mean(result.particles[:, 2])
    logρ_post = mean(result.particles[:, 3])
    @info "  posterior means" a_post logb_post logρ_post
    @info "  truth (unconstrained)" a_truth log(b_truth) log(ρ_truth)

    # Wide tolerance vs the Kalman E2E: the PF gradient is noisy and the
    # outer cloud is tiny (n_smc=24). What this test checks is that the
    # pipeline COMPOSES end-to-end — Dual-typed PF buffers, ForwardDiff
    # through the resample step, AdvancedHMC.jl + LogDensityProblems target.
    # Convergence quality is a separate tuning concern that needs n_smc≈128,
    # K≈400, num_mcmc_steps≈8 — production budget per CLAUDE.md, not test
    # budget. The bounds below sit a couple of σ wide of the prior.
    @test abs(a_post   - a_truth)     < 1.2
    @test abs(logb_post - log(b_truth)) < 2.0
    @test abs(logρ_post - log(ρ_truth)) < 2.0
end
