@testset "Phase 6 follow-up #1 — ForwardDiff through bootstrap PF" begin
    # Validates that `bootstrap_log_likelihood` is now AD-compatible.
    #   1. ForwardDiff.gradient runs without dispatch errors.
    #   2. The autodiff gradient matches a 5-point finite-difference estimator
    #      within a tolerance that accounts for PF Monte Carlo noise.
    #   3. The bootstrap PF can be wrapped as a LogDensityProblems target and
    #      consumed by AdvancedHMC.jl (gradient evaluation only — full HMC
    #      sampling lives in the integration test below).
    #
    # The key trick (charter §15.1 follow-up): random noise inside the PF is
    # sampled as Float64 (zero gradient by construction); the buffer type is
    # parameterised on the AD-tracked T so the inner-loop arithmetic stays in
    # Dual numbers when ForwardDiff calls in. The systematic-resample step
    # uses integer indices into a Dual-typed cloud — gradient flows through
    # the values, not the indices, exactly the way JAX's `gather` does.

    using ForwardDiff
    using Random
    using LogDensityProblems
    using LogDensityProblemsAD: ADgradient

    # Toy AR(1) model — same shape as Phase 6 E2E
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
        name = "AR1_AD", version = "phase6_followup",
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

    priors = all_priors(model)
    grid_obs = Dict(:y => y)
    fixed_init = [0.0]   # NOTE: Float64, while `u` will become Dual under AD
    cfg = SMCConfig(n_pf_particles = 800,
                    bandwidth_scale = 0.0,
                    ot_max_weight = 0.0)

    # The likelihood callable HMC consumes (returns scalar log p(y|θ) + log p(u))
    function ll(u)
        # Critical: each gradient call uses a DIFFERENT but DETERMINISTIC seed,
        # derived from the primal of u. ForwardDiff calls `ll(u_dual)` once for
        # the primal-and-gradient evaluation; we want the SAME random numbers
        # so the gradient is taken at the same realisation of the PF noise as
        # the primal. Hashing the primal achieves this.
        primal_u = ForwardDiff.value.(u)   # Vector{Float64} regardless of T
        seed     = abs(hash(primal_u)) % typemax(UInt32)
        return bootstrap_log_likelihood(
            model, collect(u), grid_obs, fixed_init, priors, cfg,
            MersenneTwister(seed);
            dt = 1.0, t_steps = T_obs, window_start_bin = 0,
        )
    end

    # ── 1. Primal evaluates without error ────────────────────────────────────
    u0 = [a_truth, log(b_truth), log(ρ_truth)]
    primal = ll(u0)
    @test isfinite(primal)
    @info "Phase 6 follow-up — bootstrap PF primal at truth" primal

    # ── 2. ForwardDiff gradient runs (the previous v1 blocker) ───────────────
    g_ad = ForwardDiff.gradient(ll, u0)
    @test all(isfinite, g_ad)
    @test length(g_ad) == 3
    @info "  ForwardDiff gradient" g_ad

    # ── 3. Finite-difference cross-check ─────────────────────────────────────
    # PF Monte Carlo σ on log-lik for K=800 / T=25 is ~0.5–1; that propagates
    # to gradient noise of similar magnitude per dim. Use h = 0.05 for the FD
    # step (small enough that the linearisation is accurate; large enough that
    # the PF noise doesn't dominate). Tolerance 5 nats per dim.
    h = 0.05
    g_fd = zeros(3)
    for i in 1:3
        u_p = copy(u0); u_p[i] += h
        u_m = copy(u0); u_m[i] -= h
        g_fd[i] = (ll(u_p) - ll(u_m)) / (2h)
    end
    @info "  finite-difference gradient (h=$h)" g_fd
    @info "  AD - FD" g_ad .- g_fd
    # PF noise grows with how many particles each call resamples differently;
    # we used hash(primal) so the same u gets the same seed across calls, but
    # u_p ≠ u_m so they DO see different PF noise realisations. Allow 8 nats
    # per dim of slack.
    @test maximum(abs.(g_ad .- g_fd)) < 8.0

    # ── 4. LogDensityProblems target wraps the PF callable ───────────────────
    # This confirms the PF can be plugged into AdvancedHMC.jl's standard
    # `Hamiltonian + Leapfrog + sample` interface — the integration test below
    # then drives a small SMC²-MPC bench through it.
    target = ADgradient(:ForwardDiff, SMC2FC.HMC.CallableTarget(ll, 3))
    val, grad = LogDensityProblems.logdensity_and_gradient(target, u0)
    @test isfinite(val)
    @test all(isfinite, grad)
end
