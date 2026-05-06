@testset "Phase 2 — Bootstrap PF vs Kalman closed-form" begin
    # ── Toy model: AR(1) + Gaussian observations ─────────────────────────────
    #   x_k = a · x_{k-1} + b · ξ_k,    ξ_k ~ N(0, 1)
    #   y_k = x_k + ε_k,                 ε_k ~ N(0, ρ)
    #
    # The Kalman filter gives the exact log-likelihood log p(y_{1:T}). The
    # bootstrap PF should converge to the Kalman value as N → ∞. With
    # N = 4000 particles, T = 50, the PF Monte Carlo σ on log-lik is ~0.5–1.
    #
    # This test is language-independent: it validates the Julia bootstrap PF
    # against a closed-form reference, not against the Python implementation.
    # The Python diff test lives in test/diff_python/ (Phase 6).

    using Random
    using LogExpFunctions: logsumexp

    a_truth = 0.85
    b_truth = 0.4
    ρ_truth = 0.3

    rng = MersenneTwister(2026_05_06)
    T_obs = 50
    x = zeros(T_obs)
    y = zeros(T_obs)
    x[1] = randn(rng) * 1.0
    y[1] = x[1] + ρ_truth * randn(rng)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng)
        y[k] = x[k] + ρ_truth * randn(rng)
    end

    # ── Closed-form Kalman log-likelihood ────────────────────────────────────
    function kalman_log_lik(y::Vector{Float64}, a::Float64, b::Float64, ρ::Float64,
                            m0::Float64, P0::Float64)
        T = length(y)
        m, P = m0, P0
        ll = 0.0
        for k in 1:T
            # Predict
            m_pred = a * m
            P_pred = a^2 * P + b^2
            # Innovation
            S      = P_pred + ρ^2
            ν      = y[k] - m_pred
            ll    += -0.5 * (log(2π * S) + ν^2 / S)
            # Update
            K_gain = P_pred / S
            m      = m_pred + K_gain * ν
            P      = (1 - K_gain) * P_pred
        end
        return ll
    end

    kalman_ll = kalman_log_lik(y, a_truth, b_truth, ρ_truth, 0.0, 1.0)

    # ── Build an EstimationModel for the bootstrap PF ────────────────────────
    # State is a scalar; n_states = 1. We estimate (a, b, ρ) at TRUTH and
    # init ~ N(0, 1) — but the test only feeds the truth values, so we
    # configure the model so `priors` length matches and `params[1:3]` are
    # at the truth.

    function _propagate(y_old::AbstractVector, t, dt, params,
                         grid_obs, k, σ_diag, ξ, rng_)
        # AR(1): x_new = a * y_old + b * ξ
        a_, b_, ρ_ = params[1], params[2], params[3]
        x_new = [a_ * y_old[1] + b_ * ξ[1]]
        return x_new, 0.0   # pred_lw = 0 under bootstrap proposal
    end

    function _diffusion(params)
        # State noise scale is `b`; the propagate fn applies it directly via ξ,
        # so the per-step σ_diag passed into the buffer is just 0 (we're not
        # using σ_diag inside _propagate). Return [b] for the initialisation
        # noise sample at k=0.
        return [params[2]]
    end

    function _obs_log_weight(x_new::AbstractVector, grid_obs, k, params)
        ρ_ = params[3]
        y_k = grid_obs[:y][k]
        ν   = y_k - x_new[1]
        return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
    end

    _shard_init(time_offset, params, exog, init) = init
    _align_obs(args...) = Dict()

    model = EstimationModel(
        name = "AR1_Gauss",
        version = "test",
        n_states = 1,
        n_stochastic = 1,
        stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = [(:a, NormalPrior(0.0, 1.0)),
                        (:b, LogNormalPrior(0.0, 1.0)),
                        (:ρ, LogNormalPrior(0.0, 1.0))],
        init_state_priors = Tuple{Symbol,PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn = _propagate,
        diffusion_fn = _diffusion,
        obs_log_weight_fn = _obs_log_weight,
        align_obs_fn = _align_obs,
        shard_init_fn = _shard_init,
        exogenous_keys = Symbol[],
    )

    priors = all_priors(model)
    # Unconstrained-space coordinates of (a, b, ρ) at truth.
    # a ~ Normal → identity bijection
    # b ~ LogNormal → log
    # ρ ~ LogNormal → log
    u = [a_truth, log(b_truth), log(ρ_truth)]

    # Disable OT rescue for the test (cheaper, no impact on the language-
    # independent log-likelihood result).
    cfg = SMCConfig(n_pf_particles = 4000,
                    bandwidth_scale = 0.0,    # no Liu-West kernel contamination
                    ot_max_weight   = 0.0)

    grid_obs = Dict(:y => y)

    fixed_init = [0.0]   # x_0 prior mean

    pf_rng = MersenneTwister(11)
    pf_target = bootstrap_log_likelihood(
        model, u, grid_obs, fixed_init, priors, cfg, pf_rng;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )

    # `bootstrap_log_likelihood` returns log p(y) + log p(u). Subtract the prior
    # contribution to compare against the Kalman log-likelihood directly.
    log_prior_u = SMC2FC.log_prior_unconstrained(u, priors)
    pf_ll = pf_target - log_prior_u

    @info "Phase 2 bootstrap PF vs Kalman" pf_ll kalman_ll diff=pf_ll-kalman_ll

    # PF should be within ~3σ of Kalman. With N=4000 + bandwidth_scale=0 the
    # MC variance is small; we allow ±3.0 nats, which is ~1 nat per 17 obs.
    @test isfinite(pf_ll)
    @test abs(pf_ll - kalman_ll) < 3.0

    # Sanity: re-running with the same seed gives byte-identical result
    pf_rng2 = MersenneTwister(11)
    pf_target2 = bootstrap_log_likelihood(
        model, u, grid_obs, fixed_init, priors, cfg, pf_rng2;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )
    @test pf_target ≈ pf_target2
end
