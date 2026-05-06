@testset "Phase 6 — End-to-end SMC²: parameter recovery on AR(1)" begin
    # End-to-end smoke + correctness:
    #   simulate AR(1) data → wrap log-likelihood as the function HMC consumes
    #   → run outer adaptive-tempered SMC² → recover the (a, b, ρ) parameters.
    #
    # NOTE on the inner-PF + outer-HMC composition:
    #   The Python framework feeds the bootstrap-PF log-likelihood directly
    #   into HMC, relying on JAX + Liu-West kernel smoothing to make the
    #   resampling step pseudo-differentiable. The Julia port's bootstrap PF
    #   uses in-place `BootstrapBuffers` typed `Float64`, which ForwardDiff
    #   can't trace through. Wiring the PF as an AD-compatible target is
    #   recorded in the README's "Phase 6 follow-up" list — straightforward
    #   in principle (allocate buffers parameterised on the AD-tracked type;
    #   replace `searchsortedfirst` with a Liu-West-smoothed proxy) but
    #   non-trivial in scope.
    #
    # For this end-to-end test we use the **closed-form Kalman log-likelihood**
    # for AR(1) + Gaussian obs. This:
    #   (a) demonstrates the full Phase 1+3 composition end-to-end (transforms
    #       → outer SMC² → HMC → AR(1) prior surface),
    #   (b) is independently validated against the Julia bootstrap PF in the
    #       Phase 2 test (`Phase 2 — Bootstrap PF vs Kalman closed-form`).
    # So the framework's correctness on this problem rests on two
    # independently-tested pieces being composable, which the Phase 6
    # composition test confirms numerically.

    using Random
    using Statistics: mean, std

    rng_data = MersenneTwister(2026_05_06)
    a_truth, b_truth, ρ_truth = 0.85, 0.4, 0.3
    T_obs = 30
    x = zeros(T_obs); y = zeros(T_obs)
    x[1] = randn(rng_data); y[1] = x[1] + ρ_truth * randn(rng_data)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng_data)
        y[k] = x[k]              + ρ_truth * randn(rng_data)
    end

    # Closed-form Kalman log-likelihood (AR(1) + Gaussian obs).
    function kalman_log_lik(y, a, b, ρ)
        T = length(y)
        m, P = 0.0, 1.0
        ll = 0.0
        for k in 1:T
            m_pred = a * m
            P_pred = a^2 * P + b^2
            S      = P_pred + ρ^2
            ν      = y[k] - m_pred
            ll    += -0.5 * (log(2π * S) + ν^2 / S)
            K      = P_pred / S
            m      = m_pred + K * ν
            P      = (1 - K) * P_pred
        end
        return ll
    end

    # Priors — match Phase 2 test
    priors = PriorType[NormalPrior(0.0, 1.0),       # a
                        LogNormalPrior(0.0, 1.0),    # b
                        LogNormalPrior(0.0, 1.0)]    # ρ

    # log-likelihood the outer SMC² consumes:
    #   ll(u) = log p(y | θ(u))   — does NOT include the prior
    function loglik_at_u(u)
        a_  = u[1]                              # NormalPrior identity
        b_  = exp(u[2])                          # LogNormalPrior bijection
        ρ_  = exp(u[3])
        return kalman_log_lik(y, a_, b_, ρ_)
    end

    cfg_outer = SMCConfig(
        n_smc_particles  = 64,
        target_ess_frac  = 0.5,
        num_mcmc_steps   = 5,
        max_lambda_inc   = 0.15,
        hmc_step_size    = 0.05,
        hmc_num_leapfrog = 8,
    )

    rng_outer = MersenneTwister(11)
    t0 = time()
    result = run_smc_window(loglik_at_u, priors, cfg_outer, rng_outer)
    elapsed_total = time() - t0

    @info "Phase 6 end-to-end (Kalman likelihood)" n_temp=result.n_temp elapsed=elapsed_total

    @test result.n_temp ≥ 2
    @test size(result.particles) == (cfg_outer.n_smc_particles, 3)
    @test all(isfinite, result.particles)

    # Recover AR(1) parameters in unconstrained space:
    #   E[u₁]    ≈ a_truth
    #   E[exp u₂] ≈ b_truth
    #   E[exp u₃] ≈ ρ_truth
    a_post     = mean(result.particles[:, 1])
    b_post     = mean(exp.(result.particles[:, 2]))
    ρ_post     = mean(exp.(result.particles[:, 3]))
    @info "  posterior means" a_post b_post ρ_post
    @info "  truth"           a_truth b_truth ρ_truth

    @test abs(a_post - a_truth)     < 0.15
    @test abs(b_post - b_truth)     < 0.20
    @test abs(ρ_post - ρ_truth)     < 0.20
end
