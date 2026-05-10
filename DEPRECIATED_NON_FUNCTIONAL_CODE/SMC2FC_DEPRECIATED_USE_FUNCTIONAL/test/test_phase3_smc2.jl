@testset "Phase 3 — Outer SMC² on a Gaussian target" begin
    using LinearAlgebra: I, Diagonal, dot
    using Statistics: mean, std

    @testset "Tempering: ESS bisection finds δλ that hits target" begin
        # Synthetic per-particle log-likelihood with strong concentration.
        N = 256
        rng = MersenneTwister(0)
        ll = randn(rng, N) .* 5.0       # std = 5 → reweighting collapses ESS quickly

        # Find δλ so that ESS(δλ) ≈ 0.5 N
        δ = solve_delta_for_ess(ll, 0.5, 1.0; n_bisect_steps=40)
        ess = ess_at_delta(ll, δ)
        # Tolerance 5 % of N — bisection converges fast on monotone f
        @test abs(ess - 0.5 * N) < 0.05 * N
    end

    @testset "MassMatrix: per-dim variance with regularisation" begin
        rng = MersenneTwister(1)
        particles = randn(rng, 200, 4) .* [1.0, 2.0, 0.5, 0.001]'   # last dim degenerate
        m = estimate_mass_matrix(particles)
        @test length(m) == 4
        @test m[1] > 0.5 && m[1] < 2.0    # sample variance ~ 1
        @test m[2] > 2.0 && m[2] < 6.0    # sample variance ~ 4
        @test m[4] ≥ 1e-4                  # regularised
    end

    @testset "Sampling: prior cloud has correct marginal stats" begin
        rng = MersenneTwister(2)
        priors = PriorType[NormalPrior(0.0, 1.0),
                            LogNormalPrior(2.0, 0.5),
                            BetaPrior(2.0, 5.0)]
        P = sample_from_prior(2000, priors, rng)
        # Normal(0, 1) → mean ≈ 0, std ≈ 1
        @test abs(mean(@view P[:, 1])) < 0.1
        @test abs(std(@view P[:, 1]) - 1.0) < 0.1
        # LogNormal(2, 0.5) in unconstrained space → Normal(2, 0.5)
        @test abs(mean(@view P[:, 2]) - 2.0) < 0.1
        @test abs(std(@view P[:, 2]) - 0.5) < 0.1
    end

    @testset "Cold-start outer SMC² recovers Gaussian posterior mean" begin
        # Closed-form target: u ~ N(μ_truth, Σ_truth) directly. The "likelihood"
        # is a Gaussian centred at μ_truth, the "prior" is N(0, I). The exact
        # posterior is conjugate Gaussian with closed-form mean and variance.
        d = 4
        μ_truth = [1.5, -0.7, 2.1, 0.3]
        Σ_lik   = Diagonal(fill(0.5^2, d))    # observation precision

        # Build priors so log_prior_unconstrained(u) = -½ ‖u‖²  (standard normal).
        priors = PriorType[NormalPrior(0.0, 1.0) for _ in 1:d]

        # log-likelihood: -½ (u - μ_truth)' Σ_lik⁻¹ (u - μ_truth)
        Σinv = inv(Σ_lik)
        function loglik(u)
            δ = u .- μ_truth
            return -0.5 * dot(δ, Σinv * δ)
        end

        # Conjugate-Gaussian posterior:
        #   μ_post = (I + Σ_lik⁻¹)⁻¹ · Σ_lik⁻¹ · μ_truth
        #   Σ_post = (I + Σ_lik⁻¹)⁻¹
        # With prior precision Λ_p = I, lik precision Λ_l = Σ_lik⁻¹:
        Λp = Diagonal(fill(1.0, d))
        Σ_post = inv(Matrix(Λp) + Matrix(Σinv))
        μ_post = Σ_post * (Σinv * μ_truth)

        cfg = SMCConfig(
            n_smc_particles  = 128,
            target_ess_frac  = 0.5,
            num_mcmc_steps   = 8,
            max_lambda_inc   = 0.1,
            hmc_step_size    = 0.15,
            hmc_num_leapfrog = 8,
        )

        rng = MersenneTwister(2026_05_06)
        result = run_smc_window(loglik, priors, cfg, rng)

        @info "Phase 3 cold-start SMC²" n_temp=result.n_temp elapsed=result.elapsed
        @test result.n_temp ≥ 2
        @test size(result.particles) == (cfg.n_smc_particles, d)

        # Empirical posterior mean should be close to closed form.
        emp_mean = vec(mean(result.particles; dims=1))
        @info "  posterior mean: empirical vs closed-form" emp_mean μ_post
        @test maximum(abs.(emp_mean .- μ_post)) < 0.3
    end

    @testset "Gaussian bridge: warm-start sampling from previous posterior" begin
        rng = MersenneTwister(3)
        d = 3
        n_smc = 200
        # Synthetic "previous posterior" cloud.
        prev = rand(rng, n_smc, d) .* 2.0 .+ 1.0

        cfg = SMCConfig(n_smc_particles = n_smc, bridge_type = :gaussian)
        new_cloud = bridge_init(GaussianBridge(), prev, identity, cfg, rng;
                                 n_smc = n_smc)
        @test size(new_cloud) == (n_smc, d)
        # New cloud should have similar mean and covariance to prev (single-Gaussian fit)
        Δμ = vec(mean(new_cloud; dims=1)) .- vec(mean(prev; dims=1))
        @test maximum(abs.(Δμ)) < 0.3
    end
end
