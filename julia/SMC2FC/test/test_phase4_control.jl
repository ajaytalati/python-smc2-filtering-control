@testset "Phase 4 — Control" begin
    using LinearAlgebra: norm
    using Statistics: mean

    @testset "RBFBasis: design matrix shape and Gaussian decay" begin
        b = RBFBasis(20, 0.5, 4)
        Φ = design_matrix(b)
        @test size(Φ) == (20, 4)
        # Each row's max value is 1 only at an anchor; mid-anchor values are < 1.
        @test all(Φ .≤ 1.0)
        @test all(Φ .≥ 0.0)
    end

    @testset "RBFBasis: schedule_from_theta with output transforms" begin
        b_id  = RBFBasis(10, 1.0, 3; output=IdentityOutput())
        b_sp  = RBFBasis(10, 1.0, 3; output=SoftplusOutput())
        b_sig = RBFBasis(10, 1.0, 3; output=SigmoidOutput())

        θ = [1.0, -1.0, 0.5]
        Φ = design_matrix(b_id)
        s_id  = schedule_from_theta(b_id,  θ; Φ=Φ)
        s_sp  = schedule_from_theta(b_sp,  θ; Φ=Φ)
        s_sig = schedule_from_theta(b_sig, θ; Φ=Φ)

        @test length(s_id) == 10
        @test all(s_sp .≥ 0)              # softplus
        @test all(0 .≤ s_sig .≤ 1)         # sigmoid
    end

    @testset "calibrate_beta_max: nat-budget targeting" begin
        # Quadratic cost in 4-D — std of cost(θ) for θ ~ N(0, 1.5²·I) is
        # known analytically: std(‖θ‖²) ≈ √(8) · σ² for d=4.
        cost_fn(u) = sum(abs2, u)
        β_max, c_mean, c_std = calibrate_beta_max(
            cost_fn; theta_dim=4, sigma_prior=1.5, n_samples=2000,
            target_nats=8.0, seed=0,
        )
        @test β_max > 0
        @test β_max ≈ 8.0 / c_std  rtol=1e-6
        # Sanity: c_std ~ √(8) · 2.25 ≈ 6.36 → β_max ≈ 8 / 6.36 ≈ 1.26
        @test 0.5 < β_max < 5.0
    end

    @testset "Closed-form quadratic cost: SMC² recovers optimum" begin
        # cost(θ) = ‖θ - θ_star‖² (4-D).
        # Optimal θ = θ_star = [1, -1, 0.5, 2].
        # Posterior mean → θ_star as β → β_max.
        d = 4
        θ_star = [1.0, -1.0, 0.5, 2.0]
        cost_fn(u) = sum(abs2, u .- θ_star)

        # Schedule decoder is identity for this test (we don't need one to
        # validate the optimisation; it's exercised by RBF tests above).
        sched_fn(u) = collect(u)

        spec = ControlSpec(
            name              = "quadratic_test",
            version           = "phase4",
            dt                = 1.0,
            n_steps           = d,
            initial_state     = zeros(d),
            theta_dim         = d,
            sigma_prior       = 1.5,
            cost_fn           = cost_fn,
            schedule_from_theta = sched_fn,
        )

        cfg = SMCConfig(
            n_smc_particles  = 128,
            target_ess_frac  = 0.5,
            num_mcmc_steps   = 8,
            max_lambda_inc   = 0.2,
            hmc_step_size    = 0.1,
            hmc_num_leapfrog = 8,
        )

        rng = MersenneTwister(2026_05_06)
        result = run_tempered_smc_loop(spec, cfg, rng;
                                         calib_n=128, target_nats=6.0)

        @info "Phase 4 control SMC²" n_temp=result.n_temp elapsed=result.elapsed β_max=result.β_max
        @test result.n_temp ≥ 2
        @test size(result.particles) == (cfg.n_smc_particles, d)

        # Posterior mean θ should be close to θ_star (within ~0.7 in each dim
        # given β_max from 6 nats over 128 particles, 8 leapfrog steps, 8 HMC
        # moves per level — small budget. Production cfg gets ~0.1 tolerance.)
        emp_mean = vec(mean(result.particles; dims=1))
        @info "  posterior mean vs θ_star" emp_mean θ_star
        @test maximum(abs.(emp_mean .- θ_star)) < 0.7
    end
end
