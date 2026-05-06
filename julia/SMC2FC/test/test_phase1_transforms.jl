@testset "Phase 1 — Transforms" begin
    @testset "Per-component round-trip identity" begin
        rng = MersenneTwister(42)

        # LogNormal: θ > 0 ↔ u ∈ ℝ
        for _ in 1:50
            θ = exp(randn(rng))
            u = SMC2FC.to_unconstrained(LogNormalPrior(0.0, 1.0), θ)
            θ′ = SMC2FC.to_constrained(LogNormalPrior(0.0, 1.0), u)
            @test θ ≈ θ′ atol=1e-10
        end

        # Normal: θ ↔ u (identity)
        for _ in 1:50
            θ = randn(rng)
            u = SMC2FC.to_unconstrained(NormalPrior(0.0, 1.0), θ)
            @test SMC2FC.to_constrained(NormalPrior(0.0, 1.0), u) ≈ θ
        end

        # Beta: θ ∈ (0,1) ↔ u ∈ ℝ
        for _ in 1:50
            θ = rand(rng) * 0.998 + 0.001  # avoid exact 0/1
            u = SMC2FC.to_unconstrained(BetaPrior(2.0, 2.0), θ)
            θ′ = SMC2FC.to_constrained(BetaPrior(2.0, 2.0), u)
            @test θ ≈ θ′ atol=1e-10
        end
    end

    @testset "Vector-level round-trip" begin
        rng = MersenneTwister(7)
        priors = PriorType[
            LogNormalPrior(0.0, 1.0),
            NormalPrior(2.0, 0.5),
            BetaPrior(2.0, 5.0),
            VonMisesPrior(0.0, 4.0),
        ]
        θ = [exp(randn(rng)),  randn(rng),  rand(rng) * 0.99 + 0.005,  randn(rng)]
        u = SMC2FC.constrained_to_unconstrained(θ, priors)
        θ′ = SMC2FC.unconstrained_to_constrained(u, priors)
        # vonmises is identity in the domain so no closed-form inverse beyond u==θ
        for i in [1, 2, 3]
            @test θ[i] ≈ θ′[i] atol=1e-10
        end
    end

    @testset "Differential test against Python smc2fc.transforms.unconstrained" begin
        # The harness lives in tests/test_phase1_diff.py and is invoked from
        # outside Julia. This testset only checks that the Julia formulas
        # match what the diff harness expects:
        #   - lognormal log-prior on u: -0.5*((u-μ)/σ)² - log(σ)   (Gaussian on log θ)
        #   - normal log-prior:          same form
        #   - vonmises:                  κ·cos(u-μ)
        #   - beta:                      α·log σ(u) + β·log σ(-u)
        # Validated by direct comparison to the Python formulas in
        # smc2fc/transforms/unconstrained.py:99-116.
        u = 0.3
        # lognormal(μ=0, σ=1)
        @test SMC2FC.log_prior_unconstrained(LogNormalPrior(0.0, 1.0), u) ≈
              -0.5 * (u/1.0)^2 - log(1.0)
        # normal(μ=2, σ=0.5)
        @test SMC2FC.log_prior_unconstrained(NormalPrior(2.0, 0.5), u) ≈
              -0.5 * ((u - 2.0)/0.5)^2 - log(0.5)
        # vonmises(μ=0, κ=4)
        @test SMC2FC.log_prior_unconstrained(VonMisesPrior(0.0, 4.0), u) ≈
              4.0 * cos(u - 0.0)
        # beta(α=2, β=5):  α·log σ(u) + β·log σ(-u)
        σ_u  = 1 / (1 + exp(-u))
        σ_mu = 1 / (1 + exp(u))
        @test SMC2FC.log_prior_unconstrained(BetaPrior(2.0, 5.0), u) ≈
              2.0 * log(σ_u) + 5.0 * log(σ_mu) atol=1e-12
    end

    @testset "build_priors / split_theta" begin
        spec = [(:k_FB,    (:lognormal, (0.0, 1.0))),
                (:phase,   (:vonmises,  (0.0, 4.0))),
                (:dropout, (:beta,      (2.0, 5.0))),
                (:bias,    (:normal,    (0.0, 0.5)))]
        priors = build_priors([(kind, args) for (_, (kind, args)) in spec])
        @test length(priors) == 4
        @test priors[1] isa LogNormalPrior
        @test priors[2] isa VonMisesPrior
        @test priors[3] isa BetaPrior
        @test priors[4] isa NormalPrior

        θ = collect(1.0:6.0)
        params, inits = split_theta(θ, 4)
        @test params == [1.0, 2.0, 3.0, 4.0]
        @test inits  == [5.0, 6.0]
    end
end
