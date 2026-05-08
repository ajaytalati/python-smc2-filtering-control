"""
Tests for `Transforms.jl` — the constrained ↔ unconstrained bijections,
prior log-densities, the new tuple-of-priors fast path, and `build_priors`
/ `split_theta`.
"""

using Test
using Random
using SMC2FC_functional
using StaticArrays: SVector

include("fixtures.jl")

@testset "Transforms — per-component round trip" begin
    rng = MersenneTwister(42)

    # LogNormal: θ > 0 ↔ u ∈ ℝ
    for _ in 1:50
        θ  = exp(randn(rng))
        u  = SMC2FC_functional.to_unconstrained(LogNormalPrior(0.0, 1.0), θ)
        θ′ = SMC2FC_functional.to_constrained(LogNormalPrior(0.0, 1.0), u)
        @test θ ≈ θ′ atol=1e-10
    end

    # Normal: identity in domain
    for _ in 1:50
        θ = randn(rng)
        u = SMC2FC_functional.to_unconstrained(NormalPrior(0.0, 1.0), θ)
        @test SMC2FC_functional.to_constrained(NormalPrior(0.0, 1.0), u) ≈ θ
    end

    # Beta: θ ∈ (0,1) ↔ u ∈ ℝ
    for _ in 1:50
        θ  = rand(rng) * 0.998 + 0.001
        u  = SMC2FC_functional.to_unconstrained(BetaPrior(2.0, 2.0), θ)
        θ′ = SMC2FC_functional.to_constrained(BetaPrior(2.0, 2.0), u)
        @test θ ≈ θ′ atol=1e-10
    end
end

@testset "Transforms — vector round trip (Vector{<:PriorType})" begin
    rng = MersenneTwister(7)
    p_vec, _ = make_tiny_priors()
    θ = [exp(randn(rng)),  randn(rng),  rand(rng) * 0.99 + 0.005,  randn(rng)]
    u  = constrained_to_unconstrained(θ, p_vec)
    θ′ = unconstrained_to_constrained(u, p_vec)
    for i in 1:3        # vonmises is identity in u — round-trip exact only on the others
        @test θ[i] ≈ θ′[i] atol=1e-10
    end
end

@testset "Transforms — tuple-of-priors hot path" begin
    rng = MersenneTwister(31)
    _, p_tup = make_tiny_priors()
    θ = [exp(randn(rng)), randn(rng), rand(rng) * 0.99 + 0.005, randn(rng)]
    u_tup  = constrained_to_unconstrained(θ, p_tup)
    θ′_tup = unconstrained_to_constrained(u_tup, p_tup)
    @test u_tup isa SVector{4,Float64}
    @test θ′_tup isa SVector{4,Float64}
    for i in 1:3
        @test θ[i] ≈ θ′_tup[i] atol=1e-10
    end

    # tuple and vector forms must agree
    p_vec = PriorType[p_tup...]
    u_vec = constrained_to_unconstrained(θ, p_vec)
    @test all(isapprox.(collect(u_tup), u_vec; atol=1e-12))
end

@testset "Transforms — prior log-densities (formula spot-checks)" begin
    u = 0.3
    @test log_prior_unconstrained(LogNormalPrior(0.0, 1.0), u) ≈
          -0.5 * (u/1.0)^2 - log(1.0)
    @test log_prior_unconstrained(NormalPrior(2.0, 0.5), u) ≈
          -0.5 * ((u - 2.0)/0.5)^2 - log(0.5)
    @test log_prior_unconstrained(VonMisesPrior(0.0, 4.0), u) ≈
          4.0 * cos(u - 0.0)
    σ_u  = 1 / (1 + exp(-u))
    σ_mu = 1 / (1 + exp(u))
    @test log_prior_unconstrained(BetaPrior(2.0, 5.0), u) ≈
          2.0 * log(σ_u) + 5.0 * log(σ_mu) atol=1e-12
end

@testset "Transforms — log_prior_unconstrained on tuple ≡ on vector" begin
    rng = MersenneTwister(99)
    p_vec, p_tup = make_tiny_priors()
    u = randn(rng, 4)
    s_vec = log_prior_unconstrained(u, p_vec)
    s_tup = log_prior_unconstrained(u, p_tup)
    @test s_vec ≈ s_tup atol=1e-12
end

@testset "Transforms — build_priors / split_theta" begin
    priors = build_priors([
        (:lognormal, (0.0, 1.0)),
        (:vonmises,  (0.0, 4.0)),
        (:beta,      (2.0, 5.0)),
        (:normal,    (0.0, 0.5)),
    ])
    @test length(priors) == 4
    @test priors[1] isa LogNormalPrior
    @test priors[2] isa VonMisesPrior
    @test priors[3] isa BetaPrior
    @test priors[4] isa NormalPrior

    θ = collect(1.0:6.0)
    params, inits = split_theta(θ, 4)
    @test collect(params) == [1.0, 2.0, 3.0, 4.0]
    @test collect(inits)  == [5.0, 6.0]
end

@testset "Transforms — DimensionMismatch on length disagreement" begin
    p_vec, _ = make_tiny_priors()
    @test_throws DimensionMismatch constrained_to_unconstrained([1.0, 2.0], p_vec)
    @test_throws DimensionMismatch unconstrained_to_constrained([1.0, 2.0], p_vec)
    @test_throws DimensionMismatch log_prior_unconstrained([1.0, 2.0], p_vec)
end
