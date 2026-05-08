"""
Tests for `SMC2/Tempering.jl`, `SMC2/Sampling.jl`, `SMC2/MassMatrix.jl`,
and `SMC2/Bridge.jl` — the smaller building blocks of the outer SMC²
loop. The full `run_smc_window` entry point is covered by the
end-to-end / Python-comparison harness.
"""

using Test
using Random
using SMC2FC_functional

include("fixtures.jl")

@testset "SMC2 — ess_at_delta + solve_delta_for_ess" begin
    rng = MersenneTwister(0)
    n_smc = 64
    ll = randn(rng, n_smc)
    # ESS at δ = 0 is exactly N (uniform weights).
    @test ess_at_delta(ll, 0.0) ≈ n_smc atol=1e-8
    # δ should drop ESS as we increase it.
    @test ess_at_delta(ll, 1.0) < ess_at_delta(ll, 0.1)
    # solve_delta_for_ess returns a δ whose ESS is ≈ target_ess_frac · N.
    target = 0.5
    δ = solve_delta_for_ess(ll, target, 1.0)
    @test ess_at_delta(ll, δ) ≥ target * n_smc - 1.0
end

@testset "SMC2 — sample_from_prior shape and finiteness" begin
    rng = MersenneTwister(0)
    p_vec, _ = make_tiny_priors()
    P = sample_from_prior(32, p_vec, rng)
    @test size(P) == (32, length(p_vec))
    @test all(isfinite, P)
end

@testset "SMC2 — estimate_mass_matrix is positive and per-dim variance" begin
    rng = MersenneTwister(0)
    P = randn(rng, 64, 4) .* [1.0 2.0 0.5 1.5]
    m = estimate_mass_matrix(P)
    @test length(m) == 4
    @test all(m .> 0)
    # Mass ≈ variance up to the regularisation floor; check ordering.
    @test m[2] > m[1]
    @test m[1] > m[3]
end

@testset "SMC2 — Bridge fit_gaussian + sample_from_gaussian" begin
    rng = MersenneTwister(0)
    P = randn(rng, 128, 2)
    μ, Σ = fit_gaussian(P)
    @test length(μ) == 2
    @test size(Σ) == (2, 2)

    new_samples = sample_from_gaussian(rng, μ, Σ, 64)
    @test size(new_samples) == (64, 2)
end

@testset "SMC2 — bridge_kind translates :gaussian and :schrodinger_follmer" begin
    cfg_g = SMCConfig(; bridge_type = :gaussian)
    cfg_s = SMCConfig(; bridge_type = :schrodinger_follmer)
    @test bridge_kind(cfg_g) isa GaussianBridge
    @test bridge_kind(cfg_s) isa SchrodingerFollmerBridge

    cfg_bad = SMCConfig(; bridge_type = :unknown_kind)
    @test_throws ErrorException bridge_kind(cfg_bad)
end
