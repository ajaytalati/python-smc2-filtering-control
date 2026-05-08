"""
Tests for `Control/RBFSchedule.jl` and `Control/Calibration.jl`.

The `run_tempered_smc_loop` entry point is covered by the end-to-end /
Python-comparison harness — it requires a real model and is too slow for
unit tests here.
"""

using Test
using Random
using SMC2FC_functional

@testset "Control — RBFBasis design matrix shape and apply_output" begin
    b = RBFBasis(20, 0.1, 4)    # 20 steps, dt=0.1, 4 anchors
    Φ = design_matrix(b)
    @test size(Φ) == (20, 4)
    @test all(0 .≤ Φ .≤ 1.0)

    θ = randn(MersenneTwister(0), 4)
    s = schedule_from_theta(b, θ; Φ = Φ)
    @test length(s) == 20
    # IdentityOutput → schedule equals Φ * θ
    @test isapprox(s, Φ * θ; atol = 1e-12)

    # SoftplusOutput pushes everything ≥ 0
    b_sp = RBFBasis(20, 0.1, 4; output = SoftplusOutput())
    s_sp = schedule_from_theta(b_sp, θ)
    @test all(s_sp .≥ 0)

    # SigmoidOutput maps to (0, 1)
    b_sg = RBFBasis(20, 0.1, 4; output = SigmoidOutput())
    s_sg = schedule_from_theta(b_sg, θ)
    @test all(0 .< s_sg .< 1)
end

@testset "Control — calibrate_beta_max on a quadratic cost" begin
    cost(θ) = 0.5 * sum(abs2, θ)
    β_max, c_mean, c_std = calibrate_beta_max(
        cost; theta_dim = 4, sigma_prior = 1.0,
        n_samples = 256, target_nats = 4.0, seed = 0,
    )
    @test β_max > 0
    # For a quadratic cost on N(0,1)^4, mean cost = ½ E[‖θ‖²] = 2.
    @test isapprox(c_mean, 2.0; atol = 0.5)
    # β_max = target_nats / std should hold.
    @test isapprox(β_max, 4.0 / c_std; rtol = 1e-6)
end

@testset "Control — build_crn_noise_grids returns expected shapes" begin
    g = build_crn_noise_grids(; n_inner = 8, n_steps = 12, n_channels = 2, seed = 0)
    @test size(g[:wiener])  == (8, 12, 2)
    @test length(g[:initial]) == 8
end
