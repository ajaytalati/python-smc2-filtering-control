using Test
using Smc2fcGPU
using StaticArrays
using Random
using CUDA

@testset "Smc2fcGPU" begin
    @testset "FSAv2 dynamics" begin
        p = FSAv2_DEFAULT_PARAMS(Float64)
        # at A=A_TYP, F=F_TYP, residual factors collapse to 1
        y = SVector{3,Float64}(0.05, 0.20, 0.10)
        Phi_t = 1.0
        dy = Smc2fcGPU.drift(y, p, Phi_t)
        # dB = kappa_B*1*Phi - B/tau_B = 0.01248 - 0.05/42 = 0.011289…
        @test isapprox(dy[1], 0.01248 - 0.05 / 42.0; atol=1e-9)
        # dF = kappa_F*Phi - 1/tau_F * F = 0.030 - 0.20/(7/1.1) = 0.030 - 0.0314…
        @test isapprox(dy[2], 0.030 - 0.20 / (7.0 / 1.1); atol=1e-9)
        # noise scale at this point
        ns = Smc2fcGPU.noise_scale(y, p)
        @test isapprox(ns[1], 0.010 * sqrt(0.05 * 0.95); atol=1e-12)
        @test isapprox(ns[2], 0.012 * sqrt(0.20); atol=1e-12)
        @test isapprox(ns[3], 0.020 * sqrt(0.10 + 1e-4); atol=1e-12)
    end

    @testset "Phi-burst expansion" begin
        # Constant Phi=1.0 over 1 day → daily-integrated total = 24*Phi*dt_hours
        # times the normalisation, which should integrate back to 24*1 within
        # numerical precision, since the Gamma envelope is normalised to ∫=24*Phi.
        # (Implementation note: simulation.py:generate_phi_sub_daily defines
        # amplitude = phi_d * 24 / gamma_int, and integrates over wake hours
        # with bin width dt_hours.)
        lut = expand_phi_lut([1.0]; bins_per_day=96)
        @test length(lut) == 96
        # Sleep bins are zero
        @test lut[1] == 0.0f0
        @test lut[end] == 0.0f0
        # Wake bins are non-negative
        @test all(lut .>= 0.0f0)
        # Check a known peak: at t = tau_hours = 3 h post-wake, shape = tau*exp(-1)
        # bin index = (7+3)*4 = 40 (0-based) → entry 41 (1-based)
        @test lut[41] > 0.0f0
    end

    @testset "GPU ensemble (M1b)" begin
        if CUDA.functional()
            include("test_m1b_ensemble.jl")
        else
            @info "CUDA.functional() = false — skipping M1b GPU ensemble test"
        end
    end

    @testset "EM oracle determinism" begin
        # With zero noise, EM matches deterministic Euler exactly.
        p = FSAv2_DEFAULT_PARAMS(Float64)
        y0 = SVector{3,Float64}(0.05, 0.30, 0.10)
        n_grid = 96
        n_substeps = 10
        dt_grid = 1.0 / 96.0
        t_grid = collect(range(0.0; step=dt_grid, length=n_grid + 1))
        Phi_arr = ones(Float64, n_grid)
        noise = zeros(Float64, n_substeps, n_grid, 3)
        traj = Smc2fcGPU.em_oracle_single(y0, p, t_grid, Phi_arr, noise;
                                           n_substeps=n_substeps)
        @test size(traj) == (n_grid + 1, 3)
        @test all(isfinite, traj)
        # B should drift toward kappa_B*tau_B*Phi = 0.01248 * 42 ≈ 0.524
        # (long-run equilibrium); over 1 day from B=0.05 it should rise but not
        # overshoot.
        @test traj[end, 1] > 0.05
        @test traj[end, 1] < 0.524
        # F equilibrium = kappa_F * tau_F = 0.030 * 6.36 ≈ 0.191; from F=0.30
        # it should decay toward that.
        @test traj[end, 2] < 0.30
        @test traj[end, 2] > 0.18
    end
end
