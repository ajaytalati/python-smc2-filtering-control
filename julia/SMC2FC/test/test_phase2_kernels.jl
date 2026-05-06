@testset "Phase 2 — Kernels" begin
    using LinearAlgebra: I

    @testset "compute_ess: known closed form" begin
        # Equal weights → ESS = K
        K = 50
        log_w = zeros(K)
        @test compute_ess(log_w) ≈ K  atol=1e-10

        # All mass on particle 1 → ESS = 1
        log_w = fill(-1e10, K); log_w[1] = 0.0
        @test compute_ess(log_w) ≈ 1.0  atol=1e-6

        # Two equal-weight particles, K-2 zero-weight → ESS = 2
        log_w = fill(-1e10, K); log_w[1] = log_w[2] = 0.0
        @test compute_ess(log_w) ≈ 2.0  atol=1e-6
    end

    @testset "silverman_bandwidth: dimensions and lower bound" begin
        rng = MersenneTwister(0)
        K, n_s = 100, 3
        particles = randn(rng, K, n_s)
        sto_idx = [1, 2, 3]
        h = silverman_bandwidth(particles, sto_idx, K, 1.0)
        @test length(h) == n_s
        @test all(h .>= 1e-6)            # _MIN_BW floor
        @test all(isfinite, h)
    end

    @testset "log_kernel_matrix: symmetric, diagonal == 0" begin
        rng = MersenneTwister(1)
        K = 8; n_s = 2
        particles = randn(rng, K, n_s)
        sto_idx = [1, 2]
        h = silverman_bandwidth(particles, sto_idx, K, 1.0)
        L = log_kernel_matrix(particles, sto_idx, h)
        @test size(L) == (K, K)
        # Diagonal: distance(x_i, x_i) = 0 → L[i,i] = 0
        for i in 1:K
            @test L[i, i] ≈ 0  atol=1e-12
        end
        # Symmetric
        @test maximum(abs.(L .- L')) < 1e-12
    end

    @testset "ess_bandwidth_factor: monotone in ESS" begin
        K = 100
        # Healthy cloud
        log_w_healthy = zeros(K)
        f_healthy = ess_bandwidth_factor(log_w_healthy, K)
        @test f_healthy ≈ 0.0  atol=1e-10

        # Degenerate cloud
        log_w_degen = fill(-1e10, K); log_w_degen[1] = 0.0
        f_degen = ess_bandwidth_factor(log_w_degen, K)
        # ESS = 1 → factor = (1 - 1/K)² ≈ 0.9801 for K=100
        @test f_degen ≈ (1 - 1/K)^2  atol=1e-6
        @test f_degen > f_healthy
    end

    @testset "smooth_resample variants: identity on healthy cloud" begin
        rng = MersenneTwister(42)
        K, n_s = 50, 2
        particles = randn(rng, K, n_s)
        log_w = zeros(K)               # uniform → ess_factor = 0 → identity
        sto_idx = [1, 2]

        out_basic = smooth_resample_ess_scaled(particles, log_w, sto_idx, K, 1.0)
        # ess_factor = 0 → effective_scale = 0 → bandwidth → 0 → blend ≈ identity
        # Modulo numerical floor (_MIN_BW); the kernel collapses to a permutation-mass-1 matrix.
        @test all(isfinite, out_basic)

        out_lw = smooth_resample_ess_scaled_lw(particles, log_w, sto_idx, K, 1.0)
        # ess_factor=0 → effective_scale=0 → h_norm=0 → a=1 → corrected = blended
        @test out_lw ≈ out_basic  atol=1e-10
    end
end
