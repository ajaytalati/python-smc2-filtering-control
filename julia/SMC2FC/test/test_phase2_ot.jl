@testset "Phase 2 — Optimal Transport" begin
    @testset "Nyström kernel factor: shape and column sums" begin
        rng = MersenneTwister(0)
        N, d, r = 60, 3, 8
        x = rand(rng, N, d)
        anchor_idx = collect(1:r)
        K_NR = compute_kernel_factor(x, anchor_idx, 0.5)
        @test size(K_NR) == (N, r)
        # Column-normalised → each column sums to 1
        col_sums = vec(sum(K_NR; dims=1))
        @test all(c -> isapprox(c, 1.0; atol=1e-10), col_sums)
        @test all(K_NR .>= 0)
    end

    @testset "Sinkhorn marginals: convergence within tolerance" begin
        rng = MersenneTwister(1)
        N, r = 40, 6
        x = rand(rng, N, 3)
        anchor_idx = collect(1:r)
        K_NR = compute_kernel_factor(x, anchor_idx, 0.5)

        # Uniform → biased target marginals
        a = fill(1/N, N)
        log_w = randn(rng, N)
        b = exp.(log_w .- SMC2FC.Bootstrap.logsumexp(log_w))

        u, v = sinkhorn_scalings(a, b, K_NR; n_iter=30)
        # Approximate row-sum check: u_i * (K_approx * v)_i ≈ a_i
        Kv = factor_matvec(v, K_NR)
        row_marginals = u .* Kv
        @test maximum(abs.(row_marginals .- a)) < 1e-3  # low rank ⇒ slack
    end

    @testset "Barycentric projection: convex-combination preserved" begin
        rng = MersenneTwister(2)
        N, r = 50, 8
        # Particles in [0, 1]^3
        x = rand(rng, N, 3)
        anchor_idx = collect(1:r)
        K_NR = compute_kernel_factor(x, anchor_idx, 0.5)
        a = fill(1/N, N)
        b = fill(1/N, N)
        u, v = sinkhorn_scalings(a, b, K_NR; n_iter=20)
        new_x = barycentric_projection(u, v, x, K_NR)
        @test size(new_x) == size(x)
        @test all(0 .<= new_x .<= 1.0 + 1e-8)   # Convex-combination guarantee
    end

    @testset "ot_resample_lr: stochastic indices preserved correctly" begin
        rng = MersenneTwister(3)
        N = 64
        # 4 states; only states 1 and 3 are stochastic
        particles = rand(rng, N, 4)
        deterministic_value = 7.42
        particles[:, 2] .= deterministic_value
        particles[:, 4] .= 1.0
        log_weights = randn(rng, N)
        sto_idx = [1, 3]

        out = ot_resample_lr(particles, log_weights, rng, sto_idx;
                              ε=0.5, n_iter=10, rank=8)
        @test size(out) == size(particles)
        # Deterministic columns must be byte-identical
        @test out[:, 2] == particles[:, 2]
        @test out[:, 4] == particles[:, 4]
        @test all(0 .<= out[:, 1] .<= 1.0 + 1e-6)
    end

    @testset "Sigmoid blend: limits" begin
        rng = MersenneTwister(4)
        N, n_s = 32, 2
        sys = randn(rng, N, n_s)
        ot  = randn(rng, N, n_s)

        # Healthy ESS (uniform) → sigmoid(-large) → ot_weight ≈ 0
        log_w_healthy = zeros(N)
        out = ot_blended_resample(sys, ot, log_w_healthy;
                                    ot_max_weight=0.01,
                                    ot_threshold=2.0,
                                    ot_temperature=5.0)
        @test out ≈ sys  atol=2e-3   # ot_weight ≈ ot_max·sigmoid((2-32)/5) ≈ 0

        # Degenerate ESS → sigmoid(+large) → ot_weight ≈ ot_max
        log_w_degen = fill(-1e10, N); log_w_degen[1] = 0.0
        out2 = ot_blended_resample(sys, ot, log_w_degen;
                                     ot_max_weight=0.01,
                                     ot_threshold=2.0,
                                     ot_temperature=5.0)
        # ESS = 1, threshold = 2 → sigmoid(0.2) ≈ 0.55, ot_weight ≈ 0.0055
        @test out2 != sys
    end
end
