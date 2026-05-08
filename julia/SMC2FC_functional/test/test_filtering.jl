"""
Tests for `Filtering/Kernels.jl`, `Filtering/OT.jl`, and the
`BootstrapWorkspace` constructor in `Filtering/Bootstrap.jl`.

These exercise the **functional API surface** — every public function is
called with immutable inputs and the result is captured. Internal
buffer mutation is invisible to these tests.
"""

using Test
using Random
using SMC2FC_functional
using LinearAlgebra: norm

@testset "Filtering — compute_ess on uniform and degenerate weights" begin
    K = 10
    log_w_uniform = zeros(K)
    @test compute_ess(log_w_uniform) ≈ K atol=1e-10

    # All weight on one particle ⇒ ESS = 1
    log_w_dirac = fill(-1e6, K)
    log_w_dirac[1] = 0.0
    @test compute_ess(log_w_dirac) ≈ 1.0 atol=1e-3
end

@testset "Filtering — silverman_bandwidth and log_kernel_matrix shapes" begin
    rng = MersenneTwister(0)
    K, n_st = 32, 2
    particles = randn(rng, K, n_st)
    sto_idx   = collect(1:n_st)
    h = silverman_bandwidth(particles, sto_idx, K, 1.0)
    @test length(h) == n_st
    @test all(h .> 0)

    L = log_kernel_matrix(particles, sto_idx, h)
    @test size(L) == (K, K)
    # Diagonal is 0 (a particle has zero distance to itself).
    @test all(abs.(L[i, i] for i in 1:K) .< 1e-10)
end

@testset "Filtering — smooth_resample preserves array shape" begin
    rng = MersenneTwister(0)
    K, n_st = 16, 2
    particles = randn(rng, K, n_st)
    log_w = randn(rng, K)
    out = smooth_resample(particles, log_w, collect(1:n_st), K, 1.0)
    @test size(out) == size(particles)
end

@testset "OT — Sinkhorn scalings make the right-marginal match" begin
    rng = MersenneTwister(0)
    K, d, rank = 16, 2, 8
    x = randn(rng, K, d)
    anchor_idx = collect(1:rank)

    K_NR = compute_kernel_factor(x, anchor_idx, 1.0)
    a = fill(1.0/K, K)
    b = abs.(randn(rng, K)); b ./= sum(b)
    u, v = sinkhorn_scalings(a, b, K_NR; n_iter=50)
    # Sinkhorn fixed point: v_i · (K · u)_i = b_i (col-sum identity).
    Ku = K_NR * (K_NR' * u)
    @test isapprox(v .* Ku, b; atol=2e-2)
end

@testset "OT — ot_resample_lr returns same shape and respects deterministic dims" begin
    rng = MersenneTwister(0)
    K, n_states = 32, 3
    particles = randn(rng, K, n_states)
    log_w = randn(rng, K)
    sto_idx = [1, 2]    # state 3 is deterministic
    new_parts = ot_resample_lr(particles, log_w, rng, sto_idx; rank=8, n_iter=5)
    @test size(new_parts) == size(particles)
    # Deterministic component should be unchanged
    @test new_parts[:, 3] == particles[:, 3]
end

@testset "Filtering — BootstrapWorkspace constructor" begin
    K, n_states = 16, 3
    ws = BootstrapWorkspace{Float64}(K, n_states)
    @test ws.K == K
    @test ws.n_states == n_states
    @test size(ws.particles) == (K, n_states)
    @test size(ws.log_w) == (K,)
    # The struct itself must be immutable — re-binding K should fail.
    @test_throws ErrorException ws.K = 17
end
