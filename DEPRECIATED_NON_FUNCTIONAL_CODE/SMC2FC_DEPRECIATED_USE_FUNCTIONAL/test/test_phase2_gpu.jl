# GPU smoke tests for Phase 2 — charter §15.7 mandates that every test
# runs once on Array (CPU) and once on CUDA.CuArray (GPU) within tolerance.
#
# These tests cover the GPU-portable subset of Phase 2:
#   - Kernels.compute_ess
#   - Kernels.silverman_bandwidth
#   - Kernels.log_kernel_matrix              (rewritten as 3-axis broadcast)
#   - Kernels.smooth_resample variants       (matmul + broadcast)
#   - OT.compute_kernel_factor               (rewritten as 3-axis broadcast)
#   - OT.factor_matvec, factor_matvec_batch  (just matmul)
#   - OT.barycentric_projection              (matmul + broadcast)
#
# The full bootstrap PF on GPU requires a GPU-compatible toy model
# (`propagate_fn` etc. that dispatch on CuArray) — that lands in Phase 6
# end-to-end.

using CUDA: CuArray, CUDA

@testset "Phase 2 — GPU parity (CUDA-resident)" begin
    if !CUDA.functional()
        @info "CUDA not functional — skipping GPU parity tests"
        return
    end

    rng = MersenneTwister(2026_05_06)

    @testset "compute_ess: CPU == GPU" begin
        log_w_cpu = randn(rng, 100)
        log_w_gpu = CuArray(log_w_cpu)
        e_cpu = compute_ess(log_w_cpu)
        e_gpu = compute_ess(log_w_gpu)
        @test isapprox(e_cpu, e_gpu; atol=1e-4)   # charter §15.7 tolerance
    end

    @testset "log_kernel_matrix: CPU == GPU" begin
        K, n_s = 32, 3
        particles_cpu = randn(rng, K, n_s)
        particles_gpu = CuArray(particles_cpu)
        sto_idx = [1, 2, 3]

        h_cpu = silverman_bandwidth(particles_cpu, sto_idx, K, 1.0)
        h_gpu = silverman_bandwidth(particles_gpu, sto_idx, K, 1.0)
        @test maximum(abs.(h_cpu .- Array(h_gpu))) < 1e-4

        L_cpu = log_kernel_matrix(particles_cpu, sto_idx, h_cpu)
        L_gpu = log_kernel_matrix(particles_gpu, sto_idx, h_gpu)
        @test maximum(abs.(L_cpu .- Array(L_gpu))) < 1e-4
    end

    @testset "compute_kernel_factor: CPU == GPU" begin
        N, d, r = 64, 3, 8
        x_cpu = rand(rng, N, d)
        x_gpu = CuArray(x_cpu)
        anchor_idx = collect(1:r)

        KNR_cpu = compute_kernel_factor(x_cpu, anchor_idx, 0.5)
        KNR_gpu = compute_kernel_factor(x_gpu, anchor_idx, 0.5)
        @test maximum(abs.(KNR_cpu .- Array(KNR_gpu))) < 1e-4
    end

    @testset "Sinkhorn + barycentric: CPU == GPU" begin
        N, d, r = 64, 3, 8
        x_cpu = rand(rng, N, d)
        x_gpu = CuArray(x_cpu)
        anchor_idx = collect(1:r)
        a_cpu = fill(1/N, N); a_gpu = CuArray(a_cpu)
        b_cpu = a_cpu;        b_gpu = CuArray(b_cpu)

        KNR_cpu = compute_kernel_factor(x_cpu, anchor_idx, 0.5)
        KNR_gpu = compute_kernel_factor(x_gpu, anchor_idx, 0.5)

        u_cpu, v_cpu = sinkhorn_scalings(a_cpu, b_cpu, KNR_cpu; n_iter=10)
        u_gpu, v_gpu = sinkhorn_scalings(a_gpu, b_gpu, KNR_gpu; n_iter=10)
        @test maximum(abs.(u_cpu .- Array(u_gpu))) < 1e-4
        @test maximum(abs.(v_cpu .- Array(v_gpu))) < 1e-4

        nx_cpu = barycentric_projection(u_cpu, v_cpu, x_cpu, KNR_cpu)
        nx_gpu = barycentric_projection(u_gpu, v_gpu, x_gpu, KNR_gpu)
        @test maximum(abs.(nx_cpu .- Array(nx_gpu))) < 1e-4
    end

    @testset "smooth_resample (Liu-West path): CPU == GPU" begin
        K, n_s = 32, 2
        particles_cpu = randn(rng, K, n_s)
        log_w_cpu = randn(rng, K)
        sto_idx = [1, 2]

        out_cpu = smooth_resample(particles_cpu, log_w_cpu, sto_idx, K, 1.0)

        particles_gpu = CuArray(particles_cpu)
        log_w_gpu     = CuArray(log_w_cpu)
        out_gpu = smooth_resample(particles_gpu, log_w_gpu, sto_idx, K, 1.0)

        @test maximum(abs.(out_cpu .- Array(out_gpu))) < 1e-4
    end
end
