# M1b — GPU ensemble vs Julia CPU oracle aggregate parity.
#
# The CPU oracle (M1) has fp64 byte-parity vs JAX/diffrax `solve_sde_jax`
# when fed the same noise tensor. This test runs an ensemble of K
# trajectories on both paths with independent fresh seeds and checks
# that ensemble means/stds at the end of the window agree within MC
# noise (Monte Carlo error scales as σ/√K).
#
# Tolerance for K=16384:
#   mean diff per state ≤ 8 × σ_traj × sqrt(2/K)
#     (two independent ensembles drawn from the same distribution have
#      empirical-mean difference std σ × sqrt(2/K); 8× is the per-state
#      coverage budget covering all 3 states at ~5% combined false-positive)
#   std  diff per state ≤ 10% relative
# Two complications make per-trajectory parity impossible:
#   (a) GPU uses Random123 Philox per-thread RNG; CPU uses MersenneTwister.
#   (b) GPU is fp32; CPU is fp64. Multiplicative-noise round-off accumulates
#       over 960 substeps but the resulting bias is well below MC noise.
using Test
using Smc2fcGPU
using StaticArrays
using CUDA
using Random
using Statistics
using Base.Threads

function julia_cpu_ensemble(y0, p, Phi_arr, t_grid;
                              n_trajectories::Int=4_096,
                              n_substeps::Int=10,
                              base_seed::Int=42)
    n_t = length(t_grid)
    out = Array{Float64, 3}(undef, n_t, 3, n_trajectories)
    Threads.@threads for i in 1:n_trajectories
        rng = MersenneTwister(base_seed + i)
        n_grid = n_t - 1
        noise = randn(rng, n_substeps, n_grid, 3)
        traj = Smc2fcGPU.em_oracle_single(y0, p, t_grid, Phi_arr, noise;
                                            n_substeps=n_substeps)
        @inbounds for k in 1:n_t, j in 1:3
            out[k, j, i] = traj[k, j]
        end
    end
    out
end

@testset "M1b ensemble parity (Julia GPU vs Julia CPU oracle)" begin
    K = 16_384
    p = FSAv2_DEFAULT_PARAMS(Float64)
    y0 = SVector{3, Float64}(0.05, 0.30, 0.10)
    n_grid = 96
    dt = 1.0 / 96.0
    t_grid = collect(range(0.0; step=dt, length=n_grid + 1))
    Phi_arr = Float64.(expand_phi_lut([1.0]))

    @info "Running CPU oracle ensemble" K=K threads=Threads.nthreads()
    cpu_out = julia_cpu_ensemble(y0, p, Phi_arr, t_grid;
                                  n_trajectories=K, n_substeps=10, base_seed=42)

    # Run GPU ensemble in Float32 to match what the production kernel will
    # actually use in M2+; we compare aggregate stats only.
    p32 = FSAv2_DEFAULT_PARAMS(Float32)
    y0_32 = SVector{3, Float32}(0.05f0, 0.30f0, 0.10f0)
    t32 = Float32.(t_grid)
    Phi32 = Float32.(Phi_arr)

    @info "Running GPU ensemble"
    gpu_out = run_em_ensemble_gpu(y0_32, p32, Phi32, t32;
                                    n_trajectories=K, n_substeps=10, seed=42)
    CUDA.synchronize()

    # End-of-trajectory aggregate stats on each state
    cpu_end_means = [mean(cpu_out[end, j, :]) for j in 1:3]
    gpu_end_means = [Float64(mean(gpu_out[end, j, :])) for j in 1:3]
    cpu_end_stds  = [std(cpu_out[end, j, :])  for j in 1:3]
    gpu_end_stds  = [Float64(std(gpu_out[end, j, :]))  for j in 1:3]

    println("CPU end means: ", cpu_end_means)
    println("GPU end means: ", gpu_end_means)
    println("CPU end stds:  ", cpu_end_stds)
    println("GPU end stds:  ", gpu_end_stds)

    # MC tolerance for two independent ensembles, same SDE distribution:
    #   E[(mean_GPU - mean_CPU)^2] = 2 σ²_traj / K
    #   so std(diff) = σ_traj √(2/K). 8× is the coverage budget.
    mc_tol_mean = [8 * cpu_end_stds[j] * sqrt(2 / K) for j in 1:3]
    println("MC tolerance per state: ", mc_tol_mean)
    for j in 1:3
        @test abs(gpu_end_means[j] - cpu_end_means[j]) < mc_tol_mean[j]
    end
    # 10 % relative tolerance on stds
    for j in 1:3
        @test abs(gpu_end_stds[j] - cpu_end_stds[j]) / cpu_end_stds[j] < 0.10
    end
end
