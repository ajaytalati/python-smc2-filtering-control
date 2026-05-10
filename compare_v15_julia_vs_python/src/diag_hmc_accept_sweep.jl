#!/usr/bin/env julia
# diag_hmc_accept_sweep.jl
#
# Sweep HMC step size, leapfrog count, FD-gradient step, and num_mcmc
# moves to find a config that produces non-zero acceptance — the bench
# is currently logging 0% accept across every tempering level of every
# baseline run.
#
# Setup mirrors `version_1_5_Julia/tools/test_gpu_pf.jl`: build a 1-day
# truth-Φ=1 obs window, place M chains at perturbed truth, and run
# `parallel_hmc_one_move` with each parameter combination. Per cell we
# also call `gpu_log_density` directly at the *initial* and *post-move*
# U so we can report the (proposed - current) log-density and the
# acceptance rate side by side.
#
# This is a CHEAP experiment: M=128 chains, K=200 state particles,
# T=24 bins, R=4 segments. Total runtime ~30 s on RTX 5090.
#
# Usage:
#   cd version_1_5_Julia
#   julia --project=. ../compare_v15_julia_vs_python/src/diag_hmc_accept_sweep.jl
#
# Output: TSV-like table to stdout + CSV at
#   compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv

ENV["FSA_STEP_MINUTES"] = "60"

using Printf
using Random
using Statistics
using LinearAlgebra
using StaticArrays
using StableRNGs: StableRNG

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", "version_1_5_Julia"))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: PlantState, plant_rollout, init_plant_state
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE
using .FSAHighRes.Estimation: PARAM_NAMES, PARAM_PRIOR_CONFIG
using .FSAHighRes.GPUPF: FSAGPUTarget, gpu_log_density, gpu_grads,
                          parallel_hmc_one_move


function build_truth_window(; n_bins::Int = BINS_PER_DAY,
                              Φ_const::Float64 = 1.0,
                              key::UInt64 = UInt64(0xC0FFEE))
    s0 = init_plant_state()
    Φ_seq = fill(Float32(Φ_const), n_bins)
    out = plant_rollout(s0, Φ_seq, DEFAULT_PARAMS, DT_BIN_DAYS, key)
    return (
        Phi_seq = out.Phi,
        obs_B   = out.obs_B,
        obs_F   = out.obs_F,
        obs_A   = out.obs_A,
        B_init  = INIT_STATE.B,
        F_init  = INIT_STATE.F,
        A_init  = INIT_STATE.A,
    )
end


truth_u_unc() = [log(DEFAULT_PARAMS[name]) for name in PARAM_NAMES]


function run_sweep()
    K_per_chain = 200
    M           = 128
    d_unc       = 10
    M_max       = M * (1 + 2 * d_unc)   # = 2688 for M=128, d=10
    T_steps     = BINS_PER_DAY
    R           = 4

    println("Building target: K=$K_per_chain M=$M M_max=$M_max T=$T_steps R=$R")
    target = FSAGPUTarget(
        K_per_chain = K_per_chain,
        M_max       = M_max,
        T_steps     = T_steps,
        R           = R,
        dt          = DT_BIN_DAYS,
        noise_seed  = 42,
    )

    grid_obs = build_truth_window(; n_bins = T_steps, Φ_const = 1.0,
                                    key = UInt64(0xC0FFEE))

    # Spread M chains as iid draws from the prior centred at truth.
    u_truth = truth_u_unc()
    rng_init = MersenneTwister(0xCAFE)
    prior_sigmas = [s for (_, _, _, s) in PARAM_PRIOR_CONFIG]
    U0 = repeat(reshape(u_truth, 1, :), M, 1) .+
         (reshape(prior_sigmas, 1, :) .* randn(rng_init, M, d_unc))

    prior_means = [m for (_, _, m, _) in PARAM_PRIOR_CONFIG]

    # Sweep grid
    step_sizes  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    leapfrogs   = [1, 4, 16]
    h_fds       = [1e-2, 1e-3, 1e-4]
    sweep_seed  = UInt64(0x42)

    out_csv = joinpath(@__DIR__, "..", "example_run",
                        "julia_horizon_sweep_2026-05-09",
                        "diag_hmc_accept_sweep.csv")

    open(out_csv, "w") do io
        println(io, "step_size,leapfrog,h_fd,n_chains,n_acc,accept_pct,mean_logα,p05_logα,p50_logα,p95_logα,mean_Δll")

        @printf("\n%9s %4s %7s %7s %9s %9s %9s %9s %9s\n",
                "ε", "L", "h_fd", "accept%", "mean log_α", "P05 log_α",
                "P50 log_α", "P95 log_α", "mean Δll")
        println("─"^96)

        for ε in step_sizes, L in leapfrogs, h_fd in h_fds
            # Compute val at U0 with the SAME RNG seed parallel_hmc_one_move
            # will use internally: hash((key, :leap, 0)).
            key = sweep_seed
            sub_init  = hash((key, :leap, 0))
            sub_final_seed = hash((key, :leap, L))

            # Run HMC and capture U_new + n_acc; then independently compute
            # log_α from val_init / val_final / momentum trajectory.
            # We replicate the leapfrog inline so we have access to
            # log-densities and momentum at start + end.
            rng = StableRNG(key)
            inv_mass_row = ones(1, d_unc)
            sqrt_inv_mass_row = ones(1, d_unc)
            momentum = randn(rng, M, d_unc) ./ sqrt_inv_mass_row
            p0 = copy(momentum)

            function tempered_grads(U_in, sub_key)
                out = gpu_grads(target, U_in, grid_obs, h_fd, sub_key)
                grads_prior = -(U_in .- prior_means') ./ (prior_sigmas' .^ 2)
                vals_prior  = -0.5 .* vec(sum(((U_in .- prior_means') ./ prior_sigmas') .^ 2; dims = 2))
                return (out.vals .+ vals_prior, out.grads .+ grads_prior)
            end

            val_init, grad_init = tempered_grads(U0, hash((key, :leap, 0)))
            U_new = copy(U0)
            p     = copy(momentum)
            @. p     = p     + (ε / 2) * grad_init
            @. U_new = U_new + ε * p * inv_mass_row
            for k in 2:L
                _, grad = tempered_grads(U_new, hash((key, :leap, k - 1)))
                @. p     = p     + ε * grad
                @. U_new = U_new + ε * p * inv_mass_row
            end
            val_final, grad_final = tempered_grads(U_new, hash((key, :leap, L)))
            @. p = p + (ε / 2) * grad_final

            K0    = 0.5 .* vec(sum(p0 .^ 2 .* inv_mass_row; dims = 2))
            K_new = 0.5 .* vec(sum(p  .^ 2 .* inv_mass_row; dims = 2))
            log_α = (val_final .- K_new) .- (val_init .- K0)
            Δll   = val_final .- val_init

            # Replace inf/nan with -1e6 for summary stats
            log_α_clean = map(x -> isfinite(x) ? x : -1e6, log_α)
            u_unif = rand(rng, M)
            n_acc = count(log.(u_unif) .< log_α)
            accept_pct = 100 * n_acc / M

            mean_logα = mean(log_α_clean)
            p05_logα  = quantile(log_α_clean, 0.05)
            p50_logα  = quantile(log_α_clean, 0.50)
            p95_logα  = quantile(log_α_clean, 0.95)
            mean_Δll  = mean(filter(isfinite, Δll))

            @printf("%9.4f %4d %7.0e %7.1f %9.2e %9.2e %9.2e %9.2e %9.2e\n",
                    ε, L, h_fd, accept_pct, mean_logα, p05_logα,
                    p50_logα, p95_logα, mean_Δll)
            @printf(io, "%.6g,%d,%.6g,%d,%d,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                    ε, L, h_fd, M, n_acc, accept_pct, mean_logα,
                    p05_logα, p50_logα, p95_logα, mean_Δll)
        end
    end
    println("\nWrote: $out_csv")
end


run_sweep()
