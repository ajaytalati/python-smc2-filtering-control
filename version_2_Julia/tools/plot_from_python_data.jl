#!/usr/bin/env julia
# Plot the 30-panel param-trace figure by reading the Python `data.npz`
# directly. Used as a sanity check that the Julia plotter is correct
# BEFORE we have Julia-side SMC² output.
#
# Output is labelled "Julia plotter — Python posterior data" so it
# cannot be confused with a Julia-run reproduction.
#
# Usage:
#   julia --project=. tools/plot_from_python_data.jl \
#       <python_data.npz> <out.png>

using NPZ
using Plots
using Statistics

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Dynamics: TRUTH_PARAMS
using .FSAHighRes.Estimation: PARAM_NAMES


function main(argv::Vector{String})
    if length(argv) < 2
        println("Usage: julia plot_from_python_data.jl <python_data.npz> <out.png>")
        exit(1)
    end
    d = NPZ.npzread(argv[1])

    # posterior_particles :: (n_strides, N_SMC, n_params)
    pp = d["posterior_particles"]
    n_strides, N_SMC, n_params = size(pp)
    @info "loaded posterior_particles: shape=$((n_strides, N_SMC, n_params))"

    @assert n_params == length(PARAM_NAMES) "param count mismatch"

    # Truth values (from the canonical TRUTH_PARAMS).
    truth = Float64[
        TRUTH_PARAMS.tau_B, TRUTH_PARAMS.tau_F, TRUTH_PARAMS.kappa_B,
        TRUTH_PARAMS.kappa_F, TRUTH_PARAMS.epsilon_A, TRUTH_PARAMS.lambda_A,
        TRUTH_PARAMS.mu_0, TRUTH_PARAMS.mu_B, TRUTH_PARAMS.mu_F,
        TRUTH_PARAMS.mu_FF, TRUTH_PARAMS.eta,
        # HR
        62.0, 12.0, 3.0, -2.5, 2.0,
        # Sleep
        3.0, 2.0, 0.5,
        # Stress
        30.0, 20.0, 8.0, -4.0, 4.0,
        # Steps
        5.5, 0.8, 0.5, 0.3, -0.8, 0.5,
    ]
    @assert length(truth) == n_params

    # Compute per-stride mean / p05 / p95.
    means = Matrix{Float64}(undef, n_strides, n_params)
    p05   = Matrix{Float64}(undef, n_strides, n_params)
    p95   = Matrix{Float64}(undef, n_strides, n_params)
    @inbounds for s in 1:n_strides, j in 1:n_params
        col = view(pp, s, :, j)
        means[s, j] = mean(col)
        p05[s, j]   = quantile(col, 0.05)
        p95[s, j]   = quantile(col, 0.95)
    end

    # 6×5 grid of 30 panels.
    n_cols = 5
    n_rows = ceil(Int, n_params / n_cols)
    plt = plot(layout=(n_rows, n_cols),
                size=(1500, n_rows * 200),
                legend=false, dpi=120,
                plot_title="Julia plotter — Python posterior data " *
                           "(version_2/outputs/.../data.npz)")
    x = collect(0:n_strides - 1)
    @inbounds for j in 1:n_params
        plot!(plt[j], x, p95[:, j]; fillrange=p05[:, j],
              fillalpha=0.30, linealpha=0.0, color=:blue)
        plot!(plt[j], x, means[:, j]; color=:magenta, linewidth=1.5)
        hline!(plt[j], [truth[j]]; color=:cyan, linestyle=:dash,
               linewidth=1.0)
        title!(plt[j], string(PARAM_NAMES[j]); titlefontsize=8)
    end

    savefig(plt, argv[2])
    @info "wrote: $(argv[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
