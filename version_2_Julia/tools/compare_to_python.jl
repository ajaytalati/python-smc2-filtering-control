#!/usr/bin/env julia
# Quantitative gate: Julia vs Python posterior means on the 6 identifiable
# params (≤ 5% relative error required).
#
# Reads:
#   - Python data: version_2/outputs/.../T14d_replanK2_h60min_no_infoaware/data.npz
#   - Julia data:  version_2_Julia/outputs/.../T14d_replanK2_h60min_no_infoaware/data.jld2
#
# Reports per-param relative error over the final 5 windows. Pass = ≤ 5%
# on the six ID-strong params (HR_base, S_base, β_C_HR, k_F, κ_B_HR, μ_step0).
#
# Usage:
#   julia --project=. tools/compare_to_python.jl <python_data.npz> <julia_data.jld2>

using NPZ
using JLD2
using Statistics
using Printf


const IDENTIFIABLE_PARAMS = [
    "HR_base", "S_base", "beta_C_HR", "k_F", "kappa_B_HR", "mu_step0",
]


function compare_runs(python_path::AbstractString, julia_path::AbstractString;
                       last_n_strides::Int = 5)
    py_data = NPZ.npzread(python_path)
    jl_data = JLD2.jldopen(julia_path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end

    py_particles = py_data["particles_per_stride"]   # shape: (n_strides, N_SMC, n_params)
    jl_particles = jl_data["particles_per_stride"]   # Vector{Matrix} per stride
    py_param_names = String.(py_data["param_names"])
    jl_param_names = String.(jl_data["param_names"])

    @assert py_param_names == jl_param_names "Param-name vectors disagree"
    n_strides = length(jl_particles)
    @assert size(py_particles, 1) == n_strides "Stride counts differ"

    n_params = length(jl_param_names)

    py_means_final = zeros(Float64, n_params)
    jl_means_final = zeros(Float64, n_params)
    @inbounds for j in 1:n_params
        py_window = py_particles[end - last_n_strides + 1 : end, :, j]
        py_means_final[j] = mean(py_window)
        jl_window = vcat([jl_particles[s][:, j]
                          for s in n_strides - last_n_strides + 1 : n_strides]...)
        jl_means_final[j] = mean(jl_window)
    end

    println("Param            Python mean    Julia mean    Rel. err")
    println("-" ^ 66)
    fail_count = 0
    for j in 1:n_params
        rel_err = abs(jl_means_final[j] - py_means_final[j]) /
                  max(abs(py_means_final[j]), 1e-12)
        marker = " "
        if jl_param_names[j] in IDENTIFIABLE_PARAMS
            marker = rel_err <= 0.05 ? "✓" : "✗"
            rel_err > 0.05 && (fail_count += 1)
        end
        @printf("%-15s  %12.4f  %12.4f  %8.4f%%  %s\n",
                jl_param_names[j], py_means_final[j], jl_means_final[j],
                100 * rel_err, marker)
    end
    println()
    println(fail_count == 0 ?
            "PASS: all 6 identifiable params within 5%." :
            "FAIL: $fail_count of 6 identifiable params exceed 5%.")
    return fail_count == 0
end


function main()
    if length(ARGS) < 2
        println("Usage: julia compare_to_python.jl <python_data.npz> <julia_data.jld2>")
        exit(1)
    end
    ok = compare_runs(ARGS[1], ARGS[2])
    exit(ok ? 0 : 1)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
