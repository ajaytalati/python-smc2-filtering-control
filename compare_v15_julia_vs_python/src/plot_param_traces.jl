#!/usr/bin/env julia
# 30-panel parameter-trace plot — matches Python's
# `version_2/outputs/.../E5_full_mpc_T14d_param_traces.png` format:
#   - light-blue shaded 5–95 percentile envelope
#   - blue mean line
#   - red dashed truth line
#   - x-axis: end-of-window in days
#   - title: "FSA-v2 posterior parameter traces — T=Xd, h=Ymin, n_strides=Z, ..."

using JLD2
using Plots
using Statistics

function load_run_data(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end

function plot_param_traces(data::Dict; out_path::AbstractString,
                            T_total_days::Real = 14.0,
                            step_minutes::Int = 60,
                            stride_bins::Int = 12)
    pp = haskey(data, "posterior_particles") ?
         data["posterior_particles"] : data["particles_per_stride"]
    truth_params = data["truth_params"]
    param_names  = data["param_names"]
    n_strides    = data["n_strides"]
    n_params     = length(param_names)

    means = zeros(n_strides, n_params)
    p05   = zeros(n_strides, n_params)
    p95   = zeros(n_strides, n_params)
    if pp isa AbstractArray{<:Real,3}
        @inbounds for s in 1:n_strides, j in 1:n_params
            col = view(pp, s, :, j)
            means[s, j] = mean(col)
            p05[s, j]   = quantile(col, 0.05)
            p95[s, j]   = quantile(col, 0.95)
        end
    else
        @inbounds for s in 1:n_strides
            ps = pp[s]
            for j in 1:n_params
                col = view(ps, :, j)
                means[s, j] = mean(col)
                p05[s, j]   = quantile(col, 0.05)
                p95[s, j]   = quantile(col, 0.95)
            end
        end
    end

    # x = end-of-window in days
    dt_days = step_minutes / (60 * 24)
    x = [(s - 1) * stride_bins * dt_days + 24 * dt_days for s in 1:n_strides]

    # Match Python colours
    blue_line = RGB(0x1f/255, 0x77/255, 0xb4/255)
    blue_fill = RGB(0xae/255, 0xc7/255, 0xe8/255)
    red_truth = RGB(0xd6/255, 0x27/255, 0x28/255)

    n_cols = 5
    n_rows = ceil(Int, n_params / n_cols)
    plt = plot(layout=(n_rows, n_cols),
                size=(1500, n_rows * 200),
                legend=false, dpi=120,
                fontfamily="Helvetica",
                framestyle=:box,
                grid=true, gridalpha=0.3,
                plot_title="FSA-v2 posterior parameter traces — " *
                           "T=$(Int(T_total_days))d, h=$(step_minutes)min, " *
                           "n_strides=$n_strides",
                plot_titlefontsize=11)

    @inbounds for j in 1:n_params
        plot!(plt[j], x, p95[:, j]; fillrange=p05[:, j],
              fillalpha=0.40, linealpha=0.0, color=blue_fill)
        plot!(plt[j], x, means[:, j]; color=blue_line, linewidth=1.4)
        hline!(plt[j], [truth_params[j]]; color=red_truth, linestyle=:dash, linewidth=1.0)
        title!(plt[j], string(param_names[j]); titlefontsize=9)
        xlims!(plt[j], 0.0, T_total_days)
        # Only put x-label on bottom row, but tick labels on every panel.
        if j > n_params - n_cols
            xlabel!(plt[j], "time (days)"; labelfontsize=9)
        end
        plot!(plt[j]; xticks=0:2:Int(T_total_days),
              xtickfontsize=8, ytickfontsize=7)
    end

    savefig(plt, out_path)
    return out_path
end


function main()
    if length(ARGS) < 2
        println("Usage: julia plot_param_traces.jl <data.jld2> <out.png> [T_days] [step_minutes] [stride_bins]")
        exit(1)
    end
    T_days = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 14.0
    step_min = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 60
    stride_bins = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12
    data = load_run_data(ARGS[1])
    plot_param_traces(data; out_path=ARGS[2],
                      T_total_days=T_days,
                      step_minutes=step_min,
                      stride_bins=stride_bins)
    @info "wrote: $(ARGS[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
