#!/usr/bin/env julia
# v5 posterior parameter-trace plot — 37 panels, 5/95 quantile band +
# median + truth horizontal line, one panel per estimated parameter.
#
# Mirrors `_plot_param_traces_v15` from v1.5's `bench_postproc.jl` but
# scaled up to v5's 37 estimated parameters (vs v1.5's 10). Layout:
# 5 columns × 8 rows = 40 cells (3 left blank).
#
# Reads a v5 `data.jld2` written by `tools_v5/bench/bench_postproc.jl`.
# Required keys: `posterior_particles` (n_strides × n_smc × 37 in
# constrained space), `posterior_window_mask` (Bool vector per stride),
# `param_names` (37 strings), `truth_params_dict` (Dict{String,Float64}
# with all 50 keys, of which 37 match the estimated names).
# `STRIDE_BINS`, `WINDOW_BINS`, `BINS_PER_DAY` are used to label the
# x-axis (end-of-window in days).
#
# Usage:
#   julia --project=. tools_v5/plot_param_traces_v5.jl <data.jld2> <out.png>

using JLD2
using Plots
using Statistics


function load_run_data(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


"""
    plot_param_traces_v5(data; out_path)

Render the per-parameter posterior trace across rolling windows.
Each panel: 5/95% quantile band + median + truth horizontal line.

`data` is the Dict returned by `load_run_data` (or the in-memory
equivalent that the bench's `save_bench_outputs` wrote).
"""
function plot_param_traces_v5(data::Dict; out_path::AbstractString,
                                title_prefix::AbstractString = "FSA-v5 posterior parameter traces")
    posterior  = data["posterior_particles"]            # (n_strides, n_smc, n_params)
    mask       = collect(Bool, data["posterior_window_mask"])
    param_names = String.(data["param_names"])
    truth      = data["truth_params_dict"]               # Dict{String, Float64}

    BINS_PER_DAY = data["BINS_PER_DAY"]
    STRIDE_BINS  = data["STRIDE_BINS"]
    WINDOW_BINS  = data["WINDOW_BINS"]

    n_strides = size(posterior, 1)
    end_t_days = ((collect(0:n_strides - 1) .* STRIDE_BINS) .+ WINDOW_BINS) ./ BINS_PER_DAY
    end_t_days = end_t_days[mask]

    valid = posterior[mask, :, :]
    n_valid, n_smc, n_params = size(valid)

    # Per-stride 5/50/95 quantiles per parameter.
    q05 = [quantile(vec(valid[s, :, p]), 0.05) for s in 1:n_valid, p in 1:n_params]
    q50 = [quantile(vec(valid[s, :, p]), 0.50) for s in 1:n_valid, p in 1:n_params]
    q95 = [quantile(vec(valid[s, :, p]), 0.95) for s in 1:n_valid, p in 1:n_params]

    # 5 cols × 8 rows = 40 cells; 37 params + 3 blank fillers.
    n_cols = 5
    n_rows = (n_params + n_cols - 1) ÷ n_cols

    panels = Plots.Plot[]
    for i in 1:n_params
        name = param_names[i]
        p = plot(end_t_days, q50[:, i];
                  ribbon = (q50[:, i] .- q05[:, i], q95[:, i] .- q50[:, i]),
                  fillalpha = 0.30, color = :steelblue, lw = 1.4,
                  label = "median", title = name, titlefontsize = 9,
                  legend = false, grid = true, gridalpha = 0.3,
                  tickfontsize = 7,
                  xlabel = i > n_params - n_cols ? "end of window (days)" : "",
                  xguidefontsize = 7)
        if haskey(truth, name)
            hline!(p, [truth[name]]; color = :red, ls = :dash, lw = 1.0,
                    label = "truth")
        end
        push!(panels, p)
    end
    # Pad the grid to n_rows·n_cols with empty placeholder panels.
    for _ in n_params + 1:n_rows * n_cols
        push!(panels, plot(framestyle = :none, ticks = false, legend = false))
    end

    fig = plot(panels...; layout = (n_rows, n_cols),
                size = (n_cols * 320, n_rows * 220), dpi = 120,
                plot_title = "$title_prefix — n_strides=$n_strides " *
                              "(valid=$(n_valid))",
                plot_titlefontsize = 10)
    savefig(fig, out_path)
    return out_path
end


function main()
    if length(ARGS) < 2
        println("Usage: julia plot_param_traces_v5.jl <data.jld2> <out.png>")
        exit(1)
    end
    data = load_run_data(ARGS[1])
    plot_param_traces_v5(data; out_path = ARGS[2])
    @info "wrote: $(ARGS[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
