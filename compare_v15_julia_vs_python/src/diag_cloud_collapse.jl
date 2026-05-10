#!/usr/bin/env julia
# diag_cloud_collapse.jl
#
# Read each `data.jld2` written by `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`
# and plot the per-stride **per-parameter cloud standard deviation** of the
# 512-particle posterior, in unconstrained (log) space.
#
# The saved `posterior_particles[s, m, j]` array is in *constrained* space
# (`exp(U)` per the bench, line 739). All 10 v1.5 priors are LogNormal, so
# we recover the unconstrained sample as `log(particle)`. The HMC moves
# operate on the unconstrained vector, so log-space std is the natural
# units for a "did the cloud collapse" check.
#
# Usage:
#   julia --project=version_1_5_Julia compare_v15_julia_vs_python/src/diag_cloud_collapse.jl \
#       --sweep-dir compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09 \
#       --out-png   <out>.png \
#       --label-prefix v15-baseline-no-LW
#
# OR pass a single `--data-file <path/to/data.jld2>` (and optional
# --T-days) for a one-off plot of a single run (used after Part B).

using JLD2
using Plots
using Statistics
using Printf

function _cloud_std_per_stride(pp::AbstractArray{<:Real, 3},
                                mask::AbstractVector{Bool})
    n_strides, n_smc, n_params = size(pp)
    out = fill(NaN, n_strides, n_params)
    @inbounds for s in 1:n_strides
        mask[s] || continue
        for j in 1:n_params
            col = view(pp, s, :, j)
            # constrained → unconstrained via log (LogNormal prior bijection)
            lcol = log.(max.(col, 1e-300))
            out[s, j] = std(lcol)
        end
    end
    return out
end


function _load_run(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


function _stride_x_days(n_strides::Int, stride_bins::Int, dt_days::Float64,
                         window_bins::Int)
    return [(s - 1) * stride_bins * dt_days + window_bins * dt_days
            for s in 1:n_strides]
end


function plot_cloud_std_one_run(data::Dict; out_path::AbstractString,
                                  T_total_days::Real,
                                  label::AbstractString = "")
    pp        = data["posterior_particles"]
    mask      = collect(Bool, data["posterior_window_mask"])
    pnames    = String.(data["param_names"])
    n_strides = size(pp, 1)
    stride_bins = Int(data["STRIDE_BINS"])
    window_bins = Int(data["WINDOW_BINS"])
    dt_days     = Float64(data["dt_days"])

    stds = _cloud_std_per_stride(pp, mask)
    x    = _stride_x_days(n_strides, stride_bins, dt_days, window_bins)
    n_params = size(stds, 2)

    n_cols = 5
    n_rows = ceil(Int, n_params / n_cols)
    plt = plot(layout=(n_rows, n_cols),
               size=(1500, n_rows * 200),
               legend=false, dpi=120,
               framestyle=:box, grid=true, gridalpha=0.3,
               plot_title="Cloud std (log-space) per stride — " *
                          (isempty(label) ? "" : label * " — ") *
                          "T=$(Int(T_total_days))d",
               plot_titlefontsize=11)
    for j in 1:n_params
        plot!(plt[j], x, stds[:, j]; color=:blue, linewidth=1.4)
        title!(plt[j], pnames[j]; titlefontsize=9)
        xlims!(plt[j], 0.0, T_total_days)
        ylims!(plt[j], 0.0, max(0.40, maximum(filter(!isnan, stds[:, j]))) * 1.05)
        if j > n_params - n_cols
            xlabel!(plt[j], "time (days)"; labelfontsize=9)
        end
    end
    savefig(plt, out_path)
    return out_path
end


function plot_cloud_std_sweep(sweep_dir::AbstractString;
                                out_path::AbstractString,
                                label_prefix::AbstractString = "v15-baseline-no-LW")
    horizon_dirs = sort(filter(d -> isdir(joinpath(sweep_dir, d)) &&
                                       startswith(d, "T") && endswith(d, "_seed42"),
                                  readdir(sweep_dir)))
    @info "diag_cloud_collapse: found horizons" horizon_dirs

    runs = Tuple{Int, Matrix{Float64}, Vector{Float64}, Vector{String}}[]
    for hdir in horizon_dirs
        T_days = parse(Int, replace(replace(hdir, r"^T" => ""), r"d_seed42$" => ""))
        data_path = joinpath(sweep_dir, hdir, "data.jld2")
        isfile(data_path) || (@warn "missing data.jld2 in $hdir"; continue)
        data = _load_run(data_path)
        pp   = data["posterior_particles"]
        mask = collect(Bool, data["posterior_window_mask"])
        pnames = String.(data["param_names"])
        stds = _cloud_std_per_stride(pp, mask)
        n_strides = size(pp, 1)
        stride_bins = Int(data["STRIDE_BINS"])
        window_bins = Int(data["WINDOW_BINS"])
        dt_days     = Float64(data["dt_days"])
        x = _stride_x_days(n_strides, stride_bins, dt_days, window_bins)
        push!(runs, (T_days, stds, x, pnames))
    end
    isempty(runs) && error("no runs found in $sweep_dir")

    pnames = runs[1][4]
    n_params = length(pnames)
    n_cols = 5
    n_rows = ceil(Int, n_params / n_cols)
    plt = plot(layout=(n_rows, n_cols),
               size=(1500, n_rows * 200),
               legend=:topright, legendfontsize=6,
               dpi=120, framestyle=:box, grid=true, gridalpha=0.3,
               plot_title="$label_prefix — cloud std (log-space) vs time, " *
                          "all 6 horizons overlaid",
               plot_titlefontsize=11)

    horizon_palette = Dict(14 => :red, 28 => :orange, 42 => :gold,
                           56 => :green, 70 => :blue, 84 => :purple)

    for j in 1:n_params
        for (T_days, stds, x, _) in runs
            col = get(horizon_palette, T_days, :gray)
            plot!(plt[j], x, stds[:, j];
                  color=col, linewidth=1.2,
                  label = j == 1 ? "T=$(T_days)d" : "")
        end
        title!(plt[j], pnames[j]; titlefontsize=9)
        ymax = 0.0
        for (_, stds, _, _) in runs
            v = filter(!isnan, stds[:, j])
            isempty(v) || (ymax = max(ymax, maximum(v)))
        end
        ylims!(plt[j], 0.0, max(0.40, ymax) * 1.05)
        xmax = maximum(maximum(r[3]) for r in runs)
        xlims!(plt[j], 0.0, xmax)
        if j > n_params - n_cols
            xlabel!(plt[j], "time (days)"; labelfontsize=9)
        end
    end
    savefig(plt, out_path)
    return out_path
end


function _parse_args(args::Vector{String})
    out = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--") && i + 1 <= length(args)
            out[a[3:end]] = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    return out
end


function main()
    args = _parse_args(ARGS)
    if haskey(args, "data-file")
        T_days = haskey(args, "T-days") ? parse(Float64, args["T-days"]) : 14.0
        out_png = get(args, "out-png", replace(args["data-file"], ".jld2" => "_cloud_std.png"))
        label = get(args, "label", "")
        data = _load_run(args["data-file"])
        path = plot_cloud_std_one_run(data; out_path=out_png,
                                       T_total_days=T_days, label=label)
        @info "wrote $path"
    elseif haskey(args, "sweep-dir")
        out_png = get(args, "out-png",
                      joinpath(args["sweep-dir"], "cloud_std_per_stride_all_horizons.png"))
        label_prefix = get(args, "label-prefix", "v15-baseline-no-LW")
        path = plot_cloud_std_sweep(args["sweep-dir"];
                                      out_path=out_png,
                                      label_prefix=label_prefix)
        @info "wrote $path"
    else
        println("Usage:")
        println("  --sweep-dir <dir> [--out-png <p>] [--label-prefix <s>]")
        println("  --data-file <path> [--T-days <n>] [--out-png <p>] [--label <s>]")
        exit(1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
