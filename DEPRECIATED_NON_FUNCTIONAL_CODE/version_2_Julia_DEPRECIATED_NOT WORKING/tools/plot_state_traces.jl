#!/usr/bin/env julia
# 4-panel state-trajectory plot — matches Python's E5_full_mpc_T14d_traces.png:
#   - top-left:    B trajectory (MPC blue, baseline grey dashed)
#   - top-right:   F trajectory (MPC dark-red, baseline grey dashed) + F_max
#   - bottom-left: A trajectory (MPC green, baseline grey dashed) with mean A
#                  annotations
#   - bottom-right: applied daily Φ schedule (orange step plot per stride)

using JLD2
using Plots
using Statistics

function load_run_data(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


function plot_state_traces(data::Dict; out_path::AbstractString)
    traj_mpc  = data["trajectory_mpc"]
    traj_base = data["trajectory_baseline"]
    daily_phi = data["daily_phi_per_stride"]
    BINS_PER_DAY = data["BINS_PER_DAY"]
    STRIDE_BINS  = data["STRIDE_BINS"]
    dt_days = data["dt_days"]
    F_max = haskey(data, "F_max") ? Float64(data["F_max"]) : 0.40

    n_bins = size(traj_mpc, 1)
    t_days = collect(0:n_bins - 1) .* dt_days
    n_strides = length(daily_phi)

    # Each stride starts at (s-1)*STRIDE_BINS, in days.
    stride_start_days = [(s - 1) * STRIDE_BINS * dt_days for s in 1:n_strides]

    mean_A_mpc  = mean(traj_mpc[:, 3])
    mean_A_base = mean(traj_base[:, 3])

    blue   = RGB(0x1f/255, 0x77/255, 0xb4/255)
    darkred = RGB(0x8b/255, 0x00/255, 0x00/255)
    green  = RGB(0x2c/255, 0xa0/255, 0x2c/255)
    orange = RGB(0xff/255, 0x7f/255, 0x0e/255)
    grey   = RGB(0x7f/255, 0x7f/255, 0x7f/255)
    redline = RGB(0xd6/255, 0x27/255, 0x28/255)

    plt = plot(layout=(2, 2), size=(1300, 700), dpi=120,
                fontfamily="Helvetica", framestyle=:box,
                grid=true, gridalpha=0.3,
                plot_title="Stage F — FSA-v2 closed-loop MPC, T=$(round(t_days[end], digits=1))d. " *
                           "mean A $(round(mean_A_mpc, digits=3)) vs baseline $(round(mean_A_base, digits=3))",
                plot_titlefontsize=10)

    # B
    plot!(plt[1], t_days, traj_base[:, 1]; color=grey, linestyle=:dash, alpha=0.7,
          label="B (baseline)", linewidth=1.0)
    plot!(plt[1], t_days, traj_mpc[:, 1]; color=blue, lw=1.6, label="B (MPC)")
    title!(plt[1], "B trajectory"; titlefontsize=11)
    xlabel!(plt[1], "time (days)"; labelfontsize=8)
    ylabel!(plt[1], "B"; labelfontsize=8)

    # F
    plot!(plt[2], t_days, traj_base[:, 2]; color=grey, linestyle=:dash, alpha=0.7,
          label="F (baseline)", linewidth=1.0)
    plot!(plt[2], t_days, traj_mpc[:, 2]; color=darkred, lw=1.6, label="F (MPC)")
    hline!(plt[2], [F_max]; color=redline, linestyle=:dash, label="F_max", linewidth=1.0)
    title!(plt[2], "F trajectory"; titlefontsize=11)
    xlabel!(plt[2], "time (days)"; labelfontsize=8)
    ylabel!(plt[2], "F"; labelfontsize=8)

    # A
    plot!(plt[3], t_days, traj_base[:, 3]; color=grey, linestyle=:dash, alpha=0.7,
          label="A (baseline)", linewidth=1.0)
    plot!(plt[3], t_days, traj_mpc[:, 3]; color=green, lw=1.6, label="A (MPC)")
    title!(plt[3], "A trajectory  (mean MPC: $(round(mean_A_mpc, digits=3)), " *
                   "baseline: $(round(mean_A_base, digits=3)))"; titlefontsize=10)
    xlabel!(plt[3], "time (days)"; labelfontsize=8)
    ylabel!(plt[3], "A"; labelfontsize=8)

    # Daily Φ — one point per stride (~12-h apart)
    plot!(plt[4], stride_start_days, daily_phi; color=orange, lw=2.0,
          markershape=:circle, markersize=3, label="applied daily Φ")
    hline!(plt[4], [1.0]; color=grey, linestyle=:dash, label="baseline Φ=1.0", linewidth=1.0)
    title!(plt[4], "MPC-applied Φ schedule across $(n_strides) strides"; titlefontsize=11)
    xlabel!(plt[4], "time (days)"; labelfontsize=8)
    ylabel!(plt[4], "daily Φ"; labelfontsize=8)
    ylims!(plt[4], 0.0, max(1.5, maximum(daily_phi) * 1.1))

    savefig(plt, out_path)
    return out_path
end


function main()
    if length(ARGS) < 2
        println("Usage: julia plot_state_traces.jl <data.jld2> <out.png>")
        exit(1)
    end
    data = load_run_data(ARGS[1])
    plot_state_traces(data; out_path=ARGS[2])
    @info "wrote: $(ARGS[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
