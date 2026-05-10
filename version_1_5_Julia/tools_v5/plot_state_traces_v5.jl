#!/usr/bin/env julia
# v5 state-trajectory plot — mirrors v1.5's `plot_state_traces.jl`
# (4-panel) but extended to v5's 6D state + bimodal Φ:
#
#   row 1:  B  trajectory (MPC blue, baseline grey dashed)
#           S  trajectory (MPC blue, baseline grey dashed)
#   row 2:  F  trajectory (MPC dark-red, baseline grey dashed) + F_max
#           A  trajectory (MPC green, baseline grey dashed) + mean A
#   row 3:  K_FB trajectory  (MPC purple)
#           K_FS trajectory  (MPC purple)
#   row 4:  applied daily Φ_B schedule (orange step plot per stride)
#           applied daily Φ_S schedule (orange step plot per stride)
#
# Reads a v5 `data.jld2` written by `tools_v5/bench/bench_postproc.jl`
# (schema_version 1.0-v5). Required keys: `trajectory_mpc`,
# `trajectory_baseline`, `Phi_B_per_bin_mpc`, `Phi_S_per_bin_mpc`,
# `BINS_PER_DAY`, `STRIDE_BINS`, `dt_days`. Optional: `F_max`.
#
# Usage:
#   julia --project=. tools_v5/plot_state_traces_v5.jl <data.jld2> <out.png>

using JLD2
using Plots
using Statistics


function load_run_data(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


function plot_state_traces_v5(data::Dict; out_path::AbstractString)
    traj_mpc  = data["trajectory_mpc"]
    traj_base = data["trajectory_baseline"]
    Phi_B_mpc = data["Phi_B_per_bin_mpc"]
    Phi_S_mpc = data["Phi_S_per_bin_mpc"]
    BINS_PER_DAY = data["BINS_PER_DAY"]
    STRIDE_BINS  = data["STRIDE_BINS"]
    dt_days = data["dt_days"]
    F_max = haskey(data, "F_max") ? Float64(data["F_max"]) : 0.40

    n_bins = size(traj_mpc, 1)
    t_days = collect(0:n_bins - 1) .* dt_days
    n_strides = max(1, n_bins ÷ STRIDE_BINS)

    # Per-stride mean of Φ (for the per-stride trace at the bottom).
    daily_Phi_B_per_stride = Float32[]
    daily_Phi_S_per_stride = Float32[]
    for s in 1:n_strides
        lo = (s - 1) * STRIDE_BINS + 1
        hi = min(s * STRIDE_BINS, n_bins)
        push!(daily_Phi_B_per_stride, Float32(mean(Phi_B_mpc[lo:hi])))
        push!(daily_Phi_S_per_stride, Float32(mean(Phi_S_mpc[lo:hi])))
    end
    stride_start_days = [(s - 1) * STRIDE_BINS * dt_days for s in 1:n_strides]

    # State column index: [B, S, F, A, K_FB, K_FS].
    mean_A_mpc  = mean(traj_mpc[:, 4])
    mean_A_base = mean(traj_base[:, 4])

    # v1.5-matched palette + extras for the new state components.
    blue    = RGB(0x1f/255, 0x77/255, 0xb4/255)
    teal    = RGB(0x17/255, 0xbe/255, 0xcf/255)
    darkred = RGB(0x8b/255, 0x00/255, 0x00/255)
    green   = RGB(0x2c/255, 0xa0/255, 0x2c/255)
    purple  = RGB(0x94/255, 0x67/255, 0xbd/255)
    orange  = RGB(0xff/255, 0x7f/255, 0x0e/255)
    grey    = RGB(0x7f/255, 0x7f/255, 0x7f/255)
    redline = RGB(0xd6/255, 0x27/255, 0x28/255)

    plt = plot(layout = (4, 2), size = (1300, 1500), dpi = 120,
                fontfamily = "Helvetica", framestyle = :box,
                grid = true, gridalpha = 0.3,
                plot_title = "FSA-v5 closed-loop MPC, T=$(round(t_days[end], digits=1))d. " *
                              "mean A $(round(mean_A_mpc, digits=3)) " *
                              "vs baseline $(round(mean_A_base, digits=3))",
                plot_titlefontsize = 10)

    # B
    plot!(plt[1], t_days, traj_base[:, 1]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "B (baseline)", linewidth = 1.0)
    plot!(plt[1], t_days, traj_mpc[:, 1]; color = blue, lw = 1.6, label = "B (MPC)")
    title!(plt[1], "B  (aerobic fitness)"; titlefontsize = 11)
    xlabel!(plt[1], "time (days)"; labelfontsize = 8); ylabel!(plt[1], "B"; labelfontsize = 8)

    # S
    plot!(plt[2], t_days, traj_base[:, 2]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "S (baseline)", linewidth = 1.0)
    plot!(plt[2], t_days, traj_mpc[:, 2]; color = teal, lw = 1.6, label = "S (MPC)")
    title!(plt[2], "S  (strength capacity)"; titlefontsize = 11)
    xlabel!(plt[2], "time (days)"; labelfontsize = 8); ylabel!(plt[2], "S"; labelfontsize = 8)

    # F
    plot!(plt[3], t_days, traj_base[:, 3]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "F (baseline)", linewidth = 1.0)
    plot!(plt[3], t_days, traj_mpc[:, 3]; color = darkred, lw = 1.6, label = "F (MPC)")
    hline!(plt[3], [F_max]; color = redline, linestyle = :dash,
            label = "F_max", linewidth = 1.0)
    title!(plt[3], "F  (unified fatigue)"; titlefontsize = 11)
    xlabel!(plt[3], "time (days)"; labelfontsize = 8); ylabel!(plt[3], "F"; labelfontsize = 8)

    # A
    plot!(plt[4], t_days, traj_base[:, 4]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "A (baseline)", linewidth = 1.0)
    plot!(plt[4], t_days, traj_mpc[:, 4]; color = green, lw = 1.6, label = "A (MPC)")
    title!(plt[4], "A  (autonomic; mean MPC=$(round(mean_A_mpc, digits=3)) " *
                    "base=$(round(mean_A_base, digits=3)))"; titlefontsize = 10)
    xlabel!(plt[4], "time (days)"; labelfontsize = 8); ylabel!(plt[4], "A"; labelfontsize = 8)

    # K_FB
    plot!(plt[5], t_days, traj_base[:, 5]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "K_FB (baseline)", linewidth = 1.0)
    plot!(plt[5], t_days, traj_mpc[:, 5]; color = purple, lw = 1.6, label = "K_FB (MPC)")
    title!(plt[5], "K_FB  (aerobic fatigue gain)"; titlefontsize = 11)
    xlabel!(plt[5], "time (days)"; labelfontsize = 8); ylabel!(plt[5], "K_FB"; labelfontsize = 8)

    # K_FS
    plot!(plt[6], t_days, traj_base[:, 6]; color = grey, linestyle = :dash,
            alpha = 0.7, label = "K_FS (baseline)", linewidth = 1.0)
    plot!(plt[6], t_days, traj_mpc[:, 6]; color = purple, lw = 1.6, label = "K_FS (MPC)")
    title!(plt[6], "K_FS  (strength fatigue gain)"; titlefontsize = 11)
    xlabel!(plt[6], "time (days)"; labelfontsize = 8); ylabel!(plt[6], "K_FS"; labelfontsize = 8)

    # Daily Φ_B per stride
    plot!(plt[7], stride_start_days, daily_Phi_B_per_stride;
            color = orange, lw = 2.0, markershape = :circle, markersize = 3,
            label = "applied daily Φ_B")
    hline!(plt[7], [1.0]; color = grey, linestyle = :dash,
            label = "baseline Φ=1.0", linewidth = 1.0)
    title!(plt[7], "MPC-applied Φ_B schedule across $(n_strides) strides"; titlefontsize = 11)
    xlabel!(plt[7], "time (days)"; labelfontsize = 8); ylabel!(plt[7], "daily Φ_B"; labelfontsize = 8)
    ylims!(plt[7], 0.0, max(1.5, maximum(daily_Phi_B_per_stride) * 1.1))

    # Daily Φ_S per stride
    plot!(plt[8], stride_start_days, daily_Phi_S_per_stride;
            color = orange, lw = 2.0, markershape = :circle, markersize = 3,
            label = "applied daily Φ_S")
    hline!(plt[8], [1.0]; color = grey, linestyle = :dash,
            label = "baseline Φ=1.0", linewidth = 1.0)
    title!(plt[8], "MPC-applied Φ_S schedule across $(n_strides) strides"; titlefontsize = 11)
    xlabel!(plt[8], "time (days)"; labelfontsize = 8); ylabel!(plt[8], "daily Φ_S"; labelfontsize = 8)
    ylims!(plt[8], 0.0, max(1.5, maximum(daily_Phi_S_per_stride) * 1.1))

    savefig(plt, out_path)
    return out_path
end


function main()
    if length(ARGS) < 2
        println("Usage: julia plot_state_traces_v5.jl <data.jld2> <out.png>")
        exit(1)
    end
    data = load_run_data(ARGS[1])
    plot_state_traces_v5(data; out_path = ARGS[2])
    @info "wrote: $(ARGS[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
