#!/usr/bin/env julia
# Phase 2 comparison plot: A trajectories + cumulative ∫A dt + F + daily Φ
# overlaying runs A (closed-loop saturated), B (open-loop saturated),
# C (closed-loop default) against the Φ=1 baseline.
#
# Usage:
#   julia --project=<v2> plot_area_comparison.jl <phase2_dir>
#
# Reads: <dir>/cum_A_run_{A,B,C}.csv, <dir>/phi_run_{A,B,C}.csv

using Plots
using Printf

"""
    read_csv_simple(path)

Plain-Julia CSV reader. Returns Dict{String, Vector{Float64}} keyed by header.
Assumes single header row, all numeric columns.
"""
function read_csv_simple(path::AbstractString)
    open(path, "r") do io
        header = split(strip(readline(io)), ",")
        cols = Dict{String, Vector{Float64}}(h => Float64[] for h in header)
        for line in eachline(io)
            isempty(strip(line)) && continue
            vals = split(line, ",")
            for (i, h) in enumerate(header)
                push!(cols[h], parse(Float64, vals[i]))
            end
        end
        return cols, header
    end
end

function main(out_dir::String)
    runs = ["run_A", "run_B", "run_C"]
    labels = Dict(
        "run_A" => "A: closed-loop @ saturated (N*, K*)",
        "run_B" => "B: open-loop @ saturated (N*, K*)",
        "run_C" => "C: closed-loop @ default (N=32, K=400)",
    )
    colors = Dict(
        "run_A" => RGB(0x1f/255, 0x77/255, 0xb4/255),
        "run_B" => RGB(0x2c/255, 0xa0/255, 0x2c/255),
        "run_C" => RGB(0xd6/255, 0x27/255, 0x28/255),
    )
    grey = RGB(0x7f/255, 0x7f/255, 0x7f/255)
    redline = RGB(0xd6/255, 0x27/255, 0x28/255)
    F_max = 0.40

    plt = plot(layout=(2, 2), size=(1500, 900), dpi=120,
                fontfamily="Helvetica", framestyle=:box,
                grid=true, gridalpha=0.3,
                plot_title="Phase 2 — does saturation help maximise ∫A dt?",
                plot_titlefontsize=12)

    # Panel 1: A trajectories.
    title!(plt[1], "A trajectories: MPC vs baseline (Φ=1)"; titlefontsize=11)
    xlabel!(plt[1], "time (days)"; labelfontsize=8)
    ylabel!(plt[1], "A"; labelfontsize=8)
    base_plotted = false
    for r in runs
        path = joinpath(out_dir, "cum_A_$(r).csv")
        isfile(path) || continue
        d, _ = read_csv_simple(path)
        if !base_plotted
            plot!(plt[1], d["t_days"], d["A_base"]; color=grey, linestyle=:dash,
                  alpha=0.7, lw=1.0, label="A baseline (Φ=1)")
            base_plotted = true
        end
        plot!(plt[1], d["t_days"], d["A_mpc"]; color=colors[r], lw=1.6,
              label="A — $(labels[r])")
    end

    # Panel 2: cumulative ∫A(t) dt.
    title!(plt[2], "Cumulative ∫A(t) dt — area under the A curve"; titlefontsize=11)
    xlabel!(plt[2], "time (days)"; labelfontsize=8)
    ylabel!(plt[2], "∫A dt (A·d)"; labelfontsize=8)
    base_plotted = false
    for r in runs
        path = joinpath(out_dir, "cum_A_$(r).csv")
        isfile(path) || continue
        d, _ = read_csv_simple(path)
        if !base_plotted
            plot!(plt[2], d["t_days"], d["cum_A_base"]; color=grey, linestyle=:dash,
                  alpha=0.7, lw=1.0, label="cum baseline")
            base_plotted = true
        end
        plot!(plt[2], d["t_days"], d["cum_A_mpc"]; color=colors[r], lw=1.6,
              label="cum — $(labels[r])")
    end

    # Panel 3: F trajectories.
    title!(plt[3], "F trajectories (F_max = 0.40)"; titlefontsize=11)
    xlabel!(plt[3], "time (days)"; labelfontsize=8)
    ylabel!(plt[3], "F"; labelfontsize=8)
    hline!(plt[3], [F_max]; color=redline, linestyle=:dash, lw=1.0, label="F_max")
    for r in runs
        path = joinpath(out_dir, "cum_A_$(r).csv")
        isfile(path) || continue
        d, _ = read_csv_simple(path)
        plot!(plt[3], d["t_days"], d["F_mpc"]; color=colors[r], lw=1.4,
              label="F — $(labels[r])")
    end

    # Panel 4: daily Φ.
    title!(plt[4], "Applied daily Φ"; titlefontsize=11)
    xlabel!(plt[4], "day"; labelfontsize=8)
    ylabel!(plt[4], "daily Φ"; labelfontsize=8)
    hline!(plt[4], [1.0]; color=grey, linestyle=:dash, lw=1.0, label="Φ=1 baseline")
    for r in runs
        path = joinpath(out_dir, "phi_$(r).csv")
        isfile(path) || continue
        d, _ = read_csv_simple(path)
        plot!(plt[4], d["day"], d["daily_phi"]; color=colors[r], lw=1.6,
              markershape=:circle, markersize=3, label="Φ — $(labels[r])")
    end

    out_png = joinpath(out_dir, "area_under_A_comparison.png")
    savefig(plt, out_png)
    println("wrote $out_png")
end

if length(ARGS) < 1
    println("Usage: julia plot_area_comparison.jl <phase2_dir>")
    exit(1)
end
main(ARGS[1])
