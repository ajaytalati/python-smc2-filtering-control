# tools/PlotStage2ThetaTraces.jl — Stage 2 controller-anchor traces.
#
# Port of `version_3/tools/plot_stage2_theta_traces.py`. Stage 2 has no
# filter (no SDE-param posterior), but the controller's HMC posterior
# over its 16 RBF schedule anchors IS a posterior trace — it's the
# controller's "decision parameters" evolving across the replans. This
# is the Stage 2 analogue of Stage 3's `Stage3_param_traces.png`.
#
# Layout: 16 panels in a 4×4 grid. Top half = the 8 Φ_B anchor params,
# bottom half = the 8 Φ_S anchor params. Each panel shows the mean θ
# (across SMC particles) at each replan, with a reference line at zero.

module PlotStage2ThetaTraces

using Plots
using NPZ
using Printf

export plot_stage2_theta_traces

"""
    plot_stage2_theta_traces(run_dir)

Read `replan_records.npz` from `run_dir` and write
`Stage2_theta_traces.png` + `Stage2_applied_phi_trace.png` alongside it.
"""
function plot_stage2_theta_traces(run_dir::AbstractString)
    rec = NPZ.npzread(joinpath(run_dir, "replan_records.npz"))
    mean_thetas    = rec["mean_thetas"]      # (n_replans, 2*n_anchors)
    mean_schedules = rec["mean_schedules"]   # (n_replans, n_steps, 2)

    n_replans, theta_dim = size(mean_thetas)
    n_anchors = theta_dim ÷ 2
    rep_idx = 0:(n_replans-1)

    # ── Panel 1: 16 θ-anchor traces (4×4 grid) ─────────────────────────
    plots = Plots.Plot[]
    for k in 0:(theta_dim - 1)
        anchor_idx = k % n_anchors
        dim_label  = k < n_anchors ? "Φ_B" : "Φ_S"
        p = plot(rep_idx, mean_thetas[:, k+1];
                  seriestype = :path,
                  marker = :circle, ms = 3, lw = 1.2,
                  color = :steelblue,
                  legend = false,
                  title  = "θ[$k] = $(dim_label) anchor #$anchor_idx",
                  titlefontsize = 7, tickfontsize = 6,
                  xlabel = k >= 12 ? "replan idx" : "",
                  ylabel = (k % 4) == 0 ? "mean θ" : "",
                  xguidefontsize = 7, yguidefontsize = 7)
        hline!(p, [0.0]; color = :gray, lw = 0.5, ls = :dot, label = "")
        push!(plots, p)
    end
    fig1 = plot(plots...; layout = (4, 4), size = (1500, 1100),
                plot_title = "Stage 2 ($(basename(run_dir)))\ncontroller HMC posterior mean of θ_ctrl across $n_replans replans",
                plot_titlefontsize = 10)
    out1 = joinpath(run_dir, "Stage2_theta_traces.png")
    savefig(fig1, out1)
    println("  wrote $out1")

    # ── Panel 2: applied (Φ_B, Φ_S) daily-mean trace ───────────────────
    n_steps = size(mean_schedules, 2)
    bins_per_day = min(96, n_steps)
    first_day = mean_schedules[:, 1:bins_per_day, :]   # (n_replans, 96, 2)
    daily_mean_phi = dropdims(mean(first_day; dims=2); dims=2)  # (n_replans, 2)

    fig2 = plot(rep_idx, daily_mean_phi[:, 1];
                 marker = :circle, ms = 4, lw = 1.5,
                 color = :steelblue, label = "Φ_B (aerobic)",
                 title = "Stage 2 applied daily-mean Φ across replans",
                 xlabel = "replan idx", ylabel = "daily-mean Φ")
    plot!(fig2, rep_idx, daily_mean_phi[:, 2];
           marker = :diamond, ms = 4, lw = 1.5,
           color = :darkorange, label = "Φ_S (strength)")
    out2 = joinpath(run_dir, "Stage2_applied_phi_trace.png")
    savefig(fig2, out2)
    println("  wrote $out2")
    return out1, out2
end

function main(argv::Vector{<:AbstractString} = ARGS)
    isempty(argv) && error("usage: julia PlotStage2ThetaTraces.jl <run_dir>")
    plot_stage2_theta_traces(argv[1])
end

# Pull `mean` from Statistics for the daily-mean reduction.
using Statistics: mean

end # module PlotStage2ThetaTraces
