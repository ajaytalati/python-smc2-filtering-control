#!/usr/bin/env julia
# Read a bench's data.jld2 and emit Phase 2 metrics:
#   - ∫A_MPC dt, ∫A_baseline dt, gain%
#   - F-violation %
#   - Φ peak day & value, Φ mean
#   - Oscillation amplitude (closed-loop only)
#   - Cumulative ∫A(t) saved as a CSV alongside
#
# Usage:
#   julia --project=. extract_metrics.jl <run_label> <data.jld2> <out_dir>
#
# Appends one row to <out_dir>/PHASE2_METRICS.csv.

using JLD2
using Statistics
using Printf

function moving_average(x::AbstractVector, window::Int)
    n = length(x)
    out = similar(x, Float64)
    for i in 1:n
        lo = max(1, i - window ÷ 2)
        hi = min(n, i + window ÷ 2)
        out[i] = mean(x[lo:hi])
    end
    return out
end

function main(run_label::String, data_path::String, out_dir::String)
    F_MAX = 0.40

    data = JLD2.jldopen(data_path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end

    traj_mpc  = Float64.(data["trajectory_mpc"])     # (n_bins, 3) — B, F, A
    traj_base = Float64.(data["trajectory_baseline"])
    daily_phi = Float64.(data["daily_phi_per_stride"])
    dt_days   = Float64(data["dt_days"])
    n_bins    = size(traj_mpc, 1)
    T_total_d = n_bins * dt_days

    A_MPC_traj  = traj_mpc[:, 3]
    A_BASE_traj = traj_base[:, 3]
    F_MPC_traj  = traj_mpc[:, 2]

    int_A_mpc  = sum(A_MPC_traj)  * dt_days
    int_A_base = sum(A_BASE_traj) * dt_days
    gain_pct   = 100 * (int_A_mpc - int_A_base) / int_A_base

    f_viol_pct = 100 * count(F_MPC_traj .> F_MAX) / n_bins

    phi_peak_day_idx = argmax(daily_phi)
    phi_peak_day     = phi_peak_day_idx - 1   # 0-indexed
    phi_peak_val     = daily_phi[phi_peak_day_idx]
    phi_mean         = mean(daily_phi)
    phi_min          = minimum(daily_phi)
    phi_max          = maximum(daily_phi)

    # Oscillation amplitude
    smoothed = moving_average(daily_phi, 3)
    osc_std  = std(daily_phi .- smoothed)
    osc_pct  = phi_mean > 0 ? 100 * osc_std / phi_mean : 0.0

    # Mean A (for cross-ref with §2.15 numbers in mean-A units)
    mean_A_mpc  = mean(A_MPC_traj)
    mean_A_base = mean(A_BASE_traj)

    # Cumulative ∫A(t) for plotting
    cum_A_mpc  = cumsum(A_MPC_traj)  .* dt_days
    cum_A_base = cumsum(A_BASE_traj) .* dt_days
    cum_csv = joinpath(out_dir, "cum_A_$(run_label).csv")
    open(cum_csv, "w") do io
        println(io, "t_days,A_mpc,A_base,cum_A_mpc,cum_A_base,F_mpc,B_mpc")
        for k in 1:n_bins
            t = (k - 1) * dt_days
            @printf(io, "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t, A_MPC_traj[k], A_BASE_traj[k],
                    cum_A_mpc[k], cum_A_base[k],
                    F_MPC_traj[k], traj_mpc[k, 1])
        end
    end

    # Per-day Φ trace CSV
    phi_csv = joinpath(out_dir, "phi_$(run_label).csv")
    open(phi_csv, "w") do io
        println(io, "day,daily_phi")
        for d in 1:length(daily_phi)
            @printf(io, "%d,%.6f\n", d - 1, daily_phi[d])
        end
    end

    # Append to PHASE2_METRICS.csv
    metrics_csv = joinpath(out_dir, "PHASE2_METRICS.csv")
    write_header = !isfile(metrics_csv)
    open(metrics_csv, "a") do io
        if write_header
            println(io, "label,T_total_days,int_A_mpc_Ad,int_A_base_Ad,gain_pct,mean_A_mpc,mean_A_base,F_violation_pct,phi_peak_day,phi_peak_val,phi_mean,phi_min,phi_max,osc_amp_pct,n_bins,dt_days,data_path")
        end
        @printf(io, "%s,%.4f,%.4f,%.4f,%.2f,%.6f,%.6f,%.3f,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%d,%.6f,%s\n",
                run_label, T_total_d, int_A_mpc, int_A_base, gain_pct,
                mean_A_mpc, mean_A_base, f_viol_pct,
                phi_peak_day, phi_peak_val, phi_mean, phi_min, phi_max,
                osc_pct, n_bins, dt_days, data_path)
    end

    # Console echo
    println("="^72)
    @printf("Run: %s\n", run_label)
    @printf("Data: %s\n", data_path)
    println("-"^72)
    @printf("∫A_MPC dt   = %.4f A·d   (mean A = %.4f over %.2f d)\n",
            int_A_mpc, mean_A_mpc, T_total_d)
    @printf("∫A_base dt  = %.4f A·d   (mean A = %.4f)\n", int_A_base, mean_A_base)
    @printf("Gain        = %+.2f %%   (Python target: +51 %%)\n", gain_pct)
    @printf("F-violation = %.3f %%    (target ≤ 1 %%)\n", f_viol_pct)
    @printf("Φ peak      = %.3f at day %d   (Python: 1.2 by day 13)\n", phi_peak_val, phi_peak_day)
    @printf("Φ summary   = min %.3f, mean %.3f, max %.3f\n", phi_min, phi_mean, phi_max)
    @printf("Oscillation = %.2f %% of mean Φ\n", osc_pct)
    println("="^72)
end

if length(ARGS) < 3
    println("Usage: julia extract_metrics.jl <run_label> <data.jld2> <out_dir>")
    exit(1)
end
main(ARGS[1], ARGS[2], ARGS[3])
