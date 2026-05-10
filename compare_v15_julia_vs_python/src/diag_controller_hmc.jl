#!/usr/bin/env julia
# diag_controller_hmc.jl
#
# Aggregate + plot the controller-HMC diagnostics written by
# `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` when
# `--collect-ctrl-diagnostics true`.
#
# Reads each horizon's `controller_diagnostics.csv` under a sweep root,
# produces:
#   * per-horizon multi-panel plots (β-ladder, accept-rate-per-level,
#     ChEES L picked per level, ESJD/εL per level, ΔlogD per level);
#   * cross-horizon summary CSV with one row per horizon (mean/median
#     accept rate, mean ChEES L, mean levels-per-replan, mean β_max,
#     mean ESJD/εL, mean ΔlogD).
#
# Usage:
#   julia --project=version_1_5_Julia \
#       compare_v15_julia_vs_python/src/diag_controller_hmc.jl \
#       --sweep-dir <SWEEP_ROOT>
#
# OR for a single run:
#   julia ... diag_controller_hmc.jl --csv <run_dir>/controller_diagnostics.csv
#       [--T-days 14] [--out-dir <dir>]
#
# Conventions:
#   * Sweep root assumed to contain `T<N>d_seed42/` subdirs, each with
#     `controller_diagnostics.csv`.
#   * Output: <sweep_root>/docs/controller_hmc_diagnostics_<T>d.png
#     per horizon, plus <sweep_root>/docs/controller_hmc_summary.csv.

using Plots
using Statistics
using Printf


# ── CSV reader (manual; avoids CSV.jl dependency) ──────────────────────────

"""
    read_diagnostics_csv(path) -> NamedTuple of column vectors

The CSV has variable-length list fields (`chees_L_candidates`,
`chees_scores`) encoded as semicolon-separated strings — we leave them
as strings here and parse on demand.
"""
function read_diagnostics_csv(path::AbstractString)
    isfile(path) || error("not found: $path")
    lines = readlines(path)
    isempty(lines) && error("empty CSV: $path")
    header = split(lines[1], ',')
    n_cols = length(header)
    rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]
    @assert all(length(r) == n_cols for r in rows) "ragged CSV: $path"
    cols = Dict{String, Vector{String}}()
    for (j, h) in enumerate(header)
        cols[String(h)] = [String(r[j]) for r in rows]
    end
    return cols
end


_to_float(s::AbstractString) = parse(Float64, s)
_to_int(s::AbstractString)   = parse(Int, s)


function plot_controller_diagnostics_one_horizon(cols::Dict{String,Vector{String}};
                                                    out_path::AbstractString,
                                                    T_days::Real)
    n_rows = length(cols["level"])
    if n_rows == 0
        @warn "no controller-diagnostics rows for T=$(T_days)d"
        return out_path
    end

    stride_idx = parse.(Int,     cols["stride_idx"])
    level      = parse.(Int,     cols["level"])
    beta_pre   = parse.(Float64, cols["beta_pre"])
    beta_post  = parse.(Float64, cols["beta_post"])
    beta_max   = parse.(Float64, cols["beta_max"])
    chees_L    = parse.(Int,     cols["chees_L_chosen"])
    accept_frac = parse.(Float64, cols["accept_frac"])
    delta_logD = parse.(Float64, cols["delta_log_density"])
    esjd_per_eL = parse.(Float64, cols["esjd_per_eps_L"])
    eps_step   = parse.(Float64, cols["eps_step_size"])

    # x-axis: tempering-level index, with replan boundaries marked
    # by stride_idx changes. Use a global level counter for the plot.
    n_levels_total = length(level)
    x_global = 1:n_levels_total

    plt = plot(layout = (3, 2), size = (1300, 850),
                legend = false, dpi = 120, framestyle = :box,
                grid = true, gridalpha = 0.3,
                plot_title = @sprintf("Controller-HMC diagnostics — T=%dd, n_replans=%d, n_levels=%d",
                                       Int(T_days),
                                       length(unique(stride_idx)),
                                       n_levels_total),
                plot_titlefontsize = 11)

    # 1. β-ladder: β_post relative to β_max (so y ∈ [0, 1])
    plot!(plt[1], x_global, beta_post ./ beta_max;
          color = :blue, linewidth = 1.5,
          marker = :circle, markersize = 2)
    title!(plt[1], "β_post / β_max  (one tempering ladder per replan)";
           titlefontsize = 9)
    xlabel!(plt[1], "global level index"; labelfontsize = 8)
    ylims!(plt[1], 0.0, 1.05)

    # 2. Acceptance fraction
    plot!(plt[2], x_global, 100 .* accept_frac;
          color = :darkred, linewidth = 1.5,
          marker = :circle, markersize = 2)
    hline!(plt[2], [50.0, 80.0]; color = :gray, linestyle = :dash,
           linewidth = 0.8)
    title!(plt[2], "HMC acceptance % per level"; titlefontsize = 9)
    xlabel!(plt[2], "global level index"; labelfontsize = 8)
    ylims!(plt[2], 0.0, 105.0)

    # 3. ChEES-picked leapfrog count
    plot!(plt[3], x_global, chees_L;
          color = :darkgreen, linewidth = 1.2, marker = :circle,
          markersize = 3, seriestype = :scatter)
    title!(plt[3], "ChEES-picked leapfrog count L"; titlefontsize = 9)
    xlabel!(plt[3], "global level index"; labelfontsize = 8)
    yticks!(plt[3], [16, 32, 64, 128, 256, 512])

    # 4. ESJD per ε·L (mixing efficiency)
    plot!(plt[4], x_global, esjd_per_eL;
          color = :purple, linewidth = 1.5,
          marker = :circle, markersize = 2)
    title!(plt[4], "ESJD / (ε·L)  per level  — mixing efficiency";
           titlefontsize = 9)
    xlabel!(plt[4], "global level index"; labelfontsize = 8)

    # 5. ΔlogD per level (energy progression)
    plot!(plt[5], x_global, delta_logD;
          color = :orange, linewidth = 1.5,
          marker = :circle, markersize = 2)
    title!(plt[5], "Δlog-density per level (mean-across-chains)"; titlefontsize = 9)
    xlabel!(plt[5], "global level index"; labelfontsize = 8)

    # 6. Per-replan summary: levels-per-replan and accept-rate-per-replan
    by_replan = Dict{Int, Tuple{Int, Float64}}()
    for s in unique(stride_idx)
        idx = findall(==(s), stride_idx)
        by_replan[s] = (length(idx), mean(accept_frac[idx]))
    end
    replans_sorted = sort(collect(keys(by_replan)))
    levels_per = [by_replan[r][1]  for r in replans_sorted]
    acc_per    = [100 * by_replan[r][2] for r in replans_sorted]
    plot!(plt[6], replans_sorted, levels_per;
          color = :blue,  linewidth = 1.5, marker = :circle,
          markersize = 3, label = "levels")
    plot!(plt[6], replans_sorted, acc_per;
          color = :darkred, linewidth = 1.5, marker = :diamond,
          markersize = 3, label = "accept %", linestyle = :dot)
    title!(plt[6], "per replan: # tempering levels  +  mean accept %";
           titlefontsize = 9)
    xlabel!(plt[6], "stride_idx"; labelfontsize = 8)
    plot!(plt[6], legend = :topright, legendfontsize = 7)

    savefig(plt, out_path)
    return out_path
end


function summarise_one_horizon(cols::Dict{String,Vector{String}};
                                 T_days::Real)
    n_rows = length(cols["level"])
    if n_rows == 0
        return (T_days = T_days, n_replans = 0, n_levels_total = 0,
                mean_accept_pct = NaN, median_accept_pct = NaN,
                mean_chees_L = NaN, mean_levels_per_replan = NaN,
                mean_beta_max = NaN, mean_esjd_per_eL = NaN,
                mean_delta_logD = NaN, mean_wall_per_level = NaN)
    end
    accept_frac = parse.(Float64, cols["accept_frac"])
    chees_L     = parse.(Int,     cols["chees_L_chosen"])
    stride_idx  = parse.(Int,     cols["stride_idx"])
    beta_max    = parse.(Float64, cols["beta_max"])
    esjd_per_eL = parse.(Float64, cols["esjd_per_eps_L"])
    delta_logD  = parse.(Float64, cols["delta_log_density"])
    wall_lvl    = parse.(Float64, cols["wall_seconds_level"])
    n_replans   = length(unique(stride_idx))
    return (
        T_days                 = Float64(T_days),
        n_replans              = n_replans,
        n_levels_total         = n_rows,
        mean_accept_pct        = 100 * mean(accept_frac),
        median_accept_pct      = 100 * median(accept_frac),
        mean_chees_L           = mean(chees_L),
        mean_levels_per_replan = n_rows / n_replans,
        mean_beta_max          = mean(beta_max),
        mean_esjd_per_eL       = mean(esjd_per_eL),
        mean_delta_logD        = mean(delta_logD),
        mean_wall_per_level    = mean(wall_lvl),
    )
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
    if haskey(args, "csv")
        T_days = haskey(args, "T-days") ? parse(Float64, args["T-days"]) : 14.0
        run_dir = dirname(args["csv"])
        out_dir = get(args, "out-dir", run_dir)
        mkpath(out_dir)
        cols = read_diagnostics_csv(args["csv"])
        png_path = joinpath(out_dir, "controller_hmc_diagnostics_T$(Int(T_days))d.png")
        plot_controller_diagnostics_one_horizon(cols; out_path = png_path,
                                                  T_days = T_days)
        @info "wrote $png_path"
        s = summarise_one_horizon(cols; T_days = T_days)
        @info "summary: $s"
    elseif haskey(args, "sweep-dir")
        sweep_dir = args["sweep-dir"]
        out_dir = get(args, "out-dir", joinpath(sweep_dir, "docs"))
        mkpath(out_dir)
        horizon_dirs = sort(filter(d -> isdir(joinpath(sweep_dir, d)) &&
                                          startswith(d, "T") && endswith(d, "_seed42"),
                                     readdir(sweep_dir)))
        @info "diag_controller_hmc: found horizons $horizon_dirs"

        summaries = NamedTuple[]
        for hd in horizon_dirs
            T_days = parse(Int, replace(replace(hd, r"^T" => ""),
                                          r"d_seed42$" => ""))
            csv = joinpath(sweep_dir, hd, "controller_diagnostics.csv")
            if !isfile(csv)
                @warn "missing controller_diagnostics.csv in $hd; skipping"
                continue
            end
            cols = read_diagnostics_csv(csv)
            png_path = joinpath(out_dir,
                                  "controller_hmc_diagnostics_T$(T_days)d.png")
            plot_controller_diagnostics_one_horizon(cols; out_path = png_path,
                                                      T_days = T_days)
            @info "wrote $png_path"
            push!(summaries, summarise_one_horizon(cols; T_days = T_days))
        end

        # Cross-horizon summary CSV.
        summary_csv = joinpath(out_dir, "controller_hmc_summary.csv")
        open(summary_csv, "w") do io
            println(io,
                "T_days,n_replans,n_levels_total,",
                "mean_accept_pct,median_accept_pct,",
                "mean_chees_L,mean_levels_per_replan,",
                "mean_beta_max,mean_esjd_per_eL,",
                "mean_delta_logD,mean_wall_per_level")
            for s in summaries
                println(io,
                    "$(s.T_days),$(s.n_replans),$(s.n_levels_total),",
                    "$(s.mean_accept_pct),$(s.median_accept_pct),",
                    "$(s.mean_chees_L),$(s.mean_levels_per_replan),",
                    "$(s.mean_beta_max),$(s.mean_esjd_per_eL),",
                    "$(s.mean_delta_logD),$(s.mean_wall_per_level)")
            end
        end
        @info "wrote $summary_csv"
    else
        println("Usage:")
        println("  --sweep-dir <dir>          # all horizons in a sweep")
        println("  --csv <file>  [--T-days N] # single run")
        println("  [--out-dir <dir>]")
        exit(1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
