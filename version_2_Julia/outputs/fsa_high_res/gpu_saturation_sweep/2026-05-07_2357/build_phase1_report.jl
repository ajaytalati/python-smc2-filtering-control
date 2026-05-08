#!/usr/bin/env julia
# Read RESULTS.csv from Phase 1 sweep, pick the saturated cell, write
# RESULTS.md with the 4×3 table and recommendation.
#
# Saturation score: util_median × peak_mem_fraction. Cells that did not
# complete at least 4 strides are excluded (no clean per-stride wall).
#
# Usage:
#   julia build_phase1_report.jl <sweep_dir>

using Printf
using Statistics

GPU_TOTAL_MB = 32607     # RTX 5090

function parse_float(s::AbstractString)
    s = strip(s)
    s == "-" || isempty(s) ? NaN : parse(Float64, s)
end

function read_results(path::AbstractString)
    rows = Dict{String,Any}[]
    open(path, "r") do io
        header = split(strip(readline(io)), ",")
        for line in eachline(io)
            isempty(strip(line)) && continue
            vals = split(line, ",")
            d = Dict{String,Any}()
            for (i, h) in enumerate(header)
                d[h] = String(vals[i])
            end
            push!(rows, d)
        end
    end
    return rows
end

function main(sweep_dir::String)
    csv_path = joinpath(sweep_dir, "RESULTS.csv")
    rows = read_results(csv_path)

    rows_sorted = sort(rows; by = r -> (parse(Int, r["N"]), parse(Int, r["K"])))

    # Compute saturation score per row
    enriched = []
    for r in rows_sorted
        N = parse(Int, r["N"])
        K = parse(Int, r["K"])
        M_max = parse(Int, r["M_max"])
        strides = parse(Int, r["strides"])
        util_med = parse_float(r["util_median_pct"])
        util_max = parse_float(r["util_max_pct"])
        peak_mem_mb = parse_float(r["peak_mem_mb"])
        per_stride = parse_float(r["per_stride_wall_s"])
        wall_total = parse_float(r["wall_total_s"])
        status = r["status"]
        peak_mem_gb = isnan(peak_mem_mb) ? NaN : peak_mem_mb / 1024
        mem_frac = isnan(peak_mem_mb) ? NaN : 100 * peak_mem_mb / GPU_TOTAL_MB
        # Saturation score: util × mem_frac, both as fractions 0-100.
        # Penalise cells with < 4 strides (didn't complete the budget).
        score = if strides >= 4 && !isnan(util_med) && !isnan(mem_frac)
            util_med * mem_frac / 100
        else
            -Inf
        end
        push!(enriched, (
            N=N, K=K, M_max=M_max, strides=strides,
            util_med=util_med, util_max=util_max,
            peak_mem_mb=peak_mem_mb, peak_mem_gb=peak_mem_gb,
            mem_frac=mem_frac,
            per_stride=per_stride, wall_total=wall_total,
            status=status, score=score,
        ))
    end

    # Find saturated cell (highest score among complete cells)
    valid = filter(r -> isfinite(r.score), enriched)
    sat = isempty(valid) ? nothing : valid[argmax([r.score for r in valid])]

    # Write RESULTS.md
    md_path = joinpath(sweep_dir, "RESULTS.md")
    open(md_path, "w") do io
        println(io, "# RTX 5090 saturation sweep — N × K grid")
        println(io)
        println(io, "**Config**: `--step-minutes 60 --replan-K 2 --T-days 14 --seed 42`")
        println(io, "**Hardware**: RTX 5090 (32 GB)")
        println(io, "**Sweep dir**: `$sweep_dir`")
        println(io)

        if sat !== nothing
            println(io, "## Recommendation")
            println(io)
            @printf(io, "Use `--N-smc %d --K-per-chain %d`. ", sat.N, sat.K)
            @printf(io, "Median util %.1f %%, peak memory %.2f GB / %.0f GB (%.1f %% of card), per-stride wall %.1f s.\n",
                    sat.util_med, sat.peak_mem_gb, GPU_TOTAL_MB / 1024, sat.mem_frac, sat.per_stride)
            println(io)
        else
            println(io, "## Recommendation")
            println(io)
            println(io, "No cell completed enough strides to give a clean recommendation. See raw table below.")
            println(io)
        end

        println(io, "## Phase 1 results table")
        println(io)
        println(io, "| N | K | M_max | strides | util_med % | util_max % | peak_mem_GB | mem_% | per_stride_s | wall_total_s | status |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
        for r in enriched
            mark = (sat !== nothing && r.N == sat.N && r.K == sat.K) ? " ⭐" : ""
            @printf(io, "| %d%s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s |\n",
                    r.N, mark, r.K, r.M_max, r.strides,
                    isnan(r.util_med)    ? "—" : @sprintf("%.1f", r.util_med),
                    isnan(r.util_max)    ? "—" : @sprintf("%.0f", r.util_max),
                    isnan(r.peak_mem_gb) ? "—" : @sprintf("%.2f", r.peak_mem_gb),
                    isnan(r.mem_frac)    ? "—" : @sprintf("%.1f", r.mem_frac),
                    isnan(r.per_stride)  ? "—" : @sprintf("%.1f", r.per_stride),
                    isnan(r.wall_total)  ? "—" : @sprintf("%.0f", r.wall_total),
                    r.status)
        end
        println(io)

        println(io, "## Saturation diagnosis")
        println(io)
        # Util plateau: scan along K at each N
        println(io, "Per-N util progression (low K → high K):")
        for N in [32, 64, 128, 256]
            row_n = filter(r -> r.N == N, enriched)
            if length(row_n) >= 2
                utils = [isnan(r.util_med) ? "—" : @sprintf("%.0f", r.util_med) for r in row_n]
                ks = [r.K for r in row_n]
                @printf(io, "- N=%d: K=%s → util %s %%\n", N, join(ks, "/"), join(utils, "/"))
            end
        end
        println(io)
        println(io, "Per-K util progression (low N → high N):")
        for K in [400, 800, 1600]
            row_k = filter(r -> r.K == K, enriched)
            if length(row_k) >= 2
                utils = [isnan(r.util_med) ? "—" : @sprintf("%.0f", r.util_med) for r in row_k]
                ns = [r.N for r in row_k]
                @printf(io, "- K=%d: N=%s → util %s %%\n", K, join(ns, "/"), join(utils, "/"))
            end
        end
        println(io)

        # Look for plateau signature: util stops climbing while wall keeps growing
        valid_with_strides = filter(r -> r.strides >= 4 && isfinite(r.util_med), enriched)
        if length(valid_with_strides) >= 2
            max_util = maximum(r -> r.util_med, valid_with_strides)
            @printf(io, "Maximum median util observed across complete cells: **%.1f %%**.\n", max_util)
            if max_util < 60
                println(io, "Util well below 60 % even at the saturating cell — strongly suggests the per-chain host-loop ceiling diagnosed in §2.12 of the writeup is real. Source-level fix needed to push beyond this.")
            elseif max_util < 80
                println(io, "Util in the 60–80 % range. Some headroom left, but climbing further likely needs a source-level batching change (§2.12).")
            else
                println(io, "Util ≥ 80 %. Operating well into the saturation regime.")
            end
            println(io)
        end

        # What's left on the table
        if sat !== nothing && sat.peak_mem_gb < 24
            @printf(io, "Memory used at the saturated cell: %.2f / 32 GB (%.1f %%). Significant headroom remains — the next move would be to push N or K further, but per-stride wall is already at %.1f s and Phase 2 needs a full 14d closed-loop run, which would take roughly %.1f min.\n",
                    sat.peak_mem_gb, sat.mem_frac, sat.per_stride, sat.per_stride * 27 / 60)
        end

        println(io)
        println(io, "## Files")
        println(io)
        for r in enriched
            cell = "N$(r.N)_K$(r.K)"
            println(io, "- `$cell/bench.log` — bench stdout/stderr")
            println(io, "- `$cell/gpu_log.csv` — nvidia-smi time series")
            println(io, "- `$cell/bench_output/data.jld2` — bench output (if saved)")
        end
    end

    println("wrote $md_path")
    if sat !== nothing
        println("\nSATURATED CELL: N=$(sat.N) K=$(sat.K)")
        @printf("  util_med = %.1f %%  peak_mem = %.2f GB  per_stride = %.1f s\n",
                sat.util_med, sat.peak_mem_gb, sat.per_stride)
        # Emit machine-readable for the Phase 2 driver
        sat_path = joinpath(sweep_dir, "SATURATED_CELL.txt")
        open(sat_path, "w") do io
            @printf(io, "N=%d\nK=%d\n", sat.N, sat.K)
        end
    end
end

if length(ARGS) < 1
    println("Usage: julia build_phase1_report.jl <sweep_dir>")
    exit(1)
end
main(ARGS[1])
