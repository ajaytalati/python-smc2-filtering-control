#!/usr/bin/env julia
# diag_filter_hmc_accept.jl
#
# Parse a bench.log written by version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl
# for the per-tempering-level filter HMC log lines. Format (post-2026-05-09
# unification — note the L=N field):
#
#   [ Info:     [ N] λ X.XXX → Y.YYY (Δλ=Z.ZZZ) L=L accept=A%
#
# Emits a CSV with one row per logged level, plus a summary CSV with
# one row per horizon when run in --sweep-dir mode.
#
# Usage:
#   julia ... diag_filter_hmc_accept.jl --bench-log <run>/bench.log [--T-days N] [--out-dir <dir>]
#   julia ... diag_filter_hmc_accept.jl --sweep-dir <SWEEP_ROOT>
#
# When --sweep-dir is used, walks T<N>d_seed42 subdirs, parses each
# bench.log, writes per-horizon `filter_hmc_accept_T<N>d.csv` and a
# cross-horizon `filter_hmc_accept_summary.csv`.

using Statistics
using Printf


function parse_bench_log(path::AbstractString)
    isfile(path) || error("not found: $path")
    rows = NamedTuple[]
    # Filter HMC level log: "    [ N] λ X.XXX → Y.YYY (Δλ=Z.ZZZ) L=L accept=A%"
    pattern = r"\[\s*(\d+)\]\s+λ\s+([\d.\-eE]+)\s+→\s+([\d.\-eE]+)\s+\(Δλ=([\d.\-eE]+)\)\s+L=(\d+)\s+accept=([\d.\-]+)%"
    # Stride start: "[stride NN/MM] T.Ts ..."
    stride_pat = r"\[stride\s+(\d+)/\d+\]"
    current_stride = 0
    for line in eachline(path)
        m_stride = match(stride_pat, line)
        if m_stride !== nothing
            current_stride = parse(Int, m_stride.captures[1])
            continue
        end
        m = match(pattern, line)
        m === nothing && continue
        push!(rows, (
            stride       = current_stride,
            level        = parse(Int,     m.captures[1]),
            lambda_pre   = parse(Float64, m.captures[2]),
            lambda_post  = parse(Float64, m.captures[3]),
            delta_lambda = parse(Float64, m.captures[4]),
            chees_L      = parse(Int,     m.captures[5]),
            accept_pct   = parse(Float64, m.captures[6]),
        ))
    end
    return rows
end


function write_per_level_csv(rows, out_path)
    open(out_path, "w") do io
        println(io, "stride,level,lambda_pre,lambda_post,delta_lambda,chees_L,accept_pct")
        for r in rows
            println(io,
                "$(r.stride),$(r.level),$(r.lambda_pre),$(r.lambda_post),",
                "$(r.delta_lambda),$(r.chees_L),$(r.accept_pct)")
        end
    end
    return out_path
end


function summarise(rows; T_days = NaN)
    if isempty(rows)
        return (T_days = T_days, n_levels = 0,
                mean_accept_pct = NaN, median_accept_pct = NaN,
                min_accept_pct = NaN, max_accept_pct = NaN,
                mean_chees_L = NaN, mean_levels_per_stride = NaN)
    end
    accept = [r.accept_pct for r in rows]
    L_vec  = [r.chees_L    for r in rows]
    n_strides_with_levels = length(unique([r.stride for r in rows]))
    return (
        T_days                 = Float64(T_days),
        n_levels               = length(rows),
        mean_accept_pct        = mean(accept),
        median_accept_pct      = median(accept),
        min_accept_pct         = minimum(accept),
        max_accept_pct         = maximum(accept),
        mean_chees_L           = mean(L_vec),
        mean_levels_per_stride = length(rows) / max(1, n_strides_with_levels),
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
    if haskey(args, "bench-log")
        T_days = haskey(args, "T-days") ? parse(Float64, args["T-days"]) : NaN
        out_dir = get(args, "out-dir", dirname(args["bench-log"]))
        mkpath(out_dir)
        rows = parse_bench_log(args["bench-log"])
        per_level_csv = joinpath(out_dir,
            isnan(T_days) ? "filter_hmc_accept.csv" :
                            "filter_hmc_accept_T$(Int(T_days))d.csv")
        write_per_level_csv(rows, per_level_csv)
        @info "wrote $per_level_csv ($(length(rows)) rows)"
        s = summarise(rows; T_days = T_days)
        @info "summary: $s"
    elseif haskey(args, "sweep-dir")
        sweep_dir = args["sweep-dir"]
        out_dir = get(args, "out-dir", joinpath(sweep_dir, "docs"))
        mkpath(out_dir)
        horizon_dirs = sort(filter(d -> isdir(joinpath(sweep_dir, d)) &&
                                          startswith(d, "T") && endswith(d, "_seed42"),
                                     readdir(sweep_dir)))
        @info "diag_filter_hmc_accept: found horizons $horizon_dirs"
        summaries = NamedTuple[]
        for hd in horizon_dirs
            T_days = parse(Int, replace(replace(hd, r"^T" => ""), r"d_seed42$" => ""))
            log_path = joinpath(sweep_dir, hd, "bench.log")
            if !isfile(log_path)
                @warn "missing bench.log in $hd; skipping"
                continue
            end
            rows = parse_bench_log(log_path)
            per_level_csv = joinpath(out_dir,
                "filter_hmc_accept_T$(T_days)d.csv")
            write_per_level_csv(rows, per_level_csv)
            @info "wrote $per_level_csv ($(length(rows)) rows)"
            push!(summaries, summarise(rows; T_days = T_days))
        end
        summary_csv = joinpath(out_dir, "filter_hmc_accept_summary.csv")
        open(summary_csv, "w") do io
            println(io,
                "T_days,n_levels,",
                "mean_accept_pct,median_accept_pct,",
                "min_accept_pct,max_accept_pct,",
                "mean_chees_L,mean_levels_per_stride")
            for s in summaries
                println(io,
                    "$(s.T_days),$(s.n_levels),",
                    "$(s.mean_accept_pct),$(s.median_accept_pct),",
                    "$(s.min_accept_pct),$(s.max_accept_pct),",
                    "$(s.mean_chees_L),$(s.mean_levels_per_stride)")
            end
        end
        @info "wrote $summary_csv"
    else
        println("Usage:")
        println("  --bench-log <file>  [--T-days N]  [--out-dir <dir>]")
        println("  --sweep-dir <dir>                 [--out-dir <dir>]")
        exit(1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
