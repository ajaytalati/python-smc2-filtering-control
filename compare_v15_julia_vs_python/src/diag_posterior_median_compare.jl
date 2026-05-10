#!/usr/bin/env julia
# diag_posterior_median_compare.jl
#
# Honest numerical comparison between two runs of the v1.5 bench. Loads
# `data.jld2::posterior_particles` from each and reports THREE metrics
# per parameter (per the 2026-05-09 critique that a single-stride
# median is a noisy snapshot):
#
#   1. time-averaged median across all FIRED strides after warmup
#      (default warmup = first 4 fired strides);
#   2. fraction of fired strides where the 5–95 % credible band
#      *covers truth* (the "honest coverage" — answers "did the
#      posterior bracket truth at this stride?");
#   3. cloud log-std at the final fired stride (a marker of whether
#      the cloud is alive or collapsed — collapsed clouds are bad
#      regardless of where the median lies).
#
# All measurements; no eyeballing.
#
# Usage:
#   julia ... diag_posterior_median_compare.jl \
#       --run-A <runA_dir> --label-A "name A" \
#       --run-B <runB_dir> --label-B "name B" \
#       --T-days <N> --out-csv <out>.csv [--warmup-strides 4]

using JLD2
using Statistics
using Printf


function _load(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


"""
Per-parameter aggregated metrics across the FIRED strides of one run,
after `warmup` strides. `pp` is the bench's `posterior_particles`
array of shape `(n_strides, n_smc, n_params)` in *constrained* (i.e.
`exp(U)`) space.
"""
function _aggregate_run(data::Dict; warmup::Int = 4)
    pp     = data["posterior_particles"]
    mask   = collect(Bool, data["posterior_window_mask"])
    pn     = String.(data["param_names"])
    truths = data["truth_params_dict"]
    n_strides, n_smc, n_params = size(pp)

    # The fired strides are those with mask=true. Apply warmup AFTER mask.
    fired_idx = findall(identity, mask)
    isempty(fired_idx) && error("no filter strides fired")
    keep_idx = fired_idx[(min(warmup + 1, length(fired_idx))):end]

    # Per-stride median (in constrained space) per param.
    med_per_stride = fill(NaN, length(keep_idx), n_params)
    p05_per_stride = fill(NaN, length(keep_idx), n_params)
    p95_per_stride = fill(NaN, length(keep_idx), n_params)
    for (k, s) in enumerate(keep_idx), j in 1:n_params
        col = view(pp, s, :, j)
        med_per_stride[k, j] = median(col)
        p05_per_stride[k, j] = quantile(col, 0.05)
        p95_per_stride[k, j] = quantile(col, 0.95)
    end

    # Final-stride log-std (cloud-alive indicator).
    last_fired = fired_idx[end]
    log_std_final = Float64[]
    for j in 1:n_params
        col_log = log.(max.(view(pp, last_fired, :, j), 1e-300))
        push!(log_std_final, std(col_log))
    end

    truths_vec = Float64[Float64(truths[String(p)]) for p in pn]
    n_keep = length(keep_idx)

    # Time-averaged median per param.
    median_time_avg = vec(mean(med_per_stride; dims = 1))

    # Fraction of kept strides where 5–95 % band covers truth.
    coverage_pct = zeros(n_params)
    for j in 1:n_params
        n_cover = count(k -> p05_per_stride[k, j] <= truths_vec[j] <= p95_per_stride[k, j],
                         1:n_keep)
        coverage_pct[j] = 100 * n_cover / n_keep
    end

    return (
        param_names      = pn,
        truths           = truths_vec,
        median_time_avg  = median_time_avg,
        coverage_pct     = coverage_pct,
        log_std_final    = log_std_final,
        n_strides_kept   = n_keep,
        last_fired       = last_fired,
        warmup_used      = warmup,
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
    for k in ("run-A", "run-B", "out-csv")
        haskey(args, k) || error("missing --$k")
    end
    label_A = get(args, "label-A", "A")
    label_B = get(args, "label-B", "B")
    T_days  = haskey(args, "T-days") ? parse(Float64, args["T-days"]) : NaN
    warmup  = haskey(args, "warmup-strides") ? parse(Int, args["warmup-strides"]) : 4

    data_A = _load(joinpath(args["run-A"], "data.jld2"))
    data_B = _load(joinpath(args["run-B"], "data.jld2"))
    a = _aggregate_run(data_A; warmup = warmup)
    b = _aggregate_run(data_B; warmup = warmup)
    @assert a.param_names == b.param_names
    @assert a.truths == b.truths

    open(args["out-csv"], "w") do io
        # CSV
        println(io, "param,truth,",
                    "median_timeavg_$(label_A),median_timeavg_$(label_B),",
                    "rel_err_timeavg_$(label_A)_pct,rel_err_timeavg_$(label_B)_pct,",
                    "coverage_pct_$(label_A),coverage_pct_$(label_B),",
                    "log_std_final_$(label_A),log_std_final_$(label_B),",
                    "n_strides_kept_$(label_A),n_strides_kept_$(label_B)")

        # Stdout pretty-print
        @printf("%-10s %10s | %10s %10s %8s %8s | %7s %7s | %7s %7s\n",
                "param", "truth",
                "med_$(label_A)", "med_$(label_B)",
                "rerrA%", "rerrB%",
                "covA%", "covB%",
                "logsdA", "logsdB")
        println("─"^110)
        for j in 1:length(a.param_names)
            p = a.param_names[j]
            t = a.truths[j]
            mA = a.median_time_avg[j];  mB = b.median_time_avg[j]
            errA = 100 * abs(mA - t) / max(abs(t), 1e-12)
            errB = 100 * abs(mB - t) / max(abs(t), 1e-12)
            covA = a.coverage_pct[j];    covB = b.coverage_pct[j]
            sdA  = a.log_std_final[j];   sdB  = b.log_std_final[j]
            @printf("%-10s %10.5f | %10.5f %10.5f %8.2f %8.2f | %7.1f %7.1f | %7.4f %7.4f\n",
                    p, t, mA, mB, errA, errB, covA, covB, sdA, sdB)
            println(io, "$p,$t,$mA,$mB,$errA,$errB,$covA,$covB,$sdA,$sdB,",
                        "$(a.n_strides_kept),$(b.n_strides_kept)")
        end
    end
    @info "wrote $(args["out-csv"])  (T = $T_days d, warmup = $warmup fired strides; A kept $(a.n_strides_kept) strides, B kept $(b.n_strides_kept) strides; final fired stride: A=$(a.last_fired) B=$(b.last_fired))"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
