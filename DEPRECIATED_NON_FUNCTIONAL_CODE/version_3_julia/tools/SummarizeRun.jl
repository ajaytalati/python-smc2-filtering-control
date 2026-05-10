# tools/SummarizeRun.jl — print a markdown summary of a bench run.
#
# Port of `version_3/tools/summarize_run.py`. Reads a run dir's
# manifest.json and prints a CHANGELOG-ready markdown summary.

module SummarizeRun

using JSON3
using Printf

"""
    summarize(run_dir::AbstractString)

Print a markdown summary of `run_dir/manifest.json` to stdout.
Mirrors `summarize_run.summarize` in Python.
"""
function summarize(run_dir::AbstractString)
    manifest_path = joinpath(run_dir, "manifest.json")
    isfile(manifest_path) || error("no manifest.json under $run_dir")
    m = JSON3.read(read(manifest_path, String))

    sc = get(m, :scenario, Dict())
    s  = get(m, :summary, Dict())
    cfg = get(m, :ctrl_cfg, Dict())

    println("# Run ", basename(run_dir))
    println()
    @printf "- **bench**: %s\n"        get(m, :bench, "?")
    @printf "- **stage**: %s\n"        get(m, :stage, "?")
    @printf "- **cost variant**: %s\n" get(m, :cost_variant, "?")
    @printf "- **scenario**: %s\n"     get(sc, :name, "?")
    @printf "- **baseline_phi**: %s\n" get(sc, :baseline_phi, "?")
    @printf "- **T_total_days**: %s\n" get(m, :T_total_days, "?")
    @printf "- **n_strides**: %s\n"    get(m, :n_strides, "?")
    @printf "- **n_replans**: %s\n"    get(m, :n_replans, "?")
    println()

    println("## Controller config")
    for k in (:n_smc, :n_inner, :num_mcmc_steps, :hmc_num_leapfrog, :sigma_prior)
        haskey(cfg, k) && @printf "- %s = %s\n" k cfg[k]
    end
    println()

    println("## Summary")
    for k in (:weighted_violation_rate, :posthoc_mean_A_integral,
              :A_integral_observed, :mean_A_traj,
              :total_compute_s, :total_compute_min)
        haskey(s, k) && @printf "- %s = %s\n" k s[k]
    end
    if haskey(s, :gates)
        println("- **gates**: $(s.gates)")
    end
end

main(argv::Vector{<:AbstractString} = ARGS) =
    isempty(argv) ? error("usage: julia SummarizeRun.jl <run_dir>") :
                    summarize(argv[1])

export summarize, main

end # module SummarizeRun
