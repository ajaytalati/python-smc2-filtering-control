#!/usr/bin/env julia
# Merge Phase 1 + Phase 2 metrics into the final RESULTS.md.
# Run after both phases have completed.
#
# Usage:
#   julia build_final_report.jl <sweep_dir>

using Printf

function parse_float(s::AbstractString)
    s = strip(s)
    s == "-" || isempty(s) ? NaN : parse(Float64, s)
end

function read_phase2(path::AbstractString)
    isfile(path) || return nothing
    rows = []
    open(path, "r") do io
        header_line = readline(io)
        header = split(strip(header_line), ",")
        for line in eachline(io)
            isempty(strip(line)) && continue
            vals = split(strip(line), ",")
            d = Dict{String,String}()
            for (i, h) in enumerate(header)
                d[h] = String(vals[i])
            end
            push!(rows, d)
        end
    end
    return rows
end

function main(sweep_dir::String)
    phase1_md = read(joinpath(sweep_dir, "RESULTS.md"), String)
    phase2_path = joinpath(sweep_dir, "closed_loop_at_saturation", "PHASE2_METRICS.csv")
    phase2 = read_phase2(phase2_path)

    final_path = joinpath(sweep_dir, "RESULTS.md")
    open(final_path, "w") do io
        write(io, phase1_md)
        println(io)
        println(io, "---")
        println(io)
        println(io, "# Phase 2 — closed-loop accuracy at the saturated cell")
        println(io)

        if phase2 === nothing
            println(io, "Phase 2 not yet run. Once Phase 1 picks the saturated `(N*, K*)`, run:")
            println(io)
            println(io, "```bash")
            println(io, "./run_phase2.sh <N*> <K*>")
            println(io, "```")
        else
            println(io, "**Headline:** the controller's job is to maximise ∫A dt (area under the A curve).")
            println(io)
            println(io, "Reference points:")
            println(io, "- Python target: gain ≈ +51 %")
            println(io, "- Julia closed-loop (current): gain ≈ +2.3 %")
            println(io)

            println(io, "## Phase 2 results table")
            println(io)
            println(io, "| label | ∫A_MPC dt (A·d) | ∫A_base dt (A·d) | **gain %** | F-viol % | Φ peak (day) | Φ mean | Osc % |")
            println(io, "|---|---|---|---|---|---|---|---|")
            for r in phase2
                @printf(io, "| %s | %.4f | %.4f | %+.2f | %.3f | %.3f (d%s) | %.4f | %.2f |\n",
                        r["label"],
                        parse_float(r["int_A_mpc_Ad"]),
                        parse_float(r["int_A_base_Ad"]),
                        parse_float(r["gain_pct"]),
                        parse_float(r["F_violation_pct"]),
                        parse_float(r["phi_peak_val"]),
                        r["phi_peak_day"],
                        parse_float(r["phi_mean"]),
                        parse_float(r["osc_amp_pct"]))
            end
            println(io)

            # Diagnosis
            gains = Dict(r["label"] => parse_float(r["gain_pct"]) for r in phase2)
            A_g = get(gains, "run_A", NaN)
            B_g = get(gains, "run_B", NaN)
            C_g = get(gains, "run_C", NaN)

            println(io, "## Diagnosis")
            println(io)
            if !isnan(A_g) && !isnan(B_g) && !isnan(C_g)
                @printf(io, "- C (closed-loop, default N=32 K=400): **%+.2f %%**\n", C_g)
                @printf(io, "- A (closed-loop, saturated): **%+.2f %%**\n", A_g)
                @printf(io, "- B (open-loop, saturated): **%+.2f %%**\n", B_g)
                println(io)
                if A_g > C_g + 5
                    println(io, "**Saturation HELPS the closed-loop optimiser.** A's gain materially exceeds C's. Recommend the saturated flags as the production operating point.")
                elseif abs(A_g - C_g) <= 5 && abs(B_g - C_g) <= 5
                    println(io, "**Saturation is INVARIANT** — A ≈ B ≈ C. The cost-surface gap is in the kernel / cost surface, not in sampling noise. Saturating particle counts does not close the gap to Python; the next move is the source-level bug §2.15 row 10/11 is tracking. Saturated flags still maximise GPU usage, but won't close the area gap on their own.")
                elseif B_g > A_g + 5
                    println(io, "**Saturation can find more area, but the closed-loop replan loop is throwing it away** — B's gain materially exceeds A's. This is the closed-loop oscillation bug §2.15 row 11. Recommend `--open-loop true` for production until the replan stability fix is in.")
                else
                    println(io, "Mixed signal — see numbers above. Manual interpretation needed.")
                end
            end
            println(io)

            println(io, "## Comparison plot")
            println(io)
            println(io, "![area_under_A_comparison](closed_loop_at_saturation/area_under_A_comparison.png)")
            println(io)

            println(io, "## Files")
            println(io)
            for r in phase2
                println(io, "- `closed_loop_at_saturation/$(r["label"])/` — bench output")
                println(io, "- `closed_loop_at_saturation/cum_A_$(r["label"]).csv` — A trajectory + cumulative integral")
                println(io, "- `closed_loop_at_saturation/phi_$(r["label"]).csv` — daily Φ schedule")
            end
        end
    end
    println("wrote $final_path")
end

if length(ARGS) < 1
    println("Usage: julia build_final_report.jl <sweep_dir>")
    exit(1)
end
main(ARGS[1])
