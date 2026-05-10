# tools/DiagnoseViolationRate.jl — read-only chance-constraint diagnostic.
#
# Port of `version_3/tools/diagnose_violation_rate.py`. Three
# evaluators of the violation indicator on a completed bench run:
#
#   1. Per-bin (production formulation): `(A_traj < A_sep_per_bin)`
#      with A_sep computed at the per-bin instantaneous Phi. On
#      bursty schedules this fires vacuously during rest periods
#      (Phi=0 → A_sep=+inf → indicator=1), giving the historical
#      ~96% rate.
#   2. Per-day (alternative): A_sep at daily-mean Phi; indicator on
#      daily-mean A.
#   3. Active-bin only: indicator restricted to bins where
#      Phi_B + Phi_S > threshold.
#
# Read-only — does not run the controller. Pure CPU.

module DiagnoseViolationRate

using NPZ
using JSON3
using Printf
using ..FSAv5: FSAv5State, BimodalPhi, find_a_sep,
                BINS_PER_DAY, TRUTH_PARAMS_V5

const TRUTH = TRUTH_PARAMS_V5

function _classify(asep::AbstractVector{Float64})
    n = length(asep)
    healthy   = count(isneginf, asep)
    collapsed = count(isposinf, asep)
    bistable  = count(isfinite, asep)
    return (healthy=healthy, collapsed=collapsed, bistable=bistable, total=n)
end

function _fmt(c)
    n = c.total
    return @sprintf("healthy=%d/%d (%.1f%%), bistable=%d/%d (%.1f%%), collapsed=%d/%d (%.1f%%)",
                     c.healthy, n, 100*c.healthy/n,
                     c.bistable, n, 100*c.bistable/n,
                     c.collapsed, n, 100*c.collapsed/n)
end

"""
    diagnose(run_dir::AbstractString)

Print a 3-formulation regime breakdown for the given bench run. The
run dir must contain `trajectory.npz` with keys `full_phi` and
`trajectory`. Optionally `manifest.json` for context printing.

Mirrors `diagnose_violation_rate.diagnose` in Python.
"""
function diagnose(run_dir::AbstractString)
    println("=== ", basename(run_dir), " ===")

    manifest_path = joinpath(run_dir, "manifest.json")
    if isfile(manifest_path)
        m = JSON3.read(read(manifest_path, String))
        sc = get(m, :scenario, Dict())
        s  = get(m, :summary, Dict())
        println("  cost: $(get(m, :cost_variant, "?"))  scenario: $(get(sc, :name, "?"))  T_days: $(get(m, :T_total_days, "?"))")
        if haskey(s, :weighted_violation_rate)
            @printf "  reported weighted_violation_rate: %.4f\n" s.weighted_violation_rate
        end
    end

    z      = NPZ.npzread(joinpath(run_dir, "trajectory.npz"))
    fphi   = Float64.(z["full_phi"])         # (n_bins, 2)
    traj   = Float64.(z["trajectory"])       # (n_bins, 6)
    A_traj = traj[:, 4]
    n_bins = size(fphi, 1)
    n_days = n_bins ÷ BINS_PER_DAY

    @printf "  n_bins=%d, n_days=%d\n" n_bins n_days
    @printf "  per-bin Phi: PhiB median=%.3f mean=%.3f max=%.3f\n" median(fphi[:,1]) mean(fphi[:,1]) maximum(fphi[:,1])
    @printf "               PhiS median=%.3f mean=%.3f max=%.3f\n" median(fphi[:,2]) mean(fphi[:,2]) maximum(fphi[:,2])
    @printf "  A_traj: min=%.3f max=%.3f mean=%.3f\n" minimum(A_traj) maximum(A_traj) mean(A_traj)
    println()

    p = TRUTH

    # [1] Per-bin (production)
    asep_bin = [find_a_sep(BimodalPhi(fphi[t, 1], fphi[t, 2]), p)
                 for t in 1:n_bins]
    cl = _classify(asep_bin)
    ind_bin = Float64.(A_traj .< asep_bin)
    println("[1] PER-BIN (production formulation)")
    println("    A_sep regime distribution: ", _fmt(cl))
    @printf "    weighted_violation_rate: %.4f\n" mean(ind_bin)
    fires = ind_bin .> 0.5
    n_fires = sum(fires)
    if n_fires > 0
        n_collapsed = count(isposinf, asep_bin[fires])
        n_finite    = count(isfinite, asep_bin[fires])
        @printf "    Of %d firing bins: %d are A_sep=+inf (vacuous), %d finite (real bistable)\n" n_fires n_collapsed n_finite
    end
    println()

    # [2] Per-day (alternative)
    if n_days > 0
        phi_d = reshape(fphi[1:n_days*BINS_PER_DAY, :], BINS_PER_DAY, n_days, 2)
        phi_d_mean = vec(mean(phi_d, dims=1))   # length n_days * 2 then reshape
        phi_d_mat  = reshape(phi_d_mean, n_days, 2)
        A_d_mat    = vec(mean(reshape(A_traj[1:n_days*BINS_PER_DAY], BINS_PER_DAY, n_days), dims=1))
        asep_d     = [find_a_sep(BimodalPhi(phi_d_mat[i, 1], phi_d_mat[i, 2]), p)
                       for i in 1:n_days]
        cl_d = _classify(asep_d)
        ind_d = Float64.(A_d_mat .< asep_d)
        println("[2] PER-DAY (alternative: A_sep on daily-mean Phi)")
        println("    A_sep regime distribution: ", _fmt(cl_d))
        @printf "    daily violation rate: %.4f  (%d/%d days)\n" mean(ind_d) Int(sum(ind_d)) n_days
        println("    Per-day breakdown:")
        for i in 1:n_days
            ai   = asep_d[i]
            tag  = isneginf(ai) ? "-inf  HEALTHY" :
                    isposinf(ai) ? "+inf  COLLAPSED" :
                    @sprintf "%6.3f bistable" ai
            viol = ind_d[i] > 0.5 ? "VIOL" : "    "
            @printf "      Day %2d: PhiB=%.3f PhiS=%.3f A_d=%.3f A_sep=%s  %s\n"  i phi_d_mat[i,1] phi_d_mat[i,2] A_d_mat[i] tag viol
        end
        println()
    end

    # [3] Active bins only
    threshold = 0.05
    active = (fphi[:, 1] .+ fphi[:, 2]) .> threshold
    if any(active)
        asep_act = asep_bin[active]
        A_act    = A_traj[active]
        cl_a     = _classify(asep_act)
        ind_a    = Float64.(A_act .< asep_act)
        @printf "[3] ACTIVE BINS ONLY  (sum(Phi) > %.2f, %d/%d bins)\n" threshold sum(active) n_bins
        println("    A_sep regime distribution: ", _fmt(cl_a))
        @printf "    active-bin violation rate: %.4f\n" mean(ind_a)
        println()
    end
end

# ── Main entry ─────────────────────────────────────────────────────────────

function main(argv::Vector{<:AbstractString} = ARGS)
    run_dir = isempty(argv) ?
        # Find the latest run dir under outputs/fsa_v5/experiments/
        let exp = joinpath(pwd(), "outputs", "fsa_v5", "experiments")
            isdir(exp) || error("No experiments dir at $exp")
            runs = sort([joinpath(exp, e) for e in readdir(exp) if startswith(e, "run") &&
                          isfile(joinpath(exp, e, "trajectory.npz"))])
            isempty(runs) ? error("no run dirs with trajectory.npz") : last(runs)
        end :
        argv[1]
    diagnose(run_dir)
end

# Helpers
using Statistics: mean, median

export diagnose, main

end # module DiagnoseViolationRate
