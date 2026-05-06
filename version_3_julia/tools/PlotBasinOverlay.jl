# tools/PlotBasinOverlay.jl — basin-overlay diagnostic.
#
# Port of `version_3/tools/plot_basin_overlay.py`. Builds the regime
# classification numerically via `FSAv5.find_a_sep` (returns -Inf /
# finite / +Inf for healthy / bistable / collapsed at each (Φ_B, Φ_S)
# grid point), then overlays the controller's chosen daily-mean (Φ_B,
# Φ_S) path. Mirrors Figure 2 (`fig:full-bifurcation`) of
# `LaTex_docs/FSA_version_5_technical_guide.tex`.
#
# Used by both Stage 2 and Stage 3 as their final save step. Plot
# backend = GR (Plots.jl default; no display server needed).

module PlotBasinOverlay

using Plots
using JSON3
using NPZ
using Printf

using ..FSAv5: BimodalPhi, find_a_sep, TRUTH_PARAMS_V5

export plot_basin_overlay, plot_from_run_dir

# ── Regime classification grid ─────────────────────────────────────────
"""
    classify_regime_grid(; n_grid=41, phi_max=1.5, params=TRUTH_PARAMS_V5)
        -> (phi_B_axis, phi_S_axis, regime_codes)

Codes:  0 = healthy  (A_sep == -Inf)
        1 = bistable (finite A_sep)
        2 = collapsed (A_sep == +Inf)
"""
function classify_regime_grid(; n_grid::Int = 41,
                                phi_max::Float64 = 1.5,
                                params = TRUTH_PARAMS_V5)
    phi_axis = collect(range(0.0, phi_max; length = n_grid))
    regime   = zeros(Int, n_grid, n_grid)
    @inbounds for j in 1:n_grid, i in 1:n_grid
        ϕ = BimodalPhi(phi_axis[i], phi_axis[j])
        a = find_a_sep(ϕ, params)
        regime[j, i] = a == -Inf ? 0 : (a == Inf ? 2 : 1)
    end
    return phi_axis, phi_axis, regime
end

# ── Main plotting routine ──────────────────────────────────────────────
"""
    plot_basin_overlay(applied_phi, out_path; title=nothing, baseline_phi=nothing,
                        n_grid=41, phi_max=1.5)

Render the basin-overlay diagnostic. `applied_phi` is `(n_strides, 2)`
of `(Φ_B, Φ_S)` per stride.
"""
function plot_basin_overlay(applied_phi::AbstractMatrix{<:Real},
                             out_path::AbstractString;
                             title::Union{Nothing,String} = nothing,
                             baseline_phi::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                             n_grid::Int = 41,
                             phi_max::Float64 = 1.5)
    size(applied_phi, 2) == 2 ||
        error("applied_phi must be (n_strides, 2); got $(size(applied_phi))")

    pb_axis, ps_axis, regime = classify_regime_grid(n_grid=n_grid, phi_max=phi_max)

    # Three-tone colormap: healthy / bistable / collapsed.
    plt = heatmap(pb_axis, ps_axis, regime;
                   color   = [:lightgreen, :khaki, :lightcoral],
                   clim    = (0, 2),
                   xlabel  = "Φ_B (aerobic stimulus)",
                   ylabel  = "Φ_S (strength stimulus)",
                   title   = something(title, "Basin overlay: applied (Φ_B, Φ_S) path"),
                   xlims   = (0.0, phi_max),
                   ylims   = (0.0, phi_max),
                   aspect_ratio = :equal,
                   colorbar = false,
                   alpha    = 0.6)

    # Controller path
    plot!(plt, applied_phi[:, 1], applied_phi[:, 2];
           seriestype = :path, marker = :circle, ms = 3, lw = 1.8,
           color = :steelblue, label = "controller path")
    scatter!(plt, [applied_phi[1, 1]], [applied_phi[1, 2]];
              ms = 8, color = :darkgreen, label = "start")
    scatter!(plt, [applied_phi[end, 1]], [applied_phi[end, 2]];
              ms = 8, color = :purple, label = "end")
    if baseline_phi !== nothing
        scatter!(plt, [baseline_phi[1]], [baseline_phi[2]];
                  ms = 7, marker = :xcross, color = :black,
                  label = @sprintf("baseline (%.2f,%.2f)", baseline_phi...))
    end

    # Region labels
    annotate!(plt, 0.20, 0.20, text("healthy\nisland", 9, :darkgreen, :bold))
    annotate!(plt, 1.15, 0.10, text("collapsed",        9, :darkred,   :bold))
    annotate!(plt, 0.55, 0.85, text("bistable\nannulus",8, :goldenrod, :bold))

    savefig(plt, out_path)
    return out_path
end

# ── CLI helper: plot from a saved run dir's trajectory.npz ──────────
function plot_from_run_dir(run_dir::AbstractString)
    npz_path = joinpath(run_dir, "trajectory.npz")
    isfile(npz_path) || error("trajectory.npz not found at $npz_path")
    data = NPZ.npzread(npz_path)
    haskey(data, "applied_phi_per_stride") ||
        error("trajectory.npz missing 'applied_phi_per_stride' key: $(keys(data))")
    applied = data["applied_phi_per_stride"]

    title = basename(run_dir)
    baseline_phi = nothing
    manifest_path = joinpath(run_dir, "manifest.json")
    if isfile(manifest_path)
        m = JSON3.read(read(manifest_path, String))
        sce = get(m, :scenario, Dict())
        cv  = get(m, :cost_variant, "?")
        title = "$(basename(run_dir))\ncost=$(cv), scenario=$(get(sce, :name, "?"))"
        bp = get(sce, :baseline_phi, nothing)
        bp !== nothing && (baseline_phi = (bp[1], bp[2]))
    end

    out = plot_basin_overlay(applied, joinpath(run_dir, "basin_overlay.png");
                              title=title, baseline_phi=baseline_phi)
    println("Wrote $out")
    return out
end

function main(argv::Vector{<:AbstractString} = ARGS)
    isempty(argv) && error("usage: julia PlotBasinOverlay.jl <run_dir>")
    plot_from_run_dir(argv[1])
end

end # module PlotBasinOverlay
