# Visualise the FSA-v5 separatrix A_sep(Φ_B, Φ_S) under TRUTH_PARAMS_V5.
#
# Three regimes (tech guide §4.2):
#   -Inf → mono-stable healthy  (no separatrix needed)
#   finite → bistable           (separatrix between A=0 and healthy A*)
#   +Inf → mono-stable collapsed (no positive equilibrium)
#
# The kernel currently uses a CONSTANT A_thr (default 0.3 via CLI).
# This script shows where that constant under- vs over-approximates the
# principled per-bin A_sep(Φ_t).
#
# Run from version_1_5_Julia/:
#   julia --project=. models/fsa_v5/model_notes_and_docs/plot_a_sep_landscape.jl

using Plots
using Printf

const _V5_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_V5_DIR, "FSAv5.jl"))

using .FSAv5: TRUTH_PARAMS_V5
using .FSAv5.CostV5: find_a_sep

# ── Config ────────────────────────────────────────────────────────────
const N_GRID    = 120
const PHI_MAX   = 1.5
const A_THR_CONST = 0.3        # the kernel's constant approximation

phi_grid = range(0.0, PHI_MAX; length = N_GRID)

# Probe Φ values from tech guide §4.5 sanity table
probes = [
    (0.0, 0.0,  "sedentary (0,0)"),
    (0.3, 0.0,  "aerobic-only (0.3,0)"),
    (0.0, 0.3,  "strength-only (0,0.3)"),
    (0.3, 0.3,  "balanced (0.3,0.3)"),
    (1.0, 1.0,  "over-training (1,1)"),
]

# ── Compute A_sep on the (Φ_B, Φ_S) grid ──────────────────────────────
A_sep = Matrix{Float64}(undef, N_GRID, N_GRID)
print("Computing find_a_sep on $(N_GRID)x$(N_GRID) grid... ")
elapsed = @elapsed begin
    @inbounds for i in 1:N_GRID, j in 1:N_GRID
        A_sep[i, j] = find_a_sep((phi_grid[i], phi_grid[j]), TRUTH_PARAMS_V5)
    end
end
println(@sprintf("done in %.2fs", elapsed))

# ── Probe-point summary ──────────────────────────────────────────────
println("\nA_sep at canonical probes:")
for (phi_B, phi_S, label) in probes
    a = find_a_sep((phi_B, phi_S), TRUTH_PARAMS_V5)
    tag = a == Inf  ? "+Inf  (mono-stable collapsed)" :
          a == -Inf ? "-Inf  (mono-stable healthy)" :
          @sprintf("%.4f (bistable)", a)
    @printf "  %-30s  A_sep = %s\n" label tag
end

# Frequency of each regime on the grid
n_healthy   = count(==( -Inf), A_sep)
n_collapsed = count(==(  Inf), A_sep)
n_bistable  = count(isfinite, A_sep)
@printf "\nGrid regime counts (out of %d): healthy=%d  bistable=%d  collapsed=%d\n" (
    N_GRID*N_GRID) n_healthy n_bistable n_collapsed

# Range of A_sep in the bistable annulus
finite_vals = filter(isfinite, vec(A_sep))
if !isempty(finite_vals)
    @printf "Bistable A_sep range: min=%.4f  max=%.4f  median=%.4f\n" (
        minimum(finite_vals)) maximum(finite_vals) (sort(finite_vals)[length(finite_vals) ÷ 2 + 1])
end

# ── Panel A: A_sep heatmap with sentinels masked ─────────────────────
A_sep_finite_only = map(x -> isfinite(x) ? x : NaN, A_sep)

p1 = heatmap(
    phi_grid, phi_grid, A_sep_finite_only';
    color = :viridis, clim = (0.0, 0.6),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "A_sep(Φ) — finite values only",
    aspect_ratio = :equal,
    colorbar_title = "A_sep",
)

# Overlay healthy region (-Inf) as light blue, collapsed (+Inf) as salmon
healthy_mask = map(x -> x == -Inf ? 1.0 : NaN, A_sep)
collapsed_mask = map(x -> x == Inf  ? 1.0 : NaN, A_sep)
heatmap!(p1, phi_grid, phi_grid, healthy_mask';
    color = cgrad([:lightblue, :lightblue]), colorbar = false)
heatmap!(p1, phi_grid, phi_grid, collapsed_mask';
    color = cgrad([:salmon, :salmon]), colorbar = false)

for (phi_B, phi_S, label) in probes
    scatter!(p1, [phi_B], [phi_S]; color = :black, markersize = 5,
             markershape = :circle, label = nothing)
    annotate!(p1, phi_B + 0.03, phi_S + 0.04, text(label, 7, :left, :black))
end
xlims!(p1, 0.0, PHI_MAX); ylims!(p1, 0.0, PHI_MAX)

# ── Panel B: 1D slice along the diagonal Φ_B = Φ_S ───────────────────
diag_phi = range(0.0, PHI_MAX; length = 400)
diag_A_sep = [find_a_sep((p, p), TRUTH_PARAMS_V5) for p in diag_phi]

# Visualise sentinels by clipping to plot bounds
diag_clip = map(diag_A_sep) do x
    x == -Inf ? -0.05 :       # below the axis
    x == Inf  ?  0.70 :       # above the expected range
    x
end

p2 = plot(diag_phi, diag_clip;
    label = "A_sep along Φ_B = Φ_S", lw = 2, color = :darkblue,
    xlabel = "Φ along the diagonal", ylabel = "A_sep (clipped)",
    title  = "1D slice: sedentary → healthy → over-training",
    ylim   = (-0.10, 0.75), legend = :topright)
hline!(p2, [A_THR_CONST]; label = "constant A_thr = 0.3",
    color = :red, linestyle = :dash, lw = 2)
hline!(p2, [0.0]; color = :gray, lw = 0.5, label = nothing)

annotate!(p2, 1.0, 0.65, text("+Inf (collapsed) clipped → 0.70", 8, :red, :left))
annotate!(p2, 0.5, -0.07, text("-Inf (healthy) clipped → -0.05", 8, :green, :left))

# ── Panel C: regime classification (3 categories) ────────────────────
regime_int = map(x -> x == -Inf ? 1 : (x == Inf ? 3 : 2), A_sep)

p3 = heatmap(phi_grid, phi_grid, Float64.(regime_int');
    color = cgrad([:lightblue, :gold, :salmon]; categorical = true),
    clim = (1, 3),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "Regime: blue=healthy, gold=bistable, red=collapsed",
    aspect_ratio = :equal, colorbar = false)
for (phi_B, phi_S, label) in probes
    scatter!(p3, [phi_B], [phi_S]; color = :black, markersize = 5,
             markershape = :circle, label = nothing)
    annotate!(p3, phi_B + 0.03, phi_S + 0.04, text(label, 7, :left, :black))
end
xlims!(p3, 0.0, PHI_MAX); ylims!(p3, 0.0, PHI_MAX)

# ── Panel D: A_sep − A_thr (only finite cells) ───────────────────────
diff_arr = map(x -> isfinite(x) ? (x - A_THR_CONST) : NaN, A_sep)

p4 = heatmap(phi_grid, phi_grid, diff_arr';
    color = :RdBu, clim = (-0.3, 0.3),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "A_sep − 0.3   (red = A_sep < 0.3, constant is stricter)",
    aspect_ratio = :equal,
    colorbar_title = "Δ")
xlims!(p4, 0.0, PHI_MAX); ylims!(p4, 0.0, PHI_MAX)

# ── Combine into a 2×2 figure ────────────────────────────────────────
plt = plot(p1, p2, p3, p4;
    layout = (2, 2), size = (1400, 1100),
    plot_title = "FSA-v5 A_sep landscape under TRUTH_PARAMS_V5\n" *
                 "(constant A_thr = 0.3 reference)")

out_path = joinpath(@__DIR__, "a_sep_landscape.png")
savefig(plt, out_path)
println("\nSaved figure to $out_path")
