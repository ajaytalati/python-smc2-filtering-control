# Visualise the FSA-v5 bifurcation parameter μ̄(A=0; Φ_B, Φ_S) under
# TRUTH_PARAMS_V5.
#
# Why this matters for the HMC controller:
#   - The principled A_sep(Φ) has discontinuities at the regime
#     boundaries (transcritical and saddle-node curves) — see
#     `a_sep_landscape.png`. In the mono-collapsed region (≈94% of
#     the Φ-plane) the clamped A_sep is constant in Φ, so the chance
#     penalty has *zero gradient* w.r.t. Φ from the threshold side.
#     HMC sees no escape direction.
#   - μ̄(0; Φ) is the underlying bifurcation parameter. It is a
#     smooth (polynomial-rational) function of Φ over the whole plane,
#     positive inside the healthy island and negative outside. Its
#     gradient ∇_Φ μ̄ points uphill toward the healthy island
#     everywhere → exactly the "smooth escape gradient" HMC needs.
#   - A candidate cost-augmentation term such as
#       λ_island · ∫ softplus(−μ̄(0; Φ_t)) dt
#     would rest on this smooth landscape rather than on the
#     piecewise A_sep.
#
# Run from version_1_5_Julia/:
#   julia --project=. models/fsa_v5/model_notes_and_docs/plot_mu_bar_landscape.jl

using Plots
using Printf
using Statistics: quantile

const _V5_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_V5_DIR, "FSAv5.jl"))

using .FSAv5: TRUTH_PARAMS_V5
using .FSAv5.CostV5: mu_bar

# ── Config ────────────────────────────────────────────────────────────
const N_GRID  = 200          # heatmap resolution
const N_QUIV  = 22           # quiver resolution (must divide neatly)
const PHI_MAX = 1.5
const SOFTPLUS_BETA = 50.0   # for the softplus(-μ̄) panel; matches β_chance

phi_grid = range(0.0, PHI_MAX; length = N_GRID)

probes = [
    (0.0, 0.0,  "sedentary (0,0)"),
    (0.3, 0.0,  "aerobic-only (0.3,0)"),
    (0.0, 0.3,  "strength-only (0,0.3)"),
    (0.3, 0.3,  "balanced (0.3,0.3)"),
    (1.0, 1.0,  "over-training (1,1)"),
]

# ── Compute μ̄(0; Φ) on the grid ──────────────────────────────────────
mu = Matrix{Float64}(undef, N_GRID, N_GRID)
print("Computing μ̄(0; Φ) on $(N_GRID)x$(N_GRID) grid... ")
elapsed = @elapsed begin
    @inbounds for i in 1:N_GRID, j in 1:N_GRID
        mu[i, j] = mu_bar(0.0, (phi_grid[i], phi_grid[j]), TRUTH_PARAMS_V5)
    end
end
println(@sprintf("done in %.3fs", elapsed))

# Probe summary — should match tech guide §4.5 sanity table
println("\nμ̄(0; Φ) at canonical probes (tech guide §4.5 reference values):")
ref_table = Dict(
    "sedentary (0,0)"        => -0.180,
    "aerobic-only (0.3,0)"   => -0.059,
    "strength-only (0,0.3)"  => -0.093,
    "balanced (0.3,0.3)"     =>  0.011,
    "over-training (1,1)"    => -1.608,
)
for (phi_B, phi_S, label) in probes
    val = mu_bar(0.0, (phi_B, phi_S), TRUTH_PARAMS_V5)
    expected = get(ref_table, label, NaN)
    flag = isnan(expected) ? "" :
           (abs(val - expected) < 0.01 ? " ✓" :
            @sprintf(" ✗ (expected %.3f)", expected))
    @printf "  %-30s  μ̄ = %+.4f%s\n" label val flag
end

mu_min = minimum(mu); mu_max = maximum(mu)
@printf "\nμ̄ range over grid: [%.3f, %.3f]\n" mu_min mu_max
@printf "Cells with μ̄ > 0 (healthy island): %d / %d (%.2f%%)\n" (
    count(>(0.0), mu)) (N_GRID*N_GRID) (100 * count(>(0.0), mu) / (N_GRID*N_GRID))

# ── Panel A: μ̄ heatmap with zero-contour ─────────────────────────────
# Symmetric clim around 0 so the diverging colormap reads naturally.
clim_abs = max(abs(mu_min), abs(mu_max))
clim_use = min(clim_abs, 0.5)   # clamp so the small healthy island is readable

p1 = heatmap(
    phi_grid, phi_grid, mu';
    color = :RdBu, clim = (-clim_use, clim_use),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "μ̄(0; Φ)   — blue: μ̄ > 0 (healthy)   red: μ̄ < 0 (collapsed)",
    aspect_ratio = :equal,
    colorbar_title = "μ̄",
)
contour!(p1, phi_grid, phi_grid, mu';
    levels = [0.0], color = :black, linewidth = 2.5, label = "μ̄ = 0")
for (phi_B, phi_S, label) in probes
    scatter!(p1, [phi_B], [phi_S]; color = :black, markersize = 5,
             markershape = :circle, label = nothing)
    annotate!(p1, phi_B + 0.03, phi_S + 0.04,
              text(label, 7, :left, :black))
end
xlims!(p1, 0.0, PHI_MAX); ylims!(p1, 0.0, PHI_MAX)

# ── Panel B: 1D diagonal slice ────────────────────────────────────────
diag_phi = range(0.0, PHI_MAX; length = 600)
diag_mu  = [mu_bar(0.0, (p, p), TRUTH_PARAMS_V5) for p in diag_phi]

p2 = plot(diag_phi, diag_mu;
    label = "μ̄(0; Φ_B = Φ_S)", lw = 2, color = :darkblue,
    xlabel = "Φ along the diagonal", ylabel = "μ̄(0; Φ)",
    title  = "1D slice — smooth, single positive lobe = healthy band",
    legend = :topright)
hline!(p2, [0.0]; color = :black, linestyle = :dash, lw = 1, label = "μ̄ = 0")
annotate!(p2, 0.10, 0.06, text("collapsed (μ̄ < 0)", 9, :red, :left))
annotate!(p2, 0.30, 0.06, text("healthy band\n(μ̄ > 0)", 9, :green, :left))
annotate!(p2, 1.10, 0.06, text("over-training (μ̄ < 0)", 9, :red, :left))

# ── Panel C: gradient field ∇_Φ μ̄ on a coarser grid (quiver) ────────
quiv_grid = range(0.05, PHI_MAX - 0.05; length = N_QUIV)
qx = Float64[]; qy = Float64[]; ux = Float64[]; uy = Float64[]
δ = 1e-3
for ϕB in quiv_grid, ϕS in quiv_grid
    g_B = (mu_bar(0.0, (ϕB + δ, ϕS), TRUTH_PARAMS_V5) -
           mu_bar(0.0, (ϕB - δ, ϕS), TRUTH_PARAMS_V5)) / (2δ)
    g_S = (mu_bar(0.0, (ϕB, ϕS + δ), TRUTH_PARAMS_V5) -
           mu_bar(0.0, (ϕB, ϕS - δ), TRUTH_PARAMS_V5)) / (2δ)
    # Normalise per-arrow so direction is readable; scale length later.
    nrm = sqrt(g_B^2 + g_S^2)
    if nrm > 0
        push!(qx, ϕB); push!(qy, ϕS)
        # 0.06 chosen so arrows fit in one grid cell
        push!(ux, 0.06 * g_B / nrm)
        push!(uy, 0.06 * g_S / nrm)
    end
end

p3 = heatmap(phi_grid, phi_grid, mu';
    color = :RdBu, clim = (-clim_use, clim_use),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "∇_Φ μ̄ direction (unit-normalised arrows)",
    aspect_ratio = :equal, colorbar = false)
contour!(p3, phi_grid, phi_grid, mu';
    levels = [0.0], color = :black, linewidth = 2.0, label = "μ̄ = 0")
quiver!(p3, qx, qy; quiver = (ux, uy), color = :black, lw = 0.7)
xlims!(p3, 0.0, PHI_MAX); ylims!(p3, 0.0, PHI_MAX)

# ── Panel D: candidate cost integrand softplus(-μ̄) / β ───────────────
# Numerically stable: softplus(x) = max(x,0) + log1p(exp(-|x|))
function softplus_stable(x)
    return max(x, 0.0) + log1p(exp(-abs(x)))
end
sp_integrand = map(mu) do m
    softplus_stable(SOFTPLUS_BETA * (-m)) / SOFTPLUS_BETA
end

# Clip top of color scale so the small-magnitude bowl near the island
# isn't washed out by the over-training corner.
sp_clim_top = quantile(vec(sp_integrand), 0.95)

p4 = heatmap(phi_grid, phi_grid, sp_integrand';
    color = cgrad(:viridis; rev = true),
    clim  = (0.0, sp_clim_top),
    xlabel = "Φ_B", ylabel = "Φ_S",
    title  = "Candidate cost: (1/β) softplus(-β·μ̄)",
    aspect_ratio = :equal,
    colorbar_title = "integrand")
contour!(p4, phi_grid, phi_grid, mu';
    levels = [0.0], color = :white, linewidth = 1.5)
for (phi_B, phi_S, label) in probes
    scatter!(p4, [phi_B], [phi_S]; color = :white, markersize = 4,
             markershape = :circle, markerstrokecolor = :black,
             label = nothing)
end
xlims!(p4, 0.0, PHI_MAX); ylims!(p4, 0.0, PHI_MAX)

# ── Combine ──────────────────────────────────────────────────────────
plt = plot(p1, p2, p3, p4;
    layout = (2, 2), size = (1500, 1200),
    plot_title = "FSA-v5 μ̄(0; Φ) — smooth bifurcation landscape\n" *
                 "Candidate gradient-providing term for HMC escape")

out_path = joinpath(@__DIR__, "mu_bar_landscape.png")
savefig(plt, out_path)
println("\nSaved figure to $out_path")
