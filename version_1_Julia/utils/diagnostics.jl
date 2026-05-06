# Plot helpers that mirror `smc2fc.control.diagnostics` in the Python code,
# matched to the panel layout in the Python `outputs/` PNGs.

module Diagnostics

using Plots
using Printf
using Statistics: mean, std

export plot_cost_histogram!, plot_cost_histogram, TAB10

# Matplotlib `tab10` colors (Python's default categorical palette).
# Used so the Julia + Python histograms look identical.
const TAB10 = [
    RGB(0x1f/255, 0x77/255, 0xb4/255),    # blue
    RGB(0xff/255, 0x7f/255, 0x0e/255),    # orange
    RGB(0x2c/255, 0xa0/255, 0x2c/255),    # green
    RGB(0xd6/255, 0x27/255, 0x28/255),    # red
    RGB(0x94/255, 0x67/255, 0xbd/255),    # purple
    RGB(0x8c/255, 0x56/255, 0x4b/255),    # brown
    RGB(0xe3/255, 0x77/255, 0xc2/255),    # pink
    RGB(0x7f/255, 0x7f/255, 0x7f/255),    # gray
    RGB(0xbc/255, 0xbd/255, 0x22/255),    # olive
    RGB(0x17/255, 0xbe/255, 0xcf/255),    # cyan
]

const STEELBLUE = RGB(0x46/255, 0x82/255, 0xb4/255)

"""
    plot_cost_histogram!(p, particle_costs; references, title, xlabel="cost")

Bin the per-particle costs and overlay reference vertical lines, matching
the Python `plot_cost_histogram` from `smc2fc/control/diagnostics.py:17`.
`references` is a `Vector{Pair{String, Float64}}` so the order is stable
(dict iteration order in Julia is preserved by default but using a vector
guarantees the colour mapping matches Python's `enumerate(references.items())`).
"""
function plot_cost_histogram!(p, particle_costs::AbstractVector{<:Real};
                                references::AbstractVector{<:Pair{<:AbstractString,<:Real}},
                                title::AbstractString = "",
                                xlabel::AbstractString = "cost")
    histogram!(p, particle_costs;
                bins = 30, color = STEELBLUE, alpha = 0.7,
                label = "SMC² per-particle cost",
                xlabel = xlabel, ylabel = "density",
                title  = title, grid = true)
    for (i, (label, value)) in enumerate(references)
        vline!(p, [value]; ls = :dash, lw = 2,
                color = TAB10[((i - 1) % 10) + 1],
                label = @sprintf("%s = %.3g", label, value))
    end
    return p
end

"""
    plot_cost_histogram(particle_costs; references, title, xlabel) -> Plot

Stand-alone variant (creates its own figure).
"""
function plot_cost_histogram(particle_costs::AbstractVector{<:Real}; kwargs...)
    p = plot()
    plot_cost_histogram!(p, particle_costs; kwargs...)
    return p
end

end # module Diagnostics
