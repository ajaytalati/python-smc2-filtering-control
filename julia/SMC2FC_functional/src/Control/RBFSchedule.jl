"""
    Control/RBFSchedule.jl

Gaussian-RBF schedule basis with output transforms. Port of
`smc2fc/control/rbf_schedules.py`.

The Python reference branches on a string `output: str`
(`'identity' | 'softplus' | 'sigmoid'`). Here, the same dispatch happens
at compile time via `RBFOutput` marker types.
"""
module RBFSchedule

using LogExpFunctions: log1pexp     # softplus = log(1 + exp(x))

export RBFOutput, IdentityOutput, SoftplusOutput, SigmoidOutput
export RBFBasis, design_matrix, schedule_from_theta

# ── Output-transform marker types ───────────────────────────────────────────

"""Abstract base for RBF output transforms."""
abstract type RBFOutput end

"""Identity output transform — schedules in `(-∞, ∞)`."""
struct IdentityOutput <: RBFOutput end

"""Softplus output transform — schedules in `[0, ∞)`."""
struct SoftplusOutput <: RBFOutput end

"""Sigmoid output transform — schedules in `[0, 1]`."""
struct SigmoidOutput  <: RBFOutput end

"""
    apply_output(::RBFOutput, x) -> AbstractArray

Apply the output transform to an array. Multiple-dispatched on the
output marker type.
"""
apply_output(::IdentityOutput, x::AbstractArray) = x
apply_output(::SoftplusOutput, x::AbstractArray) = log1pexp.(x)
apply_output(::SigmoidOutput,  x::AbstractArray) = 1.0 ./ (1.0 .+ exp.(.-x))

# ── RBFBasis struct ─────────────────────────────────────────────────────────

"""
    RBFBasis{O<:RBFOutput}

Gaussian-RBF schedule basis over the time grid `t_k = k·dt` for
`k = 0..n_steps-1`. Anchors are evenly spaced over `[0, n_steps·dt]`;
RBF width is `width_factor` times the anchor spacing.

# Fields
- `n_steps::Int`: number of grid points.
- `dt::Float64`: grid spacing.
- `n_anchors::Int`: number of RBF anchors.
- `width_factor::Float64`: width as a multiple of anchor spacing.
- `output::O`: output-transform marker.
"""
struct RBFBasis{O<:RBFOutput}
    n_steps::Int
    dt::Float64
    n_anchors::Int
    width_factor::Float64
    output::O
end

"""
    RBFBasis(n_steps, dt, n_anchors; width_factor=1.0, output=IdentityOutput())

Convenience constructor with keyword defaults matching the Python style.
"""
RBFBasis(n_steps::Integer, dt::Real, n_anchors::Integer;
          width_factor::Real = 1.0,
          output::RBFOutput = IdentityOutput()) =
    RBFBasis(Int(n_steps), Float64(dt), Int(n_anchors), Float64(width_factor), output)

"""
    design_matrix(b::RBFBasis) -> Matrix{Float64}

Build the Gaussian-RBF design matrix Φ such that
`Φ[t, a] = exp(-½ ((t·dt − c_a) / w)²)`.

# Arguments
- `b::RBFBasis`: basis specification.

# Returns
- `Matrix{Float64}`: `(n_steps, n_anchors)` design matrix.

# Notes
- Pre-compute it once and pass into `schedule_from_theta` to avoid
    rebuilding in the inner loop.
"""
function design_matrix(b::RBFBasis)
    T_total = b.n_steps * b.dt
    centres = collect(range(0.0, T_total; length = b.n_anchors))
    width   = (T_total / max(b.n_anchors, 1)) * b.width_factor
    t_grid  = collect(range(0.0; length = b.n_steps, step = b.dt))
    Φ = zeros(Float64, b.n_steps, b.n_anchors)
    @inbounds for a in 1:b.n_anchors, t in 1:b.n_steps
        δ = (t_grid[t] - centres[a]) / width
        Φ[t, a] = exp(-0.5 * δ * δ)
    end
    return Φ
end

"""
    schedule_from_theta(b::RBFBasis, θ; Φ=nothing) -> Vector

Build the schedule grid from RBF coefficients.

# Arguments
- `b::RBFBasis`: basis spec.
- `θ::AbstractVector`: RBF coefficients, length `n_anchors`.

# Keyword arguments
- `Φ`: optional pre-built design matrix from `design_matrix(b)`. If
    `nothing`, this function builds it on the fly (allocating).

# Returns
- `Vector`: post-transformed schedule of length `n_steps`.

# Notes
- For closures used inside SMC² rollouts, build `Φ` once outside the
    closure and capture it — significant allocation savings.
"""
function schedule_from_theta(b::RBFBasis, θ::AbstractVector;
                              Φ = nothing)
    P = Φ === nothing ? design_matrix(b) : Φ
    raw = P * θ
    return apply_output(b.output, raw)
end

end # module RBFSchedule
