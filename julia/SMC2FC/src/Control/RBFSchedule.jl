# Control/RBFSchedule.jl — Gaussian-RBF schedule basis with output transforms.
#
# Port of `smc2fc/control/rbf_schedules.py`. The Python file branches on
# `output: str` ('identity' | 'softplus' | 'sigmoid'). Per charter §15.1
# those become methods of `apply_output` dispatched on marker types
# `IdentityOutput`, `SoftplusOutput`, `SigmoidOutput`.

module RBFSchedule

using LogExpFunctions: log1pexp     # softplus = log(1 + exp(x))

export RBFOutput, IdentityOutput, SoftplusOutput, SigmoidOutput
export RBFBasis, design_matrix, schedule_from_theta

# ── Output-transform marker types ────────────────────────────────────────────

abstract type RBFOutput end
struct IdentityOutput <: RBFOutput end
struct SoftplusOutput <: RBFOutput end
struct SigmoidOutput  <: RBFOutput end

apply_output(::IdentityOutput, x::AbstractArray) = x
apply_output(::SoftplusOutput, x::AbstractArray) = log1pexp.(x)
apply_output(::SigmoidOutput,  x::AbstractArray) = 1.0 ./ (1.0 .+ exp.(.-x))

# ── RBFBasis struct ──────────────────────────────────────────────────────────

"""
    RBFBasis(n_steps, dt, n_anchors; width_factor=1.0, output=IdentityOutput())

Gaussian-RBF schedule basis over the time grid `t_k = k·dt` for `k = 0..n_steps-1`.
Anchors are evenly spaced over `[0, n_steps·dt]`; RBF width is `width_factor`
times the anchor spacing.

`output` controls the post-transform: `IdentityOutput()` for unconstrained
schedules, `SoftplusOutput()` for u ≥ 0, `SigmoidOutput()` for u ∈ [0, 1].
"""
struct RBFBasis{O<:RBFOutput}
    n_steps::Int
    dt::Float64
    n_anchors::Int
    width_factor::Float64
    output::O
end

# Convenience constructor matching the Python keyword-arg style.
RBFBasis(n_steps::Integer, dt::Real, n_anchors::Integer;
          width_factor::Real = 1.0,
          output::RBFOutput = IdentityOutput()) =
    RBFBasis(Int(n_steps), Float64(dt), Int(n_anchors), Float64(width_factor), output)

"""
    design_matrix(b::RBFBasis) -> (n_steps, n_anchors) Matrix

Build the Gaussian-RBF design matrix Φ where
`Φ[t, a] = exp(-½ ((t·dt − c_a)/w)²)`.
Pre-compute it once and pass it into `schedule_from_theta` so the inner
loop does not reallocate.
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
    schedule_from_theta(b::RBFBasis, θ::AbstractVector; Φ=nothing) -> Vector

Build the schedule grid `(n_steps,)` from RBF coefficients `θ` (shape
`(n_anchors,)`). If `Φ` is supplied (recommended), reuse it; otherwise
build it on the fly.
"""
function schedule_from_theta(b::RBFBasis, θ::AbstractVector;
                              Φ = nothing)
    P = Φ === nothing ? design_matrix(b) : Φ
    raw = P * θ
    return apply_output(b.output, raw)
end

end # module RBFSchedule
