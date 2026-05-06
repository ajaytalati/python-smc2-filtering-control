# FSAv5/Schedule.jl — RBF schedule decoder for the FSA-v5 controller.
#
# Maps line-by-line to:
#   • LEAN  spec : `FSA_model_dev/lean/Fsa/V5/Schedule.lean`
#   • Python    : `models/fsa_high_res/control.py:51-68` (`_make_schedule`)
#                 `smc2fc/control/rbf_schedules.py:41-49` (RBFSchedule.design_matrix)
#
# Bug 5 structural prevention (charter Part I §4.2): the controller's
# decision variable `RBFCoeffs` (length 2*n_anchors) is a different
# Julia type from `DynParams` (the SMC² inference target) and from
# `Vector{BimodalPhi}` (the schedule output). All three are
# non-unifiable; the historical scalar-vs-vector confusion cannot
# type-check.

# ── Standard logistic sigmoid ──────────────────────────────────────────────

@inline _sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))

"""
    c_phi(phi_default::Float64, phi_max::Float64)::Float64

Inverse-sigmoid bias such that `phi_max * sigmoid(c_phi) == phi_default`,
i.e. `θ = 0` decodes to the default Φ. Mirrors `c_Phi` at
`control.py:57-58` and `Fsa.V5.c_phi` in `Schedule.lean`.
"""
@inline function c_phi(phi_default::Float64, phi_max::Float64)::Float64
    p = phi_default / phi_max
    return log(p / (1.0 - p))
end

# ── Gaussian-RBF design matrix ─────────────────────────────────────────────

"""
    design_matrix(n_steps::Int, dt::Float64,
                   n_anchors::Int, width_factor::Float64)::Matrix{Float64}

Build the Gaussian RBF design matrix `Φ_design[t, a] =
exp(-0.5 ((t·dt - centre_a) / width)^2)`. Returns shape `(n_steps,
n_anchors)`.

Centres: `n_anchors` evenly spaced over `[0, T_total]`, where
`T_total = n_steps * dt`. Width: `(T_total / max(n_anchors, 1)) *
width_factor`.

Mirrors `RBFSchedule.design_matrix` in
`smc2fc/control/rbf_schedules.py:41-49` and `Fsa.V5.designMatrix` in
`Schedule.lean`.
"""
function design_matrix(n_steps::Int, dt::Float64,
                        n_anchors::Int, width_factor::Float64)::Matrix{Float64}
    T_total = n_steps * dt
    centres = n_anchors == 1 ? [0.0] :
              collect(range(0.0, T_total; length = n_anchors))
    denom = n_anchors == 0 ? 1.0 : Float64(n_anchors)
    width = (T_total / denom) * width_factor

    out = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for t in 1:n_steps
        t_val = (t - 1) * dt
        for a in 1:n_anchors
            z = (t_val - centres[a]) / width
            out[t, a] = exp(-0.5 * z * z)
        end
    end
    return out
end

# ── Schedule decoder ───────────────────────────────────────────────────────

"""
    schedule_from_theta(theta::AbstractVector{Float64},
                         design::AbstractMatrix{Float64},
                         c_phi_val::Float64,
                         phi_max::Float64,
                         n_anchors::Int)::Vector{BimodalPhi}

Decode `θ ∈ ℝ^(2·n_anchors)` plus a pre-computed `Φ_design` matrix
into a bimodal `Vector{BimodalPhi}` schedule of length `n_steps`. The
first half of `θ` weighs the aerobic channel (Φ_B), the second the
strength channel (Φ_S). Both channels share the design matrix; the
coefficient slice differs.

Output for every `t` is `phi_max * sigmoid(c_phi_val + Φ_design[t,
:] · θ_channel)`.

Mirrors `Fsa.V5.scheduleFromTheta` in `Schedule.lean` and the body
of `schedule_from_theta` at `control.py:60-68`.
"""
function schedule_from_theta(theta::AbstractVector{Float64},
                              design::AbstractMatrix{Float64},
                              c_phi_val::Float64,
                              phi_max::Float64,
                              n_anchors::Int)::Vector{BimodalPhi}
    n_steps = size(design, 1)
    @assert size(design, 2) == n_anchors "design matrix has $(size(design,2)) columns, expected $n_anchors"
    @assert length(theta) == 2 * n_anchors "theta has length $(length(theta)), expected $(2 * n_anchors)"

    out = Vector{BimodalPhi}(undef, n_steps)
    @inbounds for t in 1:n_steps
        raw_B = c_phi_val
        raw_S = c_phi_val
        for a in 1:n_anchors
            d_ta   = design[t, a]
            raw_B += theta[a] * d_ta
            raw_S += theta[n_anchors + a] * d_ta
        end
        out[t] = BimodalPhi(phi_max * _sigmoid(raw_B), phi_max * _sigmoid(raw_S))
    end
    return out
end

export c_phi, design_matrix, schedule_from_theta
