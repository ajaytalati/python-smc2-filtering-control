# FSA-v5 RBF schedule decoder — pure Julia transcription of
# `version_1_5_LEAN/Fsa/V5/Schedule.lean`.
#
# Provides:
#   - `sigmoid(x)`      — logistic σ(x) = 1 / (1 + e^{-x})
#   - `c_phi(default, max)` — inverse-sigmoid bias so θ = 0 decodes to
#       Φ = Φ_default
#   - `schedule_from_theta(theta, phi_design, c_phi, phi_max, n_anchors)`
#       — decode an RBF coefficient vector + design matrix into a
#       per-bin bimodal `(Phi_B, Phi_S)` schedule
#   - `design_matrix(n_steps, dt, n_anchors, width_factor)`
#       — Gaussian RBF design matrix `(n_steps, n_anchors)`
#
# Diff-tested against the Lean reference at machine precision.

module ScheduleV5

export sigmoid, c_phi, schedule_from_theta, design_matrix


"""
    sigmoid(x) -> Float64

Standard logistic sigmoid `σ(x) = 1 / (1 + e^{-x})`. Mirrors
`Fsa.V5.sigmoid` (`Fsa/V5/Schedule.lean:41-42`).
"""
@inline sigmoid(x::Real) = 1.0 / (1.0 + exp(-x))

"""
    c_phi(phi_default, phi_max) -> Float64

Inverse-sigmoid bias for an RBF schedule centred on `Phi_default`.
Returns `log(p / (1 - p))` where `p = Phi_default / Phi_max`. With
this bias, `sigmoid(c_phi) * phi_max = phi_default`, i.e. the
all-zeros θ decodes to the default Φ. Mirrors `Fsa.V5.c_phi`
(`Fsa/V5/Schedule.lean:48-50`).
"""
@inline function c_phi(phi_default::Real, phi_max::Real)
    p = phi_default / phi_max
    return log(p / (1.0 - p))
end


"""
    schedule_from_theta(theta, phi_design, cPhi, phi_max, n_anchors)
        -> Vector{NTuple{2, Float64}}

Decode a length-`(2 · n_anchors)` RBF coefficient vector `theta`
(first half = aerobic-channel weights, second half = strength) plus
an `(n_steps, n_anchors)` design matrix into a bimodal per-bin
schedule. Returns a length-`n_steps` `Vector` of `(Phi_B, Phi_S)`
tuples.

Mirrors `Fsa.V5.scheduleFromTheta` (`Fsa/V5/Schedule.lean:59-78`).

The two channels share the same design matrix; only the coefficient
slice differs (theta[1:n_anchors] for B, theta[n_anchors+1:end] for S).
"""
function schedule_from_theta(theta::AbstractVector{<:Real},
                              phi_design::AbstractMatrix{<:Real},
                              cPhi::Real,
                              phi_max::Real,
                              n_anchors::Integer)
    n_steps = size(phi_design, 1)
    out = Vector{NTuple{2, Float64}}(undef, n_steps)
    @inbounds for t in 1:n_steps
        raw_B = float(cPhi)
        raw_S = float(cPhi)
        for a in 1:n_anchors
            raw_B += theta[a]                 * phi_design[t, a]
            raw_S += theta[n_anchors + a]     * phi_design[t, a]
        end
        out[t] = (phi_max * sigmoid(raw_B), phi_max * sigmoid(raw_S))
    end
    return out
end


"""
    design_matrix(n_steps, dt, n_anchors, width_factor) -> Matrix{Float64}

Gaussian RBF design matrix of shape `(n_steps, n_anchors)`.

Centres are `n_anchors` evenly-spaced points in `[0, T_total]` where
`T_total = n_steps · dt`. Width is `(T_total / max(n_anchors, 1)) · width_factor`.
The basis is `Phi[t, a] = exp(−0.5 · ((t · dt − centre_a) / width)²)`.

Mirrors `Fsa.V5.designMatrix` (`Fsa/V5/Schedule.lean:100-110`).
"""
function design_matrix(n_steps::Integer, dt::Real,
                        n_anchors::Integer, width_factor::Real)
    T_total = n_steps * dt
    centres = _linspace(0.0, T_total, n_anchors)
    denom   = n_anchors == 0 ? 1.0 : float(n_anchors)
    width   = (T_total / denom) * width_factor
    out = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for t in 1:n_steps
        t_t = (t - 1) * dt
        for a in 1:n_anchors
            z = (t_t - centres[a]) / width
            out[t, a] = exp(-0.5 * z * z)
        end
    end
    return out
end

# linspace helper mirroring Fsa/V5/Schedule.lean:91-96.
function _linspace(a::Real, b::Real, n::Integer)
    if n == 0
        return Float64[]
    elseif n == 1
        return Float64[float(a)]
    else
        step = (b - a) / (n - 1)
        return [float(a) + (i - 1) * step for i in 1:n]
    end
end

end # module ScheduleV5
