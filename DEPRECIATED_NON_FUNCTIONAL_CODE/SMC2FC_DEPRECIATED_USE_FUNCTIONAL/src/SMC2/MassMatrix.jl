# SMC2/MassMatrix.jl — diagonal-mass-matrix estimation for the per-tempering-
# level HMC kernel.
#
# Direct port of `smc2fc/core/mass_matrix.py`. Comment from Python (kept):
#   "Full mass matrices fail on this problem: the PF likelihood landscape
#   punishes correlated HMC proposals and acceptance collapses to zero by
#   λ ≈ 0.3. The diagonal approximation is stable and adapts per-level
#   from the current particle cloud's per-dimension variance."

module MassMatrix

using Statistics: var

export estimate_mass_matrix

"""
    estimate_mass_matrix(particles::AbstractMatrix; regularisation=1e-4) -> Vector

Diagonal inverse mass-matrix from per-dimension particle variance.
Returns a `Vector{T}` of length `d_theta` (AdvancedHMC's `DiagEuclideanMetric`
expects this layout). Variance is floored at `regularisation` to avoid
zero-mass dimensions.
"""
function estimate_mass_matrix(particles::AbstractMatrix{T};
                               regularisation::Real = 1e-4) where {T<:Real}
    v = vec(var(particles; dims=1))
    return max.(v, T(regularisation))
end

end # module MassMatrix
