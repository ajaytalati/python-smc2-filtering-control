"""
    SMC2/MassMatrix.jl

Diagonal mass-matrix estimation for the per-tempering-level HMC kernel.
Direct port of `smc2fc/core/mass_matrix.py`.

The full-mass-matrix variant is intentionally not implemented: in the
Python reference it caused HMC acceptance to collapse to zero around
λ ≈ 0.3 because the PF likelihood landscape punishes correlated
proposals. The diagonal approximation adapts per-level from the current
particle cloud's per-dimension variance and is stable.
"""
module MassMatrix

using Statistics: var

export estimate_mass_matrix

"""
    estimate_mass_matrix(particles; regularisation=1e-4) -> Vector

Diagonal inverse mass-matrix from per-dimension particle variance.

# Arguments
- `particles::AbstractMatrix{T}`: `(n_smc, d_theta)` outer particle cloud,
    where each row is one parameter sample.

# Keyword arguments
- `regularisation::Real = 1e-4`: floor on each variance to avoid
    zero-mass dimensions.

# Returns
- `Vector{T}`: length `d_theta`. The layout AdvancedHMC's
    `DiagEuclideanMetric` consumes.
"""
function estimate_mass_matrix(particles::AbstractMatrix{T};
                               regularisation::Real = 1e-4) where {T<:Real}
    v = vec(var(particles; dims=1))
    return max.(v, T(regularisation))
end

end # module MassMatrix
