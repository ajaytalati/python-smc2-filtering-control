"""
    SMC2/Sampling.jl

Initial-particle sampling from the prior in unconstrained space.
Port of `smc2fc/core/sampling.py`.

The Python reference uses indicator-array branching (`is_ln`, `is_norm`)
because JAX cannot dispatch on prior type. Here, multiple dispatch on
`PriorType` does the same job at compile time.
"""
module Sampling

using Random: AbstractRNG
using ...SMC2FC_functional: PriorType, LogNormalPrior, NormalPrior,
                            VonMisesPrior, BetaPrior

export sample_from_prior, sample_from_prior_one

"""
    sample_from_prior_one(rng, prior) -> Float64

Draw a single sample in **unconstrained** space from `prior`.

# Arguments
- `rng::AbstractRNG`: random number generator.
- `prior::PriorType`: the prior (multiple-dispatched).

# Returns
- `Float64`: scalar in unconstrained space.

# Notes
- For `VonMisesPrior` we use a wide-normal proposal centred on the mean
    direction; rejection sampling is overkill at the seeding step.
- For `BetaPrior` the unconstrained space is the logit, so a standard
    normal seeding suffices.
"""
sample_from_prior_one(rng::AbstractRNG, p::LogNormalPrior) = p.μ + p.σ * randn(rng)
sample_from_prior_one(rng::AbstractRNG, p::NormalPrior)    = p.μ + p.σ * randn(rng)
sample_from_prior_one(rng::AbstractRNG, p::VonMisesPrior)  =
    p.μ + (1 / sqrt(max(p.κ, 1e-3))) * randn(rng)
sample_from_prior_one(rng::AbstractRNG, p::BetaPrior)      = randn(rng)

"""
    sample_from_prior(n_particles, priors, rng) -> Matrix{Float64}

Draw `n_particles` from the unconstrained-space prior.

# Arguments
- `n_particles::Integer`: number of particles to draw.
- `priors::Vector{<:PriorType}`: prior list, length `d_theta`.
- `rng::AbstractRNG`: random number generator.

# Returns
- `Matrix{Float64}`: `(n_particles, d_theta)` cloud — the layout
    AdvancedHMC.jl, the tempered-SMC reweight step, and
    `MassMatrix.estimate_mass_matrix` all expect.
"""
function sample_from_prior(n_particles::Integer,
                            priors::Vector{<:PriorType},
                            rng::AbstractRNG)
    d  = length(priors)
    P  = zeros(Float64, n_particles, d)
    @inbounds for j in 1:d, i in 1:n_particles
        P[i, j] = sample_from_prior_one(rng, priors[j])
    end
    return P
end

end # module Sampling
