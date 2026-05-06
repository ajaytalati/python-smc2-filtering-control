# SMC2/Sampling.jl — initial particle sampling from the prior in
# unconstrained space.
#
# Port of `smc2fc/core/sampling.py`. The Python version uses indicator-array
# branching (`is_ln`, `is_norm`) because JAX cannot dispatch on prior type.
# The Julia version uses multiple dispatch on `PriorType`, the same way
# Transforms.jl does.

module Sampling

using Random: AbstractRNG
using ...SMC2FC: PriorType, LogNormalPrior, NormalPrior, VonMisesPrior, BetaPrior

export sample_from_prior, sample_from_prior_one

# Per-prior sampling in unconstrained space.
sample_from_prior_one(rng::AbstractRNG, p::LogNormalPrior) = p.μ + p.σ * randn(rng)
sample_from_prior_one(rng::AbstractRNG, p::NormalPrior)    = p.μ + p.σ * randn(rng)
# vonmises in unconstrained space ≡ identity domain ⇒ use a wide normal
# proposal centred on the mean direction. Rejection sampling at runtime is
# overkill for the seeding step.
sample_from_prior_one(rng::AbstractRNG, p::VonMisesPrior)  = p.μ + (1 / sqrt(max(p.κ, 1e-3))) * randn(rng)
# Beta in unconstrained (logit) space ⇒ standard normal seeding is fine.
sample_from_prior_one(rng::AbstractRNG, p::BetaPrior)      = randn(rng)

"""
    sample_from_prior(n_particles::Integer, priors::Vector{<:PriorType}, rng)

Draw `n_particles` from the unconstrained-space prior. Returns a
`(n_particles, d_theta)` matrix — the layout `AdvancedHMC.jl`,
`tempered_smc`'s reweight step, and `MassMatrix.estimate_mass_matrix`
all expect.
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
