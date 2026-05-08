"""
    Control/Calibration.jl

Generic calibration helpers for the controller side. Port of
`smc2fc/control/calibration.py`.

Both helpers are pure functions: they take immutable inputs and return
fresh outputs (a tuple, a `Dict`).
"""
module Calibration

using Random: AbstractRNG, MersenneTwister
using Statistics: mean, std

export calibrate_beta_max, build_crn_noise_grids

"""
    calibrate_beta_max(cost_fn;
                        theta_dim, sigma_prior,
                        prior_mean=0.0, n_samples=256,
                        target_nats=8.0, seed=0)
        -> (β_max, prior_cost_mean, prior_cost_std)

Auto-calibrate `β_max` so the prior-cloud cost spread maps to
`target_nats` of tempering nats (default 8 ≈ 16 effective tempering levels).

# Arguments
- `cost_fn`: `θ -> Float64`. Evaluated per-row of a θ sample matrix.

# Keyword arguments
- `theta_dim::Integer`: dimension of θ.
- `sigma_prior::Real`: σ in the `N(prior_mean, σ²·I)` prior.
- `prior_mean::Union{Real,AbstractVector{<:Real}} = 0.0`: scalar or vector
    prior mean.
- `n_samples::Integer = 256`: number of prior draws.
- `target_nats::Real = 8.0`: target tempering nats.
- `seed::Integer = 0`: PRNG seed.

# Returns
- `(β_max::Float64, prior_cost_mean::Float64, prior_cost_std::Float64)`.

# Throws
- `AssertionError`: if `prior_mean` is a vector and its length disagrees
    with `theta_dim`.

# Notes
- `β_max = target_nats / std(costs)`, with the std floored at `1e-6` to
    prevent division-by-zero on degenerate cost landscapes.
"""
function calibrate_beta_max(cost_fn;
                              theta_dim::Integer,
                              sigma_prior::Real,
                              prior_mean::Union{Real,AbstractVector{<:Real}} = 0.0,
                              n_samples::Integer = 256,
                              target_nats::Real = 8.0,
                              seed::Integer = 0)
    rng = MersenneTwister(seed)
    μ_vec = prior_mean isa Real ?
        fill(Float64(prior_mean), theta_dim) :
        Float64.(prior_mean)
    @assert length(μ_vec) == theta_dim "prior_mean length mismatch"

    samples = reshape(μ_vec, 1, :) .+
              Float64(sigma_prior) .* randn(rng, n_samples, theta_dim)

    costs = [cost_fn(@view samples[i, :]) for i in 1:n_samples]

    cost_mean = mean(costs)
    cost_std  = std(costs)
    β_max     = Float64(target_nats) / max(cost_std, 1e-6)
    return (β_max, cost_mean, cost_std)
end


"""
    build_crn_noise_grids(; n_inner, n_steps, n_channels=1, seed=0) -> Dict

Build fixed Gaussian noise arrays for common-random-numbers cost
evaluation. Sharing noise across all SMC² particles makes cost
differences reflect θ-differences only — a standard variance-reduction
trick.

# Keyword arguments
- `n_inner::Integer`: inner Monte Carlo samples per cost evaluation.
- `n_steps::Integer`: time-grid length.
- `n_channels::Integer = 1`: number of independent Wiener channels.
- `seed::Integer = 0`: PRNG seed.

# Returns
- `Dict{Symbol,Array}`:
    - `:wiener => Array{Float64,3}` of shape `(n_inner, n_steps, n_channels)`.
    - `:initial => Vector{Float64}` of length `n_inner`.
"""
function build_crn_noise_grids(; n_inner::Integer,
                                  n_steps::Integer,
                                  n_channels::Integer = 1,
                                  seed::Integer = 0)
    rng = MersenneTwister(seed)
    wiener  = randn(rng, n_inner, n_steps, n_channels)
    initial = randn(rng, n_inner)
    return Dict(:wiener => wiener, :initial => initial)
end

end # module Calibration
