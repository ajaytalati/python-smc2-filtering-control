# Control/Calibration.jl — generic calibration helpers.
#
# Port of `smc2fc/control/calibration.py`:
#   - calibrate_beta_max: auto-set β_max from prior-cloud cost spread.
#   - build_crn_noise_grids: fixed Wiener-increment + initial-condition
#     arrays for common-random-numbers cost evaluation.

module Calibration

using Random: AbstractRNG, MersenneTwister
using Statistics: mean, std

export calibrate_beta_max, build_crn_noise_grids

"""
    calibrate_beta_max(cost_fn; theta_dim, sigma_prior,
                        prior_mean=0.0, n_samples=256, target_nats=8.0,
                        seed=0) -> (β_max, prior_cost_mean, prior_cost_std)

Auto-calibrate `β_max` so the prior-cloud cost spread maps to
`target_nats` of tempering nats (default 8 ≈ 16 effective tempering levels).

Sample `n_samples` θ from `N(prior_mean, sigma_prior² · I_d)`, evaluate
`cost_fn` on each, then set `β_max = target_nats / std(costs)`.
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

    samples = reshape(μ_vec, 1, :) .+ Float64(sigma_prior) .* randn(rng, n_samples, theta_dim)

    # cost_fn evaluated per row. Each call returns a scalar.
    costs = [cost_fn(@view samples[i, :]) for i in 1:n_samples]

    cost_mean = mean(costs)
    cost_std  = std(costs)
    β_max     = Float64(target_nats) / max(cost_std, 1e-6)
    return (β_max, cost_mean, cost_std)
end


"""
    build_crn_noise_grids(; n_inner, n_steps, n_channels=1, seed=0)
        -> Dict(:wiener => (n_inner, n_steps, n_channels),
                :initial => (n_inner,))

Fixed Gaussian noise arrays for common-random-numbers cost evaluation.
Same noise across all SMC² particles → cost differences reflect θ
differences, not noise differences (variance reduction).
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
