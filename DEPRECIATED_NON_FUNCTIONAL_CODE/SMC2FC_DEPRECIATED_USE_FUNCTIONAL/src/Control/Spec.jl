# Control/Spec.jl — ControlSpec: model-side contract for SMC²-as-controller.
#
# Direct port of `smc2fc/control/control_spec.py`. The Python `@dataclass`
# becomes a parametric Julia struct; function-typed fields use parametric
# function types so the JIT specialises on the closure type and inlines
# the call (no boxing penalty in the cost rollout).
#
# Charter §15.5.

module Spec

using ...SMC2FC: PriorType

export ControlSpec

"""
    ControlSpec(name, version, dt, n_steps;
                 initial_state, truth_params, theta_dim,
                 cost_fn, schedule_from_theta,
                 sigma_prior=1.5, prior_mean=0.0,
                 n_substeps=1,
                 acceptance_gates=Dict(),
                 diagnostic_plot_fn=nothing)

Complete specification for an SMC²-as-controller task. Mirrors
`smc2fc.estimation_model.EstimationModel` for the filter side.

Required:
  - `cost_fn::Fcost` — `θ -> scalar mean cost`. Built by the model's
    `build_control_spec()` helper. Captures CRN noise grids, dynamics,
    and cost coefficients in its closure.
  - `schedule_from_theta::Fsched` — `θ -> (n_steps,) Vector`. Same closure
    as `cost_fn` but exposes the raw schedule for diagnostic plots.

The search-space prior is Gaussian:
    `θ ~ N(prior_mean, sigma_prior² · I)`
in unconstrained space. Wide priors (`sigma_prior ≈ 1.5`) keep the
prior-cost cloud informative for `calibrate_beta_max`.
"""
Base.@kwdef struct ControlSpec{Fcost,Fsched,Fplot}
    name::String
    version::String

    # Time grid
    dt::Float64
    n_steps::Int
    n_substeps::Int = 1

    # Initial conditions + truth
    initial_state::Vector{Float64}
    truth_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()

    # Search space
    theta_dim::Int
    sigma_prior::Float64 = 1.5
    prior_mean::Vector{Float64} = Float64[]   # empty → broadcast 0.0

    # Core API
    cost_fn::Fcost
    schedule_from_theta::Fsched

    # Diagnostics (charter §15.5: gates + optional plot fn)
    acceptance_gates::Dict{Symbol,Function} = Dict{Symbol,Function}()
    diagnostic_plot_fn::Fplot = nothing
end

end # module Spec
