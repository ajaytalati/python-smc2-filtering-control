"""
    Control/Spec.jl

`ControlSpec` — model-side contract for SMC²-as-controller. Direct port
of `smc2fc/control/control_spec.py`.

The Python `@dataclass` becomes a parametric Julia struct; function-typed
fields use parametric function types so the JIT specialises on the
closure type and inlines the call (no boxing penalty in the cost rollout).
"""
module Spec

using ...SMC2FC_functional: PriorType

export ControlSpec

"""
    ControlSpec{Fcost,Fsched,Fplot}

Complete specification for an SMC²-as-controller task. The control-side
analogue of `EstimationModel` for the filter side.

# Fields
- `name::String`, `version::String`: identifiers.
- `dt::Float64`, `n_steps::Int`, `n_substeps::Int = 1`: time grid.
- `initial_state::Vector{Float64}`: initial latent state.
- `truth_params::Dict{Symbol,Float64}`: ground-truth parameters (used by
    the cost closure; identification is upstream of this struct).
- `theta_dim::Int`: dimension of the controller's search vector.
- `sigma_prior::Float64 = 1.5`: standard deviation of the
    `θ ~ N(prior_mean, σ²·I)` prior in unconstrained space.
- `prior_mean::Vector{Float64} = []`: empty broadcasts `0.0`.
- `cost_fn::Fcost`: `θ -> scalar` mean cost. Captures CRN noise grids,
    dynamics, and cost coefficients in its closure.
- `schedule_from_theta::Fsched`: `θ -> Vector(n_steps)`. Same closure
    as `cost_fn` but exposes the raw schedule for diagnostic plots.
- `acceptance_gates::Dict{Symbol,Function}`: gate predicates for charter
    §15.5 acceptance plots.
- `diagnostic_plot_fn::Fplot`: optional plotting callable.

# Notes
- The wide-prior default (`sigma_prior ≈ 1.5`) keeps the prior-cost cloud
    informative for `Calibration.calibrate_beta_max`.
"""
Base.@kwdef struct ControlSpec{Fcost,Fsched,Fplot}
    name::String
    version::String

    dt::Float64
    n_steps::Int
    n_substeps::Int = 1

    initial_state::Vector{Float64}
    truth_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}()

    theta_dim::Int
    sigma_prior::Float64 = 1.5
    prior_mean::Vector{Float64} = Float64[]

    cost_fn::Fcost
    schedule_from_theta::Fsched

    acceptance_gates::Dict{Symbol,Function} = Dict{Symbol,Function}()
    diagnostic_plot_fn::Fplot = nothing
end

end # module Spec
