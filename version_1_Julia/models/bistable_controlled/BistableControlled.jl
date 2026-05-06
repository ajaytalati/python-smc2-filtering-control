# Top-level module for the controlled bistable model.

module BistableControlled

include("_dynamics.jl")
include("simulation.jl")
include("estimation.jl")
include("control.jl")

using .Dynamics
using .Simulation
using .Estimation
using .Control

export Dynamics, Simulation, Estimation, Control
export PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A
export simulate_em, simulate_diffeq
export build_estimation_model
export build_basin_cost_fn

const PARAM_SET_A   = Simulation.PARAM_SET_A
const INIT_STATE_A  = Simulation.INIT_STATE_A
const EXOGENOUS_A   = Simulation.EXOGENOUS_A

simulate_em(args...; kwargs...) = Simulation.simulate_em(args...; kwargs...)
simulate_diffeq(args...; kwargs...) = Simulation.simulate_diffeq(args...; kwargs...)
build_estimation_model(args...; kwargs...) = Estimation.build_estimation_model(args...; kwargs...)
build_basin_cost_fn(args...; kwargs...) = Control.build_basin_cost_fn(args...; kwargs...)

end # module BistableControlled
