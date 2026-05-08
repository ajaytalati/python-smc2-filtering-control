# Top-level module for the FSA-v2 (high-res, Banister-coupled) model.
# Mirror of `version_1/models/fsa_high_res/__init__.py`.

module FSAHighRes

include("_dynamics.jl")
include("simulation.jl")
include("control.jl")
include("gpu_control.jl")

using .Dynamics
using .Simulation
using .Control
using .GPUControl

export Dynamics, Simulation, Control, GPUControl
export TRUTH_PARAMS, INIT_STATE, EXOGENOUS
export simulate_em
export build_control, ControlBundle, schedule_from_theta_fsa
export FSAv1ControlGPUTarget, gpu_cost_log_density_batched, make_log_density_fn

const TRUTH_PARAMS = Dynamics.TRUTH_PARAMS
const INIT_STATE   = Simulation.INIT_STATE
const EXOGENOUS    = Simulation.EXOGENOUS

simulate_em(args...; kwargs...)            = Simulation.simulate_em(args...; kwargs...)
build_control(args...; kwargs...)          = Control.build_control(args...; kwargs...)
schedule_from_theta_fsa(args...; kwargs...) = Control.schedule_from_theta_fsa(args...; kwargs...)

end # module FSAHighRes
