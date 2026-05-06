# Top-level module for the scalar OU LQG test model. Mirrors the Python
# `version_1/models/scalar_ou_lqg/__init__.py` entry point.

module ScalarOULQG

include("_dynamics.jl")
include("simulation.jl")
include("kalman.jl")
include("lqr.jl")
include("estimation.jl")
include("control.jl")

using .Dynamics
using .Simulation
using .Kalman
using .LQR
using .Estimation
using .Control

export Dynamics, Simulation, Kalman, LQR, Estimation, Control
export build_open_loop_cost_fn, build_state_feedback_cost_fn
export lqr_riccati, lqr_optimal_cost,
        lqg_optimal_cost_mc, open_loop_zero_control_cost_mc
export PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A
export simulate_em, simulate_diffeq
export kalman_filter, kalman_log_likelihood
export build_estimation_model

# Re-export simulation constants for convenience
const PARAM_SET_A = Simulation.PARAM_SET_A
const INIT_STATE_A = Simulation.INIT_STATE_A
const EXOGENOUS_A = Simulation.EXOGENOUS_A

simulate_em(args...; kwargs...) = Simulation.simulate_em(args...; kwargs...)
simulate_diffeq(args...; kwargs...) = Simulation.simulate_diffeq(args...; kwargs...)
kalman_filter(args...; kwargs...) = Kalman.kalman_filter(args...; kwargs...)
kalman_log_likelihood(args...; kwargs...) = Kalman.kalman_log_likelihood(args...; kwargs...)
build_estimation_model(args...; kwargs...) = Estimation.build_estimation_model(args...; kwargs...)
build_open_loop_cost_fn(args...; kwargs...) = Control.build_open_loop_cost_fn(args...; kwargs...)
build_state_feedback_cost_fn(args...; kwargs...) = Control.build_state_feedback_cost_fn(args...; kwargs...)
lqr_riccati(args...; kwargs...) = LQR.lqr_riccati(args...; kwargs...)
lqr_optimal_cost(args...; kwargs...) = LQR.lqr_optimal_cost(args...; kwargs...)
lqg_optimal_cost_mc(args...; kwargs...) = LQR.lqg_optimal_cost_mc(args...; kwargs...)
open_loop_zero_control_cost_mc(args...; kwargs...) = LQR.open_loop_zero_control_cost_mc(args...; kwargs...)

end # module ScalarOULQG
