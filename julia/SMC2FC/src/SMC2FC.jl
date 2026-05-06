module SMC2FC

# Phase 1 — Foundations.
# Subsequent phases extend this module with Filtering, SMC2, Control, Simulator.
# Charter: LaTex_docs/julia_port_charter.pdf §15.2.

include("Types.jl")
include("Config.jl")
include("Transforms.jl")
include("EstimationModel.jl")

# Phase 2 — Filtering
include("Filtering/Kernels.jl")
include("Filtering/OT.jl")
include("Filtering/Bootstrap.jl")
using .Kernels
using .OT
using .Bootstrap

# Phase 3 — Outer SMC²
include("SMC2/MassMatrix.jl")
include("SMC2/Sampling.jl")
include("SMC2/Tempering.jl")
include("SMC2/HMC.jl")
include("SMC2/Bridge.jl")
include("SMC2/TemperedSMC.jl")
using .MassMatrix
using .Sampling
using .Tempering
using .HMC
using .Bridge
using .TemperedSMC

# Phase 4 — Control
include("Control/RBFSchedule.jl")
include("Control/Spec.jl")
include("Control/Calibration.jl")
include("Control/TemperedSMC.jl")
using .RBFSchedule
using .Spec
using .Calibration
using .ControlLoop

# Phase 5 — Plant + Simulator
include("Simulator/SDEModel.jl")
include("Simulator/Observations.jl")
using .SDEModelWrap
using .Observations

# Re-exports — public API.
# Types
export State, DynParams, Particle, ParticleCloud
export GPUFilterState, CPUParameterCloud
export BridgeKind, GaussianBridge, SchrodingerFollmerBridge
export ChanceConstraintMode, SoftSurrogate, HardIndicator
export AbstractBackend, CPUBackend, CUDABackend
export SFQ1Mode, SFQ1ImportanceSampling, SFQ1AnnealedSMC

# Config
export SMCConfig, RollingConfig, MissingDataConfig

# Transforms
export PriorType, LogNormalPrior, NormalPrior, VonMisesPrior, BetaPrior
export to_unconstrained, to_constrained, log_prior_unconstrained
export build_priors, split_theta

# EstimationModel interface
export EstimationModel, n_params, n_init_states, n_dim, all_names, all_priors

# Filtering — Phase 2
export compute_ess, silverman_bandwidth, log_kernel_matrix, ess_bandwidth_factor
export smooth_resample_basic, smooth_resample_ess_scaled
export smooth_resample, smooth_resample_ess_scaled_lw
export compute_kernel_factor, factor_matvec, factor_matvec_batch
export sinkhorn_scalings, barycentric_projection
export ot_resample_lr, ot_blended_resample
export bootstrap_log_likelihood, BootstrapBuffers

# SMC2 — Phase 3
export estimate_mass_matrix
export sample_from_prior, sample_from_prior_one
export ess_at_delta, solve_delta_for_ess
export hmc_step_chain, build_target
export bridge_init, bridge_kind, fit_gaussian, sample_from_gaussian
export run_smc_window, run_smc_window_bridge, TemperedSMCResult

# Control — Phase 4
export RBFOutput, IdentityOutput, SoftplusOutput, SigmoidOutput
export RBFBasis, design_matrix, schedule_from_theta
export ControlSpec
export calibrate_beta_max, build_crn_noise_grids
export run_tempered_smc_loop, ControlResult

# Simulator — Phase 5
export simulate_sde, build_sde_problem
export ObsChannel, generate_all_channels

end # module SMC2FC
