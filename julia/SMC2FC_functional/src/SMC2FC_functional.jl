"""
    SMC2FC_functional

A parallel Julia rewrite of `julia/SMC2FC/` with a **functional API
surface** and Google-style docstrings. The two libraries are designed to
run side-by-side in the same Julia session so the new implementation can
be checked against the existing port and the Python reference.

# Layout

- `Types.jl`, `Config.jl`, `EstimationModel.jl`, `Transforms.jl` —
    foundations.
- `Filtering/{Kernels, OT, Bootstrap, GPUSegmentedPF}.jl` — inner PF.
- `SMC2/{Tempering, Sampling, MassMatrix, HMC, Bridge, TemperedSMC}.jl`
    — outer SMC² + warm-start bridge.
- `Control/{Spec, RBFSchedule, Calibration, TemperedSMC, GPUControlSMC}.jl`
    — control-as-inference.
- `Simulator/{SDEModel, Observations}.jl` — forward simulation.

# Functional contract

- Every public function takes immutable inputs and returns immutable
    outputs.
- Pre-allocated buffers, where needed, live behind an *immutable*
    `Workspace` container of mutable arrays (see `BootstrapWorkspace`).
- No `!`-mutating function is exposed in the public API. Internal `!`
    helpers stay module-private.
- GPU kernels (`GPUSegmentedPF`, `GPUControlSMC`) are inherently
    imperative; their wrappers are functional.

See `claude_plans/Surgical_Julian_rewrite_of_Julia_SMC2FC_port_2026-05-08_1406.md`
for the design plan.
"""
module SMC2FC_functional

# Phase 1 — Foundations.
include("Types.jl")
include("Config.jl")
include("Transforms.jl")
include("EstimationModel.jl")

# Phase 2 — Filtering
include("Filtering/Kernels.jl")
include("Filtering/OT.jl")
include("Filtering/Bootstrap.jl")
include("Filtering/GPUSegmentedPF.jl")
using .Kernels
using .OT
using .Bootstrap
using .GPUSegmentedPF

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
include("Control/GPUControlSMC.jl")
using .RBFSchedule
using .Spec
using .Calibration
using .ControlLoop
using .GPUControlSMC

# Phase 5 — Plant + Simulator
include("Simulator/SDEModel.jl")
include("Simulator/Observations.jl")
using .SDEModelWrap
using .Observations

# ── Re-exports — public API ──────────────────────────────────────────────────

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
export constrained_to_unconstrained, unconstrained_to_constrained
export build_priors, split_theta

# EstimationModel
export EstimationModel, n_params, n_init_states, n_dim, all_names, all_priors

# Filtering
export compute_ess, silverman_bandwidth, log_kernel_matrix, ess_bandwidth_factor
export smooth_resample_basic, smooth_resample_ess_scaled
export smooth_resample, smooth_resample_ess_scaled_lw
export compute_kernel_factor, factor_matvec, factor_matvec_batch
export sinkhorn_scalings, barycentric_projection
export ot_resample_lr, ot_blended_resample
export bootstrap_log_likelihood, BootstrapWorkspace

# SMC2
export estimate_mass_matrix
export sample_from_prior, sample_from_prior_one
export ess_at_delta, solve_delta_for_ess
export hmc_step_chain, build_target
export bridge_init, bridge_kind, fit_gaussian, sample_from_gaussian
export run_smc_window, run_smc_window_bridge, TemperedSMCResult

# Control
export RBFOutput, IdentityOutput, SoftplusOutput, SigmoidOutput
export RBFBasis, design_matrix, schedule_from_theta
export ControlSpec
export calibrate_beta_max, build_crn_noise_grids
export run_tempered_smc_loop, ControlResult

# Simulator
export simulate_sde, build_sde_problem
export ObsChannel, generate_all_channels

end # module SMC2FC_functional
