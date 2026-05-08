# FSA v1.5 model aggregator — purely functional bridge between v1
# (open-loop only) and v2 (closed-loop with multi-channel obs).
#
# Submodules in load order:
#   1. Dynamics    — v1's G0 SDE (verbatim)
#   2. Simulation  — BINS_PER_DAY, params, pure Gaussian obs sampler
#   3. Plant       — pure plant_step / plant_rollout (no mutable struct)
#   4. GPUControl  — v1's GPU control kernel (verbatim, only control file)
#   5. Estimation  — pure propagate / obs_log_weight + 14 priors
#   6. (gpu_pf — to be added AFTER FIM gate passes)

module FSAHighRes

include("_dynamics.jl")
include("simulation.jl")
include("_plant.jl")
include("gpu_control.jl")
include("estimation.jl")
include("gpu_pf.jl")

using .Dynamics
using .Simulation
using .Plant
using .GPUControl
using .Estimation
using .GPUPF


# ── Re-exports ────────────────────────────────────────────────────────────

# Dynamics
export TRUTH_PARAMS, drift, diffusion_state_dep, em_step_substepped

# Simulation
export BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE, PINNED_PARAMS,
       sample_obs_bfa, params_v15_to_v1_nt, fill_pinned_nt

# Plant — purely functional
export PlantState, plant_step, plant_rollout, init_plant_state

# GPU controller (v1's, verbatim)
export FSAControlGPUTarget, gpu_cost_log_density_batched, make_log_density_fn

# Estimation
export PARAM_NAMES, PARAM_PRIOR_CONFIG, propagate, obs_log_weight

# GPU PF — pure facade
export FSAGPUTarget, gpu_log_density, gpu_grads, parallel_hmc_one_move

end # module FSAHighRes
