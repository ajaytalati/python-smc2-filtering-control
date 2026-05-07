# Top-level module for the FSA-v2 (high-res, Banister-coupled) Julia port.
# Mirror of `version_2/models/fsa_high_res/__init__.py`.
#
# Submodules (in load order — _phi_burst before simulation/plant because both
# depend on its BINS_PER_DAY constant):
#   1. Dynamics         — drift + state-dependent diffusion (G1-reparametrized)
#   2. PhiBurst         — sub-daily Φ-envelope expansion
#   3. Simulation       — DEFAULT_PARAMS, INIT_STATE, EXOGENOUS, 4 obs samplers
#   4. Plant            — StepwisePlant (closed-loop simulator)
#   5. Control          — RBF schedule + cost functional
#   6. Estimation       — priors + locally-guided propagate_fn (Pitt-Shephard)

module FSAHighRes

include("_dynamics.jl")
include("_phi_burst.jl")
include("simulation.jl")
include("_plant.jl")
include("cpu_control.jl")
include("estimation.jl")
include("gpu_pf.jl")
include("gpu_control.jl")

using .Dynamics
using .PhiBurst
using .Simulation
using .Plant
using .Control
using .Estimation
using .GPUPF
using .GPUControl

# ── Re-exports — mirror Python's `from fsa_high_res import *` ────────────

# Top-level constants
export A_TYP, F_TYP, PHI_TYP, BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS
export TRUTH_PARAMS, DEFAULT_PARAMS, INIT_STATE, EXOGENOUS

# Dynamics
export drift, drift_indexed, diffusion_state_dep, em_step_substepped

# PhiBurst
export build_per_day_envelope, expand_daily_phi_to_subdaily, sleep_mask_from_hours

# Simulation
export circadian, simulate_em
export gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps

# Plant
export StepwisePlant, advance!, finalise

# Control
export build_rbf, schedule_from_theta_fsa, ControlBundle, build_control

# Estimation
export PARAM_NAMES, _PI, PARAM_PRIOR_CONFIG, INIT_STATE_PRIOR_CONFIG, COLD_START_INIT
export propagate_fn, diffusion_fn, obs_log_weight_fn, obs_log_prob_fn, align_obs_fn
export forward_sde_stochastic, get_init_theta, shard_init_fn
export build_estimation_model

# Re-export the framework's TRUTH_PARAMS under the canonical name.
const TRUTH_PARAMS  = Dynamics.TRUTH_PARAMS
const DEFAULT_PARAMS = Simulation.DEFAULT_PARAMS
const INIT_STATE    = Simulation.INIT_STATE
const EXOGENOUS     = Simulation.EXOGENOUS

end # module FSAHighRes
