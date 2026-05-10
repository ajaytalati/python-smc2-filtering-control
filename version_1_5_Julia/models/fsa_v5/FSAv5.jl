# FSA v5 model — Julia port aggregator.
#
# Parallel to `version_1_5_Julia/models/fsa_high_res/FSAHighRes.jl` but
# for the v5 surface. Loading this file brings every public v5 module
# into scope for downstream consumers (bench drivers, the differential
# test, future GPU-side code).
#
# Each submodule is a thin wrapper around the Lean reference at
# `version_1_5_LEAN/Fsa/V5/`; bit-equivalence is enforced by
# `version_1_5_Julia/diff_test/test_lean_diff_v5.jl`.

module FSAv5

# Order matters: simulation_v5 publishes constants used by the
# pure-math modules; everything else depends on it. GPU modules come
# last because they pull in CUDA / KernelAbstractions / SMC2FC_functional
# and are heavier to load — keep load-time of the CPU surface fast.
include("simulation_v5.jl")
include("_dynamics_v5.jl")
include("obs_v5.jl")
include("_plant_v5.jl")
include("estimation_v5.jl")
include("cost_v5.jl")
include("schedule_v5.jl")
include("gpu_pf_v5.jl")
include("gpu_control_v5.jl")

using .SimulationV5
using .DynamicsV5
using .ObsV5
using .PlantV5
using .EstimationV5
using .CostV5
using .ScheduleV5
using .GPUPFv5
using .GPUControlV5

# Re-export the public surface for `using FSAv5` ergonomics.

export BINS_PER_DAY, DT_BIN_DAYS
export A_TYP, F_TYP
export TRUTH_PARAMS_V4, TRUTH_PARAMS_V5
export DEFAULT_OBS_PARAMS_V5
export DEFAULT_INIT, TRAINED_ATHLETE_INIT
export FROZEN_PARAMS_V5
export PARAM_KEYS_V5, OBS_PARAM_KEYS_V5
export sample_obs_v5, params_dict_to_nt

export drift_v5, diffusion_v5, em_step_v5

export hr_mean, sleep_prob, stress_mean, steps_log_mean, volume_load_mean

export PlantState6D, plant_step_v5, plant_rollout_v5
export init_plant_state_sedentary, init_plant_state_trained

export PARAM_NAMES_V5, PARAM_PRIOR_CONFIG_V5
export obs_log_weight_v5, propagate_v5

export mu_bar, find_a_sep, a_sep_grid

export sigmoid, c_phi, schedule_from_theta, design_matrix

# GPU surface (Phase 4)
export FSAv5GPUTarget, gpu_log_density_v5, gpu_propagate_one!
export FSAv5ControlGPUTarget, gpu_cost_log_density_batched_v5
export make_log_density_fn_v5, build_rbf_design_v5, gpu_cost_one!

end # module FSAv5
