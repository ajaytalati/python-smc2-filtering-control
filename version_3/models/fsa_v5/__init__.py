"""FSA high-resolution model — v5 boundary re-exports.

This is the single import surface that the ``smc2fc`` repository (and any
external consumer) is expected to read from. Importing
``models.fsa_high_res`` exposes:

  * Truth parameter dicts             (TRUTH_PARAMS, TRUTH_PARAMS_V5)
  * Operating-point reference         (A_TYP, F_TYP)
  * Drift / diffusion (canonical)     (drift_jax, diffusion_state_dep)
  * SDE model objects                 (HIGH_RES_FSA_V4_MODEL,
                                        HIGH_RES_FSA_V5_MODEL)
  * Param sets + initial state        (DEFAULT_PARAMS, DEFAULT_PARAMS_V5,
                                        DEFAULT_INIT, BINS_PER_DAY,
                                        DT_BIN_DAYS)
  * Circadian regressors              (circadian, circadian_jax)
  * Estimation models                 (HIGH_RES_FSA_V5_ESTIMATION,
                                        HIGH_RES_FSA_V4_ESTIMATION)
  * Control spec builders             (build_control_spec — v4-style;
                                        build_control_spec_v5 — v5-style;
                                        evaluate_chance_constrained_cost
                                        — the v5 main novelty)
  * Plant simulator                   (StepwisePlant)

The v5 objects are the canonical defaults; the v4 ones are kept as thin
back-compat aliases. See ``LaTex_docs/sections/15_fim_analysis_v5.tex``
for the full parameter list, the maximum-pinning policy applied in the
v5 ``EstimationModel``, and §11.6 for the rationale.
"""

# Canonical dynamics + parameter dicts ─────────────────────────────────
from version_3.models.fsa_v5._dynamics import (
    TRUTH_PARAMS,
    TRUTH_PARAMS_V5,
    A_TYP, F_TYP,
    drift_jax,
    diffusion_state_dep,
)

# Forward simulator (SDEModel) + obs samplers + circadian + bin grid ──
from version_3.models.fsa_v5.simulation import (
    HIGH_RES_FSA_V4_MODEL,
    HIGH_RES_FSA_V5_MODEL,
    DEFAULT_PARAMS,
    DEFAULT_PARAMS_V5,
    DEFAULT_INIT,
    BINS_PER_DAY,
    DT_BIN_DAYS,
    circadian,
    circadian_jax,
)

# Estimation models (v5 default + v4 alias) ────────────────────────────
from version_3.models.fsa_v5.estimation import (
    HIGH_RES_FSA_V5_ESTIMATION,
    HIGH_RES_FSA_V4_ESTIMATION,
    PARAM_PRIOR_CONFIG,
    INIT_STATE_PRIOR_CONFIG,
)

# Control: gradient-OT spec + v5 chance-constraint cost ────────────────
from version_3.models.fsa_v5.control import (
    build_control_spec,
    build_control_spec_v5,
)
from version_3.models.fsa_v5.control_v5 import (
    evaluate_chance_constrained_cost,        # back-compat alias = hard
    evaluate_chance_constrained_cost_hard,   # for pure-SMC² importance weighting
    evaluate_chance_constrained_cost_soft,   # for HMC with sigmoid surrogate
    find_A_sep_v5,                           # legacy NumPy/Brent (debug)
)

# Closed-loop plant simulator (6D v5) ──────────────────────────────────
from version_3.models.fsa_v5._plant import StepwisePlant


__all__ = [
    # dynamics
    'TRUTH_PARAMS', 'TRUTH_PARAMS_V5', 'A_TYP', 'F_TYP',
    'drift_jax', 'diffusion_state_dep',
    # simulation
    'HIGH_RES_FSA_V4_MODEL', 'HIGH_RES_FSA_V5_MODEL',
    'DEFAULT_PARAMS', 'DEFAULT_PARAMS_V5', 'DEFAULT_INIT',
    'BINS_PER_DAY', 'DT_BIN_DAYS',
    'circadian', 'circadian_jax',
    # estimation
    'HIGH_RES_FSA_V5_ESTIMATION', 'HIGH_RES_FSA_V4_ESTIMATION',
    'PARAM_PRIOR_CONFIG', 'INIT_STATE_PRIOR_CONFIG',
    # control
    'build_control_spec', 'build_control_spec_v5',
    'evaluate_chance_constrained_cost',
    'evaluate_chance_constrained_cost_hard',
    'evaluate_chance_constrained_cost_soft',
    'find_A_sep_v5',
    # plant
    'StepwisePlant',
]
