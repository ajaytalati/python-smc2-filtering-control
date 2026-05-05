"""
version_4/models/fsa_high_res/simulation.py — FSA-v4 (Variable Dose).
=========================================================================

Extension of the FSA-v3 model to include dynamic fatigue gains (Busso 2003).
  - **6D Latent State**: [B, S, F, A, KFB, KFS]
  - **2D Control Input**: [Phi_B, Phi_S]
  - **Unified Fatigue**: dF = (KFB·Phi_B + KFS·Phi_S - ...) dt
  - **Variable Dose**: Sensitivity KFi increases with training.
"""

import math
import numpy as np

from smc2fc.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_STATE,
)

from version_3.models.fsa_v5._dynamics import A_TYP, F_TYP


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4
EPS_S_FROZEN = 1.0e-4

import os as _os
_STEP_MIN = int(_os.environ.get('FSA_STEP_MINUTES', '15'))
if (60 * 24) % _STEP_MIN != 0:
    raise ValueError(f"FSA_STEP_MINUTES={_STEP_MIN} must divide 1440")
BINS_PER_DAY = (60 * 24) // _STEP_MIN
DT_BIN_DAYS = 1.0 / BINS_PER_DAY
DT_BIN_HOURS = 24.0 / BINS_PER_DAY


# =========================================================================
# Circadian forcing
# =========================================================================

def circadian(t_days, phi=0.0):
    return np.cos(2.0 * np.pi * t_days + phi)


def circadian_jax(t_days, phi=0.0):
    import jax.numpy as jnp
    return jnp.cos(2.0 * jnp.pi * t_days + phi)


# =========================================================================
# DRIFT — v4 Variable Dose (6D state)
# =========================================================================

def _bin_lookup_2d(t_days, array_2d, dt_bin_days=DT_BIN_DAYS):
    k = int(t_days / dt_bin_days)
    k = max(0, min(k, array_2d.shape[0] - 1))
    return array_2d[k]


def drift(t, y, params, aux):
    """Numpy drift for FSA-v5 (6D state, Hill deconditioning, recovers v4
    when ``params['mu_dec_B'] = params['mu_dec_S'] = 0``).

    KEEP IN SYNC with ``models.fsa_high_res._dynamics.drift_jax`` — that
    JAX function is the single source of truth, and this numpy mirror
    exists only because some legacy paths (the scipy ODE solver via
    ``simulator.sde_solver_diffrax`` and the StateSpec deterministic
    fallback) need a numpy-callable drift.

    Args:
        t: float, time in days (used to look up Phi at the current bin).
        y: shape (6,) state ``[B, S, F, A, K_FB, K_FS]``.
        params: dict of scalar parameters; must include v5 Hill keys
            (``B_dec``, ``S_dec``, ``mu_dec_B``, ``mu_dec_S``, ``n_dec``).
        aux: tuple ``(Phi_arr,)`` — per-bin stimulus schedule, shape
            (n_bins, 2).

    Returns:
        shape (6,) numpy array of time-derivatives.
    """
    (Phi_arr,) = aux
    p = params
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]

    Phi_t = _bin_lookup_2d(t, Phi_arr)
    Phi_B, Phi_S = Phi_t[0], Phi_t[1]

    F_dev = F - F_TYP
    # FSA-v5 Hill deconditioning — penalise low chronic capacity.
    # See LaTeX §10.2, equation (eq:v5-mubar). Defaults to 0 in v4.
    n   = p['n_dec']
    Bn  = max(B, 0.0) ** n
    Sn  = max(S, 0.0) ** n
    Bdn = p['B_dec'] ** n
    Sdn = p['S_dec'] ** n
    dec_B = p['mu_dec_B'] * Bdn / (Bn + Bdn)
    dec_S = p['mu_dec_S'] * Sdn / (Sn + Sdn)
    mu = (p['mu_0'] + p['mu_B'] * B + p['mu_S'] * S
          - p['mu_F'] * F - p['mu_FF'] * F_dev * F_dev
          - dec_B - dec_S)

    a_factor_B = (1.0 + p['epsilon_AB'] * A) / (1.0 + p['epsilon_AB'] * A_TYP)
    dB = p['kappa_B'] * a_factor_B * Phi_B - B / p['tau_B']

    a_factor_S = (1.0 + p['epsilon_AS'] * A) / (1.0 + p['epsilon_AS'] * A_TYP)
    dS = p['kappa_S'] * a_factor_S * Phi_S - S / p['tau_S']

    a_factor_F = (1.0 + p['lambda_A'] * A) / (1.0 + p['lambda_A'] * A_TYP)
    dF = (KFB * Phi_B + KFS * Phi_S - a_factor_F / p['tau_F'] * F)

    dA = mu * A - p['eta'] * A * A * A

    dKFB = (p['KFB_0'] - KFB) / p['tau_K'] + p['mu_K'] * Phi_B
    dKFS = (p['KFS_0'] - KFS) / p['tau_K'] + p['mu_K'] * Phi_S

    return np.array([dB, dS, dF, dA, dKFB, dKFS])


def drift_jax(t, y, args):
    """JAX drift for FSA-v5 (6D state, Hill deconditioning).

    Inline mirror of ``models.fsa_high_res._dynamics.drift_jax`` — same
    equations, but adapted for the SDEModel calling convention here
    (which passes ``args = (params_dict, Phi_arr)`` and a scalar time
    ``t`` instead of a pre-looked-up Phi vector).

    Maps to LaTeX §11.1.
    """
    import jax.numpy as jnp
    p, Phi_arr = args
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]

    k = jnp.clip((t / DT_BIN_DAYS).astype(jnp.int32), 0, Phi_arr.shape[0] - 1)
    Phi_t = Phi_arr[k]
    Phi_B, Phi_S = Phi_t[0], Phi_t[1]

    F_dev = F - F_TYP
    # FSA-v5 Hill deconditioning subtraction. Identical structure to
    # _dynamics.drift_jax (lines marked "v5 Hill" in that file).
    n   = p['n_dec']
    Bn  = jnp.power(jnp.maximum(B, 0.0), n)
    Sn  = jnp.power(jnp.maximum(S, 0.0), n)
    Bdn = jnp.power(p['B_dec'], n)
    Sdn = jnp.power(p['S_dec'], n)
    dec_B = p['mu_dec_B'] * Bdn / (Bn + Bdn)
    dec_S = p['mu_dec_S'] * Sdn / (Sn + Sdn)
    mu = (p['mu_0'] + p['mu_B'] * B + p['mu_S'] * S
          - p['mu_F'] * F - p['mu_FF'] * F_dev * F_dev
          - dec_B - dec_S)

    a_factor_B = (1.0 + p['epsilon_AB'] * A) / (1.0 + p['epsilon_AB'] * A_TYP)
    dB = p['kappa_B'] * a_factor_B * Phi_B - B / p['tau_B']

    a_factor_S = (1.0 + p['epsilon_AS'] * A) / (1.0 + p['epsilon_AS'] * A_TYP)
    dS = p['kappa_S'] * a_factor_S * Phi_S - S / p['tau_S']

    a_factor_F = (1.0 + p['lambda_A'] * A) / (1.0 + p['lambda_A'] * A_TYP)
    dF = (KFB * Phi_B + KFS * Phi_S - a_factor_F / p['tau_F'] * F)

    dA = mu * A - p['eta'] * A * A * A

    dKFB = (p['KFB_0'] - KFB) / p['tau_K'] + p['mu_K'] * Phi_B
    dKFS = (p['KFS_0'] - KFS) / p['tau_K'] + p['mu_K'] * Phi_S

    return jnp.array([dB, dS, dF, dA, dKFB, dKFS])


# =========================================================================
# DIFFUSION — 6D
# =========================================================================

def diffusion_diagonal(params):
    return np.array([params['sigma_B'],
                     params['sigma_S'],
                     params['sigma_F'],
                     params['sigma_A'],
                     params['sigma_K'],
                     params['sigma_K']])


def noise_scale_fn(y, params):
    del params
    B = np.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    S = np.clip(y[1], EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
    F = max(y[2], 0.0)
    A = max(y[3], 0.0)
    KFB = max(y[4], 0.0)
    KFS = max(y[5], 0.0)
    return np.array([math.sqrt(B * (1.0 - B)),
                     math.sqrt(S * (1.0 - S)),
                     math.sqrt(F),
                     math.sqrt(A + EPS_A_FROZEN),
                     math.sqrt(KFB),
                     math.sqrt(KFS)])


def noise_scale_fn_jax(y, params):
    import jax.numpy as jnp
    del params
    B = jnp.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    S = jnp.clip(y[1], EPS_S_FROZEN, 1.0 - EPS_S_FROZEN)
    F = jnp.maximum(y[2], 0.0)
    A = jnp.maximum(y[3], 0.0)
    KFB = jnp.maximum(y[4], 0.0)
    KFS = jnp.maximum(y[5], 0.0)
    return jnp.array([jnp.sqrt(B * (1.0 - B)),
                      jnp.sqrt(S * (1.0 - S)),
                      jnp.sqrt(F),
                      jnp.sqrt(A + EPS_A_FROZEN),
                      jnp.sqrt(KFB),
                      jnp.sqrt(KFS)])


# =========================================================================
# AUX / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    del params, init_state, t_grid
    return (np.asarray(exogenous['Phi_arr'], dtype=np.float32),)


def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.asarray(exogenous['Phi_arr'], dtype=np.float32))


def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['S_0'], init_dict['F_0'], 
                     init_dict['A_0'], init_dict['KFB_0'], init_dict['KFS_0']])


# =========================================================================
# OBSERVATION CHANNELS — same as v3
# =========================================================================

def _sleep_prob(A, C, k_C, k_A, c_tilde):
    z = k_C * C + k_A * A - c_tilde
    return 1.0 / (1.0 + np.exp(-z))


def gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    A = trajectory[:, 3]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
    labels = (rng.random(len(t_grid)) < p).astype(np.int32)
    return {'t_idx': np.arange(len(t_grid), dtype=np.int32), 'sleep_label': labels}


def gen_obs_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    A = trajectory[:, 3]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    hr_mean = (params['HR_base'] - params['kappa_B_HR'] * B + params['alpha_A_HR'] * A + params['beta_C_HR'] * C)
    hr_obs = hr_mean + rng.normal(0.0, params['sigma_HR'], size=len(t_grid))
    sleep_label = prior_channels['obs_sleep']['sleep_label']
    idx_present = np.where(sleep_label == 1)[0]
    return {'t_idx': idx_present.astype(np.int32), 'obs_value': hr_obs[idx_present].astype(np.float32)}


def gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux
    rng = np.random.default_rng(seed)
    F = trajectory[:, 2]
    A = trajectory[:, 3]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    s_mean = (params['S_base'] + params['k_F'] * F - params['k_A_S'] * A + params['beta_C_S'] * C)
    s_obs = s_mean + rng.normal(0.0, params['sigma_S'], size=len(t_grid))
    sleep_label = prior_channels['obs_sleep']['sleep_label']
    idx_present = np.where(sleep_label == 0)[0]
    return {'t_idx': idx_present.astype(np.int32), 'obs_value': s_obs[idx_present].astype(np.float32)}


def gen_obs_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 2]
    A = trajectory[:, 3]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    log_mean = (params['mu_step0'] + params['beta_B_st'] * B - params['beta_F_st'] * F + params['beta_A_st'] * A + params['beta_C_st'] * C)
    log_obs = log_mean + rng.normal(0.0, params['sigma_st'], size=len(t_grid))
    step_count = np.maximum(np.exp(log_obs) - 1.0, 0.0)
    sleep_label = prior_channels['obs_sleep']['sleep_label']
    idx_present = np.where(sleep_label == 0)[0]
    return {'t_idx': idx_present.astype(np.int32), 'obs_value': step_count[idx_present].astype(np.float32)}


def gen_obs_volumeload(trajectory, t_grid, params, aux, prior_channels, seed):
    del aux
    rng = np.random.default_rng(seed)
    S = trajectory[:, 1]
    F = trajectory[:, 2]
    vl_mean = params['beta_S_VL'] * S - params['beta_F_VL'] * F
    vl_obs = vl_mean + rng.normal(0.0, params['sigma_VL'], size=len(t_grid))
    sleep_label = prior_channels['obs_sleep']['sleep_label']
    wake_mask = (sleep_label == 0)
    idx_present = []
    bins_per_day = BINS_PER_DAY
    for d in range(0, len(t_grid)//bins_per_day, 2):
        day_start = d * bins_per_day
        day_end = (d + 1) * bins_per_day
        day_wake_indices = np.where(wake_mask[day_start:day_end])[0] + day_start
        if len(day_wake_indices) > 0:
            mid_wake = day_wake_indices[len(day_wake_indices)//2]
            idx_present.append(mid_wake)
    idx_present = np.array(idx_present, dtype=np.int32)
    return {'t_idx': idx_present, 'obs_value': vl_obs[idx_present].astype(np.float32)}


def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    del trajectory, params, prior_channels, seed
    (Phi_arr,) = aux
    return {'t_idx': np.arange(len(t_grid), dtype=np.int32), 'Phi_value': Phi_arr[:len(t_grid)]}


def gen_C_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    del trajectory, aux, prior_channels, seed
    phi = float(params.get('phi', 0.0))
    val = np.cos(2.0 * np.pi * np.asarray(t_grid, dtype=np.float32) + phi).astype(np.float32)
    return {'t_idx': np.arange(len(t_grid), dtype=np.int32), 'C_value': val}


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    B = trajectory[:, 0]; S = trajectory[:, 1]; F = trajectory[:, 2]; A = trajectory[:, 3]
    KFB = trajectory[:, 4]; KFS = trajectory[:, 5]
    return {
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'S_min': float(S.min()), 'S_max': float(S.max()),
        'F_min': float(F.min()), 'F_max': float(F.max()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'KFB_min': float(KFB.min()), 'KFS_min': float(KFS.min()),
        'all_finite': bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETERS — v4 Variable Dose
# =========================================================================

DEFAULT_PARAMS = {
    # ── Dynamics (v4 + v5-Hill) ──
    'tau_B':      42.0,
    'kappa_B':     0.01248,
    'epsilon_AB':  0.40,
    'tau_S':      60.0,
    'kappa_S':     0.00816,
    'epsilon_AS':  0.20,
    'tau_F':       6.3636,
    'lambda_A':    1.00,
    'KFB_0':       0.030,
    'KFS_0':       0.050,
    'tau_K':       21.0,
    'mu_K':        0.005,
    'mu_0':        0.036,
    'mu_B':        0.30,
    'mu_S':        0.15,
    'mu_F':        0.26,
    'mu_FF':       0.40,
    'eta':         0.20,
    'sigma_B':     0.010,
    'sigma_S':     0.008,
    'sigma_F':     0.012,
    'sigma_A':     0.020,
    'sigma_K':     0.005,
    'phi':         0.0,
    # ── FSA-v5 Hill deconditioning (v4-recovering defaults: mu_dec_*=0) ──
    # Override these with the TRUTH_PARAMS_V5 values from _dynamics.py
    # (B_dec=S_dec=0.07, mu_dec_B=mu_dec_S=0.10) to enable the v5
    # closed-island basin topology (LaTeX §10).
    'B_dec':       0.07,
    'S_dec':       0.07,
    'mu_dec_B':    0.0,        # 0 = v4 numerics; set to 0.10 for v5
    'mu_dec_S':    0.0,        # 0 = v4 numerics; set to 0.10 for v5
    'n_dec':       4.0,
    # ── Observation channels ──
    'HR_base':     62.0, 'kappa_B_HR': 12.0, 'alpha_A_HR': 3.0, 'beta_C_HR': -2.5, 'sigma_HR': 2.0,
    'k_C':         3.0, 'k_A': 2.0, 'c_tilde': 0.5,
    'S_base':      30.0, 'k_F': 20.0, 'k_A_S': 8.0, 'beta_C_S': -4.0, 'sigma_S': 4.0,
    'mu_step0':    5.5, 'beta_B_st': 0.8, 'beta_F_st': 0.5, 'beta_A_st': 0.3, 'beta_C_st': -0.8, 'sigma_st': 0.5,
    'beta_S_VL':   100.0, 'beta_F_VL': 20.0, 'sigma_VL': 10.0,
}

# FSA-v5 parameter set: same as DEFAULT_PARAMS but with the Hill deconditioning
# turned ON. This is the concrete realisation of TRUTH_PARAMS_V5 from
# _dynamics.py, suitable for SDEModel-style calls (which expect params as a
# flat dict including observation coefficients).
DEFAULT_PARAMS_V5 = dict(DEFAULT_PARAMS)
DEFAULT_PARAMS_V5.update(
    mu_dec_B=0.10,    # closed-island calibration (LaTeX §10.4 Table)
    mu_dec_S=0.10,
)

DEFAULT_INIT = {'B_0': 0.05, 'S_0': 0.10, 'F_0': 0.30, 'A_0': 0.10, 'KFB_0': 0.030, 'KFS_0': 0.050}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

# Shared 6D state spec — identical between v4 and v5; only the drift behaves
# differently (via the Hill term in the bifurcation parameter mu).
_FSA_HIGHRES_STATES = (
    StateSpec("B",   0.0,  1.0),
    StateSpec("S",   0.0,  1.0),
    StateSpec("F",   0.0, 10.0),
    StateSpec("A",   0.0,  5.0),
    StateSpec("KFB", 0.0,  1.0),
    StateSpec("KFS", 0.0,  1.0),
)

# Shared observation channels — identical between v4 and v5.
_FSA_HIGHRES_CHANNELS = (
    ChannelSpec("obs_sleep",      depends_on=(),             generate_fn=gen_obs_sleep),
    ChannelSpec("obs_HR",         depends_on=("obs_sleep",), generate_fn=gen_obs_hr),
    ChannelSpec("obs_stress",     depends_on=("obs_sleep",), generate_fn=gen_obs_stress),
    ChannelSpec("obs_steps",      depends_on=("obs_sleep",), generate_fn=gen_obs_steps),
    ChannelSpec("obs_volumeload", depends_on=("obs_sleep",), generate_fn=gen_obs_volumeload),
    ChannelSpec("Phi",            depends_on=(),             generate_fn=gen_Phi_channel),
    ChannelSpec("C",              depends_on=(),             generate_fn=gen_C_channel),
)


# === FSA-v4 SDEModel (back-compat) ===========================================
# Kept as a thin alias for any consumer (test, example, tool) that still
# imports ``HIGH_RES_FSA_V4_MODEL``. Numerically identical to the v5 model
# when ``DEFAULT_PARAMS`` is used because that dict has ``mu_dec_*=0`` —
# the v5 Hill term in ``drift`` / ``drift_jax`` then evaluates to zero.
HIGH_RES_FSA_V4_MODEL = SDEModel(
    name="fsa_high_res_v4",
    version="4.0",
    states=_FSA_HIGHRES_STATES,
    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,
    channels=_FSA_HIGHRES_CHANNELS,
    verify_physics_fn=verify_physics,
    param_sets={'A': DEFAULT_PARAMS},
    init_states={'A': DEFAULT_INIT},
)


# === FSA-v5 SDEModel (the new default for the smc2fc port) ===================
# Identical structure to v4 but ships ``DEFAULT_PARAMS_V5`` with the Hill
# deconditioning enabled (mu_dec_* > 0). The drift function is the same
# object — v5 vs v4 difference is entirely encoded in the params dict.
HIGH_RES_FSA_V5_MODEL = SDEModel(
    name="fsa_high_res_v5",
    version="5.0",
    states=_FSA_HIGHRES_STATES,
    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,
    channels=_FSA_HIGHRES_CHANNELS,
    verify_physics_fn=verify_physics,
    param_sets={'A': DEFAULT_PARAMS_V5},
    init_states={'A': DEFAULT_INIT},
)
