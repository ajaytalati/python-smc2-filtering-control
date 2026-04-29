"""
version_2/models/fsa_high_res/simulation.py — FSA-v2 (Banister-coupled).
=========================================================================

Port of the public-dev `Python-Model-Development-Simulation/version_1/
models/fsa_high_res/simulation.py` (T_B-driven) adapted for the v2
Banister-coupled dynamics that Stage D Stage shipped:

  - **Drift**: B is now driven by training Φ (κ_B·(1+ε_A·A)·Φ − B/τ_B)
    rather than tracking an exogenous T_B target. The "rest cures all"
    pathology of the v1 model is closed by this coupling.
  - **Single control input Φ(t)** — no T_B array in `aux`.
  - **State-dependent sqrt-Itô diffusion** (Jacobi for B, CIR for F
    and A) — same form as v1 simulation.py's `noise_scale_fn`.

  - **Observations (4 channels, identical to v1)**:
      HR     ~ N(HR_base − kappa_B^HR·B + alpha_A^HR·A + beta_C_HR·C, σ_HR²)
                                            [Gaussian, sleep-gated]
      sleep  ~ Bernoulli(sigmoid(k_C·C + k_A·A − c_tilde))
                                            [Bernoulli, always observed]
      stress ~ N(S_base + k_F·F − k_A_S·A + beta_C_S·C, σ_S²)
                                            [Gaussian, wake-gated]
      log(steps+1) ~ N(mu_step0 + beta_B_st·B − beta_F_st·F + beta_A_st·A
                       + beta_C_st·C, σ_st²)         [log-Gaussian, wake]

  - **Circadian C(t)** = cos(2π·t + φ), φ frozen at 0 (morning chronotype).
    Deterministic, NOT a latent state — broadcast as exogenous channel.

The obs equations are identical to the v1 model — they read the latent
state and circadian, not Φ directly — so the port is mechanical for
everything except `drift` and `make_aux`.
"""

import math
import numpy as np

from smc2fc.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_STATE,
)


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4

# Time-grid constants — one "day" is 1.0 in t-units.
DT_BIN_DAYS = 1.0 / 96.0           # 15 minutes
DT_BIN_HOURS = 24.0 / 96.0         # = 0.25
BINS_PER_DAY = 96


# =========================================================================
# Circadian forcing (deterministic, not a state)
# =========================================================================

def circadian(t_days, phi=0.0):
    """C(t) = cos(2π·t + φ), t in days. Period = 1 day.

    With φ=0: peak at midnight (t_days integer), trough at noon.
    Physiological convention: morning chronotype has φ ≈ 0; sleep aligns
    with C > 0, waking activity with C < 0.
    """
    return np.cos(2.0 * np.pi * t_days + phi)


def circadian_jax(t_days, phi=0.0):
    import jax.numpy as jnp
    return jnp.cos(2.0 * jnp.pi * t_days + phi)


# =========================================================================
# Sub-daily Φ-burst (Gamma envelope, morning-loaded)
# =========================================================================

def generate_phi_sub_daily(daily_phi, seed=42,
                            wake_hour=7.0, sleep_hour=23.0,
                            tau_hours=3.0,
                            noise_frac=0.0):
    """Expand a per-day Φ schedule to a per-15-min-bin array with a
    morning-loaded activity profile.

    Profile per wake bin:
        t = hour_of_day − wake_hour
        shape(t) = t · exp(−t / tau_hours)         (Gamma(k=2) shape)
                                                    peaks at t = tau (~3h post-wake)

    Normalised so daily-integrated Φ equals 24 · daily_phi (i.e. the
    slow Banister dynamics see the same daily load as a constant-Φ
    model). Sleep hours [sleep_hour, wake_hour+24]: Φ = 0.

    `noise_frac` defaults to 0 (deterministic). Set to e.g. 0.15 to
    add multiplicative Gaussian noise per bin (the public-dev v1
    default). For closed-loop control we want Φ to be a deterministic
    output of the controller, so the default is no noise.
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_phi)
    phi = np.zeros(n_days * BINS_PER_DAY, dtype=np.float32)

    wake_duration = sleep_hour - wake_hour  # 16.0 h
    T = wake_duration
    # ∫_0^T t·exp(-t/τ) dt = τ²(1 − e^(-T/τ)·(1 + T/τ))
    gamma_integral = tau_hours ** 2 * (
        1.0 - np.exp(-T / tau_hours) * (1.0 + T / tau_hours)
    )

    for d in range(n_days):
        phi_d = float(daily_phi[d])
        amplitude = phi_d * 24.0 / max(gamma_integral, 1e-12)
        for k in range(BINS_PER_DAY):
            h = k * DT_BIN_HOURS
            if h < wake_hour or h >= sleep_hour:
                phi[d * BINS_PER_DAY + k] = 0.0
                continue
            t = h - wake_hour
            shape = t * np.exp(-t / tau_hours)
            base = amplitude * shape
            noise = rng.normal(0.0, noise_frac) if noise_frac > 0 else 0.0
            phi[d * BINS_PER_DAY + k] = max(base * (1.0 + noise), 0.0)

    return phi


def sleep_mask_from_hours(n_days, sleep_hour_lo=23.0, sleep_hour_hi=7.0):
    """Deterministic a-priori sleep mask (1 if 'nominally asleep' at bin)."""
    mask = np.zeros(n_days * BINS_PER_DAY, dtype=np.float32)
    for d in range(n_days):
        for k in range(BINS_PER_DAY):
            h = k * DT_BIN_HOURS
            in_sleep = (h >= sleep_hour_lo) or (h < sleep_hour_hi)
            mask[d * BINS_PER_DAY + k] = 1.0 if in_sleep else 0.0
    return mask


# =========================================================================
# DRIFT — v2 Banister-coupled (single Φ input, no T_B)
# =========================================================================

def _bin_lookup(t_days, array, dt_bin_days=DT_BIN_DAYS):
    k = int(t_days / dt_bin_days)
    k = max(0, min(k, len(array) - 1))
    return float(array[k])


def drift(t, y, params, aux):
    """Numpy v2 drift for scipy / Euler-Maruyama. t in days.

    aux = (Phi_arr,) — single per-bin training-strain rate array.
    """
    (Phi_arr,) = aux
    p = params
    B = y[0]; F = y[1]; A = y[2]

    Phi_t = _bin_lookup(t, Phi_arr)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = p['kappa_B'] * (1.0 + p['epsilon_A'] * A) * Phi_t - B / p['tau_B']
    dF = p['kappa_F'] * Phi_t \
         - (1.0 + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return np.array([dB, dF, dA])


def drift_jax(t, y, args):
    """JAX v2 drift. t in days, args = (params_dict, Phi_arr)."""
    import jax.numpy as jnp
    p, Phi_arr = args
    B = y[0]; F = y[1]; A = y[2]

    k = jnp.clip((t / DT_BIN_DAYS).astype(jnp.int32), 0, Phi_arr.shape[0] - 1)
    Phi_t = Phi_arr[k]

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = p['kappa_B'] * (1.0 + p['epsilon_A'] * A) * Phi_t - B / p['tau_B']
    dF = p['kappa_F'] * Phi_t \
         - (1.0 + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


# =========================================================================
# DIFFUSION — sqrt-Itô (Jacobi for B, CIR for F and A) — same as v1
# =========================================================================

def diffusion_diagonal(params):
    return np.array([params['sigma_B'],
                     params['sigma_F'],
                     params['sigma_A']])


def noise_scale_fn(y, params):
    del params
    B = np.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = max(y[1], 0.0)
    A = max(y[2], 0.0)
    return np.array([math.sqrt(B * (1.0 - B)),
                     math.sqrt(F),
                     math.sqrt(A + EPS_A_FROZEN)])


def noise_scale_fn_jax(y, params):
    import jax.numpy as jnp
    del params
    B = jnp.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = jnp.maximum(y[1], 0.0)
    A = jnp.maximum(y[2], 0.0)
    return jnp.array([jnp.sqrt(B * (1.0 - B)),
                      jnp.sqrt(F),
                      jnp.sqrt(A + EPS_A_FROZEN)])


# =========================================================================
# AUX / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """aux = (Phi_arr,) — per-bin Φ schedule for drift lookup."""
    del params, init_state, t_grid
    return (np.asarray(exogenous['Phi_arr'], dtype=np.float32),)


def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.asarray(exogenous['Phi_arr'], dtype=jnp.float32))


def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['F_0'], init_dict['A_0']])


# =========================================================================
# OBSERVATION CHANNELS — identical to v1 (independent of Φ vs T_B)
# =========================================================================

def _sleep_prob(A, C, k_C, k_A, c_tilde):
    z = k_C * C + k_A * A - c_tilde
    return 1.0 / (1.0 + np.exp(-z))


def gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """Bernoulli sleep label at every 15-min bin."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
    labels = (rng.random(len(t_grid)) < p).astype(np.int32)
    return {
        't_idx':       np.arange(len(t_grid), dtype=np.int32),
        'sleep_label': labels,
    }


def gen_obs_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    """HR, Gaussian, measured only during sleep."""
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    hr_mean = (params['HR_base']
               - params['kappa_B_HR'] * B
               + params['alpha_A_HR'] * A
               + params['beta_C_HR'] * C)
    hr_obs = hr_mean + rng.normal(0.0, params['sigma_HR'], size=len(t_grid))

    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = sleep_label.astype(np.int32)
    else:
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p > 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': hr_obs[idx_present].astype(np.float32),
    }


def gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    """Stress, Gaussian, measured only during waking."""
    del aux
    rng = np.random.default_rng(seed)
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    s_mean = (params['S_base']
              + params['k_F'] * F
              - params['k_A_S'] * A
              + params['beta_C_S'] * C)
    s_obs = s_mean + rng.normal(0.0, params['sigma_S'], size=len(t_grid))

    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = (1 - sleep_label).astype(np.int32)
    else:
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p <= 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': s_obs[idx_present].astype(np.float32),
    }


def gen_obs_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    """Step count, log-Gaussian, measured only during waking."""
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    log_mean = (params['mu_step0']
                + params['beta_B_st'] * B
                - params['beta_F_st'] * F
                + params['beta_A_st'] * A
                + params['beta_C_st'] * C)
    log_obs = log_mean + rng.normal(0.0, params['sigma_st'], size=len(t_grid))
    step_count = np.maximum(np.exp(log_obs) - 1.0, 0.0)

    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = (1 - sleep_label).astype(np.int32)
    else:
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p <= 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': step_count[idx_present].astype(np.float32),
    }


def _broadcast_to_grid(arr, n):
    arr = np.asarray(arr, dtype=np.float32)
    return arr[:n] if len(arr) >= n else np.resize(arr, n).astype(np.float32)


def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Broadcast per-bin Φ from aux."""
    del trajectory, params, prior_channels, seed
    (Phi_arr,) = aux
    return {'t_idx':     np.arange(len(t_grid), dtype=np.int32),
            'Phi_value': _broadcast_to_grid(Phi_arr, len(t_grid))}


def gen_C_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Broadcast deterministic circadian C(t) on the global time grid."""
    del trajectory, aux, prior_channels, seed
    phi = float(params.get('phi', 0.0))
    val = np.cos(2.0 * np.pi * np.asarray(t_grid, dtype=np.float32)
                 + phi).astype(np.float32)
    return {'t_idx':   np.arange(len(t_grid), dtype=np.int32),
            'C_value': val}


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def _mu_of(B, F, p):
    return p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F


def verify_physics(trajectory, t_grid, params):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    mu_traj = _mu_of(B, F, params)
    return {
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'F_min': float(F.min()), 'F_max': float(F.max()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_final': float(B[-1]), 'F_final': float(F[-1]), 'A_final': float(A[-1]),
        'mu_min': float(mu_traj.min()), 'mu_max': float(mu_traj.max()),
        'mu_crosses_zero': "yes" if (mu_traj.min() < 0 < mu_traj.max()) else "no",
        'all_finite': bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETERS — Set A v2 (Banister-coupled, frozen σ + φ; estimable rest)
# =========================================================================

DEFAULT_PARAMS = {
    # --- v2 Banister dynamics ---
    'tau_B':      42.0,    # canonical Banister chronic time constant
    'tau_F':       7.0,    # canonical acute time constant
    'kappa_B':     0.012,  # B-gain per unit Φ → B_ss ≈ 0.5 at Φ=1.0
    'kappa_F':     0.030,  # F-gain per unit Φ → F_ss ≈ 0.2 at Φ=1.0
    'epsilon_A':   0.40,   # A boosts B-gain (≤40% at A=1)
    'lambda_A':    1.00,   # A doubles F-clearance rate at A=1
    # Stuart-Landau bifurcation parameter
    'mu_0':        0.02,
    'mu_B':        0.30,
    'mu_F':        0.10,
    'mu_FF':       0.40,
    'eta':         0.20,
    # sqrt-Itô diffusion scales (frozen)
    'sigma_B':     0.010,
    'sigma_F':     0.012,
    'sigma_A':     0.020,

    # --- Circadian (frozen) ---
    'phi':         0.0,    # morning chronotype

    # --- Ch1: HR (sleep-gated) — same as v1 (renamed kappa_B → kappa_B_HR
    # to disambiguate from the dynamics kappa_B which is now Banister-gain) ---
    'HR_base':     62.0,
    'kappa_B_HR':  12.0,
    'alpha_A_HR':   3.0,
    'beta_C_HR':   -2.5,
    'sigma_HR':     2.0,

    # --- Ch2: Sleep (Bernoulli) ---
    'k_C':          3.0,
    'k_A':          2.0,
    'c_tilde':      0.5,

    # --- Ch3: Stress (wake-gated) ---
    'S_base':      30.0,
    'k_F':         20.0,
    'k_A_S':        8.0,
    'beta_C_S':    -4.0,
    'sigma_S':      4.0,

    # --- Ch4: Steps (log-Gaussian, wake-gated) ---
    'mu_step0':     5.5,
    'beta_B_st':    0.8,
    'beta_F_st':    0.5,
    'beta_A_st':    0.3,
    'beta_C_st':   -0.8,
    'sigma_st':     0.5,
}

# Stage-D initial state: de-trained (B_0=0.05), high residual fatigue
# (F_0=0.30), low autonomic amplitude (A_0=0.10).
DEFAULT_INIT = {'B_0': 0.05, 'F_0': 0.30, 'A_0': 0.10}


# =========================================================================
# SCENARIO PRESETS
# =========================================================================

# "C0 recovery" preset — constant-Φ baseline at canonical Banister
# default Φ = 1.0 (used as the open-loop reference baseline in the
# E1-E3 single-/rolling-window filter benchmarks; analogous to the
# v1 model's EXO_RECOVERY).
EXO_C0_RECOVERY = {
    'Phi_arr': np.full(BINS_PER_DAY, 1.0, dtype=np.float32),
    'T_end':   1.0,    # days — single-day default
}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

# Optional plot helper — falls back to None if sim_plots not yet ported
try:
    from models.fsa_high_res.sim_plots import plot_fsa_high_res
except ImportError:
    plot_fsa_high_res = None


HIGH_RES_FSA_V2_MODEL = SDEModel(
    name="fsa_high_res_v2",
    version="2.0",

    states=(
        StateSpec("B", 0.0, 1.0),
        StateSpec("F", 0.0, 10.0),
        StateSpec("A", 0.0, 5.0),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        # Sleep is generated FIRST so HR/stress/steps can use it for gating.
        ChannelSpec("obs_sleep",  depends_on=(),             generate_fn=gen_obs_sleep),
        ChannelSpec("obs_HR",     depends_on=("obs_sleep",), generate_fn=gen_obs_hr),
        ChannelSpec("obs_stress", depends_on=("obs_sleep",), generate_fn=gen_obs_stress),
        ChannelSpec("obs_steps",  depends_on=("obs_sleep",), generate_fn=gen_obs_steps),
        ChannelSpec("Phi",        depends_on=(),             generate_fn=gen_Phi_channel),
        ChannelSpec("C",          depends_on=(),             generate_fn=gen_C_channel),
    ),

    plot_fn=plot_fsa_high_res,
    verify_physics_fn=verify_physics,

    param_sets={'A': DEFAULT_PARAMS, 'C0_recovery': DEFAULT_PARAMS},
    init_states={'A': DEFAULT_INIT, 'C0_recovery': DEFAULT_INIT},
    exogenous_inputs={'A': EXO_C0_RECOVERY, 'C0_recovery': EXO_C0_RECOVERY},
)
