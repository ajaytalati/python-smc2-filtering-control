"""SWAT forward simulation + observation samplers.

Top-level forward-simulation entry for the 4-state SWAT model
(W, Z, a, T) with three exogenous control variates (V_h, V_n, V_c).
Provides:

- ``DEFAULT_PARAMS`` — merged dynamics + obs parameter dict for the
  healthy-baseline scenario (Set A).
- ``DEFAULT_INIT`` — initial state (W_0, Z_0, a_0, T_0).
- Set B / C / D scenario builders for amplitude collapse, recovery,
  and phase-shift pathology.
- Four observation samplers (HR Gaussian, Sleep 3-level ordinal,
  Steps Poisson, Stress Gaussian) — numpy implementations.
- ``BINS_PER_DAY`` constant matching ``_v_schedule.py``.

Time unit throughout: **days**. The bench tool sets
``FSA_STEP_MINUTES`` to control bin granularity (60 = h=1h,
matching the FSA-v2 default).
"""
from __future__ import annotations

import math
import os as _os

import numpy as np

from version_2.models.swat._dynamics import (
    A_SCALE_FROZEN,
    PHI_0_FROZEN,
    TRUTH_PARAMS,
    V_C_MAX_HOURS,
)
from version_2.models.swat._v_schedule import BINS_PER_DAY, DT_BIN_HOURS


# ── Default parameters: merged dynamics (TRUTH_PARAMS) + obs ───────────
#
# Dynamics block comes from _dynamics.TRUTH_PARAMS (already in days
# units). Obs block is added below. Numerical values mirror Repo A's
# PARAM_SET_A as of origin/main 2026-04-30.

_OBS_PARAMS = dict(
    # HR Gaussian channel (continuous, sleep-modulated via W)
    HR_base=50.0,
    alpha_HR=25.0,
    sigma_HR=8.0,

    # Sleep 3-level ordinal channel (thresholds on Z ∈ [0,1])
    # Rescaled from Z ∈ [0, 6] to Z ∈ [0, 1]:
    # c_tilde was 2.5 → 2.5/6 ≈ 0.42, delta_c was 1.5 → 1.5/6 = 0.25
    c_tilde=0.42,
    delta_c=0.25,                   # c2 = c_tilde + delta_c ≈ 0.67

    # Steps log-Gaussian channel (FSA-v2 pattern, wake-gated)
    # log(steps + 1) ~ N(mu_step0 + beta_W_steps * W, sigma_step²)
    # Per 15-min wake bin: at W=0.5, log_mean = 4.4 → ~80 steps/bin
    # ≈ 320 steps/hr (typical light-activity wake-bin rate).
    mu_step0=4.0,                   # ≈ log(54+1), baseline log-rate
    beta_W_steps=0.8,               # W coupling
    sigma_step=0.5,

    # Stress Gaussian channel (Garmin-style 0-100 score)
    s_base=30.0,
    alpha_s=40.0,                   # wake modulation
    beta_s=10.0,                    # V_n modulation
    sigma_s=15.0,
)

DEFAULT_PARAMS = {**TRUTH_PARAMS, **_OBS_PARAMS}


# ── Default initial state (Set A: healthy baseline) ──────────────────
#
# 4-state vector (W, Z, a, T). V_h, V_n, V_c are exogenous controls
# passed in at integration time, NOT part of the state.

# Z_0 was 3.5 in [0, 6]; rescaled to 3.5/6 ≈ 0.58 for [0, 1] domain.
DEFAULT_INIT = np.array([0.5, 0.58, 0.5, 0.5], dtype=np.float64)


# ── Scenario presets ────────────────────────────────────────────────
#
# Each preset is a dict of ``v_h_daily`` / ``v_n_daily`` / ``v_c_daily``
# constants representing the truth control schedule under that
# scenario. The bench tool / forward sim then drives the plant under
# this schedule starting from DEFAULT_INIT (or a scenario-specific
# init for Set C recovery).

def scenario_presets(t_total_days: int) -> dict:
    """Return the 4 canonical SWAT scenario truth schedules + inits.

    All four scenarios use constant daily controls — the bench tool
    will perturb V_c (and V_h, V_n) under closed-loop control. These
    constant-truth schedules are for forward-sim validation against
    psim's 14-day reference outputs.
    """
    # Z_0 rescaled to [0,1] domain (was 3.5 in [0,6])
    return {
        'A_healthy': {
            'init': DEFAULT_INIT.copy(),
            'v_h_daily': np.full(t_total_days, 1.0, dtype=np.float64),
            'v_n_daily': np.full(t_total_days, 0.3, dtype=np.float64),
            'v_c_daily': np.full(t_total_days, 0.0, dtype=np.float64),
        },
        'B_amplitude_collapse': {
            'init': DEFAULT_INIT.copy(),    # same start, fail mode is via control
            'v_h_daily': np.full(t_total_days, 0.2, dtype=np.float64),
            'v_n_daily': np.full(t_total_days, 3.5, dtype=np.float64),
            'v_c_daily': np.full(t_total_days, 0.0, dtype=np.float64),
        },
        'C_recovery': {
            'init': np.array([0.5, 0.58, 0.5, 0.05], dtype=np.float64),  # T_0 starts low
            'v_h_daily': np.full(t_total_days, 1.0, dtype=np.float64),
            'v_n_daily': np.full(t_total_days, 0.3, dtype=np.float64),
            'v_c_daily': np.full(t_total_days, 0.0, dtype=np.float64),
        },
        'D_phase_shift': {
            'init': DEFAULT_INIT.copy(),
            'v_h_daily': np.full(t_total_days, 1.0, dtype=np.float64),
            'v_n_daily': np.full(t_total_days, 0.3, dtype=np.float64),
            'v_c_daily': np.full(t_total_days, 6.0, dtype=np.float64),  # 6h jet lag
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────

def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def circadian(t_days, V_c_hours=0.0):
    """External light cycle C(t) — objective sun/dark signal.

    With t in days, the period is 1 day. V_c shifts the SUBJECT'S
    rhythm relative to this baseline (the C_eff inside drift_jax),
    not the light cycle itself. This helper returns the fixed
    external reference.
    """
    del V_c_hours    # external C ignores V_c (only C_eff in drift uses it)
    return np.sin(2.0 * np.pi * t_days + PHI_0_FROZEN)


# ── Observation samplers (numpy) ─────────────────────────────────────
#
# Each function takes a (T_len, 4) state trajectory + the time grid
# (in days) + parameter dict + a seed, and returns a dict with
# ``t_idx`` (bin indices) plus channel-specific ``obs_value`` keys.
# Patterns mirror FSA-v2's ``simulation.py:gen_obs_*`` API so the
# bench tool's existing closed-loop hooks apply to SWAT unchanged.


def gen_obs_hr(trajectory, t_grid_days, params, seed=42):
    """Gaussian HR channel: hr ~ N(HR_base + alpha_HR * W, sigma_HR^2).

    HR is observed at every bin; t_idx is just np.arange(T_len).
    """
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    T_len = len(t_grid_days)

    hr_mean = params['HR_base'] + params['alpha_HR'] * W
    hr = hr_mean + rng.normal(0.0, params['sigma_HR'], size=T_len)

    return {
        't_idx':     np.arange(T_len, dtype=np.int32),
        'obs_value': hr.astype(np.float64),
    }


def gen_obs_sleep(trajectory, t_grid_days, params, seed=43):
    """3-level ordinal sleep channel: {0=wake, 1=light+REM, 2=deep}.

    Two thresholds c1 < c2 on Z:
        P(level <= 0) = 1 - sigmoid(Z - c1)
        P(level <= 1) = 1 - sigmoid(Z - c2)
    Parameterised as (c_tilde, delta_c > 0) with c1 = c_tilde,
    c2 = c_tilde + delta_c.

    Sleep is observed at every bin.
    """
    rng = np.random.default_rng(seed)
    Z = trajectory[:, 1]
    T_len = len(t_grid_days)

    c1 = params['c_tilde']
    c2 = c1 + params['delta_c']

    s1 = _sigmoid(Z - c1).astype(np.float64)
    s2 = _sigmoid(Z - c2).astype(np.float64)

    draws = rng.random(size=T_len)
    labels = np.where(draws < 1.0 - s1, 0,
             np.where(draws < 1.0 - s2, 1, 2)).astype(np.int32)

    return {
        't_idx':     np.arange(T_len, dtype=np.int32),
        'obs_label': labels,
    }


def gen_obs_steps(trajectory, t_grid_days, params, sleep_label, seed=44):
    """Steps log-Gaussian channel, wake-gated (FSA-v2 pattern).

    log(steps + 1) ~ N(mu_step0 + beta_W_steps * W, sigma_step^2)

    The channel is **wake-gated** — only emitted when sleep_label == 0.
    The corresponding bin's `present_mask` flag is 1 if wake, 0
    otherwise. The bench's align_obs_fn ANDs this with steps_present
    so the filter never sees sleep-bin step values.

    Args:
        trajectory:   (T_len, 4) state.
        t_grid_days:  (T_len,) time grid in days.
        params:       dict with mu_step0, beta_W_steps, sigma_step.
        sleep_label:  (T_len,) integer labels {0=wake, 1=light, 2=deep}.
        seed:         RNG seed.

    Returns:
        dict with t_idx (one per bin), log_value (the Gaussian sample),
        and present_mask (1 if wake at this bin, 0 if light/deep).
    """
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    T_len = len(t_grid_days)

    log_mean = params['mu_step0'] + params['beta_W_steps'] * W
    log_value = log_mean + rng.normal(0.0, params['sigma_step'], size=T_len)
    present_mask = (np.asarray(sleep_label) == 0).astype(np.float64)

    return {
        't_idx':        np.arange(T_len, dtype=np.int32),
        'log_value':    log_value.astype(np.float64),
        'present_mask': present_mask,
    }


def gen_obs_stress(trajectory, t_grid_days, params, V_n_per_bin,
                     seed=45):
    """Gaussian stress-score channel.

    stress ~ N(s_base + alpha_s * W + beta_s * V_n, sigma_s^2),
    clipped to [0, 100].

    Args:
        trajectory:    (T_len, 4) state.
        t_grid_days:   (T_len,) time grid.
        params:        dict.
        V_n_per_bin:   (T_len,) — the V_n control input at each bin.
                        SWAT's stress channel reads V_n directly (it's
                        an exogenous control in our 4-state form, so
                        the bench passes the sub-daily V_n trajectory
                        in here).
        seed:          RNG seed.
    """
    rng = np.random.default_rng(seed)
    W = trajectory[:, 0]
    T_len = len(t_grid_days)

    mean = (params['s_base']
             + params['alpha_s'] * W
             + params['beta_s'] * V_n_per_bin)
    stress = mean + rng.normal(0.0, params['sigma_s'], size=T_len)
    stress = np.clip(stress, 0.0, 100.0)

    return {
        't_idx':     np.arange(T_len, dtype=np.int32),
        'obs_value': stress.astype(np.float64),
    }


# ── Forward simulation entry (smoke test) ────────────────────────────


def forward_sim_set(scenario_name: str, t_total_days: int = 14,
                     seed: int = 42) -> dict:
    """Convenience: run a deterministic forward sim of one scenario.

    Used as a smoke test against psim's stored reference outputs (for
    Sets A/B/C/D). Uses Euler-Maruyama with FSA-v2's substepping
    pattern. Returns a dict with the trajectory + the four obs
    streams.

    NOTE: this is the *deterministic* version (drift only, no noise);
    for the closed-loop bench, use ``_plant.py:StepwisePlant.advance``.
    """
    import jax
    import jax.numpy as jnp

    from version_2.models.swat._dynamics import drift_jax
    from version_2.models.swat._v_schedule import expand_three_schedules

    presets = scenario_presets(t_total_days)
    if scenario_name not in presets:
        raise KeyError(f"unknown scenario {scenario_name!r}; "
                        f"choices: {list(presets.keys())}")
    p = presets[scenario_name]

    n_bins = t_total_days * BINS_PER_DAY
    dt_days = 1.0 / BINS_PER_DAY
    t_grid_days = np.arange(n_bins, dtype=np.float64) * dt_days

    u_per_bin = expand_three_schedules(
        p['v_h_daily'], p['v_n_daily'], p['v_c_daily'])      # (n_bins, 3)

    # Deterministic Euler integration.
    y = jnp.asarray(p['init'], dtype=jnp.float64)
    params_jax = {k: jnp.float64(v) for k, v in DEFAULT_PARAMS.items()
                   if isinstance(v, (int, float))}
    traj = [y]
    for k in range(n_bins - 1):
        u_k = jnp.asarray(u_per_bin[k], dtype=jnp.float64)
        t_k = jnp.float64(t_grid_days[k])
        dy = drift_jax(y, params_jax, t_k, u_k)
        y = y + dt_days * dy
        # Boundary clip (replicates state_clip semantics)
        y = jnp.array([
            jnp.clip(y[0], 0.0, 1.0),
            jnp.clip(y[1], 0.0, A_SCALE_FROZEN),
            jnp.maximum(y[2], 0.0),
            jnp.maximum(y[3], 0.0),
        ])
        traj.append(y)
    trajectory = np.asarray(jnp.stack(traj))   # (n_bins, 4)

    sleep_ch = gen_obs_sleep(trajectory, t_grid_days, DEFAULT_PARAMS,
                              seed=seed + 1)
    return {
        'trajectory':  trajectory,
        't_grid_days': t_grid_days,
        'u_per_bin':   u_per_bin,
        'obs_HR':      gen_obs_hr(trajectory, t_grid_days, DEFAULT_PARAMS, seed=seed),
        'obs_sleep':   sleep_ch,
        'obs_steps':   gen_obs_steps(trajectory, t_grid_days, DEFAULT_PARAMS,
                                       sleep_label=sleep_ch['obs_label'],
                                       seed=seed + 2),
        'obs_stress':  gen_obs_stress(trajectory, t_grid_days, DEFAULT_PARAMS,
                                        V_n_per_bin=u_per_bin[:, 1], seed=seed + 3),
    }
