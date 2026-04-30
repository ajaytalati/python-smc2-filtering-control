"""SWAT EstimationModel — SMC² inference target for the 4-state plant.

Builds the framework's ``EstimationModel`` instance for the SWAT
4-state model, exposing:

- ``SWAT_ESTIMATION``: the frozen EstimationModel singleton consumed
  by the bench tool's filter (``smc2fc.filtering.gk_dpf_v3_lite``).
- ``PARAM_PRIOR_CONFIG``: 27 estimable parameters with priors. V_h,
  V_n, V_c are NOT in this set — they're exogenous controls passed
  via ``grid_obs`` at filter time.
- ``COLD_START_INIT``: cold-start initial state (W, Z, a, T) for the
  first window's filter prior.

Design choices vs FSA-v2:
- **State dim 4** (W, Z, a, T) instead of FSA-v2's 3 (B, F, A).
- **Three exogenous controls** (V_h, V_n, V_c) read from
  ``grid_obs[{'V_h', 'V_n', 'V_c'}]`` at each propagate step.
- **Four obs channels** with mixed likelihoods:
  - HR Gaussian (always observed)
  - Sleep 3-level ordinal (always observed)
  - Steps Poisson (per-bin count)
  - Stress Gaussian (always observed)
- **propagate_fn** is plain Euler-Maruyama for now — no Kalman
  fusion of Gaussian channels. The GK-DPF guided proposal (which
  FSA-v2 uses for the inner-PF efficiency boost) is deferred to
  Phase 1.5; it's a performance optimisation, not a correctness
  requirement. The basic SMC² posterior is still correct because
  the obs-side log-weight catches what the proposal misses.
- **mu_0, mu_E estimable**: this is the F_max-from-data analog. The
  bench's data-excitation experiments will probe whether the
  posterior on (mu_0, mu_E) recovers truth across scenarios.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.estimation_model import EstimationModel

from version_2.models.swat._dynamics import (
    A_SCALE_FROZEN,
    PHI_0_FROZEN,
    V_C_MAX_HOURS,
    diffusion_state_dep,
    drift_jax,
    state_clip,
)


HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================
# Three operating-point references (A_SCALE, phi_0, V_c_max) plus two
# pinning fixes from the Repo C identifiability analysis (see
# `Python-Model-Validation/identifiability/swat/fisher_information_analysis.md`):
#
# - **tau_T = 2.0 days (= 48 h)** — pinned to break the (mu_0, mu_E,
#   eta, tau_T) Stuart-Landau time-vs-rate scaling degeneracy. Without
#   this, the FIM is rank 24/27 with condition number ~10^36 — mu_0
#   and mu_E are not separately identifiable.
# - **lambda_amp_Z = 8.0** — pinned because (lambda_amp_W,
#   lambda_amp_Z) only enter E_dyn through the product
#   amp_W * amp_Z, so only one of the two is identifiable from data.
#
# After both pins applied, the analysis confirms FIM rank 25/25
# (condition 4.77e9 — borderline but identifiable). mu_0 and mu_E
# are then both individually identifiable — the F_max-from-data
# experiment is well-posed.

FROZEN_PARAMS = dict(
    A_scale=A_SCALE_FROZEN,
    phi_0=PHI_0_FROZEN,
    V_c_max=V_C_MAX_HOURS,
    # Pinned per Repo C FIM analysis 2026-04-27:
    tau_T=48.0 / 24.0,            # 2.0 days
    lambda_amp_Z=8.0,
)


# =========================================================================
# PARAMETER PRIORS — 27 estimable scalars
# =========================================================================
# Adapted from Repo A's PARAM_PRIOR_CONFIG. Differences from Repo A:
# - V_h, V_n REMOVED (now exogenous controls, not estimable params).
# - V_c REMOVED (now exogenous control).
# - Time-scale priors converted hours -> days where applicable.
# - Diffusion temperatures converted per-hour -> per-day.

_HOURS_PER_DAY = 24.0


def _ln_d(hours):
    """Helper: lognormal-mean of a value originally in hours, expressed in days."""
    return math.log(hours / _HOURS_PER_DAY)


PARAM_PRIOR_CONFIG = OrderedDict([
    # ── Sigmoid couplings (block F — fast subsystem) ─────────────────
    ('kappa',    ('lognormal', (math.log(6.67), 0.5))),
    ('lmbda',    ('lognormal', (math.log(32.0), 0.5))),
    ('gamma_3',  ('lognormal', (math.log(8.0),  0.5))),
    ('beta_Z',   ('lognormal', (math.log(4.0),  0.4))),

    # ── Timescales (in DAYS, converted from hours) ───────────────────
    # tau_T is PINNED — see FROZEN_PARAMS — to break the
    # (mu_0, mu_E, eta, tau_T) Stuart-Landau scaling degeneracy.
    ('tau_W',    ('lognormal', (_ln_d(2.0),  0.3))),
    ('tau_Z',    ('lognormal', (_ln_d(2.0),  0.3))),
    ('tau_a',    ('lognormal', (_ln_d(3.0),  0.3))),

    # ── Stuart-Landau bifurcation block (the F_max-analog parameters) ─
    # mu_0 + mu_E = 0 is the bifurcation point E_crit = -mu_0/mu_E = 0.5
    # in the healthy-baseline case. Both estimable so the data informs
    # the threshold (the SMC²-MPC novel feature for this port).
    ('mu_0',     ('normal',    (-0.5, 0.3))),
    ('mu_E',     ('lognormal', (math.log(1.0),  0.3))),
    ('eta',      ('lognormal', (math.log(0.5),  0.3))),
    ('alpha_T',  ('lognormal', (math.log(0.3),  0.3))),

    # ── Diffusion temperatures (per DAY) ─────────────────────────────
    ('T_W',      ('lognormal', (math.log(0.01 * _HOURS_PER_DAY), 0.5))),
    ('T_Z',      ('lognormal', (math.log(0.05 * _HOURS_PER_DAY), 0.5))),
    ('T_a',      ('lognormal', (math.log(0.01 * _HOURS_PER_DAY), 0.5))),
    ('T_T',      ('lognormal', (math.log(0.0001 * _HOURS_PER_DAY), 0.5))),

    # ── Entrainment-amplitude block (V_h-anabolic) ───────────────────
    # lambda_amp_Z is PINNED — see FROZEN_PARAMS — because
    # (lambda_amp_W, lambda_amp_Z) enter E_dyn only through their
    # product, so only one is individually identifiable.
    ('lambda_amp_W', ('lognormal', (math.log(5.0), 0.3))),
    ('V_n_scale',    ('lognormal', (math.log(2.0), 0.3))),

    # ── Obs ch1: HR Gaussian (sleep-modulated via W) ─────────────────
    ('HR_base',  ('normal',    (50.0, 5.0))),
    ('alpha_HR', ('lognormal', (math.log(25.0), 0.3))),
    ('sigma_HR', ('lognormal', (math.log(8.0),  0.3))),

    # ── Obs ch2: Sleep 3-level ordinal ───────────────────────────────
    ('c_tilde',  ('normal',    (2.5, 0.5))),
    ('delta_c',  ('lognormal', (math.log(1.5),  0.3))),

    # ── Obs ch3: Steps Poisson ───────────────────────────────────────
    ('lambda_base', ('lognormal', (math.log(0.5),  0.7))),
    ('lambda_step', ('lognormal', (math.log(200.0), 0.5))),
    ('W_thresh',    ('normal',    (0.6, 0.1))),

    # ── Obs ch4: Stress Gaussian ─────────────────────────────────────
    ('s_base',   ('normal',    (30.0, 10.0))),
    ('alpha_s',  ('normal',    (40.0, 10.0))),
    ('beta_s',   ('lognormal', (math.log(10.0), 0.3))),
    ('sigma_s',  ('lognormal', (math.log(15.0), 0.3))),
])


# =========================================================================
# INIT-STATE PRIORS (W_0, Z_0, a_0, T_0)
# =========================================================================

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('normal',    (0.5, 0.2))),
    ('Z_0',  ('normal',    (3.5, 1.0))),
    ('a_0',  ('lognormal', (math.log(0.5),  0.3))),
    ('T_0',  ('lognormal', (math.log(0.5),  0.5))),
])

# Fast lookup: parameter name -> index in the prior-vector
_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# Cold-start initial state for the very first window's prior mean.
COLD_START_INIT = jnp.array([0.5, 3.5, 0.5, 0.5], dtype=jnp.float64)


# =========================================================================
# CORE FUNCTIONS — Drift / Diffusion (already in _dynamics)
# =========================================================================
# We reuse drift_jax, diffusion_state_dep, state_clip from _dynamics.py.
# No wrapping needed — the propagate_fn below calls them directly.


# =========================================================================
# PROPAGATE_FN — Euler-Maruyama per bin
# =========================================================================
#
# Args (framework convention):
#   y          (n_states,) state at start of step
#   t          scalar time (days) at start of step
#   dt         scalar step size (days)
#   params     (n_params,) parameter VECTOR (PI-indexed)
#   grid_obs   dict of per-bin exogenous + obs values; reads V_h, V_n,
#              V_c at bin k
#   k          int — current bin index in the window
#   sigma_diag (n_states,) diffusion vector (precomputed by diffusion_fn)
#   noise      (n_states,) standard-normal noise
#   rng_key    PRNGKey (unused for plain Euler proposal but kept for API)
#
# Returns:
#   y_new      (n_states,) propagated state
#   pred_lw    scalar — predictive log-weight from proposal vs prior;
#              0 for plain Euler.

def propagate_fn(y, t, dt, params, grid_obs, k,
                  sigma_diag, noise, rng_key):
    """Euler-Maruyama propagate for SWAT 4-state.

    No Kalman fusion of Gaussian channels in this version — the
    obs-side log-weight (``obs_log_weight_fn``) catches all four
    channel likelihoods. GK-DPF guided proposal is a Phase 1.5
    optimisation.
    """
    del rng_key  # not used by plain Euler proposal

    # Build params dict from the indexed vector (jit-stable view)
    p_dict = {name: params[idx] for name, idx in _PI.items()}
    # Add frozen constants (drift_jax doesn't read them but keeps API
    # uniform across SWAT functions)
    for fname, fval in FROZEN_PARAMS.items():
        p_dict[fname] = jnp.float64(fval)

    # Read controls from grid_obs at this bin
    V_h = grid_obs['V_h'][k]
    V_n = grid_obs['V_n'][k]
    V_c = grid_obs['V_c'][k]
    u = jnp.array([V_h, V_n, V_c], dtype=jnp.float64)

    # Drift + diffusion step
    d_y = drift_jax(y, p_dict, t, u)
    sqrt_dt = jnp.sqrt(dt)
    y_new = y + dt * d_y + sigma_diag * sqrt_dt * noise
    y_new = state_clip(y_new)

    pred_lw = jnp.float64(0.0)   # plain Euler proposal -> zero predictive lw
    return y_new, pred_lw


# =========================================================================
# DIFFUSION_FN — diagonal noise vector from params
# =========================================================================

def diffusion_fn(params):
    """Return the diagonal sigma vector (n_states,) given a params VECTOR."""
    p_dict = {name: params[idx] for name, idx in _PI.items()}
    return diffusion_state_dep(jnp.zeros(4), p_dict)


# =========================================================================
# OBS_LOG_WEIGHT_FN — sum of 4-channel log-likelihoods at bin k
# =========================================================================
#
# Each channel may or may not have an obs at bin k; the ``*_present``
# masks let us skip absent obs without breaking the JAX trace.

def _gauss_log_lik(obs_value, mean, sigma):
    """log N(obs | mean, sigma^2)."""
    z = (obs_value - mean) / sigma
    return -HALF_LOG_2PI - jnp.log(sigma) - 0.5 * z * z


def _ordinal_log_lik(obs_label, Z, c1, c2):
    """3-level ordinal: P(label | Z) via thresholds c1 < c2.

        P(0=wake)   = 1 - sigmoid(Z - c1)
        P(1=light)  = sigmoid(Z - c1) - sigmoid(Z - c2)
        P(2=deep)   = sigmoid(Z - c2)
    """
    s1 = jax.nn.sigmoid(Z - c1)
    s2 = jax.nn.sigmoid(Z - c2)
    p_wake = 1.0 - s1
    p_light = s1 - s2
    p_deep = s2
    # Pick the right probability via integer label
    p = jnp.where(obs_label == 0, p_wake,
        jnp.where(obs_label == 1, p_light, p_deep))
    return jnp.log(jnp.maximum(p, 1e-30))


def _poisson_log_lik(obs_count, rate, bin_hours):
    """log Poisson(obs_count | rate * bin_hours).

    Stable form: k*log(λ) - λ - log(k!)  where λ = rate*bin_hours.
    Uses lgamma(k+1) for log(k!).
    """
    lam = rate * bin_hours
    return obs_count * jnp.log(jnp.maximum(lam, 1e-30)) - lam - jax.lax.lgamma(
        obs_count.astype(jnp.float64) + 1.0)


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Sum the four channel log-likelihoods at bin k.

    Each channel's contribution is gated by its ``*_present`` mask in
    grid_obs (1.0 if obs present at this bin, 0.0 if not).
    """
    p = {name: params[idx] for name, idx in _PI.items()}

    W = x_new[0]
    Z = x_new[1]

    log_w = jnp.float64(0.0)

    # ── Ch1: HR Gaussian ─────────────────────────────────────────────
    hr_value = grid_obs['hr_value'][k]
    hr_present = grid_obs['hr_present'][k]
    hr_mean = p['HR_base'] + p['alpha_HR'] * W
    log_w += hr_present * _gauss_log_lik(hr_value, hr_mean, p['sigma_HR'])

    # ── Ch2: Sleep 3-level ordinal ───────────────────────────────────
    sleep_label = grid_obs['sleep_label'][k]
    sleep_present = grid_obs['sleep_present'][k]
    c1 = p['c_tilde']
    c2 = c1 + p['delta_c']
    log_w += sleep_present * _ordinal_log_lik(sleep_label, Z, c1, c2)

    # ── Ch3: Steps Poisson ───────────────────────────────────────────
    steps_count = grid_obs['steps_count'][k]
    steps_present = grid_obs['steps_present'][k]
    rate = p['lambda_base'] + p['lambda_step'] * jax.nn.sigmoid(
        10.0 * (W - p['W_thresh']))
    bin_hours = grid_obs.get('steps_bin_hours', jnp.float64(1.0))
    log_w += steps_present * _poisson_log_lik(
        steps_count, rate, bin_hours)

    # ── Ch4: Stress Gaussian ─────────────────────────────────────────
    stress_value = grid_obs['stress_value'][k]
    stress_present = grid_obs['stress_present'][k]
    V_n = grid_obs['V_n'][k]
    stress_mean = p['s_base'] + p['alpha_s'] * W + p['beta_s'] * V_n
    log_w += stress_present * _gauss_log_lik(
        stress_value, stress_mean, p['sigma_s'])

    return log_w


# =========================================================================
# ALIGN_OBS_FN — numpy alignment of raw obs onto sim grid
# =========================================================================
# Takes the raw obs dict produced by the plant (per-channel t_idx +
# values) and produces a single dict with all channels aligned onto
# the simulation grid (every bin), with *_present masks.

def align_obs_fn(obs_data: dict, t_steps: int, dt: float) -> dict:
    """Align raw plant obs to the per-bin grid.

    Args:
        obs_data: dict with keys ``obs_HR``, ``obs_sleep``,
            ``obs_steps``, ``obs_stress`` (each with ``t_idx`` plus
            channel-specific value/label/count keys), and ``V_h``,
            ``V_n``, ``V_c`` (each with ``t_idx`` and ``value``).
        t_steps: number of bins in the window.
        dt: bin width in days (unused here but kept for API parity).

    Returns:
        dict with per-bin arrays for every channel:
        - hr_value, hr_present
        - sleep_label, sleep_present
        - steps_count, steps_present, steps_bin_hours
        - stress_value, stress_present
        - V_h, V_n, V_c    (exogenous controls, always present)
    """
    del dt  # bin width is implicit in t_steps

    out = {}

    # ── HR ────
    hr = obs_data['obs_HR']
    hr_value = np.zeros(t_steps, dtype=np.float64)
    hr_present = np.zeros(t_steps, dtype=np.float64)
    if len(hr['t_idx']) > 0:
        idx = np.asarray(hr['t_idx'])
        # Only fill bins that fall within this window's t_steps
        mask = (idx >= 0) & (idx < t_steps)
        hr_value[idx[mask]] = np.asarray(hr['obs_value'])[mask]
        hr_present[idx[mask]] = 1.0
    out['hr_value'] = hr_value
    out['hr_present'] = hr_present

    # ── Sleep ────
    sl = obs_data['obs_sleep']
    sleep_label = np.zeros(t_steps, dtype=np.int32)
    sleep_present = np.zeros(t_steps, dtype=np.float64)
    if len(sl['t_idx']) > 0:
        idx = np.asarray(sl['t_idx'])
        mask = (idx >= 0) & (idx < t_steps)
        sleep_label[idx[mask]] = np.asarray(sl['obs_label'])[mask]
        sleep_present[idx[mask]] = 1.0
    out['sleep_label'] = sleep_label
    out['sleep_present'] = sleep_present

    # ── Steps ────
    st = obs_data['obs_steps']
    steps_count = np.zeros(t_steps, dtype=np.int32)
    steps_present = np.zeros(t_steps, dtype=np.float64)
    if len(st['t_idx']) > 0:
        idx = np.asarray(st['t_idx'])
        mask = (idx >= 0) & (idx < t_steps)
        steps_count[idx[mask]] = np.asarray(st['obs_count'])[mask]
        steps_present[idx[mask]] = 1.0
    out['steps_count'] = steps_count
    out['steps_present'] = steps_present
    out['steps_bin_hours'] = float(st.get('bin_hours', 1.0))

    # ── Stress ────
    sr = obs_data['obs_stress']
    stress_value = np.zeros(t_steps, dtype=np.float64)
    stress_present = np.zeros(t_steps, dtype=np.float64)
    if len(sr['t_idx']) > 0:
        idx = np.asarray(sr['t_idx'])
        mask = (idx >= 0) & (idx < t_steps)
        stress_value[idx[mask]] = np.asarray(sr['obs_value'])[mask]
        stress_present[idx[mask]] = 1.0
    out['stress_value'] = stress_value
    out['stress_present'] = stress_present

    # ── Exogenous controls (always present per bin) ────
    for key in ('V_h', 'V_n', 'V_c'):
        ch = obs_data[key]
        arr = np.zeros(t_steps, dtype=np.float64)
        if len(ch['t_idx']) > 0:
            idx = np.asarray(ch['t_idx'])
            mask = (idx >= 0) & (idx < t_steps)
            arr[idx[mask]] = np.asarray(ch['value'])[mask]
        out[key] = arr

    return out


# =========================================================================
# SHARD_INIT_FN — start state for a fresh window
# =========================================================================

def shard_init_fn(time_offset, params, exogenous, global_init):
    """Construct the init state for a window starting at time_offset.

    For SWAT, the init state is just the global init (cold start) or
    the previous window's posterior mean (warm start, handled by the
    bench's bridge). This function is the framework's hook to compose
    those — for v1 we always return the global_init.
    """
    del time_offset, params, exogenous
    return global_init


# =========================================================================
# Build the EstimationModel singleton
# =========================================================================

SWAT_ESTIMATION = EstimationModel(
    name="swat",
    version="1.0.0",
    n_states=4,
    n_stochastic=4,
    stochastic_indices=(0, 1, 2, 3),
    state_bounds=(
        (0.0, 1.0),               # W
        (0.0, A_SCALE_FROZEN),    # Z
        (0.0, 10.0),              # a (soft upper)
        (0.0, 5.0),               # T (soft upper, T* ~ 1)
    ),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params=FROZEN_PARAMS,
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    exogenous_keys=('V_h', 'V_n', 'V_c'),
)
