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
  - Steps log-Gaussian, wake-gated (matches FSA-v2)
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
# pinning fixes from the Repo C identifiability analysis:
#
# - **tau_T = 2.0 days (= 48 h)** — pinned to break the Stuart-Landau
#   time-vs-rate scaling degeneracy.
# - **lambda_amp_Z = 8.0** — pinned because (lambda_amp_W, lambda_amp_Z)
#   only enter E_dyn through their product.

FROZEN_PARAMS = dict(
    # Dynamics (synchronized with _dynamics.py:TRUTH_PARAMS)
    kappa=6.67,
    lmbda=32.0,
    gamma_3=8.0,
    beta_Z=4.0,
    tau_W=2.0 / 24.0,
    tau_Z=0.25 / 24.0,           # Matches fixed _dynamics.py
    tau_a=10.0 / 24.0,           # Matches fixed _dynamics.py
    tau_T=48.0 / 24.0,           # 2.0 days (pinned)
    eta=0.5,
    mu_E=1.0,
    lambda_amp_W=5.0,
    lambda_amp_Z=8.0,            # pinned
    
    # Diffusion temperatures (per day)
    T_W=0.01 * 24.0,
    T_Z=0.05 * 24.0,
    T_a=0.01 * 24.0,
    T_T=0.0001 * 24.0,

    # Observation Constants (synchronized with simulation.py:_OBS_PARAMS)
    HR_base=50.0,
    s_base=30.0,
    alpha_HR=25.0,
    alpha_s=40.0,
    beta_W_steps=0.8,
    beta_s=10.0,
    
    A_scale=A_SCALE_FROZEN,
    phi_0=PHI_0_FROZEN,
    V_c_max=V_C_MAX_HOURS,
    sleep_sharpness=10.0,
    tau_sleep_persist_h=1.0,
)


# =========================================================================
# PARAMETER PRIORS — 11 estimable scalars (The Identifiable Subset)
# =========================================================================
# Parameters that are not frozen above and are expected to be identified
# from the 4-channel observation model.

PARAM_PRIOR_CONFIG = OrderedDict([
    # Stuart-Landau bifurcation block
    ('E_crit',       ('normal',    (0.5, 0.1))),
    ('alpha_T',      ('lognormal', (math.log(0.3),  0.3))),

    # Entrainment-amplitude block
    ('V_n_scale',    ('lognormal', (math.log(2.0), 0.3))),

    # Obs ch1: HR Gaussian
    ('delta_HR',     ('normal',    (0.0, 5.0))),
    ('sigma_HR',     ('lognormal', (math.log(8.0),  0.3))),

    # Obs ch2: Sleep 3-level ordinal
    ('c_tilde',      ('normal',    (0.417, 0.10))),
    ('delta_c',      ('lognormal', (math.log(0.25), 0.3))),

    # Obs ch3: Steps log-Gaussian
    ('mu_step0',     ('normal',    (4.0, 0.3))),
    ('sigma_step',   ('lognormal', (math.log(0.5), 0.15))),

    # Obs ch4: Stress Gaussian
    ('delta_s',      ('normal',    (0.0, 10.0))),
    ('sigma_s',      ('lognormal', (math.log(15.0), 0.3))),
])


# =========================================================================
# INIT-STATE PRIORS (W_0, Z_0, a_0, T_0)
# =========================================================================

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('W_0',  ('normal', (0.50, 0.05))),
    ('Z_0',  ('normal', (0.583, 0.05))),
    ('a_0',  ('normal', (0.50, 0.05))),
    ('T_0',  ('normal', (0.00, 0.05))),    # Pathological start T=0
])

# Fast lookup: parameter name -> index in the prior-vector
_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# Cold-start initial state for the very first window's prior mean.
COLD_START_INIT = jnp.array([0.5, 0.583, 0.5, 0.0], dtype=jnp.float64)


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
    """GK-DPF for SWAT 4-state.

    Uses 10x deterministic sub-stepping to find the prior mean, then
    performs sequential scalar Kalman fusion of the 3 Gaussian channels
    (HR, Steps, Stress) at the bin boundary to generate a guided proposal.
    """
    del sigma_diag, rng_key
    n_substeps = 10
    sub_dt = jnp.asarray(dt / float(n_substeps), dtype=y.dtype)

    # Build params dict
    p_dict = {name: params[idx].astype(y.dtype) for name, idx in _PI.items()}
    for fname, fval in FROZEN_PARAMS.items():
        p_dict[fname] = jnp.asarray(fval, dtype=y.dtype)

    # Read controls
    V_h = grid_obs['V_h'][k]
    V_n = grid_obs['V_n'][k]
    V_c = grid_obs['V_c'][k]
    u = jnp.array([V_h, V_n, V_c], dtype=y.dtype)

    # 1. Prior Mean via 10x deterministic sub-stepping
    def sub_body(y_curr, k_sub):
        t_sub = t + k_sub * sub_dt
        d_y = drift_jax(y_curr, p_dict, t_sub, u)
        y_next = y_curr + sub_dt * d_y
        return state_clip(y_next), None

    mu_prior, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps, dtype=y.dtype))

    # 2. Prior Covariance (using diffusion at start of bin)
    sigma_y = diffusion_state_dep(y, p_dict)
    var_prior = jnp.maximum(sigma_y ** 2 * dt, 1e-12)
    P_prior = jnp.diag(var_prior)

    # 3. Kalman Fusion setup
    H = jnp.array([
        [p_dict['alpha_HR'], 0.0, 0.0, 0.0],
        [p_dict['alpha_s'],  0.0, 0.0, 0.0],
        [p_dict['beta_W_steps'], 0.0, 0.0, 0.0],
    ], dtype=y.dtype)

    bias = jnp.array([
        p_dict['HR_base'] + p_dict['delta_HR'],
        p_dict['s_base'] + p_dict['delta_s'] + p_dict['beta_s'] * V_n,
        p_dict['mu_step0'],
    ], dtype=y.dtype)

    R_diag = jnp.array([
        p_dict['sigma_HR'] ** 2,
        p_dict['sigma_s'] ** 2,
        p_dict['sigma_step'] ** 2,
    ], dtype=y.dtype)

    obs_vals = jnp.array([
        grid_obs['hr_value'][k],
        grid_obs['stress_value'][k],
        grid_obs['log_steps_value'][k],
    ], dtype=y.dtype)

    obs_pres = jnp.array([
        grid_obs['hr_present'][k],
        grid_obs['stress_present'][k],
        grid_obs['steps_present'][k],
    ], dtype=y.dtype)

    # 4. Sequential Kalman updates
    def _kalman_step(carry, ch):
        mu, P, lp = carry
        h_i, b_i, r_i, y_i, pres_i = ch
        innov = y_i - (h_i @ mu + b_i)
        Ph    = P @ h_i
        S_i   = h_i @ Ph + r_i
        K_i   = Ph / S_i
        ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov ** 2 / S_i
        mu = mu + pres_i * K_i * innov
        P  = P  - pres_i * jnp.outer(K_i, Ph)
        lp = lp + pres_i * ll_i
        return (mu, P, lp), None

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
        _kalman_step,
        (mu_prior, P_prior, jnp.float64(0.0)),
        (H, bias, R_diag, obs_vals, obs_pres),
    )

    # 5. Sample from fused posterior
    P_safe = P_fused + jnp.asarray(1e-10, dtype=y.dtype) * jnp.eye(4, dtype=y.dtype)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise
    y_new = state_clip(x_new)

    # 6. Weight correction (predictive log-weight = Kalman marginal - exact Gaussian obs LL)
    preds_new = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5 * resids_new ** 2 / R_diag
                                      - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


# =========================================================================
# DIFFUSION_FN — diagonal noise vector from params
# =========================================================================

def diffusion_fn(params):
    """Return the diagonal sigma vector (n_states,) given a params VECTOR.
    Evaluated at COLD_START_INIT to ensure initial particle diversity.
    """
    p_dict = {name: params[idx] for name, idx in _PI.items()}
    # Add frozen constants
    for fname, fval in FROZEN_PARAMS.items():
        p_dict[fname] = jnp.float64(fval)
    return diffusion_state_dep(COLD_START_INIT, p_dict)


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


def _ordinal_log_lik(obs_label, Z, c1, c2, sharp, k_prev, is_first_bin):
    """3-level ordinal with sticky-HMM persistence.
    """
    s1 = jax.nn.sigmoid(sharp * (Z - c1))
    s2 = jax.nn.sigmoid(sharp * (Z - c2))
    p_wake = 1.0 - s1
    p_light = s1 - s2
    p_deep = s2
    p_marg = jnp.where(obs_label == 0, p_wake,
        jnp.where(obs_label == 1, p_light, p_deep))
    
    # dt_h = 15 mins / 60 = 0.25h
    dt_h = jnp.float64(0.25)
    tau = jnp.float64(FROZEN_PARAMS['tau_sleep_persist_h'])
    P_stay = jnp.exp(-dt_h / tau)
    
    is_same = (obs_label == k_prev)
    p_sticky = P_stay * is_same + (1.0 - P_stay) * p_marg
    
    p_eff = jax.lax.cond(is_first_bin, lambda: p_marg, lambda: p_sticky)
    return jnp.log(jnp.maximum(p_eff, 1e-30))

def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Total observation log-weight for the particle at x_new.
    Uses the exact sticky-HMM for the sleep channel.
    """
    p = {name: params[idx] for name, idx in _PI.items()}
    for fname, fval in FROZEN_PARAMS.items():
        p[fname] = jnp.float64(fval)

    W = x_new[0]
    Z = x_new[1]

    log_w = jnp.float64(0.0)

    # ── Ch1: HR Gaussian ─────────────────────────────────────────────
    hr_value = grid_obs['hr_value'][k]
    hr_present = grid_obs['hr_present'][k]
    hr_mean = (p['HR_base'] + p['delta_HR']) + p['alpha_HR'] * W
    log_w += hr_present * _gauss_log_lik(hr_value, hr_mean, p['sigma_HR'])

    # ── Ch2: Sleep 3-level ordinal (Sticky HMM) ──────────────────────
    sleep_label = grid_obs['sleep_label'][k]
    sleep_present = grid_obs['sleep_present'][k]
    c1 = p['c_tilde']
    c2 = c1 + p['delta_c']
    sharp = jnp.float64(FROZEN_PARAMS['sleep_sharpness'])
    
    k_prev = jax.lax.cond(k > 0, lambda: grid_obs['sleep_label'][k-1], lambda: jnp.int32(0))
    is_first_bin = (k == 0)
    
    log_w += sleep_present * _ordinal_log_lik(
        sleep_label, Z, c1, c2, sharp, k_prev, is_first_bin)

    # ── Ch3: Steps log-Gaussian (wake-gated) ─────────────────────────
    log_steps_value = grid_obs['log_steps_value'][k]
    steps_present = grid_obs['steps_present'][k]
    step_mean = p['mu_step0'] + p['beta_W_steps'] * W
    log_w += steps_present * _gauss_log_lik(
        log_steps_value, step_mean, p['sigma_step'])

    # ── Ch4: Stress Gaussian ─────────────────────────────────────────
    stress_value = grid_obs['stress_value'][k]
    stress_present = grid_obs['stress_present'][k]
    V_n = grid_obs['V_n'][k]
    stress_mean = (p['s_base'] + p['delta_s']) + p['alpha_s'] * W + p['beta_s'] * V_n
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
        - log_steps_value, steps_present  (steps_present is AND of
          sample-exists and wake-gate; log-Gaussian likelihood)
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

    # ── Steps (log-Gaussian, wake-gated) ────
    # gen_obs_steps returns t_idx, log_value, present_mask (1 only in
    # wake bins). steps_present here is the AND of "obs sample exists
    # at this bin" with "wake gate is open at this bin".
    st = obs_data['obs_steps']
    log_steps_value = np.zeros(t_steps, dtype=np.float64)
    steps_present = np.zeros(t_steps, dtype=np.float64)
    if len(st['t_idx']) > 0:
        idx = np.asarray(st['t_idx'])
        mask = (idx >= 0) & (idx < t_steps)
        log_vals = np.asarray(st['log_value'])[mask]
        present_vals = np.asarray(st['present_mask'])[mask].astype(np.float64)
        log_steps_value[idx[mask]] = log_vals
        steps_present[idx[mask]] = present_vals
    out['log_steps_value'] = log_steps_value
    out['steps_present'] = steps_present

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

    # ── Combined mask (optional but recommended for speed) ────
    # 1.0 if any channel has an observation at this bin
    has_any = np.maximum.reduce([
        hr_present, sleep_present, steps_present, stress_present])
    out['has_any_obs'] = has_any.astype(np.float64)

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
        (0.0, 1.0),               # W (Jacobi)
        (0.0, 1.0),               # Z (Jacobi, rescaled from [0, A_SCALE_FROZEN])
        (0.0, 1.0),               # a (Jacobi)
        (0.0, 5.0),               # T (sqrt-CIR, soft upper; T* ~ 1)
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
