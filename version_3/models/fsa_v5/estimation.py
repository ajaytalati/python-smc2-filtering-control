"""FSA-v5 EstimationModel for ``smc2fc`` integration.

This is the inference-side counterpart to ``simulation.py``. It exports two
``EstimationModel`` objects that the ``smc2fc`` repository imports as the
canonical FSA targets:

  * ``HIGH_RES_FSA_V5_ESTIMATION`` — the v5 default (closed-island
    deconditioning enabled). Maximum-pinned frozen parameter set per the
    Section 11 FIM analysis: structurally non-identifiable parameters
    (``KFB_0, KFS_0``) and parameters not informable from a typical
    short-horizon dataset (``tau_K``, all four v5 deconditioning params)
    are pinned to physiological-baseline values. **37 parameters are
    estimated**; 14 are frozen (8 dynamics-side + 6 diffusion).

  * ``HIGH_RES_FSA_V4_ESTIMATION`` — back-compatibility alias. Same
    structure as v5 but with the v5 Hill-deconditioning amplitudes
    pinned at zero (``mu_dec_B = mu_dec_S = 0``), which makes the drift
    numerically identical to FSA-v4.

Drift is **delegated** to the canonical implementation in
``models.fsa_high_res._dynamics.drift_jax``. There is no inlined drift in
this file — that would risk v4/v5 inconsistencies.

Cross-references to LaTeX docs:
  * Equations of motion           → §11.1 (full v5 model spec)
  * v5 deconditioning Hill term   → §10.2 eq. (eq:v5-mubar)
  * Pinning strategy & FIM basis  → §11.6
  * Observation model             → §11.1 (HR / Stress / Steps / VL / Sleep)
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import jax
import jax.numpy as jnp

from smc2fc.estimation_model import EstimationModel
from smc2fc._likelihood_constants import HALF_LOG_2PI

from version_3.models.fsa_v5._dynamics import (
    A_TYP, F_TYP,
    drift_jax as _drift_jax_canonical,   # single source of truth, see file docstring
)


# ===========================================================================
# FROZEN CONSTANTS — diffusion scales + state-clipping epsilons
# ===========================================================================
# These never enter the estimated parameter vector. They are baked into the
# ``frozen_params`` dict at the bottom of this file.

EPS_A_FROZEN   = 1.0e-4
EPS_B_FROZEN   = 1.0e-4
EPS_S_FROZEN   = 1.0e-4
SIGMA_B_FROZEN = 0.010
SIGMA_S_FROZEN = 0.008
SIGMA_F_FROZEN = 0.012
SIGMA_A_FROZEN = 0.020
SIGMA_K_FROZEN = 0.005
PHI_FROZEN     = 0.0          # circadian phase reference


# ===========================================================================
# FROZEN DYNAMICS — pinned per Section 11 FIM analysis
# ===========================================================================
# These dynamics-side parameters are NOT in the estimated parameter vector.
# Two flavours: V5 enables Hill deconditioning, V4 disables it.
#
# Why these specific freezes?
#
#   * KFB_0, KFS_0 — Section 11.6 result (1): structurally non-identifiable
#     because the K subsystem rapidly relaxes onto a slow manifold and the
#     data only sees the combination ``K_0 + tau_K * mu_K * Phi``, never
#     ``K_0`` alone. No amount of additional data fixes this.
#   * tau_K — Section 11.6 result (5): jointly weakly identified with
#     mu_K. Pinned to the Busso-standard 21 days.
#   * n_dec — structural Hill shape parameter; not learned.
#   * B_dec, S_dec, mu_dec_B, mu_dec_S — pinned to the §10 Figure 7
#     calibration values. Inference of these requires long detraining
#     episodes in the data (Section 11.6 result 3); for a first
#     pipeline-test on simple steady-state synthetic data they are pinned.

_FROZEN_V5_DYNAMICS = {
    'KFB_0':    0.030,
    'KFS_0':    0.050,
    'tau_K':    21.0,
    'n_dec':    4.0,
    'B_dec':    0.07,
    'S_dec':    0.07,
    'mu_dec_B': 0.10,    # closed-island calibration (LaTeX §10.4 Table 5)
    'mu_dec_S': 0.10,
}

# v4-flavour: identical to v5 except deconditioning is silent.
_FROZEN_V4_DYNAMICS = dict(_FROZEN_V5_DYNAMICS)
_FROZEN_V4_DYNAMICS['mu_dec_B'] = 0.0
_FROZEN_V4_DYNAMICS['mu_dec_S'] = 0.0


# ===========================================================================
# PRIORS — FSA-v5 estimated parameters (37 total)
# ===========================================================================
# What's been removed compared to the v4-era PARAM_PRIOR_CONFIG:
#   * KFB_0, KFS_0, tau_K — pinned (see _FROZEN_V5_DYNAMICS).
# What's deliberately NOT added even though v5 introduces them:
#   * B_dec, S_dec, mu_dec_B, mu_dec_S, n_dec — pinned (see above).
# Everything else is unchanged from v4: 18 dynamics parameters minus the 3
# pinned = 15 dynamics; plus 22 observation parameters = 37 total.

PARAM_PRIOR_CONFIG = OrderedDict([
    # ── Dynamics (15 estimated, after pinning KFB_0, KFS_0, tau_K) ──
    ('tau_B',       ('lognormal', (math.log(42.0), 0.10))),
    ('kappa_B',     ('lognormal', (math.log(0.01248), 0.20))),
    ('epsilon_AB',  ('lognormal', (math.log(0.40), 0.05))),

    ('tau_S',       ('lognormal', (math.log(60.0), 0.10))),
    ('kappa_S',     ('lognormal', (math.log(0.00816), 0.20))),
    ('epsilon_AS',  ('lognormal', (math.log(0.20), 0.05))),

    ('tau_F',       ('lognormal', (math.log(6.3636), 0.15))),
    ('lambda_A',    ('lognormal', (math.log(1.00), 0.05))),

    ('mu_K',        ('lognormal', (math.log(0.005), 0.20))),

    ('mu_0',        ('lognormal', (math.log(0.036), 0.20))),
    ('mu_B',        ('lognormal', (math.log(0.30), 0.20))),
    ('mu_S',        ('lognormal', (math.log(0.15), 0.20))),
    ('mu_F',        ('lognormal', (math.log(0.26), 0.20))),
    ('mu_FF',       ('lognormal', (math.log(0.40), 0.05))),
    ('eta',         ('lognormal', (math.log(0.20), 0.15))),

    # ── HR observation (5) ──
    ('HR_base',     ('normal',    (62.0, 2.0))),
    ('kappa_B_HR',  ('lognormal', (math.log(12.0), 0.15))),
    ('alpha_A_HR',  ('lognormal', (math.log(3.0),  0.20))),
    ('beta_C_HR',   ('normal',    (-2.5, 0.5))),
    ('sigma_HR',    ('lognormal', (math.log(2.0), 0.20))),
    # ── Sleep Bernoulli (3) ──
    ('k_C',         ('lognormal', (math.log(3.0), 0.15))),
    ('k_A',         ('lognormal', (math.log(2.0), 0.25))),
    ('c_tilde',     ('normal',    (0.5, 0.25))),
    # ── Stress observation (5) ──
    ('S_base',      ('normal',    (30.0, 3.0))),
    ('k_F',         ('lognormal', (math.log(20.0), 0.20))),
    ('k_A_S',       ('lognormal', (math.log(8.0),  0.25))),
    ('beta_C_S',    ('normal',    (-4.0, 0.8))),
    ('sigma_S',     ('lognormal', (math.log(4.0), 0.20))),
    # ── Steps observation (6) ──
    ('mu_step0',    ('normal',    (5.5, 0.3))),
    ('beta_B_st',   ('lognormal', (math.log(0.8), 0.20))),
    ('beta_F_st',   ('lognormal', (math.log(0.5), 0.25))),
    ('beta_A_st',   ('lognormal', (math.log(0.3), 0.25))),
    ('beta_C_st',   ('normal',    (-0.8, 0.2))),
    ('sigma_st',    ('lognormal', (math.log(0.5), 0.15))),
    # ── Volume Load observation (3) ──
    ('beta_S_VL',   ('lognormal', (math.log(100.0), 0.15))),
    ('beta_F_VL',   ('lognormal', (math.log(20.0),  0.20))),
    ('sigma_VL',    ('lognormal', (math.log(10.0),  0.20))),
])

# Init-state priors — none for the moment (assume DEFAULT_INIT or hand-set).
INIT_STATE_PRIOR_CONFIG = OrderedDict()

# 6D cold-start vector matching the StateSpec ordering [B, S, F, A, KFB, KFS]
COLD_START_INIT = jnp.array([0.05, 0.10, 0.30, 0.10, 0.030, 0.050])

# Index lookups for fast positional access during JIT-compiled propagate_fn.
_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# ===========================================================================
# Internal helper: build the full dynamics-params dict from estimated +
# frozen pieces. Used by every JIT-compiled function below.
# ===========================================================================

def _build_dynamics_params(params, frozen_dynamics):
    """Merge estimated parameter array + frozen dynamics dict → flat dict.

    Returns a dict containing every key the canonical ``drift_jax`` expects.

    NOTE on dtype: ``params`` arrives with whatever dtype the caller cast
    to (fp32 inside the framework's filter inner loop, fp64 in the SMC²
    outer cost). The frozen_dynamics dict has Python-float values which
    JAX defaults to fp64. If we naively merge them in we leak fp64 into
    every fp32 hot path (Hill term in drift, etc.). Cast them to match
    ``params`` dtype so the path stays consistent.
    """
    p = {name: params[_PI[name]] for name in _PK if name in _PI}
    target_dtype = params.dtype
    for fkey, fval in frozen_dynamics.items():
        p[fkey] = jnp.asarray(fval, dtype=target_dtype)
    return p


# ===========================================================================
# PROPAGATE_FN factory — Kalman-fused 6D Euler-Maruyama step
# ===========================================================================
# This is the inner SMC$^2$ inference workhorse. It takes the current
# bootstrap-particle state, applies one bin's worth of deterministic drift
# (delegated to ``_drift_jax_canonical``), folds in the diffusion, runs a
# scalar Kalman update against each Gaussian observation channel, and
# returns the noise-perturbed posterior with a log-weight correction.
#
# We expose two flavours:
#   * propagate_fn_v5: uses _FROZEN_V5_DYNAMICS (Hill deconditioning ON).
#   * propagate_fn_v4: uses _FROZEN_V4_DYNAMICS (Hill deconditioning OFF).
# Both share identical Kalman / observation logic; only the merge of
# frozen dynamics differs.

def _make_propagate_fn(frozen_dynamics):
    def propagate_fn(y, t, dt, params, grid_obs, k,
                     sigma_diag, noise, rng_key):
        del t, rng_key, sigma_diag

        # Build full dynamics dict (estimated + frozen, including v5 Hill keys).
        p = _build_dynamics_params(params, frozen_dynamics)

        B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
        Phi_k = grid_obs['Phi'][k]
        C_k   = grid_obs['C'][k]

        # ── Drift via canonical implementation (single source of truth) ──
        # Note: _drift_jax_canonical signature is (y, params_dict, Phi_t)
        # — direct, no aux-tuple wrapping. Includes the v5 Hill term
        # automatically because frozen_dynamics seeded ``mu_dec_*`` etc.
        d_y = _drift_jax_canonical(y, p, Phi_k)
        y_pred_det = y + dt * d_y

        # ── Prior covariance: state-dependent diffusion squared, scaled by dt ──
        # Same diagonal structure as _dynamics.diffusion_state_dep, with frozen
        # sigma scales (these never change during inference).
        # dtype-cast all scalar literals to the caller dtype so an fp32
        # filter inner loop stays fp32 throughout.
        s_dt = y.dtype
        eps_b = jnp.asarray(EPS_B_FROZEN, dtype=s_dt)
        eps_s = jnp.asarray(EPS_S_FROZEN, dtype=s_dt)
        eps_a = jnp.asarray(EPS_A_FROZEN, dtype=s_dt)
        one   = jnp.asarray(1.0, dtype=s_dt)
        zero_st = jnp.asarray(0.0, dtype=s_dt)
        sB2 = jnp.asarray(SIGMA_B_FROZEN ** 2, dtype=s_dt)
        sS2 = jnp.asarray(SIGMA_S_FROZEN ** 2, dtype=s_dt)
        sF2 = jnp.asarray(SIGMA_F_FROZEN ** 2, dtype=s_dt)
        sA2 = jnp.asarray(SIGMA_A_FROZEN ** 2, dtype=s_dt)
        sK2 = jnp.asarray(SIGMA_K_FROZEN ** 2, dtype=s_dt)
        var_floor = jnp.asarray(1e-12, dtype=s_dt)

        B_cl = jnp.clip(B, eps_b, one - eps_b)
        S_cl = jnp.clip(S, eps_s, one - eps_s)
        var_diag = jnp.array([
            sB2 * B_cl * (one - B_cl) * dt,
            sS2 * S_cl * (one - S_cl) * dt,
            sF2 * jnp.maximum(F, zero_st)               * dt,
            sA2 * (jnp.maximum(A, zero_st) + eps_a)     * dt,
            sK2 * jnp.maximum(KFB, zero_st)             * dt,
            sK2 * jnp.maximum(KFS, zero_st)             * dt,
        ])
        P_prior = jnp.diag(jnp.maximum(var_diag, var_floor))

        # ── Observation Jacobian H (4 Gaussian channels × 6D state) ──────
        # Linear-in-state observation model. Rows: [HR, Stress, Steps, VL].
        # Built inline (NOT via jnp.zeros + .at[].set) so the dtype is
        # inherited from the caller's `p` dict. The framework's filter
        # casts `params` to fp32 (gk_dpf_v3_lite.py:132-139); using
        # jnp.zeros((4,6)) here would default to fp64 and create a
        # carry-type mismatch in the Kalman scan below. Mirrors the
        # v2 estimation.py:194 pattern.
        H = jnp.array([
            [-p['kappa_B_HR'], 0.0,                0.0,             p['alpha_A_HR'], 0.0, 0.0],
            [ 0.0,             0.0,                p['k_F'],       -p['k_A_S'],      0.0, 0.0],
            [ p['beta_B_st'],  0.0,               -p['beta_F_st'],  p['beta_A_st'],  0.0, 0.0],
            [ 0.0,             p['beta_S_VL'],    -p['beta_F_VL'],  0.0,             0.0, 0.0],
        ])

        # Channel-specific bias (intercept + circadian regressor).
        bias = jnp.array([
            p['HR_base']    + p['beta_C_HR']  * C_k,
            p['S_base']     + p['beta_C_S']   * C_k,
            p['mu_step0']   + p['beta_C_st']  * C_k,
            0.0,
        ])
        R_diag = jnp.array([p['sigma_HR']**2, p['sigma_S']**2,
                             p['sigma_st']**2, p['sigma_VL']**2])

        obs_vals = jnp.array([
            grid_obs['hr_value'][k],     grid_obs['stress_value'][k],
            grid_obs['log_steps_value'][k], grid_obs['vl_value'][k]])
        obs_pres = jnp.array([
            grid_obs['hr_present'][k],   grid_obs['stress_present'][k],
            grid_obs['steps_present'][k], grid_obs['vl_present'][k]])

        # ── Sequential scalar Kalman fusion across the 4 Gaussian channels ──
        def _kalman_step(carry, ch):
            mu_, P, lp = carry
            h_i, b_i, r_i, y_i, pres_i = ch
            innov = y_i - (h_i @ mu_ + b_i)
            Ph    = P @ h_i
            S_i   = h_i @ Ph + r_i
            K_i   = Ph / S_i
            ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov**2 / S_i
            mu_   = mu_ + pres_i * K_i * innov
            P     = P    - pres_i * jnp.outer(K_i, Ph)
            lp    = lp   + pres_i * ll_i
            return (mu_, P, lp), None

        # Carry init: cast `0.0` to caller dtype so the scan body's input
        # carry types match its output. Without this, `0.0` is a Python
        # fp64 literal and the scan crashes when called with fp32 inputs
        # by the framework's filter (gk_dpf_v3_lite.py casts to fp32).
        zero_lp = jnp.asarray(0.0, dtype=y_pred_det.dtype)
        (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
            _kalman_step, (y_pred_det, P_prior, zero_lp),
            (H, bias, R_diag, obs_vals, obs_pres))

        # Sample from the Kalman posterior — perturb mean by Cholesky(P) * noise.
        # dtype-match the regulariser so an fp32 hot-loop call stays fp32
        # (mirrors v2 estimation.py:239 pattern).
        eye6 = jnp.eye(6, dtype=P_fused.dtype)
        eps  = jnp.asarray(1e-10, dtype=P_fused.dtype)
        P_safe = P_fused + eps * eye6
        L = jnp.linalg.cholesky(P_safe)
        x_new = mu_fused + L @ noise

        # State clipping — keep particle on the physical manifold.
        # Re-use the dtype-cast eps_*/one/zero_st from above so the post-
        # clip stays at the caller's dtype (no fp32 -> fp64 promotion).
        B_n   = jnp.clip(x_new[0], eps_b, one - eps_b)
        S_n   = jnp.clip(x_new[1], eps_s, one - eps_s)
        F_n   = jnp.maximum(x_new[2], zero_st)
        A_n   = jnp.maximum(x_new[3], zero_st)
        KFB_n = jnp.maximum(x_new[4], zero_st)
        KFS_n = jnp.maximum(x_new[5], zero_st)
        y_new = jnp.array([B_n, S_n, F_n, A_n, KFB_n, KFS_n])

        # Log-weight correction: actual obs-likelihood evaluated at the
        # post-clipping state, minus the proposal log-likelihood absorbed
        # by the Kalman fusion above. Yields the SMC importance weight.
        preds_new  = H @ y_new + bias
        resids_new = obs_vals - preds_new
        obs_ll_new = jnp.sum(obs_pres * (
            -0.5 * resids_new**2 / R_diag
            - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
        pred_lw    = log_pred_total - obs_ll_new
        return y_new, pred_lw

    return propagate_fn


propagate_fn_v5 = _make_propagate_fn(_FROZEN_V5_DYNAMICS)
propagate_fn_v4 = _make_propagate_fn(_FROZEN_V4_DYNAMICS)


def diffusion_fn(params):
    """Constant diagonal diffusion vector (length 6); does not depend on params."""
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                       SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN])


# ===========================================================================
# OBSERVATION LOG-LIKELIHOOD — same as v4, no v5 changes needed
# ===========================================================================

def _gaussian_obs_ll(y, grid_obs, k, params):
    """Per-bin Gaussian log-likelihood for HR + Stress + Steps + VL.

    Sleep is handled by a Bernoulli term inside ``obs_log_weight_fn``.
    """
    B, S, F, A = y[0], y[1], y[2], y[3]
    C_k = grid_obs['C'][k]
    p = {name: params[_PI[name]] for name in _PK if name in _PI}

    pHR = p['HR_base']    - p['kappa_B_HR'] * B + p['alpha_A_HR'] * A + p['beta_C_HR']  * C_k
    pS  = p['S_base']     + p['k_F']        * F - p['k_A_S']     * A + p['beta_C_S']   * C_k
    pST = p['mu_step0']   + p['beta_B_st']  * B - p['beta_F_st'] * F + p['beta_A_st']  * A + p['beta_C_st'] * C_k
    pVL = p['beta_S_VL']  * S - p['beta_F_VL'] * F

    def _ll(pred, obs_val, obs_pres, sigma):
        resid = obs_val - pred
        return obs_pres * (-0.5 * (resid / sigma)**2 - jnp.log(sigma) - HALF_LOG_2PI)

    lp =  _ll(pHR, grid_obs['hr_value'][k],        grid_obs['hr_present'][k],     p['sigma_HR'])
    lp += _ll(pS,  grid_obs['stress_value'][k],    grid_obs['stress_present'][k], p['sigma_S'])
    lp += _ll(pST, grid_obs['log_steps_value'][k], grid_obs['steps_present'][k],  p['sigma_st'])
    lp += _ll(pVL, grid_obs['vl_value'][k],        grid_obs['vl_present'][k],     p['sigma_VL'])
    return lp


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Full obs log-likelihood: 4 Gaussian channels + Bernoulli sleep label."""
    gauss_ll = _gaussian_obs_ll(x_new, grid_obs, k, params)
    C_k = grid_obs['C'][k]
    p = {name: params[_PI[name]] for name in _PK if name in _PI}

    z = p['k_C'] * C_k + p['k_A'] * x_new[3] - p['c_tilde']
    prob = jax.nn.sigmoid(z)
    prob_safe = jnp.clip(prob, 1e-8, 1.0 - 1e-8)
    s = grid_obs['sleep_label'][k].astype(prob_safe.dtype)
    bern_ll = grid_obs['sleep_present'][k] * (
        s * jnp.log(prob_safe) + (1.0 - s) * jnp.log(1.0 - prob_safe))
    return gauss_ll + bern_ll


def obs_log_prob_fn(y, grid_obs, k, params):
    return obs_log_weight_fn(y, grid_obs, k, params)


# ===========================================================================
# OBS-DATA ALIGNMENT — packs scattered observation streams into bin-resolved
# arrays consumed by propagate_fn / obs_log_weight_fn. Unchanged from v4.
# ===========================================================================

def align_obs_fn(obs_data, t_steps, dt):
    from version_3.models.fsa_v5.simulation import BINS_PER_DAY
    del BINS_PER_DAY, dt   # not actually needed; kept for API stability
    T = t_steps
    def _get(name): return obs_data.get(name) if isinstance(obs_data, dict) else None

    hr_val,     hr_pres   = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    s_val,      s_pres    = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    log_st_val, st_pres   = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    sl_label,   sl_pres   = np.zeros(T, dtype=np.int32),   np.zeros(T, dtype=np.float32)
    vl_val,     vl_pres   = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)

    for name, val, pres in [
        ('obs_HR',         hr_val,     hr_pres),
        ('obs_stress',     s_val,      s_pres),
        ('obs_steps',      log_st_val, st_pres),
        ('obs_sleep',      sl_label,   sl_pres),
        ('obs_volumeload', vl_val,     vl_pres),
    ]:
        ch = _get(name)
        if ch and 't_idx' in ch:
            idx = np.asarray(ch['t_idx']).astype(int)
            mask = (idx >= 0) & (idx < T)
            if name == 'obs_steps':
                val[idx[mask]] = np.log(np.asarray(ch['obs_value'])[mask] + 1.0)
            elif name == 'obs_sleep':
                val[idx[mask]] = np.asarray(ch['sleep_label'])[mask]
            else:
                val[idx[mask]] = np.asarray(ch['obs_value'])[mask]
            pres[idx[mask]] = 1.0

    p_ch = _get('Phi'); Phi_val = np.zeros((T, 2), dtype=np.float32)
    if p_ch and 'Phi_value' in p_ch:
        Phi_val[:min(len(p_ch['Phi_value']), T)] = p_ch['Phi_value'][:T]

    c_ch = _get('C'); C_val = np.zeros(T, dtype=np.float32)
    if c_ch and 'C_value' in c_ch:
        C_val[:min(len(c_ch['C_value']), T)] = c_ch['C_value'][:T]

    has_any = np.maximum.reduce([hr_pres, s_pres, st_pres, sl_pres, vl_pres])

    return {
        'hr_value': jnp.array(hr_val), 'hr_present': jnp.array(hr_pres),
        'stress_value': jnp.array(s_val), 'stress_present': jnp.array(s_pres),
        'log_steps_value': jnp.array(log_st_val), 'steps_present': jnp.array(st_pres),
        'sleep_label': jnp.array(sl_label), 'sleep_present': jnp.array(sl_pres),
        'vl_value': jnp.array(vl_val), 'vl_present': jnp.array(vl_pres),
        'Phi': jnp.array(Phi_val), 'C': jnp.array(C_val),
        'has_any_obs': jnp.array(has_any),
    }


# ===========================================================================
# FORWARD SDE & ASSEMBLY — used for diagnostics + plot generation
# ===========================================================================

def _make_forward_sde_stochastic(frozen_dynamics):
    """Factory: returns a forward-SDE simulator closing over frozen dynamics."""
    def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps, rng_key=None):
        p = _build_dynamics_params(params, frozen_dynamics)
        sqrt_dt = jnp.sqrt(dt)
        Phi_arr = jnp.asarray(exogenous['Phi'])
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        def step(carry, i):
            y, key = carry
            key, nk = jax.random.split(key)
            noise = jax.random.normal(nk, (6,))
            # Drift via canonical
            d_y = _drift_jax_canonical(y, p, Phi_arr[i])
            y_next = y + dt * d_y
            # State-dependent diffusion (mirrors _dynamics.diffusion_state_dep)
            B = jnp.clip(y[0], 1e-4, 0.999)
            S = jnp.clip(y[1], 1e-4, 0.999)
            F = jnp.maximum(y[2], 0.0)
            A = jnp.maximum(y[3], 0.0)
            KFB = jnp.maximum(y[4], 0.0)
            KFS = jnp.maximum(y[5], 0.0)
            sigma_y = jnp.array([
                SIGMA_B_FROZEN * jnp.sqrt(B * (1.0 - B)),
                SIGMA_S_FROZEN * jnp.sqrt(S * (1.0 - S)),
                SIGMA_F_FROZEN * jnp.sqrt(F),
                SIGMA_A_FROZEN * jnp.sqrt(A + 1e-4),
                SIGMA_K_FROZEN * jnp.sqrt(KFB),
                SIGMA_K_FROZEN * jnp.sqrt(KFS),
            ])
            y_next = y_next + sigma_y * sqrt_dt * noise
            # Domain clipping (Jacobi reflection for B/S, half-line for F/A/K)
            y_next = jnp.array([
                jnp.clip(y_next[0], 1e-4, 0.999),
                jnp.clip(y_next[1], 1e-4, 0.999),
                jnp.maximum(y_next[2], 0.0),
                jnp.maximum(y_next[3], 0.0),
                jnp.maximum(y_next[4], 0.0),
                jnp.maximum(y_next[5], 0.0),
            ])
            return (y_next, key), y_next

        (_, _), traj = jax.lax.scan(step, (init_state, rng_key), jnp.arange(n_steps))
        return traj
    return forward_sde_stochastic


forward_sde_stochastic_v5 = _make_forward_sde_stochastic(_FROZEN_V5_DYNAMICS)
forward_sde_stochastic_v4 = _make_forward_sde_stochastic(_FROZEN_V4_DYNAMICS)


def _make_imex_step_fn(frozen_dynamics):
    """Factory: deterministic Euler step (no noise) for IMEX-style use."""
    def imex_step_fn(y, t, dt, params, grid_obs):
        del t
        p = _build_dynamics_params(params, frozen_dynamics)
        Phi_k = grid_obs.get('Phi_k', jnp.zeros(2))
        d_y = _drift_jax_canonical(y, p, Phi_k)
        return y + dt * d_y
    return imex_step_fn


imex_step_fn_v5 = _make_imex_step_fn(_FROZEN_V5_DYNAMICS)
imex_step_fn_v4 = _make_imex_step_fn(_FROZEN_V4_DYNAMICS)


def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def get_init_theta():
    """Return prior-mode parameter vector (length = len(PARAM_PRIOR_CONFIG))."""
    def _mode(ptype, pa):
        # Lognormal mode is exp(mu - sigma^2); but the existing code uses the
        # MEAN exp(mu + sigma^2/2). Keep that convention for back-compat.
        return math.exp(pa[0] + pa[1]**2 / 2) if ptype == 'lognormal' else pa[0]
    return np.array([_mode(pt, pa) for _, (pt, pa) in PARAM_PRIOR_CONFIG.items()],
                     dtype=np.float32)


# ===========================================================================
# CANONICAL ESTIMATION MODELS
# ===========================================================================
# Two flavours, both use the same parameter prior config. The only
# difference is in the frozen_params dict (which seeds the v5 Hill term)
# and the propagate / forward-sde / imex helpers (which read from the
# matching frozen-dynamics constant).

# === FSA-v5 (Hill deconditioning ON; the smc2fc port default) ===
HIGH_RES_FSA_V5_ESTIMATION = EstimationModel(
    name="fsa_high_res_v5",
    version="5.0",
    n_states=6,
    n_stochastic=6,
    stochastic_indices=(0, 1, 2, 3, 4, 5),
    state_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 10.0),
                   (0.0, 5.0), (0.0, 1.0), (0.0, 1.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={
        # Diffusion scales (always frozen in production)
        'sigma_B': SIGMA_B_FROZEN, 'sigma_S': SIGMA_S_FROZEN,
        'sigma_F': SIGMA_F_FROZEN, 'sigma_A': SIGMA_A_FROZEN,
        'sigma_K': SIGMA_K_FROZEN, 'phi':     PHI_FROZEN,
        # Pinned dynamics (Section 11 FIM analysis)
        **_FROZEN_V5_DYNAMICS,
    },
    exogenous_keys=('Phi',),
    propagate_fn=propagate_fn_v5,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic_v5,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn_v5,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
)


# === FSA-v4 alias (Hill deconditioning OFF — recovers v4 numerics exactly) ===
HIGH_RES_FSA_V4_ESTIMATION = EstimationModel(
    name="fsa_high_res_v4",
    version="4.0",
    n_states=6,
    n_stochastic=6,
    stochastic_indices=(0, 1, 2, 3, 4, 5),
    state_bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 10.0),
                   (0.0, 5.0), (0.0, 1.0), (0.0, 1.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={
        'sigma_B': SIGMA_B_FROZEN, 'sigma_S': SIGMA_S_FROZEN,
        'sigma_F': SIGMA_F_FROZEN, 'sigma_A': SIGMA_A_FROZEN,
        'sigma_K': SIGMA_K_FROZEN, 'phi':     PHI_FROZEN,
        **_FROZEN_V4_DYNAMICS,    # mu_dec_* = 0 → Hill term silent
    },
    exogenous_keys=('Phi',),
    propagate_fn=propagate_fn_v4,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic_v4,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn_v4,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
)
