
"""
2-State Controlled Bistable model for sequential estimation + control.
======================================================================
Date:    18 April 2026
Version: 1.0

MATHEMATICAL SPECIFICATION
--------------------------

Latent SDE (two states):

    dx = [alpha * x * (a^2 - x^2) + u] dt  +  sqrt(2 sigma_x) dB_x
    du = -gamma * (u - u_target(t)) dt      +  sqrt(2 sigma_u) dB_u

    alpha      -- double-well strength (1/time)
    a          -- well separation; minima at x = +/- a
    sigma_x    -- x noise temperature (diffusivity)
    gamma      -- u mean-reversion rate (tau_u = 1/gamma)
    sigma_u    -- u noise temperature
    u_target(t)-- EXOGENOUS piecewise-constant intervention schedule

Critical tilt for saddle-node bifurcation of the x-drift:

    u_c = 2 * alpha * a^3 / (3 sqrt(3))    (0.385 for alpha = a = 1)

    |u| > u_c : x-landscape monostable (deterministic transition)
    |u| < u_c : x-landscape bistable (noise-assisted transition only)

Observation model:

    y_k = x_k + eps_k,   eps_k ~ N(0, sigma_obs^2)

Euler-Maruyama discretisation (used for both forward simulation and
particle-filter propagation):

    x_{k+1} = x_k + [alpha * x_k * (a^2 - x_k^2) + u_k] * dt
              + sqrt(2 sigma_x dt) * xi_x
    u_{k+1} = u_k + [-gamma * (u_k - u_target_k)] * dt
              + sqrt(2 sigma_u dt) * xi_u

where u_target_k is read from grid_obs['u_target'] at step k.

Estimated parameters:  alpha, a, sigma_x, gamma, sigma_u, sigma_obs  (6)
Initial states:        x_0, u_0                                      (2)
Total dimensions:      8

Exogenous input:       u_target(t) -- passed via grid_obs['u_target'],
                       constructed by align_obs_fn from the simulator's
                       'u_target' channel output.
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from smc2fc.estimation_model import EstimationModel
from smc2fc._likelihood_constants import HALF_LOG_2PI

# --- Priors -------------------------------------------------------

PARAM_PRIOR_CONFIG = OrderedDict([
    ('alpha',     ('lognormal', (0.0,              0.5))),  # median 1.0
    ('a',         ('lognormal', (0.0,              0.3))),  # median 1.0
    ('sigma_x',   ('lognormal', (-2.0,             0.5))),  # median ~0.135
    ('gamma',     ('lognormal', (math.log(2.0),    0.3))),  # median 2.0
    ('sigma_u',   ('lognormal', (-3.0,             0.5))),  # median ~0.050
    ('sigma_obs', ('lognormal', (-1.5,             0.3))),  # median ~0.22
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    # Weakly-informative priors that do not pin the initial condition
    # to either well (so the filter posterior over x_0 is genuinely
    # uncertain pre-intervention).
    ('x_0', ('normal', (-1.0, 0.5))),
    ('u_0', ('normal', ( 0.0, 0.3))),
])

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# --- SDE dynamics -------------------------------------------------

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    """Locally-optimal (guided) PF Euler-Maruyama step.

    For the OBSERVED state x:  sample from the Bayes-optimal proposal
        q*(x_{k+1} | x_k, y_{k+1}) proportional to p(x_{k+1}|x_k) p(y_{k+1}|x_{k+1})
    which is the Gaussian obtained by fusing the Euler prediction with
    the observation:

        sigma_prop^2 = 1 / (1/sigma_proc_x^2 + 1/sigma_obs^2)
        mu_prop      = sigma_prop^2 * (x_pred / sigma_proc_x^2
                                       + y_obs  / sigma_obs^2)

    For the UNOBSERVED state u:  keep bootstrap proposal from p(u_{k+1}|u_k).

    Weight correction (derived from the importance ratio):

        step_log_w_inc = log[ p(x_{k+1}|x_k) p(y_{k+1}|x_{k+1})
                              / q*(x_{k+1}|x_k, y_{k+1}) ]
                       = log p(y_{k+1}|x_k)          -- the PREDICTIVE likelihood
                       = log N(y_{k+1}; x_pred, sigma_proc_x^2 + sigma_obs^2)

    This is a SAMPLE-INDEPENDENT quantity, which is the key variance-
    reduction property of the locally-optimal proposal: it does not
    depend on the particular x_{k+1} that was drawn.

    The filter framework computes step_lw = pred_lw + obs_lw, where
    obs_lw = log p(y_{k+1}|x_{k+1}).  To make the total equal the
    predictive likelihood we return

        pred_lw = log p(y_{k+1}|x_k) - log p(y_{k+1}|x_{k+1})

    so the framework-added obs_lw cancels exactly, leaving step_lw =
    log p(y_{k+1}|x_k).

    When obs_present[k] = 0 (missing observation), we fall back to pure
    bootstrap: no guidance, pred_lw = 0, framework's obs_lw = 0 via mask.

    sigma_diag argument is kept for API compatibility but unused;
    we read process-noise scales directly from params.
    """
    del t, rng_key, sigma_diag

    alpha           = params[_PI['alpha']]
    a               = params[_PI['a']]
    gamma           = params[_PI['gamma']]
    sigma_x_param   = params[_PI['sigma_x']]
    sigma_u_param   = params[_PI['sigma_u']]
    sigma_obs_param = params[_PI['sigma_obs']]

    x = y[0]
    u = y[1]
    u_target_k = grid_obs['u_target'][k]

    # --- Deterministic Euler prediction -----------------------------
    drift_x = alpha * x * (a**2 - x**2) + u
    drift_u = -gamma * (u - u_target_k)
    x_pred = x + dt * drift_x
    u_pred = u + dt * drift_u

    # --- Noise variances --------------------------------------------
    sigma_proc_x_sq = 2.0 * sigma_x_param * dt
    sigma_proc_u_sq = 2.0 * sigma_u_param * dt
    sigma_obs_sq    = sigma_obs_param * sigma_obs_param

    # --- Locally-optimal proposal for x (observed state) ------------
    # Bayes-rule fusion of prior N(x_pred, sigma_proc_x_sq) and
    # likelihood N(y; x, sigma_obs_sq).  Closed-form Gaussian.
    obs_present = grid_obs['obs_present'][k]
    y_obs       = grid_obs['obs_value'][k]

    sum_var       = sigma_proc_x_sq + sigma_obs_sq
    sigma_prop_sq = sigma_proc_x_sq * sigma_obs_sq / sum_var
    mu_prop       = (x_pred * sigma_obs_sq + y_obs * sigma_proc_x_sq) / sum_var

    # When no obs: use bootstrap (mu = x_pred, var = sigma_proc_x_sq)
    guided      = obs_present > 0.5
    mu_x        = jnp.where(guided, mu_prop,       x_pred)
    sigma_x_sq  = jnp.where(guided, sigma_prop_sq, sigma_proc_x_sq)

    # --- Sample from proposals --------------------------------------
    x_new = mu_x   + jnp.sqrt(sigma_x_sq)      * noise[0]
    u_new = u_pred + jnp.sqrt(sigma_proc_u_sq) * noise[1]

    # --- Weight correction ------------------------------------------
    # Guided case: pred_lw = log p(y|x_k) - log p(y|x_{k+1})
    # so that pred_lw + obs_lw = log p(y|x_k) (sample-independent).
    HALF_LOG_2PI_L = 0.5 * jnp.log(2.0 * jnp.pi)
    log_predictive = (-0.5 * (y_obs - x_pred)**2 / sum_var
                      - 0.5 * jnp.log(sum_var) - HALF_LOG_2PI_L)
    log_obs_new    = (-0.5 * (y_obs - x_new)**2 / sigma_obs_sq
                      - 0.5 * jnp.log(sigma_obs_sq) - HALF_LOG_2PI_L)
    pred_lw = jnp.where(guided, log_predictive - log_obs_new, 0.0)

    return jnp.array([x_new, u_new]), pred_lw


def diffusion_fn(params):
    """Diagonal SDE diffusion coefficients [sqrt(2*sigma_x), sqrt(2*sigma_u)]."""
    return jnp.array([
        jnp.sqrt(2.0 * params[_PI['sigma_x']]),
        jnp.sqrt(2.0 * params[_PI['sigma_u']]),
    ])


def imex_step_fn(y, t, dt, params, grid_obs):
    """Deterministic Euler step (drift only, no noise) for EKF prediction."""
    del t
    alpha = params[_PI['alpha']]
    a     = params[_PI['a']]
    gamma = params[_PI['gamma']]
    x = y[0]
    u = y[1]
    # The EKF needs a u_target at step k; since imex_step_fn receives grid_obs
    # but not k explicitly, we pull it from grid_obs if available.  In the
    # generic pipeline imex_step_fn is called with the same grid_obs / k
    # bindings as propagate_fn.
    u_target_k = grid_obs.get('u_target_k', 0.0)
    x_new = x + dt * (alpha * x * (a**2 - x**2) + u)
    u_new = u + dt * (-gamma * (u - u_target_k))
    return jnp.array([x_new, u_new])


# --- Observation model --------------------------------------------

def obs_log_prob_fn(y, grid_obs, k, params):
    """Gaussian observation log-prob at step k (x only)."""
    sigma_obs = params[_PI['sigma_obs']]
    resid     = grid_obs['obs_value'][k] - y[0]
    return grid_obs['obs_present'][k] * (
        -0.5 * (resid / sigma_obs) ** 2
        - jnp.log(sigma_obs) - HALF_LOG_2PI)


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Observation log-weight for particle filter."""
    return obs_log_prob_fn(x_new, grid_obs, k, params)


def gaussian_obs_fn(y, grid_obs, k, params):
    """Per-step Gaussian observation info for the EKF (x channel only)."""
    sigma_obs = params[_PI['sigma_obs']]
    return {
        'mean':     jnp.array([y[0]]),
        'value':    jnp.array([grid_obs['obs_value'][k]]),
        'cov_diag': jnp.array([sigma_obs ** 2]),
        'present':  jnp.array([grid_obs['obs_present'][k]]),
    }


def obs_sample_fn(y, exog, k, params, rng_key):
    """Sample observation for a given state."""
    del exog, k
    sigma_obs = params[_PI['sigma_obs']]
    return {
        'obs_value': y[0] + sigma_obs * jax.random.normal(rng_key, dtype=y.dtype),
    }


# --- Grid alignment -----------------------------------------------

def align_obs_fn(obs_data, t_steps, dt_hours):
    """Align observations + the u_target exogenous schedule to the grid.

    obs_data keys expected:
        t_idx            -- standard simulator indexing
        obs_value        -- observed y = x + noise (for the x channel)
        u_target_value   -- the exogenous schedule (from the u_target channel)
    """
    del dt_hours
    T = t_steps
    val   = np.zeros(T, dtype=np.float32)
    pres  = np.zeros(T, dtype=np.float32)
    u_tgt = np.zeros(T, dtype=np.float32)

    if hasattr(obs_data, '__getitem__') and 't_idx' in obs_data:
        idx = np.asarray(obs_data['t_idx']).astype(int)
        if 'obs_value' in obs_data:
            val[idx]  = np.asarray(obs_data['obs_value']).astype(np.float32)
            pres[idx] = 1.0
        if 'u_target_value' in obs_data:
            u_tgt[idx] = np.asarray(obs_data['u_target_value']).astype(np.float32)

    return {
        'obs_value':   jnp.array(val),
        'obs_present': jnp.array(pres),
        'has_any_obs': jnp.array(pres),
        'u_target':    jnp.array(u_tgt),   # exogenous -- excluded from likelihood
    }


# --- Forward integration ------------------------------------------

def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps,
                            rng_key=None):
    """Stochastic Euler-Maruyama for generating synthetic data.

    exogenous must contain 'u_target' (dense array of length n_steps).
    """
    alpha = params[_PI['alpha']]
    a     = params[_PI['a']]
    gamma = params[_PI['gamma']]
    sigma_d = diffusion_fn(params)
    sqrt_dt = jnp.sqrt(dt)
    u_target_arr = jnp.asarray(exogenous['u_target'])
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (2,))
        x = y[0]
        u = y[1]
        u_target_k = u_target_arr[i]
        drift_x = alpha * x * (a**2 - x**2) + u
        drift_u = -gamma * (u - u_target_k)
        x_new = x + dt * drift_x + sigma_d[0] * sqrt_dt * noise[0]
        u_new = u + dt * drift_u + sigma_d[1] * sqrt_dt * noise[1]
        y_new = jnp.array([x_new, u_new])
        return (y_new, key), y_new

    (_, _), traj = jax.lax.scan(step, (init_state, rng_key),
                                 jnp.arange(n_steps))
    return traj


# --- Misc helpers --------------------------------------------------

def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def _prior_mean(ptype, pargs):
    if ptype == 'lognormal': return math.exp(pargs[0] + pargs[1]**2 / 2)
    elif ptype == 'normal':  return pargs[0]
    return 0.0


def get_init_theta():
    all_config = OrderedDict()
    all_config.update(PARAM_PRIOR_CONFIG)
    all_config.update(INIT_STATE_PRIOR_CONFIG)
    return np.array([_prior_mean(pt, pa) for _, (pt, pa) in all_config.items()],
                    dtype=np.float32)


# --- Assemble -----------------------------------------------------

BISTABLE_CTRL_ESTIMATION = EstimationModel(
    name="bistable_controlled",
    version="1.0",
    n_states=2,
    n_stochastic=2,
    stochastic_indices=(0, 1),
    state_bounds=((-5.0, 5.0), (-5.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={},
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
    obs_sample_fn=obs_sample_fn,
    gaussian_obs_fn=gaussian_obs_fn,
    exogenous_keys=('u_target',),
)