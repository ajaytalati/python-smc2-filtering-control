"""SMC² EstimationModel for the scalar OU LQG model.

Locally-optimal Gaussian-fusion proposal (Pitt-Shephard tilt) for the
single observed state x. Since the entire model is linear-Gaussian,
the locally-optimal proposal is the EXACT optimal proposal — there is
no Jensen-inequality bias, and the bootstrap-PF marginal-LL estimator
should converge to the true marginal LL at finite K.

This is the cleanest possible inner-LL test for the SMC² framework:
the answer is the analytical Kalman log-likelihood, computable via
``models.scalar_ou_lqg.bench_kalman``.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.estimation_model import EstimationModel
from smc2fc._likelihood_constants import HALF_LOG_2PI
from models.scalar_ou_lqg import _dynamics as dyn


# ─── Priors (centered at truth, broad enough to need real data) ──────

# Truth: a=1, b=1, sigma_w=0.3, sigma_v=0.2.
# Lognormal priors with σ=0.5 give 95% CI roughly [0.4, 2.7] × median.
PARAM_PRIOR_CONFIG = OrderedDict([
    ('a',       ('lognormal', (math.log(1.0),  0.5))),
    ('b',       ('lognormal', (math.log(1.0),  0.5))),
    ('sigma_w', ('lognormal', (math.log(0.3),  0.5))),
    ('sigma_v', ('lognormal', (math.log(0.2),  0.5))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict([
    ('x_0', ('normal', (0.0, 1.0))),
])

DEFAULT_FROZEN_PARAMS: dict = {}

_PI = {k: i for i, k in enumerate(PARAM_PRIOR_CONFIG)}


# ─── Grid alignment ──────────────────────────────────────────────────

def _make_align_obs_fn(u_schedule: np.ndarray | None = None):
    """Closure that bakes the control schedule into align_obs_fn."""
    def align_obs_fn(obs_data, t_steps, dt_hours):
        del dt_hours
        T = int(t_steps)
        obs_value = np.zeros(T, dtype=np.float32)
        obs_present = np.zeros(T, dtype=np.float32)
        u_value = np.zeros(T, dtype=np.float32)

        if 'obs' in obs_data and 'obs_value' in obs_data['obs']:
            ch = obs_data['obs']
            idx = np.asarray(ch['t_idx']).astype(int)
            mask = (idx >= 0) & (idx < T)
            obs_value[idx[mask]] = np.asarray(ch['obs_value'])[mask].astype(np.float32)
            obs_present[idx[mask]] = 1.0

        if u_schedule is not None:
            n = min(T, len(u_schedule))
            u_value[:n] = np.asarray(u_schedule[:n], dtype=np.float32)

        has_any_obs = obs_present
        return {
            'obs_value':   jnp.array(obs_value),
            'obs_present': jnp.array(obs_present),
            'u_value':     jnp.array(u_value),
            'has_any_obs': jnp.array(has_any_obs),
        }
    return align_obs_fn


# ─── EstimationModel factory ─────────────────────────────────────────

def diffusion_fn(params, frozen=None):
    return dyn.diffusion(params, frozen, _PI)


def _make_aux_at_step(grid_obs, k):
    return {'u_at_t': grid_obs['u_value'][k]}


def _propagate_fn(frozen):
    """Locally-optimal proposal: Gaussian-fusion of Euler prediction with
    the observation likelihood. For scalar linear-Gaussian dynamics with
    Gaussian obs this is the EXACT optimal proposal.

    Returns the Pitt-Shephard predictive log-weight; obs_log_weight_fn
    contributes 0 (CGM-style: weight is in the proposal).
    """
    def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key):
        del rng_key

        aux_k = _make_aux_at_step(grid_obs, k)

        # Stochastic Euler step: deterministic mean + diffusion noise.
        y_next, mu_prior, var_prior = dyn.imex_step_stochastic(
            y, t, dt, params, sigma_diag, noise, frozen, aux_k, _PI
        )

        sigma_v = params[_PI['sigma_v']]
        obs_pres = grid_obs['obs_present'][k]
        obs_val = grid_obs['obs_value'][k]

        # Pitt-Shephard guidance: Kalman update on x given y at step k+1.
        # In precision form: prec_post = 1/var_prior + obs_pres/sigma_v^2
        x_prec = 1.0 / jnp.maximum(var_prior[0], 1e-12)
        x_info = x_prec * mu_prior[0]
        x_info += obs_pres * obs_val / (sigma_v ** 2)
        x_prec += obs_pres / (sigma_v ** 2)
        x_var = 1.0 / x_prec
        x_mu = x_var * x_info

        x_new = x_mu + jnp.sqrt(x_var) * noise[0]
        y_next = y_next.at[0].set(x_new)

        # Predictive log-weight: log N(y_k | mu_prior, var_prior + sigma_v^2)
        pred_var = sigma_v ** 2 + var_prior[0]
        pred_mu = mu_prior[0]
        lw = obs_pres * (
            -0.5 * (obs_val - pred_mu) ** 2 / pred_var
            - 0.5 * jnp.log(pred_var) - HALF_LOG_2PI
        )
        return y_next, lw
    return propagate_fn


def _obs_log_weight_fn(frozen):
    """No additional weight — the obs is fused into the proposal."""
    del frozen
    def obs_log_weight_fn(x_new, grid_obs, k, params):
        del x_new, grid_obs, k, params
        return jnp.float64(0.0)
    return obs_log_weight_fn


def _shard_init_fn(frozen):
    def shard_init_fn(time_offset, params, exogenous, global_init):
        del time_offset, params, exogenous
        return jnp.asarray(global_init, dtype=jnp.float64)
    return shard_init_fn


def _imex_step_fn(frozen):
    def imex_step_fn(y, t, dt, params, grid_obs):
        aux_k = _make_aux_at_step(grid_obs, jnp.int32(t / dt))
        return dyn.imex_step_deterministic(y, t, dt, params, frozen, aux_k, _PI)
    return imex_step_fn


def _obs_log_prob_fn(frozen):
    del frozen
    def obs_log_prob_fn(y, grid_obs, k, params):
        return dyn.obs_log_prob(y, grid_obs, k, params, _PI)
    return obs_log_prob_fn


def _make_init_state_fn(frozen):
    def make_init_state_fn(init_estimates, params):
        del params
        # init_estimates carries [x_0] in the same order as
        # INIT_STATE_PRIOR_CONFIG.
        return jnp.array([init_estimates[0]])
    return make_init_state_fn


def _gaussian_obs_fn(frozen):
    def gaussian_obs_fn(y, grid_obs, k, params):
        return {
            'mean':     jnp.array([y[0]]),
            'value':    jnp.array([grid_obs['obs_value'][k]]),
            'cov_diag': jnp.array([params[_PI['sigma_v']] ** 2]),
            'present':  jnp.array([grid_obs['obs_present'][k]]),
        }
    return gaussian_obs_fn


def _obs_sample_fn(frozen):
    def obs_sample_fn(y, exog, k, params, rng_key):
        del exog, k
        sigma_v = params[_PI['sigma_v']]
        return {'obs_value': y[0] + sigma_v * jax.random.normal(rng_key, dtype=y.dtype)}
    return obs_sample_fn


def get_init_theta():
    """Initial theta vector at prior means (in constrained space)."""
    init = []
    for name, (kind, args) in PARAM_PRIOR_CONFIG.items():
        if kind == 'lognormal':
            init.append(math.exp(args[0] + 0.5 * args[1] ** 2))
        elif kind == 'normal':
            init.append(args[0])
        else:
            raise NotImplementedError(kind)
    for name, (kind, args) in INIT_STATE_PRIOR_CONFIG.items():
        if kind == 'normal':
            init.append(args[0])
        else:
            raise NotImplementedError(kind)
    return np.array(init, dtype=np.float64)


def make_scalar_ou_estimation(
    u_schedule: np.ndarray | None = None,
    frozen_params: dict | None = None,
) -> EstimationModel:
    """Build an EstimationModel for the scalar OU LQG model.

    Args:
        u_schedule: optional control schedule baked into align_obs_fn.
            If None, a zero schedule is used.
        frozen_params: optional override for DEFAULT_FROZEN_PARAMS.
    """
    frozen = dict(DEFAULT_FROZEN_PARAMS)
    if frozen_params:
        frozen.update(frozen_params)

    return EstimationModel(
        name="scalar_ou_lqg",
        version="1.0",
        n_states=1,
        n_stochastic=1,
        stochastic_indices=(0,),
        state_bounds=((-50.0, 50.0),),
        param_prior_config=PARAM_PRIOR_CONFIG,
        init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
        frozen_params=frozen,
        propagate_fn=_propagate_fn(frozen),
        diffusion_fn=lambda p: diffusion_fn(p, frozen),
        obs_log_weight_fn=_obs_log_weight_fn(frozen),
        align_obs_fn=_make_align_obs_fn(u_schedule),
        shard_init_fn=_shard_init_fn(frozen),
        get_init_theta_fn=get_init_theta,
        imex_step_fn=_imex_step_fn(frozen),
        obs_log_prob_fn=_obs_log_prob_fn(frozen),
        make_init_state_fn=_make_init_state_fn(frozen),
        obs_sample_fn=_obs_sample_fn(frozen),
        gaussian_obs_fn=_gaussian_obs_fn(frozen),
        exogenous_keys=(),
    )


SCALAR_OU_ESTIMATION = make_scalar_ou_estimation()
