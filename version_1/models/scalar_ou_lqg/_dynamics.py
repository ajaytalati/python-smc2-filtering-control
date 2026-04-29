"""Pure-JAX dynamics for the scalar OU LQG model.

State: y = [x]                     (1-D)
Drift: dx/dt = -a x + b u(t)
Diffusion: sigma_w
Obs: y_k = x_k + N(0, sigma_v^2)

Discrete-time Euler-Maruyama:
    x_{k+1} = (1 - a*dt) x_k + b*dt*u_k + sqrt(dt)*sigma_w*xi_k

The control input u(t) is a deterministic exogenous schedule; for the
filter side we set u_k = 0 (open-loop simulation). When the control
side replaces u_k, it's read from grid_obs['u_value'][k].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from smc2fc._likelihood_constants import HALF_LOG_2PI


# Parameter index map. Set in estimation.py and passed in via pi.

def drift_jax(y, t, params, frozen, aux, pi):
    """Scalar OU drift: dx/dt = -a*x + b*u."""
    del frozen, t
    x = y[0]
    a = params[pi['a']]
    b = params[pi['b']]
    u_t = aux['u_at_t']
    return jnp.array([-a * x + b * u_t])


def diffusion(params, frozen, pi):
    """Diagonal diffusion sqrt(noise temperature) — single state, sigma_w."""
    del frozen
    return jnp.array([params[pi['sigma_w']]])


def imex_step_deterministic(y, t, dt, params, frozen, aux, pi):
    """Explicit Euler step on the drift."""
    return y + dt * drift_jax(y, t, params, frozen, aux, pi)


def imex_step_stochastic(y, t, dt, params, sigma_diag, noise, frozen, aux, pi):
    """Euler-Maruyama step. Returns (y_next, mu_prior, var_prior)."""
    y_det = imex_step_deterministic(y, t, dt, params, frozen, aux, pi)
    y_next = y_det + sigma_diag * jnp.sqrt(dt) * noise
    mu_prior = y_det
    var_prior = (sigma_diag ** 2) * dt
    return y_next, mu_prior, var_prior


def obs_log_prob(y, grid_obs, k, params, pi):
    """Gaussian log-pdf for scalar observation at step k."""
    sigma_v = params[pi['sigma_v']]
    x_pred = y[0]
    resid = grid_obs['obs_value'][k] - x_pred
    return grid_obs['obs_present'][k] * (
        -0.5 * (resid / sigma_v) ** 2 - jnp.log(sigma_v) - HALF_LOG_2PI
    )
