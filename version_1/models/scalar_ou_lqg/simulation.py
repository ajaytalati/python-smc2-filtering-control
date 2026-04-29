"""Simulator for the scalar OU LQG model — numpy reference.

Generates synthetic open-loop or closed-loop trajectories with truth
parameters. Used to produce data for the filter side (Stage A1) and
for cost evaluation in the control side (Stage A2/A3).
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np


# ── Parameter sets ─────────────────────────────────────────────────────

PARAM_SET_A = {
    'a':       1.0,
    'b':       1.0,
    'sigma_w': 0.3,
    'sigma_v': 0.2,
    'q':       1.0,
    'r':       0.1,
    's':       1.0,
}

INIT_STATE_A = {'x_0': 0.0}

EXOGENOUS_A = {
    'dt':  0.05,
    'T':   20,           # number of grid steps
    'x0_var': 1.0,       # initial state variance
}


# ── Drift + diffusion ─────────────────────────────────────────────────

def drift(t, y, params, aux):
    """numpy drift: dx/dt = -a*x + b*u(t)."""
    del t
    x = y[0]
    u_t = aux.get('u_at_t', 0.0)
    return np.array([-params['a'] * x + params['b'] * u_t])


def diffusion_diagonal(params):
    return np.array([params['sigma_w']])


# ── Observation channel ───────────────────────────────────────────────

def gen_obs(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian observation channel: y_k = x_k + N(0, sigma_v^2)."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    x = trajectory[:, 0]
    noise = rng.normal(0.0, params['sigma_v'], size=T)
    return {
        't_idx':     np.arange(T, dtype=np.int32),
        'obs_value': (x + noise).astype(np.float32),
    }


# ── Direct trajectory simulator (numpy, used by tests + benches) ──────

def simulate(
    *,
    params: dict,
    init_state: dict,
    exogenous: dict,
    u: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Simulate one trajectory of the scalar OU LQG model.

    Args:
        params:     dict with a, b, sigma_w, sigma_v
        init_state: dict with x_0
        exogenous:  dict with dt, T, x0_var
        u:          (T,) control schedule. If None, open-loop u=0.
        seed:       RNG seed.

    Returns dict with keys: t_grid (T,), trajectory (T, 1), obs (T,),
    u (T,).
    """
    dt = float(exogenous['dt'])
    T = int(exogenous['T'])
    x0_var = float(exogenous.get('x0_var', 1.0))
    x0_mean = float(init_state['x_0'])

    rng = np.random.default_rng(seed)
    A = 1.0 - params['a'] * dt
    B = params['b'] * dt
    sw = params['sigma_w'] * math.sqrt(dt)

    if u is None:
        u = np.zeros(T)
    u = np.asarray(u, dtype=np.float64)

    x = np.zeros(T)
    x[0] = x0_mean + math.sqrt(x0_var) * rng.standard_normal()
    for k in range(1, T):
        x[k] = A * x[k - 1] + B * u[k - 1] + sw * rng.standard_normal()

    obs = x + params['sigma_v'] * rng.standard_normal(size=T)

    return {
        't_grid':     np.arange(T) * dt,
        'trajectory': x[:, None].astype(np.float64),
        'obs':        obs.astype(np.float64),
        'u':          u,
    }
