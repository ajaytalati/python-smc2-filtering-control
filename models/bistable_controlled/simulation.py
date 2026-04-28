"""models/bistable_controlled/simulation.py — Controlled Double-Well SDE.

Date:    18 April 2026
Version: 1.1

Two-state controlled bistable model: the health variable x lives in a
double-well potential tilted by a control/barrier process u.  u itself
is driven toward a piecewise-constant target schedule u_target(t) by an
Ornstein-Uhlenbeck mean-reversion (the target is exogenous, specified
by the intervention plan).

Stochastic differential equations:

    dx = [alpha * x * (a^2 - x^2) + u] dt  +  sqrt(2 sigma_x) dB_x
    du = -gamma * (u - u_target(t)) dt      +  sqrt(2 sigma_u) dB_u

Observation channel (Gaussian, dense on simulation grid):

    y_k = x(t_k) + N(0, sigma_obs^2)

A second "channel" u_target emits the deterministic schedule alongside
the observations, so the synthetic output directory preserves a single
source of truth for the intervention plan consumed by the estimator.

Critical tilt (saddle-node bifurcation):

    u_c = 2 * alpha * a^3 / (3 sqrt(3))    (0.385 for alpha=a=1)

    u > u_c : landscape is monostable (only one fixed point of x drift)
    u < u_c : landscape is bistable (three fixed points)

Schedule is 2-phase:
    Hour  0-24  : u = 0     (pre-intervention, symmetric bistable)
    Hour 24-72  : u = 0.5   (active intervention, supercritical)

No maintenance phase.  Simulation horizon = T_total = 72 hours.  At
dt = 10 min (default), the grid has 432 time steps -- chosen to keep
MCLMC tuner curvature feasible for Bayesian parameter estimation.

Changelog from v1.0:
  - Removed maintenance phase (u_maint) from schedule.
  - Renamed EXOGENOUS_A key T_end -> T_total (clearer semantics).
  - Schedule function _u_target simplified to 2 branches.
"""

import math
import numpy as np

from smc2fc.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_CONSTANT)
from models.bistable_controlled.sim_plots import plot_bistable_ctrl


# =========================================================================
# SCHEDULE — u_target(t) piecewise-constant, 2-phase
# =========================================================================

def _u_target(t, T_i, u_on):
    """Piecewise-constant 2-phase schedule; scalar t.

    Phase 1 (t < T_i):  u_target = 0      (pre-intervention)
    Phase 2 (t >= T_i): u_target = u_on   (intervention)
    """
    return u_on if t >= T_i else 0.0


# =========================================================================
# DRIFT
# =========================================================================

def drift(t, y, params, aux):
    """Controlled-bistable drift.

    aux = (T_intervention, u_on) — schedule parameters from make_aux.
    """
    T_i, u_on = aux
    p = params
    x = y[0]
    u = y[1]
    u_target = _u_target(t, T_i, u_on)
    dx = p['alpha'] * x * (p['a']**2 - x**2) + u
    du = -p['gamma'] * (u - u_target)
    return np.array([dx, du])


def drift_jax(t, y, args):
    """JAX variant of drift.  args = (params_dict_jax, T_i, u_on)."""
    import jax.numpy as jnp
    p, T_i, u_on = args
    x = y[0]
    u = y[1]
    u_target = jnp.where(t < T_i, 0.0, u_on)
    dx = p['alpha'] * x * (p['a']**2 - x**2) + u
    du = -p['gamma'] * (u - u_target)
    return jnp.array([dx, du])


# =========================================================================
# DIFFUSION
# =========================================================================

def diffusion_diagonal(params):
    """Diagonal SDE coefficients: [sqrt(2*sigma_x), sqrt(2*sigma_u)]."""
    return np.array([math.sqrt(2.0 * params['sigma_x']),
                     math.sqrt(2.0 * params['sigma_u'])])


# =========================================================================
# AUXILIARY / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """Pack schedule parameters into aux for drift()."""
    del params, init_state, t_grid
    return (exogenous['T_intervention'], exogenous['u_on'])


def make_aux_jax(params, init_state, t_grid, exogenous):
    """Build JAX args for drift_jax."""
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.float64(exogenous['T_intervention']),
            jnp.float64(exogenous['u_on']))


def make_y0(init_dict, params):
    """Build [x, u] initial state."""
    del params
    return np.array([init_dict['x_0'], init_dict['u_0']])


# =========================================================================
# OBSERVATION CHANNEL + SCHEDULE CHANNEL
# =========================================================================

def gen_obs(trajectory, t_grid, params, aux, prior_channels, seed):
    """Gaussian observation channel on x: y = x + N(0, sigma_obs^2)."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    T = len(t_grid)
    x = trajectory[:, 0]
    noise = rng.normal(0.0, params['sigma_obs'], size=T)
    return {
        't_idx':     np.arange(T, dtype=np.int32),
        'obs_value': (x + noise).astype(np.float32),
    }


def gen_u_target_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Emit the u_target schedule as a deterministic 'channel'.

    This is not a noisy observation -- it is the exogenous intervention
    plan.  We emit it as a channel so it is saved alongside the synthetic
    data and consumable by the estimator via align_obs_fn.
    """
    del trajectory, params, prior_channels, seed
    T_i, u_on = aux
    t = t_grid
    u_tgt = np.where(t < T_i, 0.0, u_on)
    return {
        't_idx':          np.arange(len(t), dtype=np.int32),
        'u_target_value': u_tgt.astype(np.float32),
    }


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def verify_physics(trajectory, t_grid, params):
    """Descriptive trajectory stats and critical-tilt computation.

    Does NOT assert the intervention succeeded (noise could frustrate
    the crossing for small sigma_u) — reports stats for qualitative
    inspection.
    """
    x = trajectory[:, 0]
    u = trajectory[:, 1]
    a = params['a']

    visits_pos = bool(np.any(x > 0.5 * a))
    visits_neg = bool(np.any(x < -0.5 * a))
    u_c = 2.0 * params['alpha'] * a**3 / (3.0 * math.sqrt(3.0))

    return {
        'x_min':                float(np.min(x)),
        'x_max':                float(np.max(x)),
        'x_final':              float(x[-1]),
        'u_min':                float(np.min(u)),
        'u_max':                float(np.max(u)),
        'u_final':              float(u[-1]),
        'critical_tilt_u_c':    float(u_c),
        'visits_positive_well': visits_pos,
        'visits_negative_well': visits_neg,
        'visits_both_wells':    visits_pos and visits_neg,
        'all_finite':           bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETER SETS
# =========================================================================

# Set A — defaults: barrier/noise ratio = 2.5 (peak/valley = 12.2),
# tau_u = 0.5 h so u tracks target quickly, u noise small (clean
# tracking), obs noise moderate at 0.2.
PARAM_SET_A = {
    'alpha':     1.0,   # double-well strength
    'a':         1.0,   # well separation: minima at x = +/- a
    'sigma_x':   0.10,  # x noise temperature
    'gamma':     2.0,   # u mean-reversion rate (tau_u = 1/gamma = 0.5 h)
    'sigma_u':   0.05,  # u noise temperature (smaller than sigma_x)
    'sigma_obs': 0.20,  # observation noise std
}

# Start in the unhealthy (-a) well with no control.  This is the
# intervention-onset scenario: subject is in the pathological attractor,
# control begins at t = T_intervention.
INIT_STATE_A = {'x_0': -1.0, 'u_0': 0.0}

# Schedule for a 3-day (72 h) simulation window:
#   Hour  0-24  : u = 0     (pre-intervention, symmetric bistable)
#   Hour 24-72  : u = 0.5   (active intervention, supercritical, > u_c)
#
# INTENDED RUN:  --T-total 72 --dt-minutes 10  ->  432 time steps.
# This step count is chosen to keep MCLMC tuner curvature feasible:
# prior runs at ~2160 steps produced eps ~ 0.006 (infeasible).  At 432
# steps the cubic nonlinearity is still the dominant curvature source,
# but step-count-linear contributions to |grad log p| are 5x smaller.
EXOGENOUS_A = {
    'T_intervention':  24.0,   # hours; intervention onset
    'T_total':         72.0,   # hours; simulation total duration (runner hint)
    'u_on':             0.5,   # supercritical tilt (> u_c = 0.385)
}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

BISTABLE_CTRL_MODEL = SDEModel(
    name="bistable_controlled",
    version="1.1",

    states=(
        StateSpec("x", -5.0, 5.0),
        StateSpec("u", -5.0, 5.0),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_CONSTANT,
    diffusion_fn=diffusion_diagonal,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("obs",      depends_on=(), generate_fn=gen_obs),
        ChannelSpec("u_target", depends_on=(), generate_fn=gen_u_target_channel),
    ),

    plot_fn=plot_bistable_ctrl,
    verify_physics_fn=verify_physics,

    param_sets={'A': PARAM_SET_A},
    init_states={'A': INIT_STATE_A},
    exogenous_inputs={'A': EXOGENOUS_A},
)