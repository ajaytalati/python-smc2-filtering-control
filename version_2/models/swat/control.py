"""SWAT ControlSpec for the SMC²-MPC closed-loop controller.

Wires the 4-state SWAT model into the framework's ``ControlSpec``
contract. The MPC chooses **three** time-varying control schedules
(V_h, V_n, V_c) over a horizon, encoded as RBF-anchor weights.

Cost functional
---------------
::

    cost(θ) = -E_w[ ∫ T(t) dt ]    # maximise integrated testosterone

That is the entire cost. **No regularisation, no operating-point
biases**. The controller figures out optimal V_h, V_n, V_c from the
T reward alone, subject to the bound transforms below.

This is by user direction (2026-04-30):

    "The controller does not have to be given these baselines — it
     should figure them out by itself. cost = -∫T dt is enough."

θ packing
---------
``theta`` has shape ``(3 * n_anchors,)``. The first ``n_anchors``
entries parameterise V_h, the next ``n_anchors`` parameterise V_n,
the final ``n_anchors`` parameterise V_c. Each chunk passes through
an RBF basis + a variate-specific bound transform.

Bounds (from the OT-Control adapter)
------
    V_h ∈ [0, 4],  V_n ∈ [0, 5],  V_c ∈ [-12, 12] hours.

Bound transforms (no logit bias — controller starts from the
sigmoid/tanh midpoint):
- V_h: V_h_max · sigmoid(θ·rbf)            θ=0 → V_h = 2.0
- V_n: V_n_max · sigmoid(θ·rbf)            θ=0 → V_n = 2.5
- V_c: V_c_max · tanh(θ·rbf)               θ=0 → V_c = 0
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import ControlSpec, RBFSchedule
from smc2fc.control.calibration import build_crn_noise_grids

from version_2.models.swat._dynamics import (
    A_SCALE_FROZEN,
    diffusion_state_dep,
    drift_jax,
    state_clip,
)
from version_2.models.swat.simulation import DEFAULT_INIT, DEFAULT_PARAMS
from version_2.models.swat._v_schedule import (
    V_H_BOUNDS,
    V_N_BOUNDS,
    V_C_BOUNDS,
)


# =========================================================================
# Schedule builders — three RBF schedules, one per variate
# =========================================================================

def _make_three_schedules(*, n_steps: int, dt: float, n_anchors: int):
    """Build (rbf, schedule_from_theta) for the three SWAT controls.

    No logit bias — at θ=0 the schedule sits at the sigmoid/tanh
    midpoint of each variate's bounds. Controller figures out the
    right V_h, V_n, V_c values from the T reward.
    """
    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors,
                       output='identity')
    design = rbf.design_matrix()        # (n_steps, n_anchors)

    V_h_max = V_H_BOUNDS[1]   # 4
    V_n_max = V_N_BOUNDS[1]   # 5
    V_c_max = V_C_BOUNDS[1]   # 12

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        """θ shape (3*n_anchors,) → (n_steps, 3) of (V_h, V_n, V_c)."""
        theta_h = theta[:n_anchors]
        theta_n = theta[n_anchors:2 * n_anchors]
        theta_c = theta[2 * n_anchors:]

        raw_h = jnp.einsum('a,ta->t', theta_h, design)
        raw_n = jnp.einsum('a,ta->t', theta_n, design)
        raw_c = jnp.einsum('a,ta->t', theta_c, design)

        V_h = V_h_max * jax.nn.sigmoid(raw_h)
        V_n = V_n_max * jax.nn.sigmoid(raw_n)
        V_c = V_c_max * jnp.tanh(raw_c)

        return jnp.stack([V_h, V_n, V_c], axis=-1)

    return rbf, schedule_from_theta


# =========================================================================
# Plant integrator for cost rollouts
# =========================================================================

def _make_em_step_fn(params, dt, n_substeps):
    sub_dt = dt / float(n_substeps)
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def em_step(y, t, u, noise):
        def sub_body(y_inner, _):
            return y_inner + sub_dt * drift_jax(y_inner, params, t, u), None
        y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

        sigma = diffusion_state_dep(y_det, params)
        y_pred = y_det + sigma * sqrt_dt * noise
        return state_clip(y_pred)

    return em_step


# =========================================================================
# Cost functional + trajectory sampler
# =========================================================================

def _build_cost_and_traj_fns(
    *,
    n_inner: int,
    n_steps: int,
    dt: float,
    n_substeps: int,
    schedule_from_theta,
    seed: int = 42,
):
    """Build (cost_fn, traj_sample_fn) closures for SWAT control.

    Cost is purely -E_w[ ∫ T(t) dt ]. No regularisation terms; the
    bound transforms in schedule_from_theta keep V_h, V_n, V_c
    physically valid, and the controller's only job is to maximise
    integrated testosterone amplitude.
    """
    p_jax = {k: jnp.asarray(float(v)) for k, v in DEFAULT_PARAMS.items()
              if isinstance(v, (int, float))}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=4, seed=seed,
    )
    fixed_w = grids['wiener']    # (n_inner, n_steps, 4)

    init_arr = jnp.asarray(DEFAULT_INIT, dtype=jnp.float64)

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        u_arr = schedule_from_theta(theta)    # (n_steps, 3)

        def trial(w_seq):
            def step(carry, k):
                y, T_acc = carry
                t_k = jnp.float64(k) * dt
                u_t = u_arr[k]
                y_next = em_step(y, t_k, u_t, w_seq[k])
                T_acc = T_acc + y[3] * dt        # integrate testosterone
                return (y_next, T_acc), None

            (_, T_acc), _ = jax.lax.scan(
                step, (init_arr, jnp.float64(0.0)), jnp.arange(n_steps),
            )
            return -T_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> jnp.ndarray:
        u_arr = schedule_from_theta(theta)
        w_seq = jax.random.normal(key, (n_steps, 4), dtype=jnp.float64)

        def step(y, k):
            t_k = jnp.float64(k) * dt
            y_next = em_step(y, t_k, u_arr[k], w_seq[k])
            return y_next, y_next

        _, traj = jax.lax.scan(step, init_arr, jnp.arange(n_steps))
        return traj    # (n_steps, 4)

    return cost_fn, traj_sample_fn


# =========================================================================
# ControlSpec builder — main entry point
# =========================================================================

def build_control_spec(
    *,
    n_steps: int,
    dt: float,
    n_anchors: int = 8,
    n_inner: int = 64,
    n_substeps: int = 4,
    sigma_prior: float = 1.5,
    seed: int = 42,
) -> ControlSpec:
    """Construct a SWAT ControlSpec for the given horizon.

    Cost is purely -E[∫T dt] — no regularisation terms.
    Bound transforms in ``schedule_from_theta`` keep V_h, V_n, V_c
    physically valid.

    Args:
        n_steps:    number of bins in the planning horizon.
        dt:         bin width in days.
        n_anchors:  RBF anchors per control variate. Total theta_dim
                     = 3 * n_anchors.
        n_inner:    Monte Carlo trials per cost evaluation.
        n_substeps: deterministic Euler substeps for stiffness.
        sigma_prior: std of the Gaussian prior over θ.
        seed:       common-random-numbers seed for variance reduction.

    Returns:
        A frozen ControlSpec ready for the MPC.
    """
    rbf, schedule_from_theta = _make_three_schedules(
        n_steps=n_steps, dt=dt, n_anchors=n_anchors,
    )
    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt,
        n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        seed=seed,
    )

    spec = ControlSpec(
        name="swat",
        version="1.0.0",
        dt=dt,
        n_steps=n_steps,
        n_substeps=n_substeps,
        initial_state=jnp.asarray(DEFAULT_INIT, dtype=jnp.float64),
        truth_params={k: float(v) for k, v in DEFAULT_PARAMS.items()
                       if isinstance(v, (int, float))},
        theta_dim=3 * n_anchors,
        sigma_prior=sigma_prior,
        prior_mean=jnp.zeros(3 * n_anchors, dtype=jnp.float64),
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
    )
    # Attach traj_sample_fn as an attribute on the spec (not a
    # ControlSpec field, but downstream diagnostic code may use it).
    object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)
    return spec
