"""SWAT ControlSpec for the SMC²-MPC closed-loop controller.

Wires the 4-state SWAT model into the framework's ``ControlSpec``
contract. The MPC chooses **three** time-varying control schedules
(V_h, V_n, V_c) over a horizon, encoded as RBF-anchor weights.

Cost functional (default):

    cost(θ) = -E_w[ ∫ T(t) dt ]                    # maximise testosterone
              + λ_h · ||V_h(t) - V_h_default||²    # intervention cost
              + λ_n · ||V_n(t) - V_n_default||²
              + λ_c · ||V_c(t)||²
              + λ_T_floor · ∫ max(T_floor - T, 0)² dt   # T-collapse barrier

The T-floor barrier penalises trajectories that cross the
Stuart-Landau bifurcation (T → 0). It plays the role of FSA-v2's
F-max barrier, but on the *outcome* state T rather than a separate
fatigue variable.

θ packing
---------
``theta`` has shape ``(3 * n_anchors,)``. The first ``n_anchors``
entries parameterise V_h, the next ``n_anchors`` parameterise V_n,
the final ``n_anchors`` parameterise V_c. Each chunk passes through
an RBF basis + a variate-specific bound transform.

Bounds
------
Per the OT-Control adapter:
    V_h ∈ [0, 4],  V_n ∈ [0, 5],  V_c ∈ [-12, 12] hours.

Smooth bound transforms:
- V_h, V_n: V_max · sigmoid(c + θ·rbf), with logit bias c centred on
  the default operating point.
- V_c: V_max · tanh(θ·rbf), centred at 0 (no phase shift).
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


# ── Operating-point reference ─────────────────────────────────────────
V_H_DEFAULT = 1.0     # healthy-baseline vitality reserve
V_N_DEFAULT = 0.3     # healthy-baseline chronic load
V_C_DEFAULT = 0.0     # no phase shift

T_FLOOR_DEFAULT = 0.05    # T below this counts as collapse


# =========================================================================
# Schedule builders — three RBF schedules, one per variate
# =========================================================================

def _make_three_schedules(*, n_steps: int, dt: float, n_anchors: int):
    """Build (rbf, schedule_from_theta) for the three SWAT controls.

    Returns a packed ``schedule_from_theta(theta) → (n_steps, 3)``
    function where the second axis is (V_h, V_n, V_c).
    """
    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors,
                       output='identity')
    design = rbf.design_matrix()        # (n_steps, n_anchors)

    # Logit bias for V_h: at θ=0, sigmoid(c_h) · V_h_max = V_h_default
    p_h = V_H_DEFAULT / V_H_BOUNDS[1]
    c_h = float(np.log(p_h / (1.0 - p_h)))

    # Logit bias for V_n: same pattern
    p_n = V_N_DEFAULT / V_N_BOUNDS[1]
    c_n = float(np.log(p_n / (1.0 - p_n)))

    # V_c: no logit bias since default = 0 (centre of tanh)
    V_c_max = V_C_BOUNDS[1]    # 12 hours

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        """θ shape (3*n_anchors,) → (n_steps, 3) of (V_h, V_n, V_c)."""
        theta_h = theta[:n_anchors]
        theta_n = theta[n_anchors:2 * n_anchors]
        theta_c = theta[2 * n_anchors:]

        raw_h = c_h + jnp.einsum('a,ta->t', theta_h, design)
        raw_n = c_n + jnp.einsum('a,ta->t', theta_n, design)
        raw_c =        jnp.einsum('a,ta->t', theta_c, design)

        V_h = V_H_BOUNDS[1] * jax.nn.sigmoid(raw_h)
        V_n = V_N_BOUNDS[1] * jax.nn.sigmoid(raw_n)
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
    T_floor: float = T_FLOOR_DEFAULT,
    lam_h: float = 0.01,
    lam_n: float = 0.01,
    lam_c: float = 0.001,
    lam_T_floor: float = 1.0,
    seed: int = 42,
):
    """Build (cost_fn, traj_sample_fn) closures for SWAT control."""
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
                y, T_acc, vh_acc, vn_acc, vc_acc, floor_acc = carry
                t_k = jnp.float64(k) * dt
                u_t = u_arr[k]
                y_next = em_step(y, t_k, u_t, w_seq[k])
                # Reward: integrated testosterone
                T_acc = T_acc + y[3] * dt
                # Intervention costs (squared deviation from defaults)
                vh_acc = vh_acc + (u_t[0] - V_H_DEFAULT) ** 2 * dt
                vn_acc = vn_acc + (u_t[1] - V_N_DEFAULT) ** 2 * dt
                vc_acc = vc_acc + u_t[2] ** 2 * dt
                # T-floor barrier
                floor_acc = floor_acc + (
                    jnp.maximum(T_floor - y[3], 0.0) ** 2 * dt)
                return (y_next, T_acc, vh_acc, vn_acc,
                         vc_acc, floor_acc), None

            init_carry = (init_arr,
                           jnp.float64(0.0), jnp.float64(0.0),
                           jnp.float64(0.0), jnp.float64(0.0),
                           jnp.float64(0.0))
            (_, T_acc, vh_acc, vn_acc, vc_acc, floor_acc), _ = jax.lax.scan(
                step, init_carry, jnp.arange(n_steps),
            )
            return (-T_acc
                    + lam_h * vh_acc
                    + lam_n * vn_acc
                    + lam_c * vc_acc
                    + lam_T_floor * floor_acc)

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
    T_floor: float = T_FLOOR_DEFAULT,
    lam_h: float = 0.01,
    lam_n: float = 0.01,
    lam_c: float = 0.001,
    lam_T_floor: float = 1.0,
    seed: int = 42,
) -> ControlSpec:
    """Construct a SWAT ControlSpec for the given horizon.

    Args:
        n_steps:    number of bins in the planning horizon.
        dt:         bin width in days.
        n_anchors:  RBF anchors per control variate. Total theta_dim
                     = 3 * n_anchors.
        n_inner:    Monte Carlo trials per cost evaluation.
        n_substeps: deterministic Euler substeps for stiffness.
        sigma_prior: std of the Gaussian prior over θ.
        T_floor:    T-collapse barrier threshold.
        lam_h, lam_n, lam_c, lam_T_floor: cost regularisation weights.
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
        T_floor=T_floor,
        lam_h=lam_h, lam_n=lam_n, lam_c=lam_c,
        lam_T_floor=lam_T_floor, seed=seed,
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
