"""SWAT ControlSpec for the SMC²-MPC closed-loop controller.

Wires the 4-state SWAT model into the framework's ``ControlSpec``
contract. The MPC chooses **three** time-varying control schedules
(V_h, V_n, V_c) over a horizon, encoded as RBF-anchor weights.

Cost functional
---------------
::

    cost(θ) = -E_w[ ∫ T(t) dt + lambda_E · ∫ E_dyn(t) dt ]

The first term is the original "maximise integrated testosterone"
objective. The second term (added 2026-05-02) is a SHAPING reward on
the entrainment quality E_dyn ∈ [0, 1] that drives the Stuart-Landau
bifurcation parameter μ(E) = μ_0 + μ_E · E. Rationale:

  - The chain (controls → E → μ → T) has a sharp bifurcation cliff at
    E_crit = -μ_0/μ_E = 0.5: below E_crit, T flatlines (zero gradient
    on ∫T dt); above, T climbs an exp() curve on a τ_T ≈ 2d timescale.
  - Adding ∫E dt removes the cliff: the controller climbs a smooth
    ramp 0 → 0.5 → 0.85 with reward at every step, and gets immediate
    (no τ_T lag) feedback on each control choice.
  - Both terms have aligned optima (V_h high, V_n low, V_c=0
    maximises both E and the eventual ∫T), so this is a Lyapunov-style
    auxiliary cost, not a different objective.
  - Magnitude balance: at healthy operating point, ∫T dt and ∫E dt
    are both ≈ 0.85 · t_total, so lambda_E ≈ 1 makes the two terms
    contribute equally over the horizon. Default is 1.0.

θ packing
---------
``theta`` has shape ``(3 * n_anchors,)``. The first ``n_anchors``
entries parameterise V_h, the next ``n_anchors`` parameterise V_n,
the final ``n_anchors`` parameterise V_c. Each chunk passes through
an RBF basis + a variate-specific bound transform.

Bounds
------
    V_h ∈ [0, 4],  V_n ∈ [0, 5],  V_c ∈ [-3, +3] hours.

V_h and V_n bounds come from `_v_schedule.V_H_BOUNDS / V_N_BOUNDS`
(shared with the plant's daily-clip path). The V_c bound follows
`_dynamics.V_C_MAX_HOURS = 3` and matches what the estimator uses
(`estimation.FROZEN_PARAMS['V_c_max'] = V_C_MAX_HOURS`). The dynamics'
`entrainment_quality()` clamps |V_c| to V_C_MAX_HOURS and the phase
term hits zero at exactly that boundary, so any planned |V_c| > 3 has
no gradient on the cost (and actively collapses T via μ = -E_crit).

Bound transforms (logit-biased so θ=0 sits at the bench `set_A`
healthy operating point V_h=1.0, V_n=0.2, V_c=0):
- V_h: V_h_max · sigmoid(c_h + θ·rbf)      θ=0 → V_h = 1.0,  c_h = logit(0.25)
- V_n: V_n_max · sigmoid(c_n + θ·rbf)      θ=0 → V_n = 0.2,  c_n = logit(0.04)
- V_c: V_c_max · tanh(θ·rbf)               θ=0 → V_c = 0     (symmetric, no bias)
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
    V_C_MAX_HOURS,
    diffusion_state_dep,
    drift_jax,
    entrainment_quality,
    state_clip,
)
from version_2.models.swat.simulation import DEFAULT_INIT, DEFAULT_PARAMS
from version_2.models.swat._v_schedule import (
    V_H_BOUNDS,
    V_N_BOUNDS,
)


# =========================================================================
# Schedule builders — three RBF schedules, one per variate
# =========================================================================

def _make_three_schedules(*, n_steps: int, dt: float, n_anchors: int):
    """Build (rbf, schedule_from_theta) for the three SWAT controls.

    Logit-bias the V_h and V_n sigmoids so that θ=0 sits at the
    bench's `set_A` healthy operating point (V_h=1.0, V_n=0.2, V_c=0)
    — i.e. the prior cloud is centred on a clinically reasonable
    starting state, not on the sigmoid mid-range default
    (V_h=2.0, V_n=2.5) which is strictly worse than every scenario
    baseline. V_c is symmetric about 0 so it stays unbiased.
    """
    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors,
                       output='identity')
    design = rbf.design_matrix()        # (n_steps, n_anchors)

    V_h_max = V_H_BOUNDS[1]    # 4
    V_n_max = V_N_BOUNDS[1]    # 5
    V_c_max = V_C_MAX_HOURS    # 3 — match _dynamics.entrainment_quality and
                                 #     estimation.FROZEN_PARAMS, which clamp
                                 #     and parametrise V_c on [-3, +3] hours.
                                 #     Outside that range the dynamics' phase
                                 #     term saturates at zero, killing the
                                 #     cost gradient on V_c.

    # Logit biases so θ=0 → set_A healthy operating point.
    #   sigmoid(c_h) = V_h_target / V_h_max → c_h = logit(0.25) = -ln(3)
    #   sigmoid(c_n) = V_n_target / V_n_max → c_n = logit(0.04) = -ln(24)
    c_h = float(jnp.log(jnp.asarray(0.25 / 0.75)))   # ≈ -1.0986
    c_n = float(jnp.log(jnp.asarray(0.04 / 0.96)))   # ≈ -3.1781

    # D4: tighter effective prior on V_c. The engine's sigma_prior is a
    # single scalar across all 24 θ-dims (`tempered_smc_loop.py:83`
    # casts to float; per-variate σ would need an engine change which
    # is out of scope — escalate per the senior-files principle).
    # Equivalent workaround: scale the V_c-block raw RBF output by
    # (sigma_eff_V_c / sigma_prior). With sigma_eff_V_c = 0.5 and
    # sigma_prior = 1.5, that's a factor of 1/3. Effect: V_c stays
    # closer to the unbiased centre 0 unless the data really pulls it.
    v_c_prior_scale = 1.0 / 3.0

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        """θ shape (3*n_anchors,) → (n_steps, 3) of (V_h, V_n, V_c)."""
        theta_h = theta[:n_anchors]
        theta_n = theta[n_anchors:2 * n_anchors]
        theta_c = theta[2 * n_anchors:]

        raw_h = jnp.einsum('a,ta->t', theta_h, design)
        raw_n = jnp.einsum('a,ta->t', theta_n, design)
        raw_c = jnp.einsum('a,ta->t', theta_c, design)

        V_h = V_h_max * jax.nn.sigmoid(c_h + raw_h)
        V_n = V_n_max * jax.nn.sigmoid(c_n + raw_n)
        V_c = V_c_max * jnp.tanh(v_c_prior_scale * raw_c)

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
    init_state: np.ndarray | None = None,
    params: dict | None = None,
    lambda_E: float = 1.0,
    seed: int = 42,
):
    """Build (cost_fn, traj_sample_fn) closures for SWAT control.

    Cost is `-E_w[ ∫T dt + lambda_E · ∫E_dyn dt ]`. The shaping term
    on E_dyn removes the bifurcation cliff (T flatlines below E_crit)
    and gives the controller dense, immediate gradient signal on each
    control choice (E_dyn is algebraic in V_h/V_n/V_c, no τ_T lag).
    Both terms have aligned optima at the healthy operating point;
    lambda_E=1.0 balances their per-horizon magnitudes.

    Args:
        init_state: 4-vector (W, Z, a, T) the rollout starts from.
            **Must be the actual current patient state** (e.g.
            posterior-mean of the filter). Defaults to DEFAULT_INIT
            (healthy) only as a fallback for diagnostic / unit-test
            use; in real closed-loop control the bench MUST pass the
            current posterior init or the rollout plans against the
            wrong state.
        params: SWAT param dict. **Should be the posterior-mean
            dynamics**, not truth, in closed-loop. Falls back to
            DEFAULT_PARAMS for diagnostics.
    """
    if params is None:
        params = DEFAULT_PARAMS
    if init_state is None:
        init_state = DEFAULT_INIT

    p_jax = {k: jnp.asarray(float(v)) for k, v in params.items()
              if isinstance(v, (int, float))}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)
    lambda_E_jax = jnp.float64(lambda_E)

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=4, seed=seed,
    )
    fixed_w = grids['wiener']    # (n_inner, n_steps, 4)

    init_arr = jnp.asarray(init_state, dtype=jnp.float64)

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        u_arr = schedule_from_theta(theta)    # (n_steps, 3)

        def trial(w_seq):
            def step(carry, k):
                y, T_acc, E_acc = carry
                t_k = jnp.float64(k) * dt
                u_t = u_arr[k]
                y_next = em_step(y, t_k, u_t, w_seq[k])
                E_t = entrainment_quality(
                    y[0], y[1], y[2], y[3],
                    u_t[0], u_t[1], u_t[2], p_jax)
                T_acc = T_acc + y[3] * dt
                E_acc = E_acc + E_t * dt
                return (y_next, T_acc, E_acc), None

            (_, T_acc, E_acc), _ = jax.lax.scan(
                step,
                (init_arr, jnp.float64(0.0), jnp.float64(0.0)),
                jnp.arange(n_steps),
            )
            return -(T_acc + lambda_E_jax * E_acc)

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
    lambda_E: float = 1.0,
    init_state: np.ndarray | None = None,
    params: dict | None = None,
    seed: int = 42,
) -> ControlSpec:
    """Construct a SWAT ControlSpec for the given horizon.

    Cost is `-E[∫T dt + lambda_E · ∫E_dyn dt]`. See module docstring
    for the rationale on the shaping term.

    Args:
        n_steps:    number of bins in the planning horizon.
        dt:         bin width in days.
        n_anchors:  RBF anchors per control variate. Total theta_dim
                     = 3 * n_anchors.
        n_inner:    Monte Carlo trials per cost evaluation.
        n_substeps: deterministic Euler substeps for stiffness.
        sigma_prior: std of the Gaussian prior over θ.
        lambda_E:   weight on the ∫E_dyn dt shaping term. 0 disables;
                    1.0 (default) balances ∫T and ∫E_dyn at healthy ops.
        init_state: 4-vector (W, Z, a, T) the cost rollout starts
                     from. **Must be the actual current patient
                     state** (filter posterior mean) for closed-loop
                     planning to be coherent — the cost function
                     bakes this into a JIT closure, so post-construction
                     ``object.__setattr__(spec, 'initial_state', …)``
                     does NOT update what the rollout uses. Defaults
                     to DEFAULT_INIT for diagnostic / unit-test use.
        params:     SWAT param dict (posterior-mean dynamics +
                     truth obs in closed loop). Same closure caveat
                     as ``init_state``. Defaults to DEFAULT_PARAMS.
        seed:       common-random-numbers seed for variance reduction.

    Returns:
        A frozen ControlSpec ready for the MPC.
    """
    if params is None:
        params = DEFAULT_PARAMS
    if init_state is None:
        init_state = DEFAULT_INIT

    rbf, schedule_from_theta = _make_three_schedules(
        n_steps=n_steps, dt=dt, n_anchors=n_anchors,
    )
    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt,
        n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        init_state=init_state,
        params=params,
        lambda_E=lambda_E,
        seed=seed,
    )

    spec = ControlSpec(
        name="swat",
        version="1.1.0",
        dt=dt,
        n_steps=n_steps,
        n_substeps=n_substeps,
        initial_state=jnp.asarray(init_state, dtype=jnp.float64),
        truth_params={k: float(v) for k, v in params.items()
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
