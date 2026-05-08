"""FSA v1.5 control task spec.

Single Φ control input; 8 RBF anchors over the planning horizon; sigmoid
output transform with logit-bias offset so θ at the prior mean produces
the canonical Banister Φ ≈ 1.0. Total search dim: θ ∈ ℝ^8.

Cost functional (mean over CRN MC noise grid):

    J(θ) = E_τ [ −∫ A(t) dt
                + λ_F · ∫ max(F(t) − F_max, 0)² dt ]

NO Φ² penalty (matching `version_1_5_Julia/models/fsa_high_res/gpu_control.jl`
which is "v1 verbatim" minus the λ_Φ term). The Stuart-Landau μ_FF·F²
term and the F-barrier already discourage overtraining endogenously.

Mirrors `version_1_Python_JAX/models/fsa_high_res/control.py` adapted
for the v1.5 parameter basis (B_inf, F_inf rather than κ_B, κ_F).
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import ControlSpec, RBFSchedule
from smc2fc.control.calibration import build_crn_noise_grids

from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS, INIT_STATE, params_v15_to_v1,
)


# ── Defaults ──────────────────────────────────────────────────────────
EXOGENOUS = dict(
    T_total=14.0,           # days — matches v1.5 Julia bench's 14-day open-loop
    dt_days=1.0 / 24.0,     # 1-hour outer step (matches default BINS_PER_DAY=24)
    n_substeps=4,
    F_max=0.40,
    Phi_max=3.0,
    Phi_default=1.0,
)


# ── RBF schedule decoder: θ ∈ ℝ^n_anchors → Φ(t) ──────────────────────
def _make_schedule(*, n_steps: int, dt: float, n_anchors: int = 8,
                    Phi_default: float = EXOGENOUS['Phi_default'],
                    Phi_max: float = EXOGENOUS['Phi_max']):
    """RBF basis + a JIT'd schedule_from_theta(θ) closure.

    Φ(t) = Phi_max · sigmoid(c_Φ + θ · design_matrix(t))
    With c_Φ = logit(Phi_default / Phi_max), θ = 0 ⇒ Φ ≡ Phi_default.
    """
    rbf = RBFSchedule(
        n_steps=n_steps, dt=dt, n_anchors=n_anchors, output='identity',
    )
    Phi_design = rbf.design_matrix()    # (n_steps, n_anchors)

    p_ratio = Phi_default / Phi_max
    c_Phi = float(np.log(p_ratio / (1.0 - p_ratio)))

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        raw = c_Phi + jnp.einsum('a,ta->t', theta, Phi_design)
        return Phi_max * jax.nn.sigmoid(raw)

    return rbf, schedule_from_theta


# ── EM step factory (v1.5 → v1 basis rotation done once per build) ────
def _make_em_step_fn(params_v15: dict, dt: float, n_substeps: int):
    p_v1 = params_v15_to_v1(params_v15)
    p_jax = {k: jnp.asarray(float(v)) for k, v in p_v1.items()}
    sub_dt = dt / float(n_substeps)
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def em_step(y, Phi_t, noise_3d):
        def sub_body(y_inner, _):
            return y_inner + sub_dt * drift_jax(y_inner, p_jax, Phi_t), None
        y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))
        sigma_y = diffusion_state_dep(y_det, p_jax)
        y_pred = y_det + sigma_y * sqrt_dt * noise_3d
        B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
        B_next = jnp.where(B_pred < 0.0, -B_pred,
                            jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
        F_next = jnp.abs(F_pred)
        A_next = jnp.abs(A_pred)
        return jnp.array([B_next, F_next, A_next])

    return em_step


# ── Cost + diagnostic-trajectory builders ──────────────────────────────
def _build_cost_and_traj_fns(*, n_inner: int, n_steps: int,
                              dt: float, n_substeps: int,
                              schedule_from_theta,
                              params_v15: dict,
                              init_state,
                              F_max: float = EXOGENOUS['F_max'],
                              lam_barrier: float = 1.0,
                              seed: int = 42):
    em_step = _make_em_step_fn(params_v15, dt, n_substeps)
    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=3, seed=seed,
    )
    fixed_w = grids['wiener']    # (n_inner, n_steps, 3)
    init_arr = jnp.array([float(init_state['B']),
                          float(init_state['F']),
                          float(init_state['A'])])

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)

        def trial(w_seq):
            def step(carry, k):
                y, A_acc, barrier_acc = carry
                Phi_t = Phi_arr[k]
                y_next = em_step(y, Phi_t, w_seq[k])
                A_acc = A_acc + y[2] * dt
                barrier_acc = barrier_acc + (
                    jnp.maximum(y[1] - F_max, 0.0) ** 2 * dt
                )
                return (y_next, A_acc, barrier_acc), None

            init_carry = (init_arr, jnp.float64(0.0), jnp.float64(0.0))
            (_, A_acc, barrier_acc), _ = jax.lax.scan(
                step, init_carry, jnp.arange(n_steps),
            )
            return -A_acc + lam_barrier * barrier_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)
        w_seq = jax.random.normal(key, (n_steps, 3), dtype=jnp.float64)

        def step(y, k):
            y_next = em_step(y, Phi_arr[k], w_seq[k])
            return y_next, y_next

        _, traj = jax.lax.scan(step, init_arr, jnp.arange(n_steps))
        return traj

    return cost_fn, traj_sample_fn


# ── ControlSpec factory ───────────────────────────────────────────────
def build_control_spec(*, T_total: float = EXOGENOUS['T_total'],
                         dt_days: float = EXOGENOUS['dt_days'],
                         n_substeps: int = EXOGENOUS['n_substeps'],
                         n_anchors: int = 8,
                         n_inner: int = 32,
                         Phi_max: float = EXOGENOUS['Phi_max'],
                         Phi_default: float = EXOGENOUS['Phi_default'],
                         F_max: float = EXOGENOUS['F_max'],
                         lam_barrier: float = 1.0,
                         params_v15: dict = None,
                         init_state: dict = None,
                         sigma_prior: float = 1.5,
                         seed: int = 42):
    """Build a `smc2fc.control.ControlSpec` for v1.5.

    Defaults match `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`.
    Supplying `params_v15` (e.g. a posterior-mean dict from the filter)
    swaps the truth params for closed-loop replans.
    """
    p_v15 = DEFAULT_PARAMS if params_v15 is None else params_v15
    init = INIT_STATE if init_state is None else init_state

    n_steps = int(round(T_total / dt_days))
    rbf, schedule_from_theta = _make_schedule(
        n_steps=n_steps, dt=dt_days, n_anchors=n_anchors,
        Phi_default=Phi_default, Phi_max=Phi_max,
    )
    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt_days,
        n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        params_v15=p_v15, init_state=init,
        F_max=F_max, lam_barrier=lam_barrier, seed=seed,
    )

    initial_state_jnp = jnp.array(
        [float(init['B']), float(init['F']), float(init['A'])],
        dtype=jnp.float64,
    )
    spec = ControlSpec(
        name='fsa_high_res_v15',
        version='1.5',
        dt=dt_days,
        n_steps=n_steps,
        n_substeps=n_substeps,
        initial_state=initial_state_jnp,
        truth_params={k: float(v) for k, v in p_v15.items()},
        theta_dim=n_anchors,
        sigma_prior=sigma_prior,
        prior_mean=0.0,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
    )
    # `traj_sample_fn` isn't a ControlSpec field; expose it as an
    # attribute for diagnostic-plot callers (matches v1's pattern).
    object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)
    return spec
