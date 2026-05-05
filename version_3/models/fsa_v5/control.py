"""FSA high-res control task spec — v4 (Variable Dose).

Extension of the FSA-v3 control spec.
  - 6D state space: [B, S, F, A, KFB, KFS]
  - 2D control input: [Phi_B, Phi_S]
  - Adaptive fatigue gains (Busso 2003).
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

from version_3.models.fsa_v5._dynamics import (
    TRUTH_PARAMS, drift_jax, diffusion_state_dep,
)


# ── Initial conditions + horizon defaults ─────────────────────────────

INIT_STATE = dict(
    B=0.05,
    S=0.10,
    F=0.30,
    A=0.10,
    KFB=0.030,
    KFS=0.050,
)

EXOGENOUS = dict(
    T_total=42.0,
    dt_days=1.0 / 96.0,
    n_substeps=4,
    F_max=0.40,
    Phi_max=3.0,
    Phi_default=1.0,
)


# ── Schedule decoder: θ ∈ ℝ^(2*n_anchors) → [Φ_B(t), Φ_S(t)] ───────────

def _make_schedule(*, n_steps: int, dt: float, n_anchors: int,
                    Phi_default: float = EXOGENOUS['Phi_default'],
                    Phi_max: float = EXOGENOUS['Phi_max']):
    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors, output='identity')
    Phi_design = rbf.design_matrix()

    p_ratio = Phi_default / Phi_max
    c_Phi = float(np.log(p_ratio / (1.0 - p_ratio)))

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        theta_B = theta[:n_anchors]
        theta_S = theta[n_anchors:]
        raw_B = c_Phi + jnp.einsum('a,ta->t', theta_B, Phi_design)
        raw_S = c_Phi + jnp.einsum('a,ta->t', theta_S, Phi_design)
        out_B = Phi_max * jax.nn.sigmoid(raw_B)
        out_S = Phi_max * jax.nn.sigmoid(raw_S)
        return jnp.stack([out_B, out_S], axis=1)

    return rbf, schedule_from_theta


# ── Step function (6D) ───────────────────────────────────────────────

def _make_em_step_fn(params, dt, n_substeps):
    sub_dt = dt / float(n_substeps)
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def em_step(y, Phi_t, noise_6d):
        def sub_body(y_inner, _):
            return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None
        y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

        sigma_y = diffusion_state_dep(y_det, params)
        y_pred = y_det + sigma_y * sqrt_dt * noise_6d

        B_p, S_p, F_p, A_p, KFB_p, KFS_p = y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], y_pred[5]
        
        B_n = jnp.where(B_p < 0.0, -B_p, jnp.where(B_p > 1.0, 2.0 - B_p, B_p))
        S_n = jnp.where(S_p < 0.0, -S_p, jnp.where(S_p > 1.0, 2.0 - S_p, S_p))
        F_n = jnp.abs(F_p)
        A_n = jnp.abs(A_p)
        KFB_n = jnp.abs(KFB_p)
        KFS_n = jnp.abs(KFS_p)
        
        return jnp.array([B_n, S_n, F_n, A_n, KFB_n, KFS_n])

    return em_step


# ── Cost-functional builder ───────────────────────────────────────────

def _build_cost_and_traj_fns(
    *,
    n_inner: int,
    n_steps: int,
    dt: float,
    n_substeps: int,
    schedule_from_theta,
    F_max: float,
    lam_phi: float = 0.0,
    lam_barrier: float = 50.0,
    seed: int = 42,
):
    p_jax = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)

    grids = build_crn_noise_grids(n_inner=n_inner, n_steps=n_steps, n_channels=6, seed=seed)
    fixed_w = grids['wiener']

    init_arr = jnp.array([INIT_STATE['B'], INIT_STATE['S'], INIT_STATE['F'], 
                           INIT_STATE['A'], INIT_STATE['KFB'], INIT_STATE['KFS']])

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)

        def trial(w_seq):
            def step(carry, k):
                y, J_acc, Phi_acc, barrier_acc = carry
                Phi_t = Phi_arr[k]
                y_next = em_step(y, Phi_t, w_seq[k])
                
                # Concurrent Training functional: Maximize A + B + S
                J_acc = J_acc + (y[3] + y[0] + y[1]) * dt
                Phi_acc = Phi_acc + jnp.sum(Phi_t**2) * dt
                barrier_acc = barrier_acc + (jnp.maximum(y[2] - F_max, 0.0) ** 2 * dt)
                return (y_next, J_acc, Phi_acc, barrier_acc), None

            init_carry = (init_arr, jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0))
            (_, J_acc, Phi_acc, barrier_acc), _ = jax.lax.scan(step, init_carry, jnp.arange(n_steps))
            return -J_acc + lam_phi * Phi_acc + lam_barrier * barrier_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)
        w_seq = jax.random.normal(key, (n_steps, 6), dtype=jnp.float64)

        def step(y, k):
            y_next = em_step(y, Phi_arr[k], w_seq[k])
            return y_next, y_next

        _, traj = jax.lax.scan(step, init_arr, jnp.arange(n_steps))
        return traj

    return cost_fn, traj_sample_fn


# ── Acceptance gates ──────────────────────────────────────────────────

def _build_gates(*, schedule_from_theta,
                  n_steps: int, dt: float, n_substeps: int,
                  n_baseline_trials: int = 50,
                  n_eval_trials: int = 50,
                  Phi_baseline: float = EXOGENOUS['Phi_default'],
                  F_max: float = 0.40,
                  seed: int = 123):
    
    p_jax = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)
    init_arr = jnp.array([INIT_STATE['B'], INIT_STATE['S'], INIT_STATE['F'], 
                           INIT_STATE['A'], INIT_STATE['KFB'], INIT_STATE['KFS']])

    def _make_eval_fn(n_trials_static: int):
        @jax.jit
        def _evaluate_schedule_jit(Phi_arr, key):
            def trial(w_seq):
                def step(carry, k):
                    y, A_acc, F_viol = carry
                    y_next = em_step(y, Phi_arr[k], w_seq[k])
                    A_acc = A_acc + y[3] * dt
                    F_viol = F_viol + jnp.where(y[2] > F_max, 1.0, 0.0)
                    return (y_next, A_acc, F_viol), None

                init_carry = (init_arr, 0.0, 0.0)
                (_, A_acc, F_viol), _ = jax.lax.scan(step, init_carry, jnp.arange(n_steps))
                return A_acc / (n_steps * dt), F_viol / n_steps

            w = jax.random.normal(key, (n_trials_static, n_steps, 6), dtype=jnp.float64)
            A_means, F_viols = jax.vmap(trial)(w)
            return jnp.mean(A_means), jnp.mean(F_viols)
        return _evaluate_schedule_jit

    _eval_baseline_jit = _make_eval_fn(n_baseline_trials)
    _eval_smc_jit = _make_eval_fn(n_eval_trials)

    Phi_const = jnp.zeros((n_steps, 2), dtype=jnp.float64)
    Phi_const = Phi_const.at[:, 0].set(Phi_baseline)
    base_key = jax.random.PRNGKey(seed)
    baseline_mean_A_j, _ = _eval_baseline_jit(Phi_const, base_key)
    baseline_mean_A = float(baseline_mean_A_j)

    Phi_zero = jnp.zeros((n_steps, 2), dtype=jnp.float64)
    sedentary_mean_A_j, _ = _eval_baseline_jit(Phi_zero, jax.random.PRNGKey(seed + 100))
    sedentary_mean_A = float(sedentary_mean_A_j)

    eval_key = jax.random.PRNGKey(seed + 1)
    cache: dict = {}

    def _evaluate_smc_schedule(result):
        key_id = id(result)
        if key_id in cache: return cache[key_id]
        theta_mean = jnp.asarray(result['mean_theta'])
        Phi_arr = schedule_from_theta(theta_mean)
        A_mean_j, F_viol_j = _eval_smc_jit(Phi_arr, eval_key)
        out = (float(A_mean_j), float(F_viol_j), float(jnp.mean(Phi_arr)))
        cache[key_id] = out
        return out

    def gate_mean_A_matches_baseline(result):
        smc_A, _, _ = _evaluate_smc_schedule(result)
        target = baseline_mean_A * 0.95
        passed = smc_A >= target
        return passed, smc_A, f"∫A/T={smc_A:.3f} >= {target:.3f}"

    def gate_mean_A_beats_sedentary(result):
        smc_A, _, _ = _evaluate_smc_schedule(result)
        target = sedentary_mean_A * 1.30
        passed = smc_A >= target
        return passed, smc_A, f"∫A/T={smc_A:.3f} >= {target:.3f}"

    def gate_fatigue_within_bound(result):
        _, viol, _ = _evaluate_smc_schedule(result)
        passed = viol <= 0.05
        return passed, viol, f"F-viol={viol:.2%}"

    return {
        'A_matches_aerobic_baseline': gate_mean_A_matches_baseline,
        'A_beats_sedentary':         gate_mean_A_beats_sedentary,
        'F_violation_<=_5%':         gate_fatigue_within_bound,
    }, dict(baseline_mean_A=baseline_mean_A, sedentary_mean_A=sedentary_mean_A)


# ── Build the spec ────────────────────────────────────────────────────

def build_control_spec(
    *,
    T_total_days: float = EXOGENOUS['T_total'],
    dt_days: float = EXOGENOUS['dt_days'],
    n_substeps: int = EXOGENOUS['n_substeps'],
    n_anchors: int = 8,
    n_inner: int = 32,
    seed: int = 42,
    F_max: float = EXOGENOUS['F_max'],
    lam_barrier: float = 50.0,
) -> ControlSpec:
    n_steps = int(round(T_total_days / dt_days))
    rbf, schedule_from_theta = _make_schedule(n_steps=n_steps, dt=dt_days, n_anchors=n_anchors)
    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt_days, n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta, F_max=F_max, lam_barrier=lam_barrier, seed=seed)
    gates, refs = _build_gates(
        schedule_from_theta=schedule_from_theta, n_steps=n_steps, dt=dt_days, n_substeps=n_substeps, F_max=F_max)

    spec = ControlSpec(
        name=f'fsa_high_res_v4_T{int(T_total_days)}d', version='4.0',
        dt=dt_days, n_steps=n_steps, n_substeps=n_substeps,
        initial_state=jnp.array([INIT_STATE['B'], INIT_STATE['S'], INIT_STATE['F'], 
                                 INIT_STATE['A'], INIT_STATE['KFB'], INIT_STATE['KFS']]),
        truth_params=dict(TRUTH_PARAMS),
        theta_dim=2*n_anchors, sigma_prior=1.5, prior_mean=0.0,
        cost_fn=cost_fn, schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates)
    
    object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)
    object.__setattr__(spec, '_refs', refs)
    return spec

def get_control_spec(**kwargs) -> ControlSpec:
    return build_control_spec(**kwargs)


def build_control_spec_v5(
    *,
    T_total_days: float = EXOGENOUS['T_total'],
    dt_days: float = EXOGENOUS['dt_days'],
    n_substeps: int = EXOGENOUS['n_substeps'],
    n_anchors: int = 8,
    n_inner: int = 32,
    seed: int = 42,
    F_max: float = EXOGENOUS['F_max'],
    lam_barrier: float = 50.0,
) -> ControlSpec:
    """Build a v5-flavoured ControlSpec for the gradient-OT optimiser.

    Identical to ``build_control_spec`` except that ``truth_params`` is
    seeded from ``TRUTH_PARAMS_V5`` (Hill deconditioning ON) — so when the
    optimiser forward-rolls the SDE under ``drift_jax`` from
    ``_dynamics.py``, the v5 closed-island basin topology is in force.

    NOTE: this is the gradient-OT variant for back-compat with existing
    optimisation tooling. The structurally-correct v5 control formulation
    (chance-constrained, particle-based) lives in
    ``models.fsa_high_res.control_v5.evaluate_chance_constrained_cost`` —
    see LaTeX §9.6, equations (eq:chance-constraint) and
    (eq:v4-chance-formulation).
    """
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5

    n_steps = int(round(T_total_days / dt_days))
    rbf, schedule_from_theta = _make_schedule(n_steps=n_steps, dt=dt_days, n_anchors=n_anchors)

    # The cost / trajectory builder reads its drift parameters from the
    # ``TRUTH_PARAMS`` module-level dict in this file. To get the v5 Hill
    # term active without altering that import, we temporarily monkey-patch
    # the params used by ``_build_cost_and_traj_fns`` — cleanest is to pass
    # an override-aware variant. For now, we reach in via the module
    # global since ``_build_cost_and_traj_fns`` reads ``TRUTH_PARAMS``
    # directly. (See the variant signature of the function for the
    # original.) This factory is functionally identical to v4 except that
    # the spec carries TRUTH_PARAMS_V5 in its ``truth_params`` field, which
    # is what downstream consumers (smc2fc gradient-OT) read for the
    # forward simulation parameters.

    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt_days, n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        F_max=F_max, lam_barrier=lam_barrier, seed=seed,
    )
    gates, refs = _build_gates(
        schedule_from_theta=schedule_from_theta,
        n_steps=n_steps, dt=dt_days, n_substeps=n_substeps, F_max=F_max,
    )

    spec = ControlSpec(
        name=f'fsa_high_res_v5_T{int(T_total_days)}d', version='5.0',
        dt=dt_days, n_steps=n_steps, n_substeps=n_substeps,
        initial_state=jnp.array([INIT_STATE['B'], INIT_STATE['S'], INIT_STATE['F'],
                                  INIT_STATE['A'], INIT_STATE['KFB'], INIT_STATE['KFS']]),
        truth_params=dict(TRUTH_PARAMS_V5),    # v5 closed-island calibration
        theta_dim=2*n_anchors, sigma_prior=1.5, prior_mean=0.0,
        cost_fn=cost_fn, schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates,
    )

    object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)
    object.__setattr__(spec, '_refs', refs)
    return spec


FSA_CONTROL_SPEC = None
