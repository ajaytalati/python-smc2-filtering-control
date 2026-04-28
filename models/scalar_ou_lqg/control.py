"""Scalar OU LQG control task specs.

Two ControlSpec instances:

  SCALAR_OU_OPEN_LOOP_SPEC      Stage A2: 20-D raw-pulse schedule
                                 u = (u_0, ..., u_19). Gate: SMC² mean
                                 cost / open-loop u=0 cost ∈ [0.95, 1.10].

  SCALAR_OU_STATE_FEEDBACK_SPEC Stage A3: state-feedback gain vector
                                 K = (K_0, ..., K_19) with inline Kalman.
                                 Gate: SMC² / MC LQG ∈ [0.95, 1.10],
                                 K RMS error vs Riccati < 25%.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import ControlSpec
from smc2fc.control.calibration import build_crn_noise_grids

from models.scalar_ou_lqg.bench_lqr import (
    lqr_riccati, lqr_optimal_cost,
    lqg_optimal_cost_monte_carlo,
    open_loop_zero_control_cost_monte_carlo,
)


# ── Truth parameters ──────────────────────────────────────────────────

TRUTH = dict(
    a=1.0, b=1.0, sigma_w=0.3, sigma_v=0.2,
    q=1.0, r=0.1, s=1.0,
    dt=0.05, T=20,
    x0_mean=0.0, x0_var=1.0,
)


# ── Cost evaluator: open-loop schedule (Stage A2) ─────────────────────

def _build_open_loop_cost_fn(*, n_inner: int, seed: int):
    A = 1.0 - TRUTH['a'] * TRUTH['dt']
    B = TRUTH['b'] * TRUTH['dt']
    sw = TRUTH['sigma_w'] * jnp.sqrt(TRUTH['dt'])
    T_steps = TRUTH['T']

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=T_steps, n_channels=1, seed=seed,
    )
    fixed_w = grids['wiener'][:, :, 0]
    rng = np.random.default_rng(seed + 1)
    fixed_x0 = jnp.asarray(
        TRUTH['x0_mean'] + jnp.sqrt(TRUTH['x0_var']) *
        rng.standard_normal((n_inner,)),
        dtype=jnp.float64,
    )

    @jax.jit
    def J(u):
        def trial(x0, w_seq):
            def step(carry, k):
                x, cost = carry
                u_k = u[k]
                cost_k = TRUTH['q'] * x ** 2 + TRUTH['r'] * u_k ** 2
                x_next = A * x + B * u_k + sw * w_seq[k]
                return (x_next, cost + cost_k), None
            (x_T, total_stage), _ = jax.lax.scan(
                step, (x0, jnp.float64(0.0)), jnp.arange(T_steps),
            )
            return total_stage + TRUTH['s'] * x_T ** 2
        return jnp.mean(jax.vmap(trial)(fixed_x0, fixed_w))

    return J


# ── Cost evaluator: state-feedback gain (Stage A3) ────────────────────

def _build_state_feedback_cost_fn(*, n_inner: int, seed: int):
    A = 1.0 - TRUTH['a'] * TRUTH['dt']
    B = TRUTH['b'] * TRUTH['dt']
    sw = TRUTH['sigma_w'] * jnp.sqrt(TRUTH['dt'])
    Q = TRUTH['sigma_w'] ** 2 * TRUTH['dt']
    R = TRUTH['sigma_v'] ** 2
    T_steps = TRUTH['T']

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=T_steps, n_channels=2, seed=seed,
    )
    fixed_w = grids['wiener'][:, :, 0]
    fixed_v = grids['wiener'][:, :, 1]
    rng = np.random.default_rng(seed + 1)
    fixed_x0 = jnp.asarray(
        TRUTH['x0_mean'] + jnp.sqrt(TRUTH['x0_var']) *
        rng.standard_normal((n_inner,)),
        dtype=jnp.float64,
    )

    @jax.jit
    def J(K):
        def trial(x0, w_seq, v_seq):
            def step(carry, k):
                x, x_hat_mean, x_hat_var, cost = carry
                # Kalman update
                y = x + TRUTH['sigma_v'] * v_seq[k]
                S = x_hat_var + R
                G = x_hat_var / S
                x_hat_mean_post = x_hat_mean + G * (y - x_hat_mean)
                x_hat_var_post = (1.0 - G) * x_hat_var
                # State-feedback action
                u_k = -K[k] * x_hat_mean_post
                stage = TRUTH['q'] * x ** 2 + TRUTH['r'] * u_k ** 2
                # Advance
                x_next = A * x + B * u_k + sw * w_seq[k]
                x_hat_mean_pred = A * x_hat_mean_post + B * u_k
                x_hat_var_pred = A * A * x_hat_var_post + Q
                return (x_next, x_hat_mean_pred, x_hat_var_pred,
                        cost + stage), None
            init_carry = (
                x0,
                jnp.float64(TRUTH['x0_mean']),
                jnp.float64(TRUTH['x0_var']),
                jnp.float64(0.0),
            )
            (x_T, _, _, total), _ = jax.lax.scan(
                step, init_carry, jnp.arange(T_steps),
            )
            return total + TRUTH['s'] * x_T ** 2
        return jnp.mean(jax.vmap(trial)(fixed_x0, fixed_w, fixed_v))

    return J


# ── Acceptance gates ──────────────────────────────────────────────────

def _build_open_loop_gates(cost_fn_eval):
    riccati_cfg = dict(
        a=TRUTH['a'], b=TRUTH['b'], q=TRUTH['q'], r=TRUTH['r'],
        s=TRUTH['s'], sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    init = dict(x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'])

    riccati = lqr_riccati(**riccati_cfg)
    lqr_perfect_cost = lqr_optimal_cost(
        riccati=riccati, **init,
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    lqg_mc = lqg_optimal_cost_monte_carlo(
        **riccati_cfg, sigma_v=TRUTH['sigma_v'], **init,
        n_trials=5000, seed=0,
    )['mean_cost']
    open_loop_mc = open_loop_zero_control_cost_monte_carlo(
        **riccati_cfg, **init, n_trials=5000, seed=0,
    )['mean_cost']

    def gate_open_loop_ratio(result):
        smc_cost = float(cost_fn_eval(jnp.asarray(result['mean_schedule'])))
        ratio = smc_cost / open_loop_mc
        passed = 0.95 <= ratio <= 1.10
        return passed, ratio, (
            f"SMC² cost {smc_cost:.3f} / open-loop {open_loop_mc:.3f} = "
            f"{ratio:.3f} ({'passes' if passed else 'fails'} [0.95, 1.10])"
        )

    def gate_lqr_ratio_diagnostic(result):
        smc_cost = float(cost_fn_eval(jnp.asarray(result['mean_schedule'])))
        ratio = smc_cost / lqr_perfect_cost
        # Diagnostic only: open-loop *can't* match LQR, so we don't gate.
        return True, ratio, (
            f"SMC² cost / LQR = {ratio:.3f} (diagnostic only — needs A3)"
        )

    return {
        'cost_in_[0.95, 1.10]_x_open_loop': gate_open_loop_ratio,
        'lqr_ratio_diagnostic':              gate_lqr_ratio_diagnostic,
    }, dict(
        lqr_perfect=lqr_perfect_cost, lqg_mc=lqg_mc, open_loop_mc=open_loop_mc,
        riccati_gains=np.asarray(riccati.gains),
    )


def _build_state_feedback_gates(cost_fn_eval):
    riccati_cfg = dict(
        a=TRUTH['a'], b=TRUTH['b'], q=TRUTH['q'], r=TRUTH['r'],
        s=TRUTH['s'], sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    init = dict(x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'])

    riccati = lqr_riccati(**riccati_cfg)
    lqr_perfect_cost = lqr_optimal_cost(
        riccati=riccati, **init,
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    lqg_mc = lqg_optimal_cost_monte_carlo(
        **riccati_cfg, sigma_v=TRUTH['sigma_v'], **init,
        n_trials=5000, seed=0,
    )['mean_cost']
    open_loop_mc = open_loop_zero_control_cost_monte_carlo(
        **riccati_cfg, **init, n_trials=5000, seed=0,
    )['mean_cost']
    riccati_gains = np.asarray(riccati.gains)

    def gate_lqg_ratio(result):
        K_mean = result['mean_theta']
        smc_cost = float(cost_fn_eval(jnp.asarray(K_mean)))
        ratio = smc_cost / lqg_mc
        passed = 0.95 <= ratio <= 1.10
        return passed, ratio, (
            f"SMC² cost {smc_cost:.3f} / MC LQG {lqg_mc:.3f} = {ratio:.3f} "
            f"({'passes' if passed else 'fails'} [0.95, 1.10])"
        )

    def gate_open_loop_ratio(result):
        K_mean = result['mean_theta']
        smc_cost = float(cost_fn_eval(jnp.asarray(K_mean)))
        ratio = smc_cost / open_loop_mc
        passed = ratio <= 0.7
        return passed, ratio, (
            f"SMC² cost / open-loop = {ratio:.3f} "
            f"({'passes' if passed else 'fails'} ≤ 0.7)"
        )

    def gate_K_rms_err(result):
        K_mean = np.asarray(result['mean_theta'])
        rms = float(np.sqrt(
            np.mean((K_mean - riccati_gains) ** 2)
            / np.mean(riccati_gains ** 2)
        ))
        passed = rms < 0.25
        return passed, rms, (
            f"K RMS error vs Riccati = {rms:.3f} "
            f"({'passes' if passed else 'fails'} < 0.25)"
        )

    return {
        'cost_in_[0.95, 1.10]_x_lqg':       gate_lqg_ratio,
        'cost_<=_0.7_x_open_loop':          gate_open_loop_ratio,
        'K_rms_err_<_0.25':                 gate_K_rms_err,
    }, dict(
        lqr_perfect=lqr_perfect_cost, lqg_mc=lqg_mc,
        open_loop_mc=open_loop_mc, riccati_gains=riccati_gains,
    )


# ── Build the specs ───────────────────────────────────────────────────

def build_open_loop_spec(*, n_inner: int = 64, seed: int = 42) -> ControlSpec:
    cost_fn = _build_open_loop_cost_fn(n_inner=n_inner, seed=seed)
    cost_fn_eval = _build_open_loop_cost_fn(n_inner=2000, seed=99)
    gates, refs = _build_open_loop_gates(cost_fn_eval)

    @jax.jit
    def schedule_from_theta(theta):
        return theta    # raw 20-D pulse schedule, identity transform

    spec = ControlSpec(
        name='scalar_ou_lqg_open_loop',
        version='1.0',
        dt=TRUTH['dt'],
        n_steps=TRUTH['T'],
        n_substeps=1,
        initial_state=jnp.array([TRUTH['x0_mean']]),
        truth_params={k: float(v) for k, v in TRUTH.items()
                       if isinstance(v, (int, float))},
        theta_dim=TRUTH['T'],
        sigma_prior=2.0,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates,
    )
    object.__setattr__(spec, '_refs', refs)
    object.__setattr__(spec, '_cost_eval', cost_fn_eval)
    return spec


def build_state_feedback_spec(*, n_inner: int = 64,
                                 seed: int = 42) -> ControlSpec:
    cost_fn = _build_state_feedback_cost_fn(n_inner=n_inner, seed=seed)
    cost_fn_eval = _build_state_feedback_cost_fn(n_inner=2000, seed=99)
    gates, refs = _build_state_feedback_gates(cost_fn_eval)

    @jax.jit
    def schedule_from_theta(theta):
        return theta    # the "schedule" here is the gain vector K_0..K_{T-1}

    spec = ControlSpec(
        name='scalar_ou_lqg_state_feedback',
        version='1.0',
        dt=TRUTH['dt'],
        n_steps=TRUTH['T'],
        n_substeps=1,
        initial_state=jnp.array([TRUTH['x0_mean']]),
        truth_params={k: float(v) for k, v in TRUTH.items()
                       if isinstance(v, (int, float))},
        theta_dim=TRUTH['T'],
        sigma_prior=3.0,
        # Riccati gains for this setup are positive ~ 0.5 to 2.1.
        # Center the prior at 1.5 so the SMC particle cloud starts in
        # the right neighbourhood. (Without this, prior includes
        # destabilising negative K and β_max calibration is dominated
        # by extreme costs from those samples.)
        prior_mean=1.5,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates,
    )
    object.__setattr__(spec, '_refs', refs)
    object.__setattr__(spec, '_cost_eval', cost_fn_eval)
    return spec


SCALAR_OU_OPEN_LOOP_SPEC      = build_open_loop_spec()
SCALAR_OU_STATE_FEEDBACK_SPEC = build_state_feedback_spec()
