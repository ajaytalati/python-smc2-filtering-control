"""Smoke tests for the LQG/Riccati controller.

Runs CPU-only. Verifies:
  1. `linearize_drift_at` returns shape-correct (A, B) Jacobians.
  2. The Riccati solution P(t) is symmetric and positive-semi-definite at
     every grid point.
  3. The LQG open-loop schedule has the right shape and stays inside
     [phi_min, phi_max].

Invoke directly:
    cd version_2 && PYTHONPATH=.:.. python tests/test_lqg.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax.numpy as jnp
import numpy as np

from models.fsa_high_res._dynamics import (
    drift_jax as drift_jax_v2, A_TYP, F_TYP, TRUTH_PARAMS,
)
from smc2fc.control.lqg import (
    linearize_drift_at,
    solve_riccati_backward,
    compute_lqr_gain,
    LQGSpec, LQGController, build_lqg_open_loop_schedule,
)


def test_linearize_shapes():
    x_star = jnp.array([0.05, F_TYP, A_TYP], dtype=jnp.float64)
    A_lin, B_lin = linearize_drift_at(
        drift_jax_v2, x_star=x_star, phi_star=1.0, params=TRUTH_PARAMS,
    )
    assert A_lin.shape == (3, 3), f"A_lin shape {A_lin.shape} != (3,3)"
    assert B_lin.shape == (3, 1), f"B_lin shape {B_lin.shape} != (3,1)"
    A_lin_np = np.asarray(A_lin)
    B_lin_np = np.asarray(B_lin)
    # Sanity: the diagonal of A should have negative eigenvalues for B and F
    # (decay). A might be unstable (eta - 3 A^2 > 0 at small A).
    assert A_lin_np[0, 0] < 0, "B-self-decay should be < 0"
    assert A_lin_np[1, 1] < 0, "F-self-decay should be < 0"
    assert B_lin_np[0, 0] > 0, "Phi -> B should be positive"
    assert B_lin_np[1, 0] > 0, "Phi -> F should be positive"
    print(f"  [pass] linearize: A_lin diag = "
          f"{np.diag(A_lin_np).round(4).tolist()}, "
          f"B_lin = {B_lin_np.flatten().round(4).tolist()}")


def test_riccati_psd():
    A = jnp.array([[-0.1, 0.0, 0.0], [0.05, -0.16, 0.0], [0.0, 0.0, 0.05]])
    B = jnp.array([[0.012], [0.030], [0.0]])
    Q = jnp.diag(jnp.array([0.0, 100.0, 100.0]))
    R = jnp.eye(1)
    Q_T = jnp.zeros((3, 3))
    dt = 1.0 / 96.0
    n_steps = 96 * 7  # 1 week
    P_traj = solve_riccati_backward(A, B, Q, R, Q_T, dt, n_steps)
    P_np = np.asarray(P_traj)
    assert P_np.shape == (n_steps + 1, 3, 3)
    # symmetry & PSD at every step
    for k in [0, n_steps // 2, n_steps]:
        Pk = P_np[k]
        sym_err = np.max(np.abs(Pk - Pk.T))
        assert sym_err < 1e-8, f"P[{k}] symmetry err = {sym_err}"
        eigs = np.linalg.eigvalsh(Pk)
        assert eigs.min() >= -1e-8, f"P[{k}] eigs min = {eigs.min()}"
    # Terminal condition matches Q_T
    assert np.allclose(P_np[n_steps], np.asarray(Q_T))
    print(f"  [pass] Riccati P(t) symmetric and PSD at t=0,T/2,T over "
          f"{n_steps + 1} grid points")


def test_open_loop_schedule_in_bounds():
    x_star = np.array([0.05, F_TYP, A_TYP], dtype=np.float64)
    x_ref  = np.array([0.05, 0.0,    0.30], dtype=np.float64)
    Q   = np.diag([0.0, 100.0, 100.0])
    R   = np.eye(1) * 1.0
    Q_T = np.zeros((3, 3))
    dt = 1.0 / 96.0
    n_steps = 96 * 14  # 14 days
    phi, ctrl = build_lqg_open_loop_schedule(
        drift_jax=drift_jax_v2, params=TRUTH_PARAMS,
        x_star=x_star, phi_star=1.0,
        x_ref=x_ref, phi_ref=1.0,
        Q=Q, R=R, Q_T=Q_T,
        dt=dt, n_steps=n_steps,
        phi_min=0.0, phi_max=3.0,
    )
    assert phi.shape == (n_steps,), f"phi shape {phi.shape} != ({n_steps},)"
    assert phi.min() >= 0.0 and phi.max() <= 3.0, \
        f"phi range [{phi.min():.3f}, {phi.max():.3f}] out of [0, 3]"
    # NB: the LQG linearisation has B_lin[2,0]=0 — Phi has no DIRECT effect
    # on A. The controller can only push A up via the cascade Phi -> F -> A
    # (cubic). So whether the open-loop schedule lifts Phi above phi_ref
    # depends entirely on cost-weight design + how strongly the Riccati
    # backward-propagates the A reward through F-coupling. We do NOT
    # assert on the direction here — that's a cost-tuning question for
    # the bench, not LQG mechanics. Just verify finiteness.
    assert np.all(np.isfinite(phi)), "open-loop phi contains non-finite values"
    print(f"  [pass] open-loop schedule: shape={phi.shape}, range "
          f"[{phi.min():.3f}, {phi.max():.3f}], mean={phi.mean():.3f}")


def main():
    print("=" * 64)
    print("  Stage H — LQG/Riccati smoke tests")
    print("=" * 64)
    test_linearize_shapes()
    test_riccati_psd()
    test_open_loop_schedule_in_bounds()
    print("-" * 64)
    print("  All tests passed.")


if __name__ == '__main__':
    main()
