"""Backward integration of the continuous-time differential Riccati equation.

For a finite-horizon LQR problem with time-invariant linearisation
(constant A, B), the Riccati equation is

    -dP/dt = A^T P + P A - P B R^{-1} B^T P + Q,    P(T) = Q_T

We integrate this backward in time on the project's outer grid (15-min
or 1-h depending on FSA_STEP_MINUTES) using a 4th-order Runge-Kutta
step. Each step is matrix-valued; for a 3-state FSA-v2 the Riccati
state is 3x3 and the integration is O(n_steps) with negligible cost
even at T=84d / dt=15min (~8000 steps).

For an infinite-horizon (T -> inf) and time-invariant (A, B), one would
instead solve the algebraic Riccati equation; we don't need that here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def _riccati_rhs(P, A, B, R_inv, Q):
    """RHS of the backward Riccati ODE: F(P) such that -dP/dt = F(P).

    Returned in the *forward* sign convention (so the integrator's RK4
    backward step subtracts dt * F(P)).
    """
    BRBt = B @ R_inv @ B.T
    return A.T @ P + P @ A - P @ BRBt @ P + Q


def solve_riccati_backward(A: jnp.ndarray,
                            B: jnp.ndarray,
                            Q: jnp.ndarray,
                            R: jnp.ndarray,
                            Q_T: jnp.ndarray,
                            dt: float,
                            n_steps: int) -> jnp.ndarray:
    """Integrate the differential Riccati equation backward from t=T to t=0.

    Parameters
    ----------
    A : (n_x, n_x)
        Linearised state Jacobian.
    B : (n_x, n_u)
        Linearised control Jacobian.
    Q, Q_T : (n_x, n_x)
        Running and terminal state cost matrices (PSD).
    R : (n_u, n_u)
        Control cost matrix (PD).
    dt : float
        Outer time-step (in days; matches DT_BIN_DAYS).
    n_steps : int
        Number of outer steps (= n_grid_bins for the planning horizon).

    Returns
    -------
    P_traj : (n_steps + 1, n_x, n_x)
        P[k] = P at time t_k = k * dt. P_traj[0] = P(0); P_traj[n_steps] = P(T) = Q_T.
        Stored in *forward* time order so callers can index by the same
        bin index used for the state trajectory.
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    B = jnp.asarray(B, dtype=jnp.float64)
    Q = jnp.asarray(Q, dtype=jnp.float64)
    R = jnp.asarray(R, dtype=jnp.float64)
    Q_T = jnp.asarray(Q_T, dtype=jnp.float64)

    R_inv = jnp.linalg.inv(R)

    # Backward RK4 step. We integrate dP/dt = -F(P) backward, i.e.
    # P(t-dt) = P(t) + dt * F(P(t))  (since -dP/dt = F means dP/dt = -F,
    # and stepping backward by dt corresponds to a forward step of -dt
    # in the dP/dt = -F formulation, i.e. P_new = P + dt * F(P_mid).)
    def back_step(P, _):
        k1 = _riccati_rhs(P,                A, B, R_inv, Q)
        k2 = _riccati_rhs(P + 0.5 * dt * k1, A, B, R_inv, Q)
        k3 = _riccati_rhs(P + 0.5 * dt * k2, A, B, R_inv, Q)
        k4 = _riccati_rhs(P +       dt * k3, A, B, R_inv, Q)
        P_back = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return P_back, P_back

    _, P_back_seq = jax.lax.scan(back_step, Q_T, jnp.arange(n_steps))
    # P_back_seq[k] is P at time T - (k+1)*dt for k=0..n_steps-1.
    # We want forward-time storage P_fwd[k] = P at t_k = k * dt:
    #   P_fwd[n_steps]   = Q_T
    #   P_fwd[n_steps-1] = P_back_seq[0]
    #   P_fwd[0]         = P_back_seq[n_steps-1]
    P_back_in_fwd = P_back_seq[::-1]
    P_fwd = jnp.concatenate([P_back_in_fwd, Q_T[None, ...]], axis=0)
    return P_fwd


@jax.jit
def compute_lqr_gain(P_traj: jnp.ndarray,
                      B: jnp.ndarray,
                      R: jnp.ndarray) -> jnp.ndarray:
    """K(t) = R^{-1} B^T P(t).

    Parameters
    ----------
    P_traj : (n_steps + 1, n_x, n_x)
    B      : (n_x, n_u)
    R      : (n_u, n_u)

    Returns
    -------
    K_traj : (n_steps + 1, n_u, n_x)
    """
    R_inv = jnp.linalg.inv(R)
    # K[n,u,z] = sum_{w,v} R_inv[u,w] * B[v,w] * P[n,v,z]   (B^T at index w)
    return jnp.einsum('uw,vw,nvz->nuz', R_inv, B, P_traj)
