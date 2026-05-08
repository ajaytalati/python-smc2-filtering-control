"""LQG controller wrapping linearisation + Riccati + open-loop schedule.

Two modes of use:

1. **Open-loop schedule** (used by the LQG-baseline bench):
   - Pre-compute K(t) once.
   - Roll the *nominal* deterministic trajectory x*(t) under a constant
     reference control Phi_ref (e.g. the deterministic skeleton at
     Phi_default = 1.0). This serves as the linearisation trajectory.
   - Emit Phi_open(t) = Phi_ref - K(t) (x*(t) - x_ref).
   - Apply Phi_open(t) to the StepwisePlant; compare to the SMC^2
     closed-loop result.

2. **Closed-loop feedback** (separation principle):
   - At each replan stride, take the SMC^2 posterior-mean state x_hat,
     compute Phi(t) = Phi_ref - K(t) (x_hat(t) - x_ref) using the
     pre-computed K(t).
   - Cheap: K(t) is computed once and cached.

We expose both via `LQGController.open_loop_schedule(...)` and
`LQGController.feedback_phi(x_hat, t_idx)`.

Cost design (Section 8.8):
- Q gives weight to A only (we want max int(A); B and F are not
  directly in the reward).
- Q_F adds a soft penalty around F_ref < F_max so the controller
  respects the F-barrier (symmetric quadratic, less efficient than the
  asymmetric soft-plus barrier in the SMC^2 cost).
- R penalises Phi deviation from Phi_ref so the policy stays close
  to the nominal regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control.lqg.linearize import linearize_drift_at
from smc2fc.control.lqg.riccati import (
    solve_riccati_backward, compute_lqr_gain,
)


# =========================================================================
# Spec dataclass
# =========================================================================

@dataclass(frozen=True)
class LQGSpec:
    """Configuration for an LQG controller.

    All matrices in the natural model time-units (drift in 1/days);
    `dt` is the outer time-step (1/BINS_PER_DAY days).
    """
    # Operating point
    x_star:    np.ndarray   # shape (n_x,)
    phi_star:  float
    # Reference cost shifts
    x_ref:     np.ndarray   # shape (n_x,)  - cost penalises (x - x_ref)
    phi_ref:   float        # cost penalises (Phi - phi_ref)
    # Cost matrices
    Q:         np.ndarray   # (n_x, n_x)
    R:         np.ndarray   # (n_u, n_u)  with n_u = 1 for FSA-v2
    Q_T:       np.ndarray   # (n_x, n_x)
    # Time grid
    dt:        float
    n_steps:   int
    # Box constraints (applied AFTER computing the LQR feedback Phi)
    phi_min:   float = 0.0
    phi_max:   float = 3.0


# =========================================================================
# Controller
# =========================================================================

class LQGController:
    """Builds K(t), holds the linearisation, emits open-loop / feedback Phi."""

    def __init__(self, spec: LQGSpec, drift_jax: Callable, params: dict):
        self.spec = spec
        self.drift_jax = drift_jax
        self.params = params

        # 1. Linearise drift at operating point.
        A_lin, B_lin = linearize_drift_at(
            drift_jax,
            x_star=jnp.asarray(spec.x_star, dtype=jnp.float64),
            phi_star=float(spec.phi_star),
            params=params,
        )
        self.A_lin = np.asarray(A_lin)
        self.B_lin = np.asarray(B_lin)

        # 2. Solve Riccati backward.
        P_traj = solve_riccati_backward(
            A=A_lin, B=B_lin,
            Q=spec.Q, R=spec.R, Q_T=spec.Q_T,
            dt=float(spec.dt), n_steps=int(spec.n_steps),
        )
        self.P_traj = np.asarray(P_traj)

        # 3. Compute K(t) = R^{-1} B^T P(t).
        K_traj = compute_lqr_gain(
            P_traj=P_traj,
            B=B_lin,
            R=jnp.asarray(spec.R, dtype=jnp.float64),
        )
        self.K_traj = np.asarray(K_traj)   # (n_steps + 1, n_u, n_x)

    # ---------------------------------------------------------------------
    # Mode 1: open-loop schedule
    # ---------------------------------------------------------------------

    def nominal_trajectory(self) -> np.ndarray:
        """Deterministic-skeleton trajectory under Phi == phi_ref starting at x_star.

        Used as the linearisation trajectory x*(t) for open-loop schedule
        construction. Returns shape (n_steps + 1, n_x).
        """
        dt = float(self.spec.dt)
        x = np.asarray(self.spec.x_star, dtype=np.float64).copy()
        traj = [x.copy()]
        p_jax = {k: float(v) for k, v in self.params.items()}
        # Use the supplied drift via jax to keep parity with linearisation,
        # but advance under the ode itself (drift only, no diffusion).
        @jax.jit
        def step(y, phi):
            return y + dt * self.drift_jax(
                jnp.asarray(y, dtype=jnp.float64),
                {k: jnp.asarray(v, dtype=jnp.float64) for k, v in p_jax.items()},
                jnp.asarray(phi, dtype=jnp.float64),
            )
        for _ in range(int(self.spec.n_steps)):
            x = np.asarray(step(x, self.spec.phi_ref))
            traj.append(x.copy())
        return np.asarray(traj)

    def open_loop_schedule(self) -> np.ndarray:
        """Return the open-loop Phi schedule of length `n_steps`.

        Phi(t) = phi_ref - K(t) (x_nominal(t) - x_ref), clipped to
        [phi_min, phi_max].
        """
        x_nom = self.nominal_trajectory()    # (n_steps + 1, n_x)
        x_ref = np.asarray(self.spec.x_ref, dtype=np.float64)
        deltas = x_nom - x_ref[None, :]      # (n_steps + 1, n_x)
        # K_traj[k] has shape (n_u, n_x); deltas[k] has shape (n_x,)
        # K_traj[k] @ deltas[k] is shape (n_u,) — for FSA-v2 n_u=1
        K_dot = np.einsum('nuz,nz->nu', self.K_traj, deltas)   # (n_steps+1, n_u)
        phi = self.spec.phi_ref - K_dot[:, 0]                  # scalar Phi
        phi = np.clip(phi, self.spec.phi_min, self.spec.phi_max)
        return phi[:int(self.spec.n_steps)]

    # ---------------------------------------------------------------------
    # Mode 2: closed-loop feedback (separation principle)
    # ---------------------------------------------------------------------

    def feedback_phi(self, x_hat: np.ndarray, t_idx: int) -> float:
        """Compute Phi at bin t_idx given current posterior-mean state x_hat."""
        K = self.K_traj[int(t_idx)]                            # (n_u, n_x)
        x_ref = np.asarray(self.spec.x_ref, dtype=np.float64)
        delta = np.asarray(x_hat, dtype=np.float64) - x_ref
        phi = float(self.spec.phi_ref - (K @ delta)[0])
        return float(np.clip(phi, self.spec.phi_min, self.spec.phi_max))


# =========================================================================
# Convenience: build an open-loop schedule from drift + cost weights
# =========================================================================

def build_lqg_open_loop_schedule(*,
                                   drift_jax: Callable,
                                   params: dict,
                                   x_star: np.ndarray,
                                   phi_star: float,
                                   x_ref: np.ndarray,
                                   phi_ref: float,
                                   Q: np.ndarray,
                                   R: np.ndarray,
                                   Q_T: np.ndarray,
                                   dt: float,
                                   n_steps: int,
                                   phi_min: float = 0.0,
                                   phi_max: float = 3.0,
                                   ) -> tuple[np.ndarray, LQGController]:
    """One-shot helper: builds the controller and returns the open-loop Phi.

    Returns (phi_schedule of length n_steps, controller object for diagnostics).
    """
    spec = LQGSpec(
        x_star=np.asarray(x_star, dtype=np.float64),
        phi_star=float(phi_star),
        x_ref=np.asarray(x_ref, dtype=np.float64),
        phi_ref=float(phi_ref),
        Q=np.asarray(Q, dtype=np.float64),
        R=np.asarray(R, dtype=np.float64),
        Q_T=np.asarray(Q_T, dtype=np.float64),
        dt=float(dt), n_steps=int(n_steps),
        phi_min=phi_min, phi_max=phi_max,
    )
    ctrl = LQGController(spec, drift_jax, params)
    phi = ctrl.open_loop_schedule()
    return phi, ctrl
