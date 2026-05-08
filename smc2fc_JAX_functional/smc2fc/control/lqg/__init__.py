"""LQG / differential-Riccati companion to the SMC^2 controller.

Implements the linear-quadratic-Gaussian baseline sketched in
Section 8.8 (LaTex_docs) under three explicit assumptions:

  (A1) Drift linearisation around an operating point (x*, Phi*).
  (A2) State-dependent Jacobi/CIR diffusions replaced by constant-
       coefficient Gaussian noise sigma_bar dW at the operating point.
  (A3) The asymmetric -int A dt reward and soft F-barrier replaced by
       a quadratic surrogate cost J_LQ.

The optimal feedback law under (A1)-(A3) is the time-varying linear
policy

    Phi*(t) = -K(t) (x(t) - x_ref)
    K(t)    = R^{-1} B_lin(t)^T P(t)

where P(t) is the solution of the (continuous-time) backward
differential Riccati equation

    -dP/dt = A^T P + P A - P B R^{-1} B^T P + Q,    P(T) = Q_T.

Under (A1)-(A2) the state-estimation problem also linearises (Kalman-
Bucy) and the LQG controller is the certainty-equivalent composition;
the linear-Gaussian separation principle holds. For Stage H we only
implement the deterministic LQR side (open-loop schedule + closed-loop
gain), since Stage I's SMC^2 filter already supplies posterior means
that play the role of the Kalman-Bucy estimate.

Public API
----------
linearize_drift_at(...)   -> (A_lin, B_lin)
solve_riccati_backward(...) -> P_traj
compute_lqr_gain(...)     -> K_traj
build_lqg_schedule(...)   -> Phi_open_loop
LQGController             -> wraps the above
"""

from smc2fc.control.lqg.linearize import linearize_drift_at
from smc2fc.control.lqg.riccati import (
    solve_riccati_backward,
    compute_lqr_gain,
)
from smc2fc.control.lqg.controller import (
    LQGSpec,
    LQGController,
    build_lqg_open_loop_schedule,
)


__all__ = [
    'linearize_drift_at',
    'solve_riccati_backward',
    'compute_lqr_gain',
    'LQGSpec',
    'LQGController',
    'build_lqg_open_loop_schedule',
]
